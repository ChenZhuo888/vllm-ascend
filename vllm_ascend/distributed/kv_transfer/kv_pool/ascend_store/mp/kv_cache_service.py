"""KV cache service orchestration independent of the RPC transport."""

import hashlib
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from .kv_cache_error import ServiceNotRegisteredError
from .registration import (
    SchedulerFactory,
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)
from .service import ServiceLifecycleManager

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

_LOOKUP_COORDINATOR_RANK = 0
_SERVICE_LEASE_TIMEOUT_S = 60.0
_LEASE_CHECK_INTERVAL_S = 5.0


class KVCacheServiceManager:
    """Own KV cache services and coordinate calls between them."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
        lease_timeout_s: float = _SERVICE_LEASE_TIMEOUT_S,
        lease_check_interval_s: float = _LEASE_CHECK_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._scheduler_factory = scheduler_factory or self._create_scheduler
        self._worker_factory = worker_factory or self._create_worker
        self._binding_lock = threading.Lock()
        self._schedulers = ServiceLifecycleManager[SchedulerIdentity, "KVPoolScheduler"](
            "Scheduler",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-scheduler-lifecycle",
        )
        self._workers = ServiceLifecycleManager[WorkerIdentity, "KVPoolWorker"](
            "Worker",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-worker-lifecycle",
        )

    @property
    def scheduler_count(self) -> int:
        return self._schedulers.count

    @property
    def worker_count(self) -> int:
        return self._workers.count

    @staticmethod
    def _create_scheduler(
        registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler
    ) -> "KVPoolScheduler":
        from .lookup_worker import MPKVPoolScheduler

        return MPKVPoolScheduler(registration, lookup_handler)

    @staticmethod
    def _create_worker(registration: WorkerRegistration) -> "KVPoolWorker":
        from .lookup_worker import LookupKVPoolWorker

        return LookupKVPoolWorker(
            registration.vllm_config,
            kv_cache_config=registration.kv_cache_config,
            rank=registration.identity.rank,
        )

    def _build_scheduler(self, registration: SchedulerRegistration) -> "KVPoolScheduler":
        return self._scheduler_factory(registration, self._lookup_worker)

    def start(self) -> None:
        self._schedulers.start()
        self._workers.start()

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> "KVPoolScheduler":
        self._validate_scheduler_registration(registration)
        scheduler = self._schedulers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._build_scheduler(registration),
        )
        self._bind_lookup_store(registration.identity)
        return scheduler

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "KVPoolWorker":
        self._validate_worker_registration(registration)
        worker = self._workers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._worker_factory(registration),
        )
        self._bind_lookup_store(
            SchedulerIdentity(registration.identity.engine_id, registration.identity.data_parallel_rank)
        )
        return worker

    def unregister_scheduler(self, identity: SchedulerIdentity, session_id: str) -> bool:
        return self._schedulers.unregister(identity, session_id)

    def unregister_worker(self, identity: WorkerIdentity, session_id: str) -> bool:
        return self._workers.unregister(identity, session_id)

    def renew_scheduler(self, identity: SchedulerIdentity, session_id: str) -> None:
        if not self._schedulers.renew(identity, session_id):
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

    def renew_worker(self, identity: WorkerIdentity, session_id: str) -> None:
        if not self._workers.renew(identity, session_id):
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")

    def lookup(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        scheduler = self._schedulers.get_active(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def close(self) -> None:
        self._workers.close()
        self._schedulers.close()

    def _lookup_worker(
        self,
        scheduler_identity: SchedulerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker_identity = self._get_lookup_worker_identity(scheduler_identity)
        worker = self._workers.get(worker_identity)
        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.lookup_scheduler(token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens)

    def _bind_lookup_store(self, scheduler_identity: SchedulerIdentity) -> None:
        with self._binding_lock:
            scheduler = self._schedulers.get(scheduler_identity)
            worker = self._workers.get(self._get_lookup_worker_identity(scheduler_identity))
            if scheduler is None or worker is None:
                return

            store = getattr(scheduler, "store_scheduler", None)
            if store is None:
                return

            bind_lookup_store = getattr(worker, "bind_lookup_store", None)
            if callable(bind_lookup_store):
                bind_lookup_store(store)

    @staticmethod
    def _get_lookup_worker_identity(scheduler_identity: SchedulerIdentity) -> WorkerIdentity:
        return WorkerIdentity(
            scheduler_identity.engine_id,
            rank=_LOOKUP_COORDINATOR_RANK,
            data_parallel_rank=scheduler_identity.data_parallel_rank,
        )

    @staticmethod
    def _close_service(service: object) -> None:
        close = getattr(service, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _validate_scheduler_registration(registration: SchedulerRegistration) -> None:
        expected_identity = SchedulerIdentity.from_vllm_config(registration.vllm_config)
        if registration.identity != expected_identity:
            raise ValueError(
                f"Scheduler identity does not match VllmConfig: {registration.identity!r} != {expected_identity!r}"
            )

    @staticmethod
    def _validate_worker_registration(registration: WorkerRegistration) -> None:
        expected_identity = WorkerIdentity.from_vllm_config(registration.vllm_config)
        if registration.identity != expected_identity:
            raise ValueError(
                f"Worker identity does not match VllmConfig: {registration.identity!r} != {expected_identity!r}"
            )
