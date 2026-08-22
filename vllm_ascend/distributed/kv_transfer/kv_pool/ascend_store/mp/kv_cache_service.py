"""KV cache service orchestration independent of the RPC transport."""

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from .kv_cache_error import ServiceNotRegisteredError
from .kv_cache_registry import KVCacheServiceRegistry
from .registration import (
    SchedulerFactory,
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

_LOOKUP_COORDINATOR_RANK = 0


class KVCacheService:
    """Own KV cache services and coordinate calls between them."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
    ):
        self._scheduler_factory = scheduler_factory or self._create_scheduler
        self._worker_factory = worker_factory or self._create_worker
        self._binding_lock = threading.Lock()
        self._registry = KVCacheServiceRegistry(self._build_scheduler, self._worker_factory)

    @property
    def scheduler_count(self) -> int:
        return self._registry.scheduler_count

    @property
    def worker_count(self) -> int:
        return self._registry.worker_count

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

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> None:
        self._registry.register_scheduler(registration, payload)
        self._bind_lookup_store(registration.identity)

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> None:
        self._registry.register_worker(registration, payload)
        self._bind_lookup_store(
            SchedulerIdentity(registration.identity.engine_id, registration.identity.data_parallel_rank)
        )

    def unregister_scheduler(self, identity: SchedulerIdentity, session_id: str) -> None:
        self._registry.unregister_scheduler(identity, session_id)

    def unregister_worker(self, identity: WorkerIdentity, session_id: str) -> None:
        self._registry.unregister_worker(identity, session_id)

    def renew_scheduler(self, identity: SchedulerIdentity, session_id: str) -> None:
        if not self._registry.touch_scheduler(identity, session_id):
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

    def renew_worker(self, identity: WorkerIdentity, session_id: str) -> None:
        if not self._registry.touch_worker(identity, session_id):
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")

    def lookup(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        scheduler = self._registry.get_scheduler(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def reap_stale(self, stale_before: float) -> tuple[int, int]:
        return self._registry.reap_stale(stale_before)

    def close(self) -> None:
        self._registry.close()

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
        worker = self._registry.get_worker(worker_identity)
        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.lookup_scheduler(token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens)

    def _bind_lookup_store(self, scheduler_identity: SchedulerIdentity) -> None:
        with self._binding_lock:
            scheduler = self._registry.get_scheduler(scheduler_identity)
            worker = self._registry.get_worker(self._get_lookup_worker_identity(scheduler_identity))
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
