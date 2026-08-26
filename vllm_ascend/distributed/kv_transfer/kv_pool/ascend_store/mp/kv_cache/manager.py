"""KV cache service orchestration independent of the RPC transport."""

import hashlib
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from ...metadata import AscendConnectorMetadata, AscendStoreKVConnectorWorkerMetadata
from ..rpc import TaskExecutor
from ..service import ServiceLifecycleManager
from .error import ServiceNotRegisteredError
from .registration import (
    SchedulerFactory,
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)
from .synchronization import NPUEventSpec
from .view import (
    BlocksView,
    ConnectorOutputView,
    RequestIdView,
    RequestView,
    SchedulerOutputView,
    WorkerKVCacheSpec,
)

if TYPE_CHECKING:
    from ...pool_scheduler import KVPoolScheduler
    from ...pool_worker import KVPoolWorker

_LOOKUP_COORDINATOR_RANK = 0
_SERVICE_LEASE_TIMEOUT_S = 60.0
_LEASE_CHECK_INTERVAL_S = 5.0


class KVCacheServiceManager:
    """Own KV cache services and coordinate calls between optional owner lanes."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
        lease_timeout_s: float = _SERVICE_LEASE_TIMEOUT_S,
        lease_check_interval_s: float = _LEASE_CHECK_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        scheduler_executor: TaskExecutor | None = None,
        worker_executor: TaskExecutor | None = None,
    ):
        self._scheduler_factory = scheduler_factory or self._create_scheduler
        self._worker_factory = worker_factory or self._create_worker
        self._worker_executor = worker_executor
        self._schedulers = ServiceLifecycleManager[SchedulerIdentity, "KVPoolScheduler"](
            "Scheduler",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-scheduler-lifecycle",
            owner_close_handler=partial(self._close_service_on_owner, scheduler_executor),
        )
        self._workers = ServiceLifecycleManager[WorkerIdentity, "KVPoolWorker"](
            "Worker",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-worker-lifecycle",
            owner_close_handler=partial(self._close_service_on_owner, worker_executor),
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
        from .pool.scheduler import MPKVPoolScheduler

        return MPKVPoolScheduler(registration, lookup_handler)

    @staticmethod
    def _create_worker(registration: WorkerRegistration) -> "KVPoolWorker":
        from .pool.worker import MPKVPoolWorker

        return MPKVPoolWorker(
            registration.vllm_config,
            kv_cache_config=registration.kv_cache_config,
            rank=registration.identity.rank,
        )

    def _build_scheduler(self, registration: SchedulerRegistration) -> "KVPoolScheduler":
        return self._scheduler_factory(registration, self._lookup_worker)

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> "KVPoolScheduler":
        self._validate_scheduler_registration(registration)
        scheduler = self._schedulers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._build_scheduler(registration),
        )
        return scheduler

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "KVPoolWorker":
        self._validate_worker_registration(registration)
        worker = self._workers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._worker_factory(registration),
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

    def register_worker_kv_caches(
        self,
        identity: WorkerIdentity,
        session_id: str,
        spec: WorkerKVCacheSpec,
    ) -> None:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        configure = getattr(worker, "configure_kv_caches", None)
        if not callable(configure):
            raise RuntimeError(f"Worker {identity!r} does not support KV cache configuration")
        configure(spec)

    def wait_for_save(
        self,
        identity: WorkerIdentity,
        session_id: str,
        metadata: AscendConnectorMetadata,
        event_spec: NPUEventSpec,
    ) -> None:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        handler = getattr(worker, "wait_for_save", None)
        if not callable(handler):
            raise RuntimeError(f"Worker {identity!r} does not support wait_for_save")
        handler(metadata, event_spec)

    def get_finished(
        self,
        identity: WorkerIdentity,
        session_id: str,
        finished_req_ids: set[str],
        metadata: AscendConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return worker.get_finished(finished_req_ids, metadata)

    def build_connector_worker_meta(
        self,
        identity: WorkerIdentity,
        session_id: str,
    ) -> AscendStoreKVConnectorWorkerMetadata | None:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return worker.build_connector_worker_meta()

    def get_kv_events(self, identity: WorkerIdentity, session_id: str) -> list[BlockStored]:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return worker.get_kv_events()

    def start_load_kv(
        self,
        identity: WorkerIdentity,
        session_id: str,
        metadata: AscendConnectorMetadata,
    ) -> None:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        worker.start_load_kv(metadata)

    def get_block_ids_with_load_errors(self, identity: WorkerIdentity, session_id: str) -> set[int]:
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return worker.get_block_ids_with_load_errors()

    def lookup(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: RequestView,
        blocks: BlocksView,
        num_external_tokens: int,
    ) -> None:
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        # The inherited method stores the view in _unfinished_requests, which
        # doubles as the request registry for later business methods.
        scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        output: SchedulerOutputView,
    ) -> tuple:
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        metadata = scheduler.build_connector_meta(output)
        take_commands = getattr(scheduler, "take_block_pool_commands", None)
        touch_block_ids = take_commands() if callable(take_commands) else []
        return metadata, touch_block_ids

    def request_finished(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        req_id: str,
        block_ids,
        all_groups: bool,
    ) -> tuple:
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        request = RequestIdView(request_id=req_id)
        if all_groups:
            return scheduler.request_finished_all_groups(request, block_ids)
        return scheduler.request_finished(request, block_ids)

    def update_connector_output(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        output: ConnectorOutputView,
    ) -> list[int]:
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        scheduler.update_connector_output(output)
        take_free = getattr(scheduler, "take_free_block_commands", None)
        return take_free() if callable(take_free) else []

    def start_lease_maintenance(self) -> None:
        self._schedulers.start_maintenance()
        self._workers.start_maintenance()

    def stop_lease_maintenance(self, wait: bool = True) -> None:
        self._workers.stop_maintenance(wait=wait)
        self._schedulers.stop_maintenance(wait=wait)

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
        callback = partial(
            self._execute_worker_lookup,
            worker_identity,
            token_len,
            block_hashes,
            kv_cache_group_ids,
            use_layerwise,
            hbm_hit_tokens,
        )
        if self._worker_executor is None:
            return callback()
        return self._worker_executor.submit(callback, worker_identity).result()

    def _execute_worker_lookup(
        self,
        worker_identity: WorkerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker = self._workers.find(worker_identity)
        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.lookup_scheduler(token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens)

    def _close_service_on_owner(
        self,
        executor: TaskExecutor | None,
        identity: SchedulerIdentity | WorkerIdentity,
        service: object,
    ) -> None:
        callback = partial(self._close_service, service)
        if executor is None:
            callback()
            return
        executor.submit(callback, identity, block=True).result()

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
