"""Server-side KV cache service orchestration."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from .kv_cache_error import ServiceNotRegisteredError
from .kv_cache_protocol import (
    ACK_RESPONSE,
    KVCacheMethod,
    decode_lookup_request,
    decode_registration,
    decode_scheduler_session,
    decode_worker_session,
    encode_lookup_response,
    lookup_affinity_key,
)
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
from .rpc import ExecutionMode, HandlerSpec, MPServer
from .service import ServiceReaper

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

_SERVICE_STALE_TIMEOUT_S = 60.0
_SERVICE_REAP_INTERVAL_S = 5.0


class KVCacheServer:
    """Own Scheduler and Worker services and expose their operations through RPC."""

    def __init__(
        self,
        bind_url: str,
        max_workers: int = 4,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
    ):
        self._registry = KVCacheServiceRegistry(
            scheduler_factory or self._create_scheduler,
            worker_factory or self._create_worker,
            self._lookup_worker,
        )
        self._rpc_server = MPServer(
            bind_url,
            max_workers=max_workers,
            handlers={
                KVCacheMethod.REGISTER_SCHEDULER: self._handle_register_scheduler,
                KVCacheMethod.REGISTER_WORKER: self._handle_register_worker,
                KVCacheMethod.UNREGISTER_SCHEDULER: self._handle_unregister_scheduler,
                KVCacheMethod.UNREGISTER_WORKER: self._handle_unregister_worker,
                KVCacheMethod.HEARTBEAT_SCHEDULER: HandlerSpec(self._handle_scheduler_heartbeat, ExecutionMode.INLINE),
                KVCacheMethod.HEARTBEAT_WORKER: HandlerSpec(self._handle_worker_heartbeat, ExecutionMode.INLINE),
                KVCacheMethod.LOOKUP: HandlerSpec(self._handle_lookup, ExecutionMode.AFFINITY, lookup_affinity_key),
            },
        )
        self._service_reaper = ServiceReaper(
            self._registry.reap_stale,
            stale_timeout_s=_SERVICE_STALE_TIMEOUT_S,
            interval_s=_SERVICE_REAP_INTERVAL_S,
            thread_name="ascend-store-kv-reaper",
        )

    def __enter__(self) -> "KVCacheServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return self._rpc_server.endpoint

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
            registration.vllm_config, kv_cache_config=registration.kv_cache_config, rank=registration.identity.rank
        )

    def _handle_register_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, SchedulerRegistration)
        self._registry.register_scheduler(registration, payloads[0])
        return (ACK_RESPONSE,)

    def _handle_register_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, WorkerRegistration)
        self._registry.register_worker(registration, payloads[0])
        return (ACK_RESPONSE,)

    def _handle_unregister_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_scheduler_session(payloads)
        self._registry.unregister_scheduler(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_unregister_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_worker_session(payloads)
        self._registry.unregister_worker(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_scheduler_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_scheduler_session(payloads)
        if not self._registry.touch_scheduler(identity, session_id):
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return (ACK_RESPONSE,)

    def _handle_worker_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_worker_session(payloads)
        if not self._registry.touch_worker(identity, session_id):
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return (ACK_RESPONSE,)

    def _lookup_worker(
        self,
        scheduler_identity: SchedulerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker_identity = WorkerIdentity(
            scheduler_identity.engine_id, rank=0, data_parallel_rank=scheduler_identity.data_parallel_rank
        )
        worker = self._registry.get_worker(worker_identity)
        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.lookup_scheduler(token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens)

    def _handle_lookup(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, request, num_computed_tokens = decode_lookup_request(payloads)
        scheduler = self._registry.get_scheduler(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

        matched_tokens, is_async = scheduler.get_num_new_matched_tokens(cast(Request, request), num_computed_tokens)
        return encode_lookup_response(matched_tokens, is_async)

    def run(self) -> None:
        self._service_reaper.start()
        try:
            self._rpc_server.run()
        finally:
            self._service_reaper.stop()

    def close(self) -> None:
        self._service_reaper.stop()
        self._rpc_server.close()
        self._registry.close()
