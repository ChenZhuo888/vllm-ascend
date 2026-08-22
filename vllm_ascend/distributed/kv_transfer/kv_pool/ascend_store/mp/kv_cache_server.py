"""Server-side KV cache service orchestration."""

from typing import cast

from vllm.v1.request import Request

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
from .kv_cache_service import KVCacheService
from .registration import (
    SchedulerFactory,
    SchedulerRegistration,
    WorkerFactory,
    WorkerRegistration,
)
from .rpc import ExecutionMode, HandlerSpec, MPServer, MPServerBusyError
from .service import ServiceBusyError, ServiceReaper

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
        self._service = KVCacheService(scheduler_factory, worker_factory)
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
            self._service.reap_stale,
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
        return self._service.scheduler_count

    @property
    def worker_count(self) -> int:
        return self._service.worker_count

    def _handle_register_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, SchedulerRegistration)
        try:
            self._service.register_scheduler(registration, payloads[0])
        except ServiceBusyError as exc:
            raise MPServerBusyError(str(exc)) from exc
        return (ACK_RESPONSE,)

    def _handle_register_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, WorkerRegistration)
        try:
            self._service.register_worker(registration, payloads[0])
        except ServiceBusyError as exc:
            raise MPServerBusyError(str(exc)) from exc
        return (ACK_RESPONSE,)

    def _handle_unregister_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_scheduler_session(payloads)
        self._service.unregister_scheduler(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_unregister_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_worker_session(payloads)
        self._service.unregister_worker(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_scheduler_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_scheduler_session(payloads)
        self._service.heartbeat_scheduler(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_worker_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_worker_session(payloads)
        self._service.heartbeat_worker(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_lookup(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, request, num_computed_tokens = decode_lookup_request(payloads)
        matched_tokens, is_async = self._service.lookup(
            identity,
            session_id,
            cast(Request, request),
            num_computed_tokens,
        )
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
        self._service.close()
