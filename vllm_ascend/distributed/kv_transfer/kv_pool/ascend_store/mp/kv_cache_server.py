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
from .kv_cache_service import KVCacheServiceManager
from .registration import (
    SchedulerFactory,
    SchedulerRegistration,
    WorkerFactory,
    WorkerRegistration,
)
from .rpc import (
    AffinityExecutor,
    BoundedThreadPoolExecutor,
    InlineExecutor,
    MPServer,
    MPServerBusyError,
    Route,
)
from .service import ServiceBusyError

_MAX_PENDING_REQUESTS = 64


class KVCacheServer:
    """Own Scheduler and Worker services and expose their operations through RPC."""

    def __init__(
        self,
        bind_url: str,
        max_workers: int = 4,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
    ):
        self._service = KVCacheServiceManager(scheduler_factory, worker_factory)
        inline_executor = InlineExecutor()
        parallel_executor = BoundedThreadPoolExecutor(max_workers, _MAX_PENDING_REQUESTS, "ascend-store-kv-parallel")
        affinity_executor = AffinityExecutor(max_workers, _MAX_PENDING_REQUESTS, "ascend-store-kv-affinity")
        self._rpc_server = MPServer(
            bind_url,
            routes=(
                Route(KVCacheMethod.REGISTER_SCHEDULER, self._handle_register_scheduler, parallel_executor),
                Route(KVCacheMethod.REGISTER_WORKER, self._handle_register_worker, parallel_executor),
                Route(KVCacheMethod.UNREGISTER_SCHEDULER, self._handle_unregister_scheduler, parallel_executor),
                Route(KVCacheMethod.UNREGISTER_WORKER, self._handle_unregister_worker, parallel_executor),
                Route(KVCacheMethod.RENEW_SCHEDULER, self._handle_renew_scheduler, inline_executor),
                Route(KVCacheMethod.RENEW_WORKER, self._handle_renew_worker, inline_executor),
                Route(KVCacheMethod.LOOKUP, self._handle_lookup, affinity_executor, lookup_affinity_key),
            ),
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

    def _handle_renew_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_scheduler_session(payloads)
        self._service.renew_scheduler(identity, session_id)
        return (ACK_RESPONSE,)

    def _handle_renew_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = decode_worker_session(payloads)
        self._service.renew_worker(identity, session_id)
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
        self._service.start()
        try:
            self._rpc_server.run()
        finally:
            self._service.close()

    def close(self) -> None:
        self._rpc_server.close()
        self._service.close()
