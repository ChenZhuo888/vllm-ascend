"""Server-side KV cache service orchestration."""

import threading
from functools import partial
from typing import cast

from vllm.v1.request import Request

from .kv_cache_protocol import (
    ACK_RESPONSE,
    KVCacheMethod,
    decode_lookup_request,
    decode_registration_request,
    decode_scheduler_session,
    decode_worker_session,
    encode_lookup_response,
    lookup_affinity_key,
    scheduler_affinity_key,
    worker_affinity_key,
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
        scheduler_executor = AffinityExecutor(max_workers, _MAX_PENDING_REQUESTS, "ascend-store-kv-scheduler")
        worker_executor = AffinityExecutor(max_workers, _MAX_PENDING_REQUESTS, "ascend-store-kv-worker")
        lease_executor = InlineExecutor()
        self._close_lock = threading.Lock()
        self._closed = False
        self._service = KVCacheServiceManager(
            scheduler_factory,
            worker_factory,
            scheduler_executor=scheduler_executor,
            worker_executor=worker_executor,
        )
        scheduler_route = partial(Route, executor=scheduler_executor, key_factory=scheduler_affinity_key)
        lookup_route = partial(Route, executor=scheduler_executor, key_factory=lookup_affinity_key)
        worker_route = partial(Route, executor=worker_executor, key_factory=worker_affinity_key)
        # Renewal only updates lifecycle metadata and must not wait behind business work.
        lease_route = partial(Route, executor=lease_executor)
        self._rpc_server = MPServer(
            bind_url,
            routes=(
                scheduler_route(KVCacheMethod.REGISTER_SCHEDULER, self._handle_register_scheduler),
                scheduler_route(KVCacheMethod.UNREGISTER_SCHEDULER, self._handle_unregister_scheduler),
                lease_route(KVCacheMethod.RENEW_SCHEDULER, self._handle_renew_scheduler),
                lookup_route(KVCacheMethod.LOOKUP, self._handle_lookup),
                worker_route(KVCacheMethod.REGISTER_WORKER, self._handle_register_worker),
                worker_route(KVCacheMethod.UNREGISTER_WORKER, self._handle_unregister_worker),
                lease_route(KVCacheMethod.RENEW_WORKER, self._handle_renew_worker),
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
        registration, serialized_registration = decode_registration_request(payloads, SchedulerRegistration)
        try:
            self._service.register_scheduler(registration, serialized_registration)
        except ServiceBusyError as exc:
            raise MPServerBusyError(str(exc)) from exc
        return (ACK_RESPONSE,)

    def _handle_register_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration, serialized_registration = decode_registration_request(payloads, WorkerRegistration)
        try:
            self._service.register_worker(registration, serialized_registration)
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
        try:
            self._service.start_lease_maintenance()
            self._rpc_server.run()
        except BaseException:
            self.abort()
            raise
        else:
            self.close()

    def request_stop(self) -> None:
        """Ask a running server to exit without waiting for shutdown to finish."""
        self._rpc_server.request_stop()

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Wait until accepted RPC requests have drained."""
        return self._rpc_server.wait_until_stopped(timeout)

    def abort(self) -> None:
        """Cancel queued RPC work without waiting for running business code."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._rpc_server.abort()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.request_stop()
            self._service.stop_lease_maintenance()
            self._rpc_server.wait_until_stopped()
            try:
                # MPServer still owns live route executors while services close on their owner lanes.
                self._service.close()
            finally:
                self._rpc_server.close()
