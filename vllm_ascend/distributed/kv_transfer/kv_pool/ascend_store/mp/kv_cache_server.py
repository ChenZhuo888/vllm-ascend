"""Server-side KV cache service orchestration."""

import threading
from functools import partial
from typing import cast

from vllm.v1.request import Request

from ..metadata import AscendStoreKVConnectorWorkerMetadata
from .kv_cache_protocol import (
    ACK_RESPONSE,
    KVCacheMethod,
    decode_build_connector_meta_request,
    decode_lookup_request,
    decode_register_kv_caches_request,
    decode_registration_request,
    decode_request_finished,
    decode_scheduler_session,
    decode_update_connector_output,
    decode_update_state_after_alloc,
    decode_worker_session,
    encode_build_connector_meta_response,
    encode_lookup_response,
    encode_request_finished_response,
    encode_update_connector_output_response,
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
from .request_view import ConnectorOutputView
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
        self._abort_requested = threading.Event()
        self._closed = False
        self._service = KVCacheServiceManager(
            scheduler_factory,
            worker_factory,
            scheduler_executor=scheduler_executor,
            worker_executor=worker_executor,
        )
        scheduler_route = partial(Route, executor=scheduler_executor, key_factory=scheduler_affinity_key)
        worker_route = partial(Route, executor=worker_executor, key_factory=worker_affinity_key)
        # Renewal only updates lifecycle metadata and must not wait behind business work.
        lease_route = partial(Route, executor=lease_executor)
        self._rpc_server = MPServer(
            bind_url,
            routes=(
                scheduler_route(KVCacheMethod.REGISTER_SCHEDULER, self._handle_register_scheduler),
                scheduler_route(KVCacheMethod.UNREGISTER_SCHEDULER, self._handle_unregister_scheduler),
                lease_route(KVCacheMethod.RENEW_SCHEDULER, self._handle_renew_scheduler),
                scheduler_route(KVCacheMethod.LOOKUP, self._handle_lookup),
                scheduler_route(KVCacheMethod.UPDATE_STATE_AFTER_ALLOC, self._handle_update_state_after_alloc),
                scheduler_route(KVCacheMethod.BUILD_CONNECTOR_META, self._handle_build_connector_meta),
                scheduler_route(KVCacheMethod.REQUEST_FINISHED, self._handle_request_finished),
                scheduler_route(KVCacheMethod.UPDATE_CONNECTOR_OUTPUT, self._handle_update_connector_output),
                worker_route(KVCacheMethod.REGISTER_WORKER, self._handle_register_worker),
                worker_route(KVCacheMethod.REGISTER_KV_CACHES, self._handle_register_kv_caches),
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

    def _handle_register_kv_caches(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, spec = decode_register_kv_caches_request(payloads)
        self._service.register_worker_kv_caches(identity, session_id, spec)
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

    def _handle_update_state_after_alloc(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, request, blocks, num_external_tokens = decode_update_state_after_alloc(payloads)
        self._service.update_state_after_alloc(identity, session_id, request, blocks, num_external_tokens)
        return (ACK_RESPONSE,)

    def _handle_build_connector_meta(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, output = decode_build_connector_meta_request(payloads)
        metadata, touch_block_ids = self._service.build_connector_meta(identity, session_id, output)
        return encode_build_connector_meta_response(metadata, touch_block_ids)

    def _handle_request_finished(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, req_id, block_ids, all_groups = decode_request_finished(payloads)
        delay_free, extra = self._service.request_finished(identity, session_id, req_id, block_ids, all_groups)
        return encode_request_finished_response(delay_free, extra)

    def _handle_update_connector_output(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id, completed_events = decode_update_connector_output(payloads)
        output = ConnectorOutputView(kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata(completed_events))
        free_block_ids = self._service.update_connector_output(identity, session_id, output)
        return encode_update_connector_output_response(free_block_ids)

    def run(self) -> None:
        try:
            self._service.start_lease_maintenance()
            self._rpc_server.run()
        except BaseException:
            self.abort()
            raise
        else:
            self.close()

    def request_stop(self) -> bool:
        """Ask a running server to drain if all accepted requests are bounded."""
        return self._rpc_server.request_stop()

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Wait until the RPC run loop stops."""
        return self._rpc_server.wait_until_stopped(timeout)

    def abort(self) -> None:
        """Cancel queued RPC work without waiting for running business code."""
        if self._closed or self._abort_requested.is_set():
            return
        self._abort_requested.set()
        self._rpc_server.abort()
        self._service.stop_lease_maintenance(wait=False)

    def close(self) -> bool:
        """Gracefully close the server, or return ``False`` when abort is required."""
        if self._closed:
            return True
        if self._abort_requested.is_set() or not self.request_stop():
            return False

        self._service.stop_lease_maintenance(wait=False)
        if not self._rpc_server.wait_for_drain():
            return False
        self._service.stop_lease_maintenance()

        with self._close_lock:
            if self._closed:
                return True
            if self._abort_requested.is_set():
                return False
            try:
                # MPServer still owns live route executors while services close on their owner lanes.
                self._service.close()
            finally:
                rpc_closed = self._rpc_server.close()
            self._closed = rpc_closed
            return rpc_closed
