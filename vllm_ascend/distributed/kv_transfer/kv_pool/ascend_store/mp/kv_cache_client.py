"""Client-side KV cache service orchestration."""

import contextlib
import logging
import threading
import uuid

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from .kv_cache_error import SERVICE_NOT_REGISTERED_PREFIX, STALE_SESSION_PREFIX, ServiceSessionExpiredError
from .kv_cache_protocol import (
    ACK_RESPONSE,
    KVCacheMethod,
    decode_lookup_response,
    encode_lookup_request,
    encode_registration,
    encode_scheduler_session,
    encode_worker_session,
)
from .registration import SchedulerRegistration, WorkerRegistration
from .rpc import (
    MPClient,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerBusyError,
    MPServerUnavailableError,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 5000
_REGISTRATION_TIMEOUT_MS = 500
_HEARTBEAT_INTERVAL_MS = 1000
_HEARTBEAT_TIMEOUT_MS = 1000


class KVCacheClient:
    """Typed KV cache RPC client with recoverable service registration."""

    def __init__(self, server_url: str):
        self._rpc_client = MPClient(server_url)
        self._registration_lock = threading.Lock()
        self._registration: tuple[SchedulerRegistration | WorkerRegistration, bytes] | None = None
        self._session_id = uuid.uuid4().hex
        self._registered = False
        self._superseded = False
        self._closed = False

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._rpc_client.is_transport_connected

    @property
    def is_registered(self) -> bool:
        with self._registration_lock:
            return (
                not self._closed and not self._superseded and self._registered and self._rpc_client.is_server_responsive
            )

    def register_scheduler(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None, page_size_bytes: int
    ) -> bool:
        registration = SchedulerRegistration.create(
            vllm_config, kv_cache_config, page_size_bytes, session_id=self._session_id
        )
        return self._configure_registration(registration)

    def register_worker(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> bool:
        registration = WorkerRegistration.create(vllm_config, kv_cache_config, session_id=self._session_id)
        return self._configure_registration(registration)

    def _configure_registration(self, registration: SchedulerRegistration | WorkerRegistration) -> bool:
        with self._registration_lock:
            if self._closed:
                raise RuntimeError("KVCacheClient is closed")
            if self._superseded:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")
            if self._registration is not None and type(self._registration[0]) is not type(registration):
                raise RuntimeError("A KVCacheClient cannot register both Scheduler and Worker services")

            self._registration = (registration, encode_registration(registration))
            self._registered = False

        registered = self._try_register()
        self._rpc_client.start_heartbeat(
            interval_ms=_HEARTBEAT_INTERVAL_MS,
            timeout_ms=_HEARTBEAT_TIMEOUT_MS,
            recovery_callback=self._recover_registration,
            heartbeat_callback=self._heartbeat_service,
        )
        return registered

    def _recover_registration(self) -> bool:
        try:
            return self._try_register()
        except ServiceSessionExpiredError:
            # The server is responsive; only this client incarnation is no longer valid.
            return True

    def _heartbeat_service(self) -> None:
        with self._registration_lock:
            if self._closed or self._superseded or self._registration is None:
                return
            registration = self._registration[0]
            registered = self._registered

        if not registered:
            with contextlib.suppress(ServiceSessionExpiredError):
                self._try_register()
            return

        if isinstance(registration, SchedulerRegistration):
            method = KVCacheMethod.HEARTBEAT_SCHEDULER
            payloads = encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.HEARTBEAT_WORKER
            payloads = encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_HEARTBEAT_TIMEOUT_MS)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            return
        except MPRemoteError as exc:
            if str(exc).startswith(SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                with contextlib.suppress(ServiceSessionExpiredError):
                    self._try_register()
                return
            if str(exc).startswith(STALE_SESSION_PREFIX):
                self._mark_superseded()
                return
            raise

        if responses != [ACK_RESPONSE]:
            raise MPProtocolError(f"{method.value} expects an OK response, got {responses!r}")

    def _try_register(self) -> bool:
        with self._registration_lock:
            if self._closed:
                return False
            if self._superseded:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")

            configured_registration = self._registration
            if configured_registration is None:
                return False
            if self._registered and self._rpc_client.is_server_responsive:
                return True

        registration, payload = configured_registration
        method = (
            KVCacheMethod.REGISTER_SCHEDULER
            if isinstance(registration, SchedulerRegistration)
            else KVCacheMethod.REGISTER_WORKER
        )

        if not self._rpc_client.is_transport_connected:
            return False

        try:
            if self._rpc_client.ping(timeout_ms=_REGISTRATION_TIMEOUT_MS) != "OK":
                raise MPProtocolError("KV cache server returned an invalid PING response")
            responses = self._rpc_client.request(method, (payload,), timeout_ms=_REGISTRATION_TIMEOUT_MS)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            self._mark_unregistered()
            return False
        except MPRemoteError as exc:
            if str(exc).startswith(STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(str(exc)) from exc
            raise

        if responses != [ACK_RESPONSE]:
            raise MPProtocolError(f"{method.value} expects an OK response, got {responses!r}")

        with self._registration_lock:
            if self._registration is not configured_registration or self._closed or self._superseded:
                return False
            self._registered = True
        return True

    def _mark_unregistered(self) -> None:
        with self._registration_lock:
            self._registered = False

    def _mark_superseded(self) -> None:
        with self._registration_lock:
            self._registered = False
            self._superseded = True

    def _raise_if_superseded(self) -> None:
        with self._registration_lock:
            if self._superseded:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")

    def _get_scheduler_registration(self) -> SchedulerRegistration:
        with self._registration_lock:
            configured_registration = self._registration

        if configured_registration is None or not isinstance(configured_registration[0], SchedulerRegistration):
            raise RuntimeError("KVCacheClient is not configured as a Scheduler client")
        return configured_registration[0]

    def lookup(
        self, request: Request, num_computed_tokens: int, timeout_ms: int = _DEFAULT_TIMEOUT_MS
    ) -> tuple[int, bool]:
        self._raise_if_superseded()
        registration = self._get_scheduler_registration()
        payloads = encode_lookup_request(registration, request, num_computed_tokens)

        if not self.is_registered and not self._try_register():
            return 0, False

        try:
            responses = self._rpc_client.request(KVCacheMethod.LOOKUP, payloads, timeout_ms=timeout_ms)
        except MPServerBusyError:
            return 0, False
        except (MPRequestTimeoutError, MPServerUnavailableError):
            self._mark_unregistered()
            return 0, False
        except MPRemoteError as exc:
            if str(exc).startswith(SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                return 0, False
            if str(exc).startswith(STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(str(exc)) from exc
            raise

        return decode_lookup_response(responses)

    def _unregister(self) -> None:
        with self._registration_lock:
            configured_registration = self._registration
            should_unregister = self._registered and not self._superseded
            self._registered = False

        if configured_registration is None or not should_unregister or not self._rpc_client.is_transport_connected:
            return

        registration = configured_registration[0]
        if isinstance(registration, SchedulerRegistration):
            method = KVCacheMethod.UNREGISTER_SCHEDULER
            payloads = encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.UNREGISTER_WORKER
            payloads = encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_REGISTRATION_TIMEOUT_MS)
            if responses != [ACK_RESPONSE]:
                raise MPProtocolError(f"{method.value} expects an OK response, got {responses!r}")
        except Exception:
            # close() is best-effort cleanup; transport recovery is no longer useful once the client is closing.
            logger.debug("Failed to unregister KV cache service during client close", exc_info=True)

    def close(self) -> None:
        with self._registration_lock:
            if self._closed:
                return
            self._closed = True

        self._rpc_client.stop_heartbeat()
        self._unregister()
        self._rpc_client.close()
