"""Client-side KV cache service orchestration."""

import contextlib
import logging
import threading
import uuid

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from .kv_cache_error import SERVICE_NOT_REGISTERED_PREFIX, STALE_SESSION_PREFIX, ServiceSessionExpiredError
from .kv_cache_protocol import (
    KVCacheMethod,
    decode_ack_response,
    decode_build_connector_meta_response,
    decode_lookup_response,
    decode_request_finished_response,
    decode_update_connector_output_response,
    encode_build_connector_meta_request,
    encode_lookup_request,
    encode_register_kv_caches_request,
    encode_registration_request,
    encode_request_finished,
    encode_scheduler_session,
    encode_update_connector_output,
    encode_update_state_after_alloc,
    encode_worker_session,
)
from .registration import SchedulerRegistration, WorkerRegistration
from .request_view import WorkerKVCacheSpec
from .rpc import (
    MPClient,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerBusyError,
    MPServerUnavailableError,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 5000
_REGISTRATION_TIMEOUT_MS = 500
_LEASE_RENEW_INTERVAL_MS = 1000
_LEASE_REQUEST_TIMEOUT_MS = 1000
_RegistrationState = tuple[SchedulerRegistration | WorkerRegistration, tuple[bytes, ...]]


class KVCacheClient:
    """Typed KV cache RPC client with recoverable service registration."""

    def __init__(self, server_url: str):
        self._rpc_client = MPClient(server_url)
        self._registration_lock = threading.Lock()
        self._lease_lock = threading.Lock()
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._registration: _RegistrationState | None = None
        self._worker_kv_cache_payloads: tuple[bytes, ...] | None = None
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
            return not self._closed and not self._superseded and self._registered

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

            self._registration = (registration, encode_registration_request(registration))
            self._registered = False

        registered = self._try_register()
        self._start_lease_loop()
        return registered

    def _maintain_lease(self) -> None:
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
            method = KVCacheMethod.RENEW_SCHEDULER
            payloads = encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.RENEW_WORKER
            payloads = encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_LEASE_REQUEST_TIMEOUT_MS)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            self._mark_unregistered()
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

        decode_ack_response(responses, method)

    def _start_lease_loop(self) -> None:
        with self._registration_lock:
            if self._closed:
                return

            with self._lease_lock:
                if self._lease_thread is not None and self._lease_thread.is_alive():
                    return

                self._lease_stop.clear()
                self._lease_thread = threading.Thread(
                    target=self._lease_loop, daemon=True, name="ascend-store-kv-lease"
                )
                self._lease_thread.start()

    def _lease_loop(self) -> None:
        interval_s = _LEASE_RENEW_INTERVAL_MS / 1000
        while not self._lease_stop.wait(interval_s):
            try:
                self._maintain_lease()
            except Exception:
                logger.exception("KV cache service lease maintenance failed")

    def _stop_lease_loop(self) -> None:
        with self._lease_lock:
            lease_thread = self._lease_thread
            if lease_thread is None:
                return
            self._lease_stop.set()

        if lease_thread is not threading.current_thread():
            lease_thread.join()

        with self._lease_lock:
            if self._lease_thread is lease_thread:
                self._lease_thread = None

    def _try_register(self) -> bool:
        with self._registration_lock:
            if self._closed:
                return False
            if self._superseded:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")

            configured_registration = self._registration
            worker_kv_cache_payloads = self._worker_kv_cache_payloads
            if configured_registration is None:
                return False
            if self._registered:
                return True

        registration, payloads = configured_registration
        method = (
            KVCacheMethod.REGISTER_SCHEDULER
            if isinstance(registration, SchedulerRegistration)
            else KVCacheMethod.REGISTER_WORKER
        )

        if not self._rpc_client.is_transport_connected:
            return False

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_REGISTRATION_TIMEOUT_MS)
            decode_ack_response(responses, method)
            if isinstance(registration, WorkerRegistration) and worker_kv_cache_payloads is not None:
                responses = self._rpc_client.request(
                    KVCacheMethod.REGISTER_KV_CACHES,
                    worker_kv_cache_payloads,
                    timeout_ms=_REGISTRATION_TIMEOUT_MS,
                )
                decode_ack_response(responses, KVCacheMethod.REGISTER_KV_CACHES)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            self._mark_unregistered()
            return False
        except MPRemoteError as exc:
            message = str(exc)
            if message.startswith(SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                return False
            if message.startswith(STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(message) from exc
            raise

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

    def _get_worker_registration(self) -> WorkerRegistration:
        with self._registration_lock:
            configured_registration = self._registration

        if configured_registration is None or not isinstance(configured_registration[0], WorkerRegistration):
            raise RuntimeError("KVCacheClient is not configured as a Worker client")
        return configured_registration[0]

    def register_kv_caches(
        self,
        spec: WorkerKVCacheSpec,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> bool:
        """Register process-neutral cache layouts for the configured Worker."""
        self._raise_if_superseded()
        registration = self._get_worker_registration()
        payloads = encode_register_kv_caches_request(registration, spec)
        if not self.is_registered and not self._try_register():
            with self._registration_lock:
                self._worker_kv_cache_payloads = payloads
            return False

        with self._registration_lock:
            self._worker_kv_cache_payloads = payloads

        responses = self._service_rpc(KVCacheMethod.REGISTER_KV_CACHES, payloads, timeout_ms)
        if responses is None:
            # Cache layout is initialization state, so a busy response must be
            # retried by the lease loop instead of becoming a permanent no-op.
            self._mark_unregistered()
            return False
        decode_ack_response(responses, KVCacheMethod.REGISTER_KV_CACHES)
        return True

    def lookup(
        self, request: Request, num_computed_tokens: int, timeout_ms: int = _DEFAULT_TIMEOUT_MS
    ) -> tuple[int, bool]:
        responses = self._scheduler_rpc(
            KVCacheMethod.LOOKUP,
            lambda registration: encode_lookup_request(registration, request, num_computed_tokens),
            timeout_ms,
        )
        return decode_lookup_response(responses) if responses is not None else (0, False)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> None:
        responses = self._scheduler_rpc(
            KVCacheMethod.UPDATE_STATE_AFTER_ALLOC,
            lambda registration: encode_update_state_after_alloc(registration, request, blocks, num_external_tokens),
            timeout_ms,
        )
        if responses is not None:
            decode_ack_response(responses, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        new_token_ids: dict[str, list[int]],
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple | None:
        """Return (metadata, touch_block_ids) or None when degraded."""
        responses = self._scheduler_rpc(
            KVCacheMethod.BUILD_CONNECTOR_META,
            lambda registration: encode_build_connector_meta_request(registration, scheduler_output, new_token_ids),
            timeout_ms,
        )
        return decode_build_connector_meta_response(responses) if responses is not None else None

    def request_finished(
        self,
        request_id: str,
        block_ids,
        all_groups: bool = False,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple[bool, dict | None]:
        responses = self._scheduler_rpc(
            KVCacheMethod.REQUEST_FINISHED,
            lambda registration: encode_request_finished(registration, request_id, block_ids, all_groups),
            timeout_ms,
        )
        return decode_request_finished_response(responses) if responses is not None else (False, None)

    def update_connector_output(
        self,
        completed_events: dict[int, int],
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> list[int]:
        """Report worker completion counts; return block ids to free locally."""
        responses = self._scheduler_rpc(
            KVCacheMethod.UPDATE_CONNECTOR_OUTPUT,
            lambda registration: encode_update_connector_output(registration, completed_events),
            timeout_ms,
        )
        return decode_update_connector_output_response(responses) if responses is not None else []

    def _scheduler_rpc(
        self,
        method: KVCacheMethod,
        encode,
        timeout_ms: int,
    ) -> list[bytes] | None:
        """Send one scheduler-scoped RPC through the shared degradation ladder.

        Returns the raw response payloads, or None when the service is
        unreachable, unregistered, or busy; a stale session still raises
        ServiceSessionExpiredError.
        """
        self._raise_if_superseded()
        registration = self._get_scheduler_registration()
        payloads = encode(registration)

        return self._service_rpc(method, payloads, timeout_ms)

    def _service_rpc(
        self,
        method: KVCacheMethod,
        payloads: tuple[bytes, ...],
        timeout_ms: int,
    ) -> list[bytes] | None:
        """Send a registered service request through the degradation ladder."""

        if not self.is_registered and not self._try_register():
            return None

        try:
            return self._rpc_client.request(method, payloads, timeout_ms=timeout_ms)
        except MPServerBusyError:
            return None
        except (MPRequestTimeoutError, MPServerUnavailableError):
            self._mark_unregistered()
            return None
        except MPRemoteError as exc:
            message = str(exc)
            if message.startswith(SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                return None
            if message.startswith(STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(message) from exc
            raise

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
            decode_ack_response(responses, method)
        except Exception:
            # close() is best-effort cleanup; transport recovery is no longer useful once the client is closing.
            logger.debug("Failed to unregister KV cache service during client close", exc_info=True)

    def close(self) -> None:
        with self._registration_lock:
            if self._closed:
                return
            self._closed = True

        self._stop_lease_loop()
        self._unregister()
        self._rpc_client.close()
