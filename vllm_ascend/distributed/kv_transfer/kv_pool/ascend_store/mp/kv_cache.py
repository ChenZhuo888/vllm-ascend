"""KV cache business facade for AscendStore multiprocessing mode."""

import enum
import logging
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from .registration import (
    KVCacheServiceRegistry,
    SchedulerFactory,
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
    decode_registration,
    encode_registration,
)
from .rpc import (
    ExecutionMode,
    HandlerSpec,
    MPClient,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServer,
    MPServerBusyError,
    MPServerUnavailableError,
)

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 5000
_REGISTRATION_TIMEOUT_MS = 500
_HEARTBEAT_INTERVAL_MS = 1000
_HEARTBEAT_TIMEOUT_MS = 1000
_SERVICE_STALE_TIMEOUT_S = 60.0
_SERVICE_REAP_INTERVAL_S = 5.0
_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_LOOKUP_HEADER_PAYLOADS = 6
_REGISTRATION_RESPONSE = b"OK"
_UNREGISTRATION_RESPONSE = b"OK"
_ASYNC_RESPONSE = b"\x01"
_SYNC_RESPONSE = b"\x00"
_SERVICE_NOT_REGISTERED_PREFIX = "ServiceNotRegisteredError:"
_STALE_SESSION_PREFIX = "StaleSessionError:"


class ServiceNotRegisteredError(RuntimeError):
    pass


class ServiceSessionExpiredError(RuntimeError):
    pass


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    UNREGISTER_SCHEDULER = "UNREGISTER_SCHEDULER"
    UNREGISTER_WORKER = "UNREGISTER_WORKER"
    HEARTBEAT_SCHEDULER = "HEARTBEAT_SCHEDULER"
    HEARTBEAT_WORKER = "HEARTBEAT_WORKER"
    LOOKUP = "LOOKUP"


@dataclass
class _LookupRequestView:
    request_id: str
    prompt_token_ids: range
    block_hashes: list[BlockHash]
    num_tokens: int


def _encode_non_negative_int(value: int, field_name: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")

    try:
        return value.to_bytes(_INTEGER_BYTES, byteorder=_BYTE_ORDER)
    except OverflowError as exc:
        raise ValueError(f"{field_name} is too large: {value}") from exc


def _decode_non_negative_int(payload: bytes, field_name: str) -> int:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"{field_name} payload must be bytes, got {type(payload).__name__}")
    if len(payload) != _INTEGER_BYTES:
        raise MPProtocolError(f"{field_name} payload must contain {_INTEGER_BYTES} bytes, got {len(payload)}")
    return int.from_bytes(payload, byteorder=_BYTE_ORDER)


def _encode_text(value: str, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value.encode()


def _decode_text(payload: bytes, field_name: str) -> str:
    try:
        value = payload.decode()
    except UnicodeDecodeError as exc:
        raise MPProtocolError(f"{field_name} payload must be valid UTF-8") from exc

    if not value:
        raise MPProtocolError(f"{field_name} payload must not be empty")
    return value


def _encode_block_hash(block_hash: BlockHash) -> bytes:
    if not isinstance(block_hash, bytes):
        raise TypeError(f"block_hash must be bytes, got {type(block_hash).__name__}")
    if not block_hash:
        raise ValueError("block_hash must not be empty")
    return block_hash


def _decode_block_hash(payload: bytes) -> BlockHash:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"block_hash payload must be bytes, got {type(payload).__name__}")
    if not payload:
        raise MPProtocolError("block_hash payload must not be empty")
    return payload


def _encode_scheduler_session(identity: SchedulerIdentity, session_id: str) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(session_id, "session_id"),
    )


def _decode_scheduler_session(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str]:
    if len(payloads) != 3:
        raise MPProtocolError(f"Scheduler session expects 3 payloads, got {len(payloads)}")
    identity = SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )
    return identity, _decode_text(payloads[2], "session_id")


def _encode_worker_session(identity: WorkerIdentity, session_id: str) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.rank, "rank"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(session_id, "session_id"),
    )


def _decode_worker_session(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    if len(payloads) != 4:
        raise MPProtocolError(f"Worker session expects 4 payloads, got {len(payloads)}")
    identity = WorkerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        rank=_decode_non_negative_int(payloads[1], "rank"),
        data_parallel_rank=_decode_non_negative_int(payloads[2], "data_parallel_rank"),
    )
    return identity, _decode_text(payloads[3], "session_id")


def _encode_lookup_request(
        registration: SchedulerRegistration, request: Request, num_computed_tokens: int
) -> tuple[bytes, ...]:
    identity = registration.identity
    prompt_token_count = len(request.prompt_token_ids)
    payloads = [
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(request.request_id, "request_id"),
        _encode_non_negative_int(prompt_token_count, "prompt_token_count"),
        _encode_non_negative_int(request.num_tokens, "num_tokens"),
        _encode_non_negative_int(num_computed_tokens, "num_computed_tokens"),
    ]
    payloads.extend(_encode_block_hash(block_hash) for block_hash in request.block_hashes)
    payloads.append(_encode_text(registration.session_id, "session_id"))
    return tuple(payloads)


def _decode_lookup_identity(payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    if len(payloads) < _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError(f"LOOKUP expects at least {_LOOKUP_HEADER_PAYLOADS} payloads, got {len(payloads)}")

    return SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )


def _lookup_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_lookup_identity(payloads)


def _decode_lookup_request(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, _LookupRequestView, int]:
    identity = _decode_lookup_identity(payloads)
    if len(payloads) == _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError("LOOKUP expects a session_id payload")

    request_id = _decode_text(payloads[2], "request_id")
    prompt_token_count = _decode_non_negative_int(payloads[3], "prompt_token_count")
    num_tokens = _decode_non_negative_int(payloads[4], "num_tokens")
    num_computed_tokens = _decode_non_negative_int(payloads[5], "num_computed_tokens")
    block_hashes = [_decode_block_hash(payload) for payload in payloads[_LOOKUP_HEADER_PAYLOADS:-1]]
    session_id = _decode_text(payloads[-1], "session_id")

    request = _LookupRequestView(
        request_id=request_id,
        prompt_token_ids=range(prompt_token_count),
        block_hashes=block_hashes,
        num_tokens=num_tokens,
    )
    return identity, session_id, request, num_computed_tokens


def _decode_async_response(payload: bytes) -> bool:
    if payload == _ASYNC_RESPONSE:
        return True
    if payload == _SYNC_RESPONSE:
        return False
    raise MPProtocolError(f"Invalid LOOKUP async response: {payload!r}")


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
                    not self._closed
                    and not self._superseded
                    and self._registered
                    and self._rpc_client.is_server_responsive
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
            try:
                self._try_register()
            except ServiceSessionExpiredError:
                pass
            return

        if isinstance(registration, SchedulerRegistration):
            method = KVCacheMethod.HEARTBEAT_SCHEDULER
            payloads = _encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.HEARTBEAT_WORKER
            payloads = _encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_HEARTBEAT_TIMEOUT_MS)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            return
        except MPRemoteError as exc:
            if str(exc).startswith(_SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                try:
                    self._try_register()
                except ServiceSessionExpiredError:
                    pass
                return
            if str(exc).startswith(_STALE_SESSION_PREFIX):
                self._mark_superseded()
                return
            raise

        if responses != [_REGISTRATION_RESPONSE]:
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
            if str(exc).startswith(_STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(str(exc)) from exc
            raise

        if responses != [_REGISTRATION_RESPONSE]:
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
        payloads = _encode_lookup_request(registration, request, num_computed_tokens)

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
            if str(exc).startswith(_SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                return 0, False
            if str(exc).startswith(_STALE_SESSION_PREFIX):
                self._mark_superseded()
                raise ServiceSessionExpiredError(str(exc)) from exc
            raise

        if len(responses) != 2:
            raise MPProtocolError(f"LOOKUP expects 2 response payloads, got {len(responses)}")

        matched_tokens = _decode_non_negative_int(responses[0], "matched_tokens")
        return matched_tokens, _decode_async_response(responses[1])

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
            payloads = _encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.UNREGISTER_WORKER
            payloads = _encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._rpc_client.request(method, payloads, timeout_ms=_REGISTRATION_TIMEOUT_MS)
            if responses != [_UNREGISTRATION_RESPONSE]:
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
                KVCacheMethod.HEARTBEAT_SCHEDULER: HandlerSpec(
                    self._handle_scheduler_heartbeat, ExecutionMode.INLINE
                ),
                KVCacheMethod.HEARTBEAT_WORKER: HandlerSpec(self._handle_worker_heartbeat, ExecutionMode.INLINE),
                KVCacheMethod.LOOKUP: HandlerSpec(self._handle_lookup, ExecutionMode.AFFINITY, _lookup_affinity_key),
            },
        )
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None

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
        return (_REGISTRATION_RESPONSE,)

    def _handle_register_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, WorkerRegistration)
        self._registry.register_worker(registration, payloads[0])
        return (_REGISTRATION_RESPONSE,)

    def _handle_unregister_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = _decode_scheduler_session(payloads)
        self._registry.unregister_scheduler(identity, session_id)
        return (_UNREGISTRATION_RESPONSE,)

    def _handle_unregister_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = _decode_worker_session(payloads)
        self._registry.unregister_worker(identity, session_id)
        return (_UNREGISTRATION_RESPONSE,)

    def _handle_scheduler_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = _decode_scheduler_session(payloads)
        if not self._registry.touch_scheduler(identity, session_id):
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return (_REGISTRATION_RESPONSE,)

    def _handle_worker_heartbeat(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, session_id = _decode_worker_session(payloads)
        if not self._registry.touch_worker(identity, session_id):
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return (_REGISTRATION_RESPONSE,)

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
        identity, session_id, request, num_computed_tokens = _decode_lookup_request(payloads)
        scheduler = self._registry.get_scheduler(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

        matched_tokens, is_async = scheduler.get_num_new_matched_tokens(cast(Request, request), num_computed_tokens)
        return (
            _encode_non_negative_int(matched_tokens, "matched_tokens"),
            _ASYNC_RESPONSE if is_async else _SYNC_RESPONSE,
        )

    def _reaper_loop(self) -> None:
        while not self._reaper_stop.wait(_SERVICE_REAP_INTERVAL_S):
            try:
                stale_before = time.monotonic() - _SERVICE_STALE_TIMEOUT_S
                self._registry.reap_stale(stale_before)
            except Exception:
                logger.exception("KV cache service reaper failed")

    def _start_reaper(self) -> None:
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True, name="ascend-store-kv-reaper")
        self._reaper_thread.start()

    def _stop_reaper(self) -> None:
        reaper_thread = self._reaper_thread
        if reaper_thread is None:
            return
        self._reaper_stop.set()
        if reaper_thread is not threading.current_thread():
            reaper_thread.join()
        if self._reaper_thread is reaper_thread:
            self._reaper_thread = None

    def run(self) -> None:
        self._start_reaper()
        try:
            self._rpc_server.run()
        finally:
            self._stop_reaper()

    def close(self) -> None:
        self._stop_reaper()
        self._rpc_server.close()
        self._registry.close()
