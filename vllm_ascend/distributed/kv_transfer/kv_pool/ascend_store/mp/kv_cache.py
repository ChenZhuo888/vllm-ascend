"""KV cache business facade for AscendStore multiprocessing mode."""

import enum
import hashlib
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
    decode_registration,
    encode_registration,
)
from .rpc import (
    MPClient,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServer,
    MPServerUnavailableError,
)

_DEFAULT_TIMEOUT_MS = 5000
_REGISTRATION_TIMEOUT_MS = 500
_HEARTBEAT_INTERVAL_MS = 1000
_HEARTBEAT_TIMEOUT_MS = 1000
_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_LOOKUP_HEADER_PAYLOADS = 5
_REGISTRATION_RESPONSE = b"OK"
_ASYNC_RESPONSE = b"\x01"
_SYNC_RESPONSE = b"\x00"
_SERVICE_NOT_REGISTERED_PREFIX = "ServiceNotRegisteredError:"


class RegistrationConflictError(RuntimeError):
    pass


class ServiceNotRegisteredError(RuntimeError):
    pass


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    LOOKUP = "LOOKUP"


class SchedulerNode(Protocol):
    store_scheduler: object

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]: ...


class WorkerNode(Protocol):
    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int: ...


WorkerLookupHandler = Callable[[SchedulerIdentity, int, Sequence[BlockHash], list[int] | None, bool, int], int]
SchedulerFactory = Callable[[SchedulerRegistration, WorkerLookupHandler], SchedulerNode]
WorkerFactory = Callable[[WorkerRegistration], WorkerNode]


@dataclass(frozen=True)
class _ClientRegistration:
    method: KVCacheMethod
    identity: SchedulerIdentity | WorkerIdentity
    payload: bytes


@dataclass(frozen=True)
class _RegisteredScheduler:
    fingerprint: bytes
    service: SchedulerNode


@dataclass(frozen=True)
class _RegisteredWorker:
    fingerprint: bytes
    service: WorkerNode


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


def _encode_lookup_request(
    identity: SchedulerIdentity, request: Request, num_computed_tokens: int
) -> tuple[bytes, ...]:
    prompt_token_count = len(request.prompt_token_ids)
    payloads = [
        _encode_text(identity.engine_id, "engine_id"),
        _encode_text(request.request_id, "request_id"),
        _encode_non_negative_int(prompt_token_count, "prompt_token_count"),
        _encode_non_negative_int(request.num_tokens, "num_tokens"),
        _encode_non_negative_int(num_computed_tokens, "num_computed_tokens"),
    ]
    payloads.extend(_encode_block_hash(block_hash) for block_hash in request.block_hashes)
    return tuple(payloads)


def _decode_lookup_request(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, _LookupRequestView, int]:
    if len(payloads) < _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError(f"LOOKUP expects at least {_LOOKUP_HEADER_PAYLOADS} payloads, got {len(payloads)}")

    identity = SchedulerIdentity(engine_id=_decode_text(payloads[0], "engine_id"))
    request_id = _decode_text(payloads[1], "request_id")
    prompt_token_count = _decode_non_negative_int(payloads[2], "prompt_token_count")
    num_tokens = _decode_non_negative_int(payloads[3], "num_tokens")
    num_computed_tokens = _decode_non_negative_int(payloads[4], "num_computed_tokens")
    block_hashes = [_decode_block_hash(payload) for payload in payloads[_LOOKUP_HEADER_PAYLOADS:]]

    request = _LookupRequestView(
        request_id=request_id,
        prompt_token_ids=range(prompt_token_count),
        block_hashes=block_hashes,
        num_tokens=num_tokens,
    )
    return identity, request, num_computed_tokens


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
        self._registration: _ClientRegistration | None = None
        self._registered = False

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
            return self._registered and self._rpc_client.is_server_responsive

    def register_scheduler(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None, page_size_bytes: int
    ) -> bool:
        registration = SchedulerRegistration.create(
            vllm_config,
            kv_cache_config,
            page_size_bytes,
        )
        return self._configure_registration(
            KVCacheMethod.REGISTER_SCHEDULER,
            registration.identity,
            encode_registration(registration),
        )

    def register_worker(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> bool:
        registration = WorkerRegistration.create(vllm_config, kv_cache_config)
        return self._configure_registration(
            KVCacheMethod.REGISTER_WORKER,
            registration.identity,
            encode_registration(registration),
        )

    def _configure_registration(
        self, method: KVCacheMethod, identity: SchedulerIdentity | WorkerIdentity, payload: bytes
    ) -> bool:
        with self._registration_lock:
            if self._registration is not None and self._registration.method != method:
                raise RuntimeError("A KVCacheClient cannot register both Scheduler and Worker services")

            self._registration = _ClientRegistration(method, identity, payload)
            self._registered = False

        registered = self._try_register()
        self._rpc_client.start_heartbeat(
            interval_ms=_HEARTBEAT_INTERVAL_MS, timeout_ms=_HEARTBEAT_TIMEOUT_MS, recovery_callback=self._try_register
        )
        return registered

    def _try_register(self) -> bool:
        with self._registration_lock:
            registration = self._registration
            if registration is None:
                return False
            if self._registered and self._rpc_client.is_server_responsive:
                return True

        if not self._rpc_client.is_transport_connected:
            return False

        try:
            if self._rpc_client.ping(timeout_ms=_REGISTRATION_TIMEOUT_MS) != "OK":
                raise MPProtocolError("KV cache server returned an invalid PING response")

            responses = self._rpc_client.request(
                registration.method, (registration.payload,), timeout_ms=_REGISTRATION_TIMEOUT_MS
            )
        except (MPRequestTimeoutError, MPServerUnavailableError):
            self._mark_unregistered()
            return False

        if responses != [_REGISTRATION_RESPONSE]:
            raise MPProtocolError(f"{registration.method.value} expects an OK response, got {responses!r}")

        with self._registration_lock:
            if self._registration != registration:
                return False
            self._registered = True
        return True

    def _mark_unregistered(self) -> None:
        with self._registration_lock:
            self._registered = False

    def _get_scheduler_identity(self) -> SchedulerIdentity:
        with self._registration_lock:
            registration = self._registration

        if registration is None or not isinstance(registration.identity, SchedulerIdentity):
            raise RuntimeError("KVCacheClient is not configured as a Scheduler client")
        return registration.identity

    def lookup(
        self, request: Request, num_computed_tokens: int, timeout_ms: int = _DEFAULT_TIMEOUT_MS
    ) -> tuple[int, bool]:
        identity = self._get_scheduler_identity()
        payloads = _encode_lookup_request(identity, request, num_computed_tokens)

        if not self.is_registered and not self._try_register():
            return 0, False

        try:
            responses = self._rpc_client.request(KVCacheMethod.LOOKUP, payloads, timeout_ms=timeout_ms)
        except (MPRequestTimeoutError, MPServerUnavailableError):
            self._mark_unregistered()
            return 0, False
        except MPRemoteError as exc:
            if str(exc).startswith(_SERVICE_NOT_REGISTERED_PREFIX):
                self._mark_unregistered()
                return 0, False
            raise

        if len(responses) != 2:
            raise MPProtocolError(f"LOOKUP expects 2 response payloads, got {len(responses)}")

        matched_tokens = _decode_non_negative_int(responses[0], "matched_tokens")
        return matched_tokens, _decode_async_response(responses[1])

    def close(self) -> None:
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
        self._registry_lock = threading.RLock()
        self._schedulers: dict[SchedulerIdentity, _RegisteredScheduler] = {}
        self._workers: dict[WorkerIdentity, _RegisteredWorker] = {}
        self._scheduler_factory = scheduler_factory or self._create_scheduler
        self._worker_factory = worker_factory or self._create_worker
        self._rpc_server = MPServer(
            bind_url,
            max_workers=max_workers,
            handlers={
                KVCacheMethod.REGISTER_SCHEDULER: self._handle_register_scheduler,
                KVCacheMethod.REGISTER_WORKER: self._handle_register_worker,
                KVCacheMethod.LOOKUP: self._handle_lookup,
            },
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
        with self._registry_lock:
            return len(self._schedulers)

    @property
    def worker_count(self) -> int:
        with self._registry_lock:
            return len(self._workers)

    @staticmethod
    def _create_scheduler(registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler) -> SchedulerNode:
        from .lookup_worker import MPKVPoolScheduler

        return MPKVPoolScheduler(registration, lookup_handler)

    @staticmethod
    def _create_worker(registration: WorkerRegistration) -> WorkerNode:
        from .lookup_worker import LookupKVPoolWorker

        return LookupKVPoolWorker(
            registration.vllm_config, kv_cache_config=registration.kv_cache_config, rank=registration.identity.rank
        )

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

    def _handle_register_scheduler(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, SchedulerRegistration)
        self._validate_scheduler_registration(registration)
        fingerprint = hashlib.sha256(payloads[0]).digest()

        with self._registry_lock:
            existing = self._schedulers.get(registration.identity)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise RegistrationConflictError(
                        f"Scheduler {registration.identity!r} is already registered with different configuration"
                    )
                return (_REGISTRATION_RESPONSE,)

            service = self._scheduler_factory(registration, self._lookup_worker)
            self._schedulers[registration.identity] = _RegisteredScheduler(fingerprint, service)
            self._bind_engine_store(registration.identity)

        return (_REGISTRATION_RESPONSE,)

    def _handle_register_worker(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        registration = decode_registration(payloads, WorkerRegistration)
        self._validate_worker_registration(registration)
        fingerprint = hashlib.sha256(payloads[0]).digest()

        with self._registry_lock:
            existing = self._workers.get(registration.identity)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise RegistrationConflictError(
                        f"Worker {registration.identity!r} is already registered with different configuration"
                    )
                return (_REGISTRATION_RESPONSE,)

            service = self._worker_factory(registration)
            self._workers[registration.identity] = _RegisteredWorker(fingerprint, service)
            scheduler_identity = SchedulerIdentity(registration.identity.engine_id)
            self._bind_engine_store(scheduler_identity)

        return (_REGISTRATION_RESPONSE,)

    def _bind_engine_store(self, scheduler_identity: SchedulerIdentity) -> None:
        scheduler = self._schedulers.get(scheduler_identity)
        if scheduler is None:
            return

        store = getattr(scheduler.service, "store_scheduler", None)
        if store is None:
            return

        for identity, worker in self._workers.items():
            if identity.engine_id != scheduler_identity.engine_id:
                continue

            bind_store = getattr(worker.service, "bind_store", None)
            if callable(bind_store):
                bind_store(store)

    def _lookup_worker(
        self,
        scheduler_identity: SchedulerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker_identity = WorkerIdentity(scheduler_identity.engine_id, rank=0)
        with self._registry_lock:
            worker = self._workers.get(worker_identity)

        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.service.lookup_scheduler(
            token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens
        )

    def _handle_lookup(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        identity, request, num_computed_tokens = _decode_lookup_request(payloads)

        with self._registry_lock:
            scheduler = self._schedulers.get(identity)

        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

        matched_tokens, is_async = scheduler.service.get_num_new_matched_tokens(
            cast(Request, request), num_computed_tokens
        )
        return (
            _encode_non_negative_int(matched_tokens, "matched_tokens"),
            _ASYNC_RESPONSE if is_async else _SYNC_RESPONSE,
        )

    def run(self) -> None:
        self._rpc_server.run()

    def close(self) -> None:
        self._rpc_server.close()
