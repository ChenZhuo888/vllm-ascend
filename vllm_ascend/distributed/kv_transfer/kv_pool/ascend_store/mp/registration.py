"""Registration and service registry for AscendStore multiprocessing mode.

Registration payloads use cloudpickle because VllmConfig and KVCacheConfig
contain framework-specific Python objects. The MP endpoint must therefore be
restricted to trusted processes.
"""

import hashlib
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import cloudpickle
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig

from .rpc import MPProtocolError

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

logger = logging.getLogger(__name__)

_LEGACY_SESSION_ID = "legacy"

RegistrationT = TypeVar("RegistrationT", bound="SchedulerRegistration | WorkerRegistration")
ServiceT = TypeVar("ServiceT")
WorkerLookupHandler = Callable[["SchedulerIdentity", int, Sequence[BlockHash], list[int] | None, bool, int], int]
SchedulerFactory = Callable[["SchedulerRegistration", WorkerLookupHandler], "KVPoolScheduler"]
WorkerFactory = Callable[["WorkerRegistration"], "KVPoolWorker"]


def _validate_engine_id(engine_id: str) -> None:
    if not isinstance(engine_id, str):
        raise TypeError(f"engine_id must be a string, got {type(engine_id).__name__}")
    if not engine_id:
        raise ValueError("engine_id must not be empty")


def _validate_rank(rank: int, field_name: str) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(rank).__name__}")
    if rank < 0:
        raise ValueError(f"{field_name} must not be negative, got {rank}")


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str):
        raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
    if not session_id:
        raise ValueError("session_id must not be empty")


@dataclass(frozen=True)
class SchedulerIdentity:
    engine_id: str
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _validate_engine_id(self.engine_id)
        _validate_rank(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "SchedulerIdentity":
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class WorkerIdentity:
    engine_id: str
    rank: int
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _validate_engine_id(self.engine_id)
        _validate_rank(self.rank, "rank")
        _validate_rank(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "WorkerIdentity":
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            rank=vllm_config.parallel_config.rank,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class SchedulerRegistration:
    identity: SchedulerIdentity
    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig | None
    page_size_bytes: int
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)

    @classmethod
    def create(
            cls,
            vllm_config: VllmConfig,
            kv_cache_config: KVCacheConfig | None,
            page_size_bytes: int,
            session_id: str = _LEGACY_SESSION_ID,
    ) -> "SchedulerRegistration":
        _validate_rank(page_size_bytes, "page_size_bytes")
        return cls(
            identity=SchedulerIdentity.from_vllm_config(vllm_config),
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            page_size_bytes=page_size_bytes,
            session_id=session_id,
        )


@dataclass(frozen=True)
class WorkerRegistration:
    identity: WorkerIdentity
    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig | None
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)

    @classmethod
    def create(
            cls, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None, session_id: str = _LEGACY_SESSION_ID
    ) -> "WorkerRegistration":
        return cls(
            identity=WorkerIdentity.from_vllm_config(vllm_config),
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            session_id=session_id,
        )


class RegistrationConflictError(RuntimeError):
    pass


class StaleSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ServiceEntry(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    service: ServiceT


@dataclass(frozen=True)
class _RegistrationFlight(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    future: Future[ServiceT]


def encode_registration(registration: SchedulerRegistration | WorkerRegistration) -> bytes:
    try:
        return cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}") from exc


def decode_registration(payloads: tuple[bytes, ...], expected_type: type[RegistrationT]) -> RegistrationT:
    if len(payloads) != 1:
        raise MPProtocolError(f"{expected_type.__name__} expects 1 payload, got {len(payloads)}")

    try:
        registration = cloudpickle.loads(payloads[0])
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {expected_type.__name__}") from exc

    if not isinstance(registration, expected_type):
        raise MPProtocolError(f"Expected {expected_type.__name__}, got {type(registration).__name__}")
    return registration


class KVCacheServiceRegistry:
    """Map Scheduler and Worker identities to their current service instances."""

    def __init__(
            self,
            scheduler_factory: SchedulerFactory,
            worker_factory: WorkerFactory,
            worker_lookup_handler: WorkerLookupHandler,
    ):
        self._lock = threading.RLock()
        self._schedulers: dict[SchedulerIdentity, _ServiceEntry["KVPoolScheduler"]] = {}
        self._workers: dict[WorkerIdentity, _ServiceEntry["KVPoolWorker"]] = {}
        self._registering_schedulers: dict[SchedulerIdentity, _RegistrationFlight["KVPoolScheduler"]] = {}
        self._registering_workers: dict[WorkerIdentity, _RegistrationFlight["KVPoolWorker"]] = {}
        self._retired_scheduler_sessions: dict[SchedulerIdentity, set[str]] = {}
        self._retired_worker_sessions: dict[WorkerIdentity, set[str]] = {}
        self._scheduler_factory = scheduler_factory
        self._worker_factory = worker_factory
        self._worker_lookup_handler = worker_lookup_handler

    @property
    def scheduler_count(self) -> int:
        with self._lock:
            return len(self._schedulers)

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> "KVPoolScheduler":
        self._validate_scheduler_registration(registration)
        identity = registration.identity
        session_id = registration.session_id
        fingerprint = hashlib.sha256(payload).digest()
        old_service = None

        with self._lock:
            self._raise_if_retired("Scheduler", identity, session_id, self._retired_scheduler_sessions)

            entry = self._schedulers.get(identity)
            if entry is not None:
                if entry.session_id == session_id:
                    self._validate_fingerprint("Scheduler", identity, entry.fingerprint, fingerprint)
                    return entry.service

                self._retire_session_locked(self._retired_scheduler_sessions, identity, entry.session_id)
                del self._schedulers[identity]
                old_service = entry.service

            flight = self._registering_schedulers.get(identity)
            if flight is not None:
                if flight.session_id != session_id:
                    raise RegistrationConflictError(
                        f"Scheduler {identity!r} is already registering session {flight.session_id!r}"
                    )
                self._validate_fingerprint("Scheduler", identity, flight.fingerprint, fingerprint)
                future = flight.future
                should_create = False
            else:
                future = Future()
                self._registering_schedulers[identity] = _RegistrationFlight(session_id, fingerprint, future)
                should_create = True

        if not should_create:
            return future.result()

        service = None
        try:
            if old_service is not None:
                self._close_service(old_service)

            service = self._scheduler_factory(registration, self._worker_lookup_handler)
            with self._lock:
                flight = self._registering_schedulers.get(identity)
                assert flight is not None and flight.future is future

                self._schedulers[identity] = _ServiceEntry(session_id, fingerprint, service)
                try:
                    self._bind_engine_store_locked(identity)
                except BaseException:
                    del self._schedulers[identity]
                    raise
                del self._registering_schedulers[identity]
        except BaseException as exc:
            with self._lock:
                flight = self._registering_schedulers.get(identity)
                if flight is not None and flight.future is future:
                    del self._registering_schedulers[identity]
            if service is not None:
                self._close_service_safely(service)
            if not future.done():
                future.set_exception(exc)
            raise

        future.set_result(service)
        return service

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "KVPoolWorker":
        self._validate_worker_registration(registration)
        identity = registration.identity
        session_id = registration.session_id
        fingerprint = hashlib.sha256(payload).digest()
        old_service = None

        with self._lock:
            self._raise_if_retired("Worker", identity, session_id, self._retired_worker_sessions)

            entry = self._workers.get(identity)
            if entry is not None:
                if entry.session_id == session_id:
                    self._validate_fingerprint("Worker", identity, entry.fingerprint, fingerprint)
                    return entry.service

                self._retire_session_locked(self._retired_worker_sessions, identity, entry.session_id)
                del self._workers[identity]
                old_service = entry.service

            flight = self._registering_workers.get(identity)
            if flight is not None:
                if flight.session_id != session_id:
                    raise RegistrationConflictError(
                        f"Worker {identity!r} is already registering session {flight.session_id!r}"
                    )
                self._validate_fingerprint("Worker", identity, flight.fingerprint, fingerprint)
                future = flight.future
                should_create = False
            else:
                future = Future()
                self._registering_workers[identity] = _RegistrationFlight(session_id, fingerprint, future)
                should_create = True

        if not should_create:
            return future.result()

        service = None
        try:
            if old_service is not None:
                self._close_service(old_service)

            service = self._worker_factory(registration)
            scheduler_identity = SchedulerIdentity(identity.engine_id, identity.data_parallel_rank)
            with self._lock:
                flight = self._registering_workers.get(identity)
                assert flight is not None and flight.future is future

                self._workers[identity] = _ServiceEntry(session_id, fingerprint, service)
                try:
                    self._bind_engine_store_locked(scheduler_identity)
                except BaseException:
                    del self._workers[identity]
                    raise
                del self._registering_workers[identity]
        except BaseException as exc:
            with self._lock:
                flight = self._registering_workers.get(identity)
                if flight is not None and flight.future is future:
                    del self._registering_workers[identity]
            if service is not None:
                self._close_service_safely(service)
            if not future.done():
                future.set_exception(exc)
            raise

        future.set_result(service)
        return service

    def unregister_scheduler(self, identity: SchedulerIdentity, session_id: str) -> bool:
        _validate_session_id(session_id)
        with self._lock:
            self._raise_if_retired("Scheduler", identity, session_id, self._retired_scheduler_sessions)
            entry = self._schedulers.get(identity)
            if entry is None:
                return False
            if entry.session_id != session_id:
                raise StaleSessionError(
                    f"Scheduler {identity!r} session {session_id!r} is stale; current session is {entry.session_id!r}"
                )

            del self._schedulers[identity]
            self._retire_session_locked(self._retired_scheduler_sessions, identity, session_id)
            service = entry.service

        self._close_service(service)
        return True

    def unregister_worker(self, identity: WorkerIdentity, session_id: str) -> bool:
        _validate_session_id(session_id)
        with self._lock:
            self._raise_if_retired("Worker", identity, session_id, self._retired_worker_sessions)
            entry = self._workers.get(identity)
            if entry is None:
                return False
            if entry.session_id != session_id:
                raise StaleSessionError(
                    f"Worker {identity!r} session {session_id!r} is stale; current session is {entry.session_id!r}"
                )

            del self._workers[identity]
            self._retire_session_locked(self._retired_worker_sessions, identity, session_id)
            service = entry.service

        self._close_service(service)
        return True

    def get_scheduler(self, identity: SchedulerIdentity, session_id: str | None = None) -> "KVPoolScheduler | None":
        with self._lock:
            if session_id is not None:
                self._raise_if_retired("Scheduler", identity, session_id, self._retired_scheduler_sessions)

            entry = self._schedulers.get(identity)
            if entry is None:
                return None
            if session_id is not None and entry.session_id != session_id:
                raise StaleSessionError(
                    f"Scheduler {identity!r} session {session_id!r} is stale; current session is {entry.session_id!r}"
                )
            return entry.service

    def get_worker(self, identity: WorkerIdentity, session_id: str | None = None) -> "KVPoolWorker | None":
        with self._lock:
            if session_id is not None:
                self._raise_if_retired("Worker", identity, session_id, self._retired_worker_sessions)

            entry = self._workers.get(identity)
            if entry is None:
                return None
            if session_id is not None and entry.session_id != session_id:
                raise StaleSessionError(
                    f"Worker {identity!r} session {session_id!r} is stale; current session is {entry.session_id!r}"
                )
            return entry.service

    def close(self) -> None:
        with self._lock:
            services = [entry.service for entry in self._workers.values()]
            services.extend(entry.service for entry in self._schedulers.values())
            self._workers.clear()
            self._schedulers.clear()
            self._registering_workers.clear()
            self._registering_schedulers.clear()
            self._retired_worker_sessions.clear()
            self._retired_scheduler_sessions.clear()

        for service in services:
            self._close_service_safely(service)

    @staticmethod
    def _retire_session_locked(retired_sessions, identity, session_id: str) -> None:
        retired_sessions.setdefault(identity, set()).add(session_id)

    @staticmethod
    def _raise_if_retired(role: str, identity, session_id: str, retired_sessions) -> None:
        if session_id in retired_sessions.get(identity, ()):
            raise StaleSessionError(f"{role} {identity!r} session {session_id!r} has been retired")

    @staticmethod
    def _close_service(service) -> None:
        close = getattr(service, "close", None)
        if callable(close):
            close()

    @classmethod
    def _close_service_safely(cls, service) -> None:
        try:
            cls._close_service(service)
        except Exception:
            logger.exception("Failed to close KV cache service %r", service)

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

    @staticmethod
    def _validate_fingerprint(
            role: str, identity: SchedulerIdentity | WorkerIdentity, existing: bytes, incoming: bytes
    ) -> None:
        if existing != incoming:
            raise RegistrationConflictError(f"{role} {identity!r} is already registered with different configuration")

    def _bind_engine_store_locked(self, scheduler_identity: SchedulerIdentity) -> None:
        scheduler = self._schedulers.get(scheduler_identity)
        if scheduler is None:
            return

        store = getattr(scheduler.service, "store_scheduler", None)
        if store is None:
            return

        for identity, worker in self._workers.items():
            if (
                    identity.engine_id != scheduler_identity.engine_id
                    or identity.data_parallel_rank != scheduler_identity.data_parallel_rank
            ):
                continue
            bind_store = getattr(worker.service, "bind_store", None)
            if callable(bind_store):
                bind_store(store)
