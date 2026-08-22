"""Registration models and KV cache service orchestration.

Registration payloads use cloudpickle because VllmConfig and KVCacheConfig
contain framework-specific Python objects. The MP endpoint must therefore be
restricted to trusted processes.
"""

import hashlib
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import cloudpickle
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig

from .rpc import MPProtocolError, MPServerBusyError
from .service import RegistrationConflictError as RegistrationConflictError
from .service import ServiceBusyError, ServiceRegistry
from .service import StaleSessionError as StaleSessionError

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

_LEGACY_SESSION_ID = "legacy"

RegistrationT = TypeVar("RegistrationT", bound="SchedulerRegistration | WorkerRegistration")
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


def _monotonic() -> float:
    return time.monotonic()


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
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        session_id: str = _LEGACY_SESSION_ID,
    ) -> "WorkerRegistration":
        return cls(
            identity=WorkerIdentity.from_vllm_config(vllm_config),
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            session_id=session_id,
        )


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
    """Orchestrate Scheduler and Worker services for the KV cache domain."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory,
        worker_factory: WorkerFactory,
        worker_lookup_handler: WorkerLookupHandler,
    ):
        self._scheduler_factory = scheduler_factory
        self._worker_factory = worker_factory
        self._worker_lookup_handler = worker_lookup_handler
        self._binding_lock = threading.Lock()
        self._schedulers = ServiceRegistry[SchedulerIdentity, "KVPoolScheduler"](
            "Scheduler", self._close_service, _monotonic
        )
        self._workers = ServiceRegistry[WorkerIdentity, "KVPoolWorker"]("Worker", self._close_service, _monotonic)

    @property
    def scheduler_count(self) -> int:
        return self._schedulers.count

    @property
    def worker_count(self) -> int:
        return self._workers.count

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> "KVPoolScheduler":
        self._validate_scheduler_registration(registration)
        try:
            scheduler = self._schedulers.register(
                registration.identity,
                registration.session_id,
                hashlib.sha256(payload).digest(),
                lambda: self._scheduler_factory(registration, self._worker_lookup_handler),
            )
        except ServiceBusyError as exc:
            raise MPServerBusyError(str(exc)) from exc

        self._bind_engine_store(registration.identity)
        return scheduler

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "KVPoolWorker":
        self._validate_worker_registration(registration)
        try:
            worker = self._workers.register(
                registration.identity,
                registration.session_id,
                hashlib.sha256(payload).digest(),
                lambda: self._worker_factory(registration),
            )
        except ServiceBusyError as exc:
            raise MPServerBusyError(str(exc)) from exc

        scheduler_identity = SchedulerIdentity(
            registration.identity.engine_id,
            registration.identity.data_parallel_rank,
        )
        self._bind_engine_store(scheduler_identity)
        return worker

    def unregister_scheduler(self, identity: SchedulerIdentity, session_id: str) -> bool:
        return self._schedulers.unregister(identity, session_id)

    def unregister_worker(self, identity: WorkerIdentity, session_id: str) -> bool:
        return self._workers.unregister(identity, session_id)

    def touch_scheduler(self, identity: SchedulerIdentity, session_id: str) -> bool:
        return self._schedulers.touch(identity, session_id)

    def touch_worker(self, identity: WorkerIdentity, session_id: str) -> bool:
        return self._workers.touch(identity, session_id)

    def get_scheduler(self, identity: SchedulerIdentity, session_id: str | None = None) -> "KVPoolScheduler | None":
        return self._schedulers.get(identity, session_id)

    def get_worker(self, identity: WorkerIdentity, session_id: str | None = None) -> "KVPoolWorker | None":
        return self._workers.get(identity, session_id)

    def reap_stale(self, stale_before: float) -> tuple[int, int]:
        return self._schedulers.reap_stale(stale_before), self._workers.reap_stale(stale_before)

    def close(self) -> None:
        self._workers.close()
        self._schedulers.close()

    def _bind_engine_store(self, scheduler_identity: SchedulerIdentity) -> None:
        with self._binding_lock:
            scheduler = self._schedulers.get(scheduler_identity)
            if scheduler is None:
                return

            store = getattr(scheduler, "store_scheduler", None)
            if store is None:
                return

            for identity, worker in self._workers.items():
                if (
                    identity.engine_id != scheduler_identity.engine_id
                    or identity.data_parallel_rank != scheduler_identity.data_parallel_rank
                ):
                    continue
                bind_store = getattr(worker, "bind_store", None)
                if callable(bind_store):
                    bind_store(store)

    @staticmethod
    def _close_service(service) -> None:
        close = getattr(service, "close", None)
        if callable(close):
            close()

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
