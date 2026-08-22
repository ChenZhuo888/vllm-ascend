"""KV cache service registration and Scheduler-Worker binding."""

import hashlib
import threading
import time
from typing import TYPE_CHECKING

from .registration import (
    SchedulerFactory,
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)
from .rpc import MPServerBusyError
from .service import ServiceBusyError, ServiceRegistry

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker


def _monotonic() -> float:
    return time.monotonic()


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
