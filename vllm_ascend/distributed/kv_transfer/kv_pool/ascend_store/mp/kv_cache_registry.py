"""KV cache service instance lifecycle management."""

import hashlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerFactory,
    WorkerIdentity,
    WorkerRegistration,
)
from .service import ServiceRegistry

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

SchedulerServiceFactory = Callable[[SchedulerRegistration], "KVPoolScheduler"]


def _monotonic() -> float:
    return time.monotonic()


class KVCacheServiceRegistry:
    """Manage Scheduler and Worker service instance lifecycles."""

    def __init__(
        self,
        scheduler_factory: SchedulerServiceFactory,
        worker_factory: WorkerFactory,
    ):
        self._scheduler_factory = scheduler_factory
        self._worker_factory = worker_factory
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
        return self._schedulers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._scheduler_factory(registration),
        )

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "KVPoolWorker":
        self._validate_worker_registration(registration)
        return self._workers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._worker_factory(registration),
        )

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
