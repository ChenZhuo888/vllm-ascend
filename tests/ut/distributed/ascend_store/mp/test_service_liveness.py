import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_protocol import encode_registration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_registry import KVCacheServiceRegistry
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import (
    RegistrationConflictError,
    ServiceBusyError,
    StaleSessionError,
)

REGISTRY_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_registry"


class _FakeScheduler:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _BlockingCloseScheduler(_FakeScheduler):
    def __init__(self, close_started: threading.Event, release_close: threading.Event):
        super().__init__()
        self._close_started = close_started
        self._release_close = release_close

    def close(self) -> None:
        super().close()
        self._close_started.set()
        if not self._release_close.wait(5):
            raise TimeoutError("Timed out waiting to release Scheduler close")


class _FakeWorker:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _make_vllm_config(engine_id: str = "engine-0", rank: int = 0, data_parallel_rank: int = 0, marker: str = ""):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id=engine_id),
        parallel_config=SimpleNamespace(rank=rank, data_parallel_rank=data_parallel_rank),
        marker=marker,
    )


def _scheduler_registration(session_id: str, marker: str = "") -> SchedulerRegistration:
    return SchedulerRegistration.create(_make_vllm_config(marker=marker), None, 0, session_id=session_id)


def _worker_registration(session_id: str) -> WorkerRegistration:
    return WorkerRegistration.create(_make_vllm_config(), None, session_id=session_id)


def _create_registry(scheduler_factory=None, worker_factory=None) -> KVCacheServiceRegistry:
    return KVCacheServiceRegistry(
        scheduler_factory or (lambda registration: _FakeScheduler()),
        worker_factory or (lambda registration: _FakeWorker()),
    )


def test_reaped_scheduler_can_recover_with_the_same_session() -> None:
    created = []

    def scheduler_factory(registration):
        scheduler = _FakeScheduler()
        created.append(scheduler)
        return scheduler

    registry = _create_registry(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        first_service = registry.register_scheduler(registration, payload)

    assert registry.reap_stale(11.0) == (1, 0)
    assert first_service.close_count == 1
    assert registry.scheduler_count == 0

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=20.0):
        second_service = registry.register_scheduler(registration, payload)

    assert second_service is created[1]
    assert second_service is not first_service
    assert registry.scheduler_count == 1


def test_reaped_session_keeps_its_configuration_fingerprint() -> None:
    registry = _create_registry()
    registration = _scheduler_registration("session-0", marker="first")
    payload = encode_registration(registration)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        registry.register_scheduler(registration, payload)
    registry.reap_stale(11.0)

    conflicting = _scheduler_registration("session-0", marker="second")
    with pytest.raises(RegistrationConflictError, match="different configuration"):
        registry.register_scheduler(conflicting, encode_registration(conflicting))


def test_new_session_after_reap_retires_the_old_session() -> None:
    registry = _create_registry()
    old_registration = _scheduler_registration("old-session")
    new_registration = _scheduler_registration("new-session")

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        registry.register_scheduler(old_registration, encode_registration(old_registration))
    registry.reap_stale(11.0)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=20.0):
        registry.register_scheduler(new_registration, encode_registration(new_registration))

    with pytest.raises(StaleSessionError, match="retired"):
        registry.register_scheduler(old_registration, encode_registration(old_registration))


def test_registration_is_busy_while_stale_service_is_closing() -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    def scheduler_factory(registration):
        return _BlockingCloseScheduler(close_started, release_close)

    registry = _create_registry(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        registry.register_scheduler(registration, payload)

    with ThreadPoolExecutor(max_workers=1) as executor:
        reap_future = executor.submit(registry.reap_stale, 11.0)
        assert close_started.wait(5), "Stale Scheduler did not start closing"
        try:
            with pytest.raises(ServiceBusyError, match="being reaped"):
                registry.register_scheduler(registration, payload)
        finally:
            release_close.set()

        assert reap_future.result(timeout=5) == (1, 0)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=20.0):
        assert registry.register_scheduler(registration, payload) is not None


def test_scheduler_access_refreshes_liveness_but_internal_worker_access_does_not() -> None:
    scheduler = _FakeScheduler()
    worker = _FakeWorker()
    registry = _create_registry(
        scheduler_factory=lambda registration: scheduler,
        worker_factory=lambda registration: worker,
    )
    scheduler_registration = _scheduler_registration("scheduler-session")
    worker_registration = _worker_registration("worker-session")

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        registry.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
        registry.register_worker(worker_registration, encode_registration(worker_registration))

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=100.0):
        assert registry.get_scheduler(scheduler_registration.identity, scheduler_registration.session_id) is scheduler
        assert registry.get_worker(worker_registration.identity) is worker

    assert registry.reap_stale(50.0) == (0, 1)
    assert scheduler.close_count == 0
    assert worker.close_count == 1


def test_unregistering_recoverable_session_retires_it() -> None:
    registry = _create_registry()
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with patch(f"{REGISTRY_MODULE}.time.monotonic", return_value=10.0):
        registry.register_scheduler(registration, payload)
    registry.reap_stale(11.0)

    assert registry.unregister_scheduler(registration.identity, registration.session_id)
    with pytest.raises(StaleSessionError, match="retired"):
        registry.register_scheduler(registration, payload)
