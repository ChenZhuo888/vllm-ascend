import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.manager import KVCacheServiceManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import encode_registration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import (
    RegistrationConflictError,
    StaleSessionError,
)


class _FakeScheduler:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False


class _FakeWorker:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return 0


def _make_vllm_config(engine_id: str = "engine-0", rank: int = 0, data_parallel_rank: int = 0, marker: str = ""):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id=engine_id),
        parallel_config=SimpleNamespace(rank=rank, data_parallel_rank=data_parallel_rank),
        marker=marker,
    )


def _scheduler_registration(session_id: str, *, data_parallel_rank: int = 0, marker: str = ""):
    return SchedulerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank, marker=marker),
        None,
        0,
        session_id=session_id,
    )


def _worker_registration(session_id: str, *, data_parallel_rank: int = 0, rank: int = 0):
    return WorkerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank, rank=rank),
        None,
        session_id=session_id,
    )


def _create_service_manager(scheduler_factory=None, worker_factory=None) -> KVCacheServiceManager:
    scheduler_factory = scheduler_factory or (lambda registration: _FakeScheduler())
    return KVCacheServiceManager(
        lambda registration, _lookup_handler: scheduler_factory(registration),
        worker_factory or (lambda registration: _FakeWorker()),
    )


def test_scheduler_factories_for_different_identities_run_in_parallel() -> None:
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def scheduler_factory(registration):
        if registration.identity.data_parallel_rank == 0:
            first_started.set()
            assert release_first.wait(5), "First Scheduler factory was not released"
        else:
            second_started.set()
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    first_registration = _scheduler_registration("session-0", data_parallel_rank=0)
    second_registration = _scheduler_registration("session-1", data_parallel_rank=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            service_manager.register_scheduler, first_registration, encode_registration(first_registration)
        )
        assert first_started.wait(5), "First Scheduler factory did not start"

        second_future = executor.submit(
            service_manager.register_scheduler, second_registration, encode_registration(second_registration)
        )
        try:
            assert second_started.wait(1), "Second Scheduler factory was blocked by the first registration"
        finally:
            release_first.set()

        assert isinstance(first_future.result(timeout=5), _FakeScheduler)
        assert isinstance(second_future.result(timeout=5), _FakeScheduler)

    assert service_manager.scheduler_count == 2


def test_concurrent_identical_scheduler_registration_shares_one_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created_schedulers = []

    def scheduler_factory(registration):
        scheduler = _FakeScheduler()
        created_schedulers.append(scheduler)
        factory_started.set()
        assert release_factory.wait(5), "Scheduler factory was not released"
        return scheduler

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_scheduler, registration, payload)
        assert factory_started.wait(5), "Scheduler factory did not start"
        second_future = executor.submit(service_manager.register_scheduler, registration, payload)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created_schedulers == [first_service]
    assert service_manager.scheduler_count == 1


def test_conflicting_scheduler_registration_fails_while_original_is_registering() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def scheduler_factory(registration):
        factory_started.set()
        assert release_factory.wait(5), "Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0", marker="first")
    conflicting = _scheduler_registration("session-0", marker="second")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service_manager.register_scheduler, registration, encode_registration(registration))
        assert factory_started.wait(5), "Scheduler factory did not start"

        try:
            with pytest.raises(RegistrationConflictError, match="different configuration"):
                service_manager.register_scheduler(conflicting, encode_registration(conflicting))
        finally:
            release_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_concurrent_scheduler_registration_shares_factory_failure_and_next_request_retries() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    attempts = 0

    def scheduler_factory(registration):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            factory_started.set()
            assert release_factory.wait(5), "Scheduler factory was not released"
            raise RuntimeError("factory failed")
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_scheduler, registration, payload)
        assert factory_started.wait(5), "Scheduler factory did not start"
        second_future = executor.submit(service_manager.register_scheduler, registration, payload)
        release_factory.set()

        with pytest.raises(RuntimeError, match="factory failed"):
            first_future.result(timeout=5)
        with pytest.raises(RuntimeError, match="factory failed"):
            second_future.result(timeout=5)

    assert attempts == 1
    assert service_manager.scheduler_count == 0

    assert isinstance(service_manager.register_scheduler(registration, payload), _FakeScheduler)
    assert attempts == 2
    assert service_manager.scheduler_count == 1


def test_new_scheduler_session_replaces_and_retires_old_session() -> None:
    created = []

    def scheduler_factory(registration):
        scheduler = _FakeScheduler()
        created.append(scheduler)
        return scheduler

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    old_registration = _scheduler_registration("old-session")
    new_registration = _scheduler_registration("new-session")

    old_service = service_manager.register_scheduler(old_registration, encode_registration(old_registration))
    new_service = service_manager.register_scheduler(new_registration, encode_registration(new_registration))

    assert old_service.close_count == 1
    assert new_service is created[1]
    assert service_manager.lookup(new_registration.identity, "new-session", SimpleNamespace(), 0) == (0, False)

    with pytest.raises(StaleSessionError, match="retired"):
        service_manager.register_scheduler(old_registration, encode_registration(old_registration))
    with pytest.raises(StaleSessionError):
        service_manager.lookup(old_registration.identity, "old-session", SimpleNamespace(), 0)


def test_old_session_is_fenced_while_new_session_is_registering() -> None:
    new_factory_started = threading.Event()
    release_new_factory = threading.Event()

    def scheduler_factory(registration):
        if registration.session_id == "new-session":
            new_factory_started.set()
            assert release_new_factory.wait(5), "New Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    old_registration = _scheduler_registration("old-session")
    new_registration = _scheduler_registration("new-session")
    service_manager.register_scheduler(old_registration, encode_registration(old_registration))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service_manager.register_scheduler, new_registration, encode_registration(new_registration)
        )
        assert new_factory_started.wait(5), "New Scheduler factory did not start"
        try:
            with pytest.raises(StaleSessionError, match="retired"):
                service_manager.register_scheduler(old_registration, encode_registration(old_registration))
        finally:
            release_new_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_different_new_session_is_rejected_while_session_transition_is_running() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def scheduler_factory(registration):
        if registration.session_id == "session-1":
            factory_started.set()
            assert release_factory.wait(5), "Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    first = _scheduler_registration("session-0")
    second = _scheduler_registration("session-1")
    third = _scheduler_registration("session-2")
    service_manager.register_scheduler(first, encode_registration(first))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service_manager.register_scheduler, second, encode_registration(second))
        assert factory_started.wait(5), "Scheduler factory did not start"
        try:
            with pytest.raises(RegistrationConflictError, match="already registering session"):
                service_manager.register_scheduler(third, encode_registration(third))
        finally:
            release_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_unregister_retires_current_session_and_closes_service() -> None:
    service_manager = _create_service_manager()
    registration = _scheduler_registration("session-0")
    service = service_manager.register_scheduler(registration, encode_registration(registration))

    assert service_manager.unregister_scheduler(registration.identity, registration.session_id)
    assert service.close_count == 1
    assert service_manager.scheduler_count == 0

    with pytest.raises(StaleSessionError, match="retired"):
        service_manager.register_scheduler(registration, encode_registration(registration))


def test_concurrent_identical_worker_registration_shares_one_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created_workers = []

    def worker_factory(registration):
        worker = _FakeWorker()
        created_workers.append(worker)
        factory_started.set()
        assert release_factory.wait(5), "Worker factory was not released"
        return worker

    service_manager = _create_service_manager(worker_factory=worker_factory)
    registration = _worker_registration("worker-session")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_worker, registration, payload)
        assert factory_started.wait(5), "Worker factory did not start"
        second_future = executor.submit(service_manager.register_worker, registration, payload)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created_workers == [first_service]
    assert service_manager.worker_count == 1
