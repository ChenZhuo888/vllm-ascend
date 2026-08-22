import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import (
    RegistrationConflictError,
    ServiceRegistry,
    StaleSessionError,
)


class _FakeService:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _create_registry(clock=lambda: 0.0) -> ServiceRegistry[str, _FakeService]:
    return ServiceRegistry("Test", lambda service: service.close(), clock)


def test_register_get_touch_and_reap_service() -> None:
    now = [10.0]
    registry = _create_registry(lambda: now[0])
    service = registry.register("service-0", "session-0", b"config", _FakeService)

    assert registry.get("service-0") is service
    assert registry.items() == (("service-0", service),)
    assert registry.count == 1

    now[0] = 20.0
    assert registry.touch("service-0", "session-0")
    assert registry.reap_stale(15.0) == 0
    assert registry.reap_stale(21.0) == 1
    assert service.close_count == 1

    recovered = registry.register("service-0", "session-0", b"config", _FakeService)
    assert recovered is not service


def test_identical_concurrent_registration_shares_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created = []

    def factory() -> _FakeService:
        service = _FakeService()
        created.append(service)
        factory_started.set()
        assert release_factory.wait(5), "Service factory was not released"
        return service

    registry = _create_registry()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(registry.register, "service-0", "session-0", b"config", factory)
        assert factory_started.wait(5), "Service factory did not start"
        second_future = executor.submit(registry.register, "service-0", "session-0", b"config", factory)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created == [first_service]


def test_registration_conflict_and_retired_session_are_rejected() -> None:
    registry = _create_registry()
    registry.register("service-0", "session-0", b"config", _FakeService)

    with pytest.raises(RegistrationConflictError, match="different configuration"):
        registry.register("service-0", "session-0", b"changed", _FakeService)

    assert registry.unregister("service-0", "session-0")
    with pytest.raises(StaleSessionError, match="retired"):
        registry.register("service-0", "session-0", b"config", _FakeService)
