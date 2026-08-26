import queue
import threading
from typing import Any

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.transfer import (
    _MPTransferThreadMixin,
)

# isort: on


class _Store:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def set_device(self) -> None:
        if self.error is not None:
            raise self.error


class _TestTransferThread(_MPTransferThreadMixin, threading.Thread):
    def __init__(self, store: _Store | None = None):
        super().__init__(daemon=True, name="test-transfer")
        self.m_store = store or _Store()
        self.ready_event = threading.Event()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.handled: list[Any] = []
        self._fatal_error: BaseException | None = None

    @staticmethod
    def _set_os_thread_name() -> None:
        return None

    def _handle_request(self, request: Any) -> None:
        self.handled.append(request)
        self.request_queue.task_done()


def test_mp_transfer_thread_drains_accepted_requests_before_stopping() -> None:
    thread = _TestTransferThread()
    thread.start()
    assert thread.ready_event.wait(timeout=1)

    thread.add_request("first")
    thread.add_request("second")
    thread.stop()

    assert thread.handled == ["first", "second"]
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="no longer accepts requests"):
        thread.add_request("late")


def test_mp_transfer_thread_reports_device_setup_failure_without_blocking_startup() -> None:
    thread = _TestTransferThread(_Store(RuntimeError("device unavailable")))
    thread.start()

    assert thread.ready_event.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert isinstance(thread._fatal_error, RuntimeError)
