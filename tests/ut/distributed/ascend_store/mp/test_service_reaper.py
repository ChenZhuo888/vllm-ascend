import threading
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import ServiceReaper

REAPER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service.reaper"


def test_reaper_passes_stale_threshold_to_registry() -> None:
    reaped = threading.Event()
    stale_thresholds = []

    def reap_stale(stale_before: float) -> None:
        stale_thresholds.append(stale_before)
        reaped.set()

    reaper = ServiceReaper(reap_stale, stale_timeout_s=60.0, interval_s=0.01)
    with patch(f"{REAPER_MODULE}.time.monotonic", return_value=100.0):
        reaper.start()
        try:
            assert reaped.wait(1), "Service reaper did not run"
        finally:
            reaper.stop()

    assert stale_thresholds[0] == 40.0
    assert not reaper.is_running


def test_reaper_is_idempotent_and_survives_callback_failure() -> None:
    callback_finished = threading.Event()
    attempts = 0

    def reap_stale(_stale_before: float) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("reap failed")
        callback_finished.set()

    reaper = ServiceReaper(reap_stale, stale_timeout_s=1.0, interval_s=0.01)
    with patch(f"{REAPER_MODULE}.logger.exception") as log_exception:
        reaper.start()
        reaper.start()
        try:
            assert callback_finished.wait(1), "Service reaper stopped after callback failure"
        finally:
            reaper.stop()
            reaper.stop()

        log_exception.assert_called_once_with("Service reaper failed")

    assert attempts >= 2
    assert not reaper.is_running


@pytest.mark.parametrize(
    ("stale_timeout_s", "interval_s", "field_name"),
    [(0.0, 1.0, "stale_timeout_s"), (1.0, 0.0, "interval_s")],
)
def test_reaper_rejects_non_positive_intervals(
    stale_timeout_s: float,
    interval_s: float,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ServiceReaper(lambda _stale_before: None, stale_timeout_s, interval_s)
