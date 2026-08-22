import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ServiceReaper:
    """Periodically remove stale services through a registry callback."""

    def __init__(
        self,
        reap_stale: Callable[[float], object],
        stale_timeout_s: float,
        interval_s: float,
        thread_name: str = "service-reaper",
    ):
        if stale_timeout_s <= 0:
            raise ValueError(f"stale_timeout_s must be greater than 0, got {stale_timeout_s}")
        if interval_s <= 0:
            raise ValueError(f"interval_s must be greater than 0, got {interval_s}")

        self._reap_stale = reap_stale
        self._stale_timeout_s = stale_timeout_s
        self._interval_s = interval_s
        self._thread_name = thread_name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name=self._thread_name)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()

        if thread is not threading.current_thread():
            thread.join()

        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._reap_stale(time.monotonic() - self._stale_timeout_s)
            except Exception:
                logger.exception("Service reaper failed")
