"""Stoppable variants of the original KV pool transfer threads."""

import threading
from typing import Any

from vllm.logger import logger

from ....kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)

_STOP_REQUEST = object()


class _MPTransferThreadMixin:
    """Drain accepted requests before stopping a Worker-owned transfer thread."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._admission_lock = threading.Lock()
        self._accepting_requests = True
        self._stop_enqueued = False

    def add_request(self, request: Any) -> None:
        with self._admission_lock:
            if not self._accepting_requests:
                raise RuntimeError(f"{self.name} is stopping and no longer accepts requests")
            self.request_queue.put(request)

    def stop(self, wait: bool = True) -> None:
        with self._admission_lock:
            self._accepting_requests = False
            if not self._stop_enqueued and self.ident is not None:
                self._stop_enqueued = True
                self.request_queue.put(_STOP_REQUEST)

        if wait and self.ident is not None and self is not threading.current_thread():
            self.join()

    def run(self) -> None:
        """Preserve the original request handling and add an ordered stop marker."""
        self._set_os_thread_name()
        try:
            self.m_store.set_device()
        except Exception as exc:
            self._record_fatal_error(exc)
            self.ready_event.set()
            return

        self.ready_event.set()
        while True:
            request = self.request_queue.get()
            if request is _STOP_REQUEST:
                self.request_queue.task_done()
                return
            try:
                self._handle_request(request)
            except Exception as exc:
                self._record_fatal_error(exc)
                return

    def _record_fatal_error(self, error: BaseException) -> None:
        self._fatal_error = error
        logger.error(
            "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
            self.name,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


class MPKVCacheStoreSendingThread(_MPTransferThreadMixin, KVCacheStoreSendingThread):
    pass


class MPKVCacheStoreRecvingThread(_MPTransferThreadMixin, KVCacheStoreRecvingThread):
    pass


class MPKVCacheStoreKeyLayerSendingThread(_MPTransferThreadMixin, KVCacheStoreKeyLayerSendingThread):
    pass


class MPKVCacheStoreKeyLayerRecvingThread(_MPTransferThreadMixin, KVCacheStoreKeyLayerRecvingThread):
    pass


class MPKVCacheStoreLayerSendingThread(_MPTransferThreadMixin, KVCacheStoreLayerSendingThread):
    pass


class MPKVCacheStoreLayerRecvingThread(_MPTransferThreadMixin, KVCacheStoreLayerRecvingThread):
    pass


__all__ = [
    "MPKVCacheStoreKeyLayerRecvingThread",
    "MPKVCacheStoreKeyLayerSendingThread",
    "MPKVCacheStoreLayerRecvingThread",
    "MPKVCacheStoreLayerSendingThread",
    "MPKVCacheStoreRecvingThread",
    "MPKVCacheStoreSendingThread",
]
