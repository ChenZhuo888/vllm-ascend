"""Bounded execution primitives used by the multiprocess RPC server."""

import queue
import threading
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .error import MPServerBusyError

_ResultT = TypeVar("_ResultT")
_STOP = object()


class TaskExecutor(Protocol):
    """Submit work with optional affinity and blocking admission."""

    def submit(
        self, fn: Callable[[], _ResultT], key: Hashable | None = None, block: bool = False
    ) -> Future[_ResultT]: ...

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None: ...


def _validate_limits(max_workers: int, max_pending_tasks: int) -> None:
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")
    if max_pending_tasks < 0:
        raise ValueError("max_pending_tasks must not be negative")


class InlineExecutor:
    """Execute tasks immediately in the submitting thread."""

    def submit(self, fn: Callable[[], _ResultT], key: Hashable | None = None, block: bool = False) -> Future[_ResultT]:
        if key is not None:
            raise ValueError("Inline executor does not accept an execution key")

        future: Future[_ResultT] = Future()
        if not future.set_running_or_notify_cancel():
            return future

        try:
            result = fn()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


class BoundedThreadPoolExecutor:
    """Thread pool with a bounded number of running and pending tasks."""

    def __init__(self, max_workers: int, max_pending_tasks: int, thread_name_prefix: str):
        _validate_limits(max_workers, max_pending_tasks)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._capacity = threading.BoundedSemaphore(max_workers + max_pending_tasks)

    def submit(self, fn: Callable[[], _ResultT], key: Hashable | None = None, block: bool = False) -> Future[_ResultT]:
        if key is not None:
            raise ValueError("Parallel executor does not accept an execution key")
        if not self._capacity.acquire(blocking=block):
            raise MPServerBusyError("Parallel executor is at capacity")

        try:
            future = self._executor.submit(fn)
        except BaseException:
            self._capacity.release()
            raise

        future.add_done_callback(self._release_capacity)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _release_capacity(self, _future: Future[_ResultT]) -> None:
        self._capacity.release()


@dataclass(frozen=True)
class _AffinityTask(Generic[_ResultT]):
    future: Future[_ResultT]
    callback: Callable[[], _ResultT]


class AffinityExecutor:
    """Execute tasks with the same key serially on the same worker thread.

    A key is deterministically mapped to one worker for the executor lifetime.
    The total number of running and pending tasks is bounded.
    """

    def __init__(self, max_workers: int, max_pending_tasks: int, thread_name_prefix: str):
        _validate_limits(max_workers, max_pending_tasks)
        self._queues: list[queue.Queue[object]] = [queue.Queue() for _ in range(max_workers)]
        self._capacity = threading.BoundedSemaphore(max_workers + max_pending_tasks)
        self._state_lock = threading.Lock()
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._run_worker,
                args=(task_queue,),
                name=f"{thread_name_prefix}-{worker_index}",
            )
            for worker_index, task_queue in enumerate(self._queues)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, fn: Callable[[], _ResultT], key: Hashable | None = None, block: bool = False) -> Future[_ResultT]:
        if key is None:
            raise ValueError("Affinity task must define an affinity key")

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Affinity executor is closed")
            if not self._capacity.acquire(blocking=block):
                raise MPServerBusyError("Affinity executor is at capacity")

            try:
                worker_index = hash(key) % len(self._queues)
            except BaseException:
                self._capacity.release()
                raise

            future: Future[_ResultT] = Future()
            future.add_done_callback(self._release_capacity)
            self._queues[worker_index].put_nowait(_AffinityTask(future, fn))
            return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._state_lock:
            if not self._closed:
                self._closed = True

                if cancel_futures:
                    for task_queue in self._queues:
                        self._cancel_pending_tasks(task_queue)

                for task_queue in self._queues:
                    task_queue.put(_STOP)

        if wait:
            current_thread = threading.current_thread()
            for thread in self._threads:
                if thread is not current_thread:
                    thread.join()

    def _run_worker(self, task_queue: queue.Queue[object]) -> None:
        while True:
            task = task_queue.get()
            try:
                if task is _STOP:
                    return

                affinity_task = task
                if not isinstance(affinity_task, _AffinityTask):
                    raise TypeError(f"Unexpected affinity task: {type(affinity_task).__name__}")
                if not affinity_task.future.set_running_or_notify_cancel():
                    continue

                try:
                    result = affinity_task.callback()
                except BaseException as exc:
                    affinity_task.future.set_exception(exc)
                else:
                    affinity_task.future.set_result(result)
            finally:
                task_queue.task_done()

    @staticmethod
    def _cancel_pending_tasks(task_queue: queue.Queue[object]) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return

            try:
                if isinstance(task, _AffinityTask):
                    task.future.cancel()
            finally:
                task_queue.task_done()

    def _release_capacity(self, _future: Future[_ResultT]) -> None:
        self._capacity.release()
