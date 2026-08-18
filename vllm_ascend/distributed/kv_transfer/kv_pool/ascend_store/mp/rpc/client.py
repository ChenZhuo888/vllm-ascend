"""Multiprocess RPC client.

Thread ownership rules:

- Public request methods may be called from application threads.
- The outbound queue is a multi-producer, single-consumer queue.
- The DEALER socket and pending request map are owned by the I/O thread.
- Future callbacks run synchronously in the I/O thread and must not call
  blocking methods on this client.
"""

import itertools
import logging
import math
import queue
import socket
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass

import zmq
from zmq.utils.monitor import recv_monitor_message

from .error import (
    MPClientClosedError,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerUnavailableError,
)
from .protocol import (
    MultipartMessage,
    ResponseStatus,
    SystemMethod,
    decode_response,
    encode_request,
    normalize_method,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OutboundRequest:
    request_id: bytes
    method: str
    frames: MultipartMessage
    future: Future[list[bytes]]
    deadline: float | None


@dataclass(frozen=True)
class _PendingRequest:
    method: str
    future: Future[list[bytes]]
    deadline: float | None


class MPClient:
    def __init__(self, server_url: str):
        self._context = zmq.Context()
        self._server_url = server_url
        self._request_ids = itertools.count()

        self._outbound_queue: queue.Queue[_OutboundRequest] = queue.Queue()
        self._pending_requests: dict[bytes, _PendingRequest] = {}

        self._close_requested = threading.Event()
        self._transport_connected = threading.Event()
        self._server_responsive = threading.Event()
        self._server_responsive.set()
        self._lifecycle_lock = threading.Lock()
        self._resources_released = False

        self._heartbeat_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_interval_ms = 0
        self._heartbeat_timeout_ms = 0
        self._recovery_callback: Callable[[], bool] | None = None

        self._io_ready = threading.Event()
        self._io_error: Exception | None = None

        self._notify_reader, self._notify_writer = socket.socketpair()
        self._notify_writer.setblocking(False)
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True, name="ascend-store-mp-client")
        self._io_thread.start()

        self._io_ready.wait()
        if self._io_error is not None:
            io_error = self._io_error
            self._notify_writer.close()
            self._context.term()
            self._resources_released = True
            raise RuntimeError("Failed to start MP client I/O thread") from io_error

    def __enter__(self) -> "MPClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_transport_connected(self) -> bool:
        """Whether ZMQ reports an active transport connection."""
        return self._transport_connected.is_set()

    @property
    def is_server_responsive(self) -> bool:
        """Whether the connected server is considered responsive.

        Without heartbeat monitoring, a transport connection is optimistically
        treated as responsive.
        """
        return self._transport_connected.is_set() and self._server_responsive.is_set()

    @property
    def is_heartbeat_running(self) -> bool:
        """Whether the heartbeat monitoring thread is running."""
        with self._heartbeat_lock:
            return self._heartbeat_thread is not None and self._heartbeat_thread.is_alive()

    def wait_until_connected(self, timeout_ms: int = 5000) -> None:
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {timeout_ms}")

        with self._lifecycle_lock:
            if self._close_requested.is_set() or self._resources_released:
                raise MPClientClosedError("MP client is closed")

        if self._transport_connected.wait(timeout_ms / 1000):
            return

        with self._lifecycle_lock:
            if self._close_requested.is_set() or self._resources_released:
                raise MPClientClosedError("MP client is closed")

            if self._io_error is not None:
                raise MPServerUnavailableError("MP client I/O thread is unavailable") from self._io_error

        raise MPServerUnavailableError(f"Timed out connecting to MP server: {self._server_url}")

    @staticmethod
    def _deadline_from_timeout(timeout_ms: int | None) -> float | None:
        if timeout_ms is None:
            return None
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {timeout_ms}")
        return time.monotonic() + timeout_ms / 1000

    def _notify_io_thread(self) -> None:
        try:
            self._notify_writer.send(b"\x01")
        except BlockingIOError:
            pass

    def submit_request(
            self,
            method: str,
            payloads: Sequence[bytes] | None = None,
            timeout_ms: int | None = None,
    ) -> Future[list[bytes]]:
        method_name = normalize_method(method)

        with self._lifecycle_lock:
            if self._close_requested.is_set() or self._resources_released:
                raise MPClientClosedError("MP client is closed")

            if not self._transport_connected.is_set():
                raise MPServerUnavailableError("MP server is unavailable")

            request_id = str(next(self._request_ids)).encode()
            frames = encode_request(request_id, method_name, payloads or ())
            future: Future[list[bytes]] = Future()
            deadline = self._deadline_from_timeout(timeout_ms)

            self._outbound_queue.put(_OutboundRequest(request_id, method_name, frames, future, deadline))
            self._notify_io_thread()

        return future

    def request(
            self,
            method: str,
            payloads: Sequence[bytes] | None = None,
            timeout_ms: int = 5000,
    ) -> list[bytes]:
        return self.submit_request(method, payloads, timeout_ms=timeout_ms).result()

    def ping(self, timeout_ms: int = 5000) -> str:
        return self.request(SystemMethod.PING, timeout_ms=timeout_ms)[0].decode()

    def echo(self, payload: bytes, timeout_ms: int = 5000) -> bytes:
        return self.request(SystemMethod.ECHO, [payload], timeout_ms=timeout_ms)[0]

    @staticmethod
    def _set_request_timeout(method: str, future: Future[list[bytes]]) -> None:
        if not future.done():
            future.set_exception(MPRequestTimeoutError(f"Timed out waiting for response to {method}"))

    def _process_outbound(self, zmq_socket: zmq.Socket) -> None:
        try:
            while True:
                request = self._outbound_queue.get_nowait()
                if not request.future.set_running_or_notify_cancel():
                    continue

                if request.deadline is not None and request.deadline <= time.monotonic():
                    self._set_request_timeout(request.method, request.future)
                    continue

                self._pending_requests[request.request_id] = _PendingRequest(
                    request.method,
                    request.future,
                    request.deadline,
                )
                zmq_socket.send_multipart(request.frames)
        except queue.Empty:
            pass

    def _process_inbound(self, zmq_socket: zmq.Socket) -> None:
        frames = zmq_socket.recv_multipart()
        if not frames:
            logger.error("Discarding malformed response without a request ID")
            return

        request_id = frames[0]
        pending = self._pending_requests.pop(request_id, None)
        if pending is None:
            logger.debug("Discarding response for inactive request ID %r", request_id)
            return

        if pending.future.done():
            return

        try:
            _, response_method, status, responses = decode_response(frames)
        except MPProtocolError as exc:
            pending.future.set_exception(exc)
            return

        if response_method != pending.method:
            pending.future.set_exception(
                MPProtocolError(
                    f"Response method mismatch: expected {pending.method!r}, got {response_method!r}"
                )
            )
            return

        if status is ResponseStatus.ERROR:
            message = responses[0].decode(errors="replace") if responses else "Unknown server error"
            pending.future.set_exception(MPRemoteError(message))
            return

        pending.future.set_result(list(responses))

    def _drain_inbound(self, zmq_socket: zmq.Socket) -> None:
        while zmq_socket.poll(timeout=0, flags=zmq.POLLIN):
            self._process_inbound(zmq_socket)

    def _handle_transport_disconnected(self, zmq_socket: zmq.Socket) -> None:
        with self._lifecycle_lock:
            if self._close_requested.is_set() or not self._transport_connected.is_set():
                return

            self._transport_connected.clear()
            self._server_responsive.clear()

        self._drain_inbound(zmq_socket)
        self._fail_pending(MPServerUnavailableError(f"MP server disconnected: {self._server_url}"))

    def _handle_transport_connected(self) -> None:
        with self._lifecycle_lock:
            if self._close_requested.is_set() or self._resources_released:
                return

            self._transport_connected.set()

        if not self.is_heartbeat_running:
            self._server_responsive.set()

    def _process_monitor_event(self, zmq_socket: zmq.Socket, monitor_socket: zmq.Socket) -> None:
        monitor_event = recv_monitor_message(monitor_socket)
        event = monitor_event["event"]

        if event == zmq.EVENT_DISCONNECTED:
            self._handle_transport_disconnected(zmq_socket)
        elif event == zmq.EVENT_CONNECTED:
            self._handle_transport_connected()

    def _next_poll_timeout_ms(self) -> int | None:
        deadlines = [request.deadline for request in self._pending_requests.values() if request.deadline is not None]
        if not deadlines:
            return None

        remaining_seconds = min(deadlines) - time.monotonic()
        return max(0, math.ceil(remaining_seconds * 1000))

    def _expire_pending_requests(self) -> None:
        now = time.monotonic()
        expired_request_ids = [
            request_id
            for request_id, request in self._pending_requests.items()
            if request.deadline is not None and request.deadline <= now
        ]

        for request_id in expired_request_ids:
            request = self._pending_requests.pop(request_id)
            self._set_request_timeout(request.method, request.future)

    def _io_loop(self) -> None:
        zmq_socket = None
        monitor_socket = None

        try:
            zmq_socket = self._context.socket(zmq.DEALER)
            monitor_socket = zmq_socket.get_monitor_socket(events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
            zmq_socket.connect(self._server_url)

            poller = zmq.Poller()
            poller.register(zmq_socket, zmq.POLLIN)
            poller.register(monitor_socket, zmq.POLLIN)
            poller.register(self._notify_reader.fileno(), zmq.POLLIN)
            self._io_ready.set()

            while True:
                timeout_ms = self._next_poll_timeout_ms()
                events = dict(poller.poll() if timeout_ms is None else poller.poll(timeout_ms))

                if self._notify_reader.fileno() in events:
                    self._notify_reader.recv(4096)

                    if self._close_requested.is_set():
                        self._fail_pending(MPClientClosedError("MP client was closed"))
                        break

                    self._process_outbound(zmq_socket)

                if zmq_socket in events:
                    self._process_inbound(zmq_socket)

                if monitor_socket in events:
                    self._process_monitor_event(zmq_socket, monitor_socket)

                self._expire_pending_requests()
        except Exception as exc:
            self._io_error = exc

            with self._lifecycle_lock:
                self._transport_connected.clear()
                self._server_responsive.clear()

            self._fail_pending(exc)
            self._io_ready.set()
        finally:
            self._transport_connected.clear()
            self._server_responsive.clear()

            if monitor_socket is not None:
                monitor_socket.close(linger=0)

            if zmq_socket is not None:
                zmq_socket.close(linger=0)

            self._notify_reader.close()

    def _fail_pending(self, exc: Exception) -> None:
        while True:
            try:
                request = self._outbound_queue.get_nowait()
                if request.future.set_running_or_notify_cancel():
                    request.future.set_exception(exc)
            except queue.Empty:
                break

        for request in self._pending_requests.values():
            if not request.future.done():
                request.future.set_exception(exc)

        self._pending_requests.clear()

    def start_heartbeat(
            self,
            interval_ms: int = 10000,
            timeout_ms: int | None = None,
            recovery_callback: Callable[[], bool] | None = None,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be greater than 0, got {interval_ms}")

        heartbeat_timeout_ms = interval_ms if timeout_ms is None else timeout_ms
        if heartbeat_timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {heartbeat_timeout_ms}")

        with self._lifecycle_lock:
            if self._close_requested.is_set() or self._resources_released:
                raise MPClientClosedError("MP client is closed")

            with self._heartbeat_lock:
                if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                    return

                self._heartbeat_interval_ms = interval_ms
                self._heartbeat_timeout_ms = heartbeat_timeout_ms
                self._recovery_callback = recovery_callback
                self._heartbeat_stop.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True,
                    name="ascend-store-mp-heartbeat",
                )
                self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            was_responsive = self._server_responsive.is_set()

            try:
                responsive = self.ping(timeout_ms=self._heartbeat_timeout_ms) == "OK"
            except Exception:
                logger.debug("MP server heartbeat failed", exc_info=True)
                responsive = False

            if self._heartbeat_stop.is_set():
                break

            if responsive and not was_responsive and self._recovery_callback is not None:
                try:
                    responsive = bool(self._recovery_callback())
                except Exception:
                    logger.exception("MP server heartbeat recovery callback failed")
                    responsive = False

            if self._heartbeat_stop.is_set():
                break

            if responsive:
                self._server_responsive.set()
                if not was_responsive:
                    logger.info("MP server is responsive again")
            else:
                self._server_responsive.clear()
                if was_responsive:
                    logger.warning("MP server is unresponsive")

            if self._heartbeat_stop.wait(self._heartbeat_interval_ms / 1000):
                break

    def stop_heartbeat(self) -> None:
        with self._heartbeat_lock:
            heartbeat_thread = self._heartbeat_thread
            if heartbeat_thread is None:
                return
            self._heartbeat_stop.set()

        if heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join()

        with self._heartbeat_lock:
            if self._heartbeat_thread is heartbeat_thread:
                self._heartbeat_thread = None

        if not self._close_requested.is_set() and self._transport_connected.is_set():
            self._server_responsive.set()

    def close(self) -> None:
        heartbeat_thread: threading.Thread | None = None

        with self._lifecycle_lock:
            if self._resources_released:
                return

            with self._heartbeat_lock:
                heartbeat_thread = self._heartbeat_thread
                self._heartbeat_stop.set()

            if not self._close_requested.is_set():
                self._close_requested.set()
                self._transport_connected.clear()
                self._server_responsive.clear()

                if self._io_thread.is_alive():
                    try:
                        self._notify_io_thread()
                    except OSError:
                        pass

        self._io_thread.join()

        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join()

        with self._heartbeat_lock:
            if self._heartbeat_thread is heartbeat_thread:
                self._heartbeat_thread = None

        with self._lifecycle_lock:
            if self._resources_released:
                return

            self._notify_writer.close()
            self._context.term()
            self._resources_released = True
