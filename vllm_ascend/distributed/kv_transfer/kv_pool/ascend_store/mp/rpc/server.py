"""Multiprocess RPC server.

The ROUTER socket is owned exclusively by the thread running ``run``.
Request handlers publish completed responses through the notification socket.
"""

import logging
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial

import zmq

from .error import MPServerBusyError
from .executor import InlineExecutor, TaskExecutor
from .protocol import (
    MultipartMessage,
    ResponseStatus,
    SystemMethod,
    decode_request,
    encode_response,
    normalize_method,
)

logger = logging.getLogger(__name__)

RequestHandler = Callable[[tuple[bytes, ...]], Iterable[bytes]]
ExecutionKeyFactory = Callable[[bytes, tuple[bytes, ...]], Hashable]
ServerResponse = MultipartMessage
_RequestKey = tuple[bytes, bytes]

_RESPONSE_LINGER_MS = 1000


@dataclass(frozen=True)
class Route:
    """Bind one RPC method to its handler and executor.

    The optional key factory runs in the server I/O thread and must remain
    lightweight. It is intended for executors that require keyed execution.
    """

    method: str
    handler: RequestHandler
    executor: TaskExecutor
    key_factory: ExecutionKeyFactory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_method(self.method))
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not callable(getattr(self.executor, "submit", None)):
            raise TypeError("executor must define submit")
        if not callable(getattr(self.executor, "shutdown", None)):
            raise TypeError("executor must define shutdown")
        if self.key_factory is not None and not callable(self.key_factory):
            raise TypeError("key_factory must be callable")


@dataclass(frozen=True)
class _AcceptedRequest:
    identity: bytes
    request_id: bytes
    method: str
    deadline_ns: int | None

    @property
    def key(self) -> _RequestKey:
        return self.identity, self.request_id


@dataclass(frozen=True)
class _ResponseEnvelope:
    frames: ServerResponse
    request_key: _RequestKey | None = None
    is_abort: bool = False


class MPServer:
    """Serve RPC routes and own their executors until the server is closed."""

    def __init__(self, bind_url: str, routes: Iterable[Route] = ()):
        system_executor = InlineExecutor()
        all_routes = (
            Route(SystemMethod.PING, self._handle_ping, system_executor),
            Route(SystemMethod.ECHO, self._handle_echo, system_executor),
            *routes,
        )
        self._executors = self._collect_executors(all_routes)
        try:
            self._routes = self._index_routes(all_routes)
        except BaseException:
            self._shutdown_executors()
            raise
        try:
            self._notify_reader, self._notify_writer = socket.socketpair()
        except BaseException:
            self._shutdown_executors()
            raise

        self._output_queue: queue.Queue[_ResponseEnvelope] = queue.Queue()
        self._response_backlog: deque[_ResponseEnvelope] = deque()
        self._notify_writer.setblocking(False)
        self._notify_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._transport_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._abort_requested = threading.Event()
        self._run_stopped = threading.Event()
        self._run_stopped.set()
        self._run_thread: threading.Thread | None = None
        self._accepted_requests: dict[_RequestKey, _AcceptedRequest] = {}
        self._graceful_deadline_ns: int | None = None
        self._run_started = False
        self._transport_closed = False
        self._closed = False

        context = None
        zmq_socket = None
        try:
            context = zmq.Context()
            zmq_socket = context.socket(zmq.ROUTER)
            self.endpoint = self._bind(zmq_socket, bind_url)
        except BaseException:
            if zmq_socket is not None:
                zmq_socket.close(linger=0)
            if context is not None:
                context.term()
            self._notify_reader.close()
            self._notify_writer.close()
            self._shutdown_executors()
            self._closed = True
            raise

        self._context = context
        self._socket = zmq_socket

    def __enter__(self) -> "MPServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _index_routes(route_definitions: Iterable[Route]) -> dict[str, Route]:
        indexed_routes = {}
        for route in route_definitions:
            if not isinstance(route, Route):
                raise TypeError(f"routes must contain Route instances, got {type(route).__name__}")
            if route.method in indexed_routes:
                raise ValueError(f"Duplicate RPC method: {route.method}")
            indexed_routes[route.method] = route
        return indexed_routes

    @staticmethod
    def _collect_executors(routes: Iterable[Route]) -> tuple[TaskExecutor, ...]:
        executors = {}
        for route in routes:
            executors.setdefault(id(route.executor), route.executor)
        return tuple(executors.values())

    def _shutdown_executors(self, wait: bool = True, cancel_futures: bool = True) -> None:
        for executor in self._executors:
            try:
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            except Exception:
                logger.exception("Failed to shut down MP server executor")

    def _close_transport(self, linger_ms: int) -> None:
        with self._transport_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
            self._socket.close(linger=linger_ms)

    @staticmethod
    def _bind(zmq_socket: zmq.Socket, bind_url: str) -> str:
        if bind_url.endswith(":*"):
            base_url = bind_url[:-2]
            port = zmq_socket.bind_to_random_port(base_url)
            return f"{base_url}:{port}"

        zmq_socket.bind(bind_url)
        return bind_url

    @staticmethod
    def _handle_ping(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if payloads:
            raise ValueError("PING does not accept payloads")
        return (b"OK",)

    @staticmethod
    def _handle_echo(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(payloads) != 1:
            raise ValueError(f"ECHO expects 1 payload, got {len(payloads)}")
        return payloads

    @staticmethod
    def _encode_error_response(identity: bytes, request_id: bytes, method: str, exc: BaseException) -> ServerResponse:
        status = ResponseStatus.BUSY if isinstance(exc, MPServerBusyError) else ResponseStatus.ERROR
        response_frames = encode_response(
            request_id,
            method,
            status,
            (f"{type(exc).__name__}: {exc}".encode(),),
        )
        return identity, *response_frames

    @staticmethod
    def _encode_abort_response(request: _AcceptedRequest) -> ServerResponse:
        response_frames = encode_response(
            request.request_id,
            request.method,
            ResponseStatus.ABORTED,
            (b"MP server was force-aborted",),
        )
        return request.identity, *response_frames

    def _execute_handler(
        self,
        identity: bytes,
        request_id: bytes,
        method: str,
        payloads: tuple[bytes, ...],
        handler: RequestHandler,
    ) -> ServerResponse:
        try:
            response_frames = encode_response(request_id, method, ResponseStatus.OK, handler(payloads))
            return identity, *response_frames
        except BaseException as exc:
            return self._encode_error_response(identity, request_id, method, exc)

    def _notify_response_ready(self) -> None:
        with self._notify_lock:
            try:
                self._notify_writer.send(b"\x01")
            except BlockingIOError:
                pass
            except OSError:
                if not self._closed:
                    logger.exception("Failed to notify MP server response loop")

    def _publish_response(self, response: ServerResponse) -> None:
        self._output_queue.put(_ResponseEnvelope(response))
        self._notify_response_ready()

    def _complete_execution(self, request_key: _RequestKey, response: ServerResponse) -> None:
        with self._state_lock:
            if request_key not in self._accepted_requests:
                return
            if self._abort_requested.is_set():
                return
            self._output_queue.put(_ResponseEnvelope(response, request_key))
        self._notify_response_ready()

    def _on_execution_done(self, request: _AcceptedRequest, future: Future[ServerResponse]) -> None:
        if future.cancelled():
            response = self._encode_error_response(
                request.identity,
                request.request_id,
                request.method,
                RuntimeError("Request execution was cancelled"),
            )
            self._complete_execution(request.key, response)
            return

        try:
            response = future.result()
        except BaseException as exc:
            logger.exception("Failed to process MP server request")
            response = self._encode_error_response(request.identity, request.request_id, request.method, exc)
        self._complete_execution(request.key, response)

    def _try_accept_request(self, request: _AcceptedRequest) -> bool:
        with self._state_lock:
            if self._closed or self._stop_requested.is_set():
                return False
            if request.key in self._accepted_requests:
                return False
            self._accepted_requests[request.key] = request
            return True

    def _reject_stopping_request(self, identity: bytes, request_id: bytes, method: str) -> None:
        if self._abort_requested.is_set():
            return
        self._publish_response(
            self._encode_error_response(identity, request_id, method, MPServerBusyError("MP server is stopping"))
        )

    def _dispatch_request(
        self,
        identity: bytes,
        request_id: bytes,
        method: str,
        payloads: tuple[bytes, ...],
        deadline_ns: int | None = None,
    ) -> None:
        request = _AcceptedRequest(identity, request_id, method, deadline_ns)
        if not self._try_accept_request(request):
            self._reject_stopping_request(identity, request_id, method)
            return

        route = self._routes.get(method)
        if route is None:
            self._complete_execution(
                request.key,
                self._encode_error_response(identity, request_id, method, ValueError(f"Unsupported method: {method}")),
            )
            return

        try:
            callback = partial(self._execute_handler, identity, request_id, method, payloads, route.handler)
            key = None if route.key_factory is None else route.key_factory(identity, payloads)
        except BaseException as exc:
            self._complete_execution(request.key, self._encode_error_response(identity, request_id, method, exc))
            return

        try:
            future = route.executor.submit(callback, key)
        except BaseException as exc:
            self._complete_execution(request.key, self._encode_error_response(identity, request_id, method, exc))
            return

        future.add_done_callback(partial(self._on_execution_done, request))

    def _receive_request(self) -> None:
        frames = self._socket.recv_multipart()
        if len(frames) < 4:
            logger.warning(
                "Discarding malformed request: expected "
                "[identity, request_id, method, deadline, *payloads], got %d frames",
                len(frames),
            )
            return

        identity, *request_frames = frames
        try:
            request_id, method, deadline_ns, payloads = decode_request(request_frames)
        except Exception:
            logger.exception("Discarding malformed MP server request")
            return

        self._dispatch_request(identity, request_id, method, payloads, deadline_ns)

    def _receive_response_notification(self) -> None:
        self._notify_reader.recv(4096)

        try:
            while True:
                self._response_backlog.append(self._output_queue.get_nowait())
        except queue.Empty:
            pass

    def _send_responses(self) -> None:
        while self._response_backlog:
            response = self._response_backlog[0]
            if response.request_key is None:
                try:
                    self._socket.send_multipart(response.frames, flags=zmq.NOBLOCK)
                except zmq.Again:
                    return
                self._response_backlog.popleft()
                continue

            with self._state_lock:
                if response.request_key not in self._accepted_requests:
                    self._response_backlog.popleft()
                    continue
                if self._abort_requested.is_set() and not response.is_abort:
                    self._response_backlog.popleft()
                    continue

                try:
                    self._socket.send_multipart(response.frames, flags=zmq.NOBLOCK)
                except zmq.Again:
                    return
                del self._accepted_requests[response.request_key]
                self._response_backlog.popleft()

    def _should_stop_run(self) -> bool:
        if not self._stop_requested.is_set():
            return False

        with self._state_lock:
            return not self._accepted_requests

    def run(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MPServer is closed")
            if self._run_started:
                raise RuntimeError("MPServer.run() can only be called once")
            self._run_started = True
            self._run_thread = threading.current_thread()
            self._run_stopped.clear()

        try:
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            poller.register(self._notify_reader.fileno(), zmq.POLLIN)
            socket_events = zmq.POLLIN

            while not self._should_stop_run():
                expected_socket_events = zmq.POLLOUT if self._response_backlog else zmq.POLLIN
                if socket_events != expected_socket_events:
                    socket_events = expected_socket_events
                    poller.modify(self._socket, socket_events)

                events = dict(poller.poll())

                if self._notify_reader.fileno() in events:
                    self._receive_response_notification()
                    self._send_responses()

                socket_event = events.get(self._socket, 0)
                if socket_event & zmq.POLLOUT:
                    self._send_responses()
                if socket_event & zmq.POLLIN and not self._response_backlog:
                    self._receive_request()
        finally:
            controlled_stop = self._stop_requested.is_set()
            self._close_transport(_RESPONSE_LINGER_MS if controlled_stop else 0)
            with self._state_lock:
                self._run_thread = None
                self._run_stopped.set()
            if self._abort_requested.is_set():
                self._release_resources(shutdown_executors=False)

    def request_stop(self) -> bool:
        """Stop accepting work if every accepted request has a finite deadline."""
        with self._state_lock:
            if self._closed:
                return not self._abort_requested.is_set()
            if self._abort_requested.is_set():
                return False
            if self._stop_requested.is_set():
                return True

            requests = tuple(self._accepted_requests.values())
            indefinite_count = sum(request.deadline_ns is None for request in requests)
            if indefinite_count:
                logger.warning(
                    "Graceful MP server shutdown rejected: %d accepted request(s) have no deadline; call abort()",
                    indefinite_count,
                )
                return False

            deadlines_ns = [request.deadline_ns for request in requests if request.deadline_ns is not None]
            self._graceful_deadline_ns = max(deadlines_ns, default=None)
            self._stop_requested.set()
        self._notify_response_ready()
        return True

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Wait until the run loop stops, whether by draining, abort, or failure."""
        with self._state_lock:
            run_thread = self._run_thread
        if run_thread is None or run_thread is threading.current_thread():
            return self._run_stopped.is_set()
        return self._run_stopped.wait(timeout)

    def wait_for_drain(self) -> bool:
        """Wait no longer than the latest accepted request deadline."""
        with self._state_lock:
            if not self._stop_requested.is_set() or self._abort_requested.is_set():
                return False
            deadline_ns = self._graceful_deadline_ns

        timeout = None if deadline_ns is None else max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        if not self.wait_until_stopped(timeout):
            return False
        with self._state_lock:
            return not self._accepted_requests and not self._abort_requested.is_set()

    def abort(self) -> None:
        """Fail accepted requests and stop without waiting for running handlers."""
        with self._state_lock:
            if self._closed or self._abort_requested.is_set():
                return
            self._abort_requested.set()
            self._stop_requested.set()
            requests = tuple(self._accepted_requests.values())
            for request in requests:
                self._output_queue.put(
                    _ResponseEnvelope(self._encode_abort_response(request), request.key, is_abort=True)
                )

        self._shutdown_executors(wait=False, cancel_futures=True)
        self._notify_response_ready()
        if self._run_stopped.is_set():
            self._release_resources(linger_ms=0, shutdown_executors=False)

    def close(self) -> bool:
        """Release resources after a bounded drain, or return ``False``."""
        if not self.request_stop():
            return False
        if not self.wait_for_drain():
            logger.warning("Graceful MP server shutdown timed out; call abort()")
            return False
        self._release_resources()
        return True

    def _release_resources(
        self,
        linger_ms: int = _RESPONSE_LINGER_MS,
        shutdown_executors: bool = True,
    ) -> None:
        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            self._close_transport(linger_ms)
            if shutdown_executors:
                aborted = self._abort_requested.is_set()
                self._shutdown_executors(wait=not aborted, cancel_futures=aborted)
            self._notify_reader.close()
            self._notify_writer.close()
            self._context.term()
