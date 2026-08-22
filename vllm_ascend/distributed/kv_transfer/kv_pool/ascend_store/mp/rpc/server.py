"""Multiprocess RPC server.

The ROUTER socket is owned exclusively by the thread running ``run``.
Request handlers publish completed responses through the notification socket.
"""

import logging
import queue
import socket
import threading
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


class MPServer:
    """Serve RPC routes and own their executors until the server is closed."""

    def __init__(self, bind_url: str, routes: Iterable[Route] = ()):
        system_executor = InlineExecutor()
        all_routes = (
            Route(SystemMethod.PING, self._handle_ping, system_executor),
            Route(SystemMethod.ECHO, self._handle_echo, system_executor),
            *routes,
        )
        self._routes = self._index_routes(all_routes)
        self._executors = self._collect_executors(all_routes)
        try:
            self._notify_reader, self._notify_writer = socket.socketpair()
        except BaseException:
            self._shutdown_executors()
            raise

        self._output_queue: queue.Queue[ServerResponse] = queue.Queue()
        self._notify_writer.setblocking(False)
        self._notify_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._transport_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._run_stopped = threading.Event()
        self._run_stopped.set()
        self._run_thread: threading.Thread | None = None
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

    def _shutdown_executors(self) -> None:
        for executor in self._executors:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                logger.exception("Failed to shut down MP server executor")

    def _close_transport(self) -> None:
        with self._transport_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
            self._socket.close(linger=0)

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
        except Exception as exc:
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
        self._output_queue.put(response)
        self._notify_response_ready()

    def _on_request_done(self, future: Future[ServerResponse]) -> None:
        if future.cancelled():
            return

        try:
            response = future.result()
        except Exception:
            logger.exception("Failed to process MP server request")
            return

        self._publish_response(response)

    def _dispatch_request(self, identity: bytes, request_id: bytes, method: str, payloads: tuple[bytes, ...]) -> None:
        with self._state_lock:
            if self._closed or self._stop_requested.is_set():
                return

        route = self._routes.get(method)
        if route is None:
            self._publish_response(
                self._encode_error_response(identity, request_id, method, ValueError(f"Unsupported method: {method}"))
            )
            return

        try:
            callback = partial(self._execute_handler, identity, request_id, method, payloads, route.handler)
            key = None if route.key_factory is None else route.key_factory(identity, payloads)
            future = route.executor.submit(callback, key)
        except Exception as exc:
            self._publish_response(self._encode_error_response(identity, request_id, method, exc))
            return

        future.add_done_callback(self._on_request_done)

    def _receive_request(self) -> None:
        frames = self._socket.recv_multipart()
        if len(frames) < 3:
            logger.warning(
                "Discarding malformed request: expected [identity, request_id, method, *payloads], got %d frames",
                len(frames),
            )
            return

        identity, *request_frames = frames
        try:
            request_id, method, payloads = decode_request(request_frames)
        except Exception:
            logger.exception("Discarding malformed MP server request")
            return

        self._dispatch_request(identity, request_id, method, payloads)

    def _send_responses(self) -> None:
        self._notify_reader.recv(4096)

        try:
            while True:
                response = self._output_queue.get_nowait()
                try:
                    self._socket.send_multipart(response, flags=zmq.NOBLOCK)
                except zmq.Again:
                    logger.warning("Dropping MP response because the outbound transport is busy")
        except queue.Empty:
            pass

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

            while not self._stop_requested.is_set():
                events = dict(poller.poll())
                if self._stop_requested.is_set():
                    break

                if self._socket in events:
                    self._receive_request()

                if self._notify_reader.fileno() in events:
                    self._send_responses()
        finally:
            # Disconnect clients before business services and executors are drained.
            # Running handlers cannot be interrupted safely, but clients must not
            # remain blocked while the server waits for them to finish.
            self._close_transport()
            with self._state_lock:
                self._run_thread = None
                self._run_stopped.set()

    def request_stop(self) -> None:
        """Ask the run loop to stop accepting requests without waiting for it."""
        with self._state_lock:
            if self._closed or self._stop_requested.is_set():
                return
            self._stop_requested.set()
        self._notify_response_ready()

    def wait_until_stopped(self) -> None:
        """Wait until the run loop has stopped and disconnected its clients."""
        with self._state_lock:
            run_thread = self._run_thread
        if run_thread is not None and run_thread is not threading.current_thread():
            self._run_stopped.wait()

    def close(self) -> None:
        with self._close_lock:
            self.request_stop()
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            self.wait_until_stopped()
            self._close_transport()
            self._shutdown_executors()
            self._notify_reader.close()
            self._notify_writer.close()
            self._context.term()
