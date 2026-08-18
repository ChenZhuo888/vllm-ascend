"""Multiprocess RPC server.

The ROUTER socket is owned exclusively by the thread running ``run``.
Request handlers execute in worker threads and publish completed responses
through the notification socket.
"""

import logging
import queue
import socket
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor

import zmq

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
ServerResponse = MultipartMessage


class MPServer:
    def __init__(
            self,
            bind_url: str,
            max_workers: int = 4,
            handlers: Mapping[str, RequestHandler] | None = None,
    ):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self.endpoint = self._bind(bind_url)

        self._handlers: dict[str, RequestHandler] = {
            SystemMethod.PING.value: self._handle_ping,
            SystemMethod.ECHO.value: self._handle_echo,
        }
        if handlers is not None:
            for method, handler in handlers.items():
                self._handlers[normalize_method(method)] = handler

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ascend-store-mp-server")
        self._output_queue: queue.Queue[ServerResponse] = queue.Queue()
        self._notify_reader, self._notify_writer = socket.socketpair()
        self._notify_writer.setblocking(False)
        self._notify_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

    def __enter__(self) -> "MPServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _bind(self, bind_url: str) -> str:
        if bind_url.endswith(":*"):
            base_url = bind_url[:-2]
            port = self._socket.bind_to_random_port(base_url)
            return f"{base_url}:{port}"

        self._socket.bind(bind_url)
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

    def _execute_request(self, identity: bytes, request_frames: list[bytes]) -> ServerResponse:
        request_id, method, payloads = decode_request(request_frames)

        try:
            handler = self._handlers.get(method)
            if handler is None:
                raise ValueError(f"Unsupported method: {method}")

            response_frames = encode_response(
                request_id,
                method,
                ResponseStatus.OK,
                handler(payloads),
            )
        except Exception as exc:
            response_frames = encode_response(
                request_id,
                method,
                ResponseStatus.ERROR,
                (f"{type(exc).__name__}: {exc}".encode(),),
            )

        return identity, *response_frames

    def _notify_response_ready(self) -> None:
        with self._notify_lock:
            try:
                self._notify_writer.send(b"\x01")
            except BlockingIOError:
                pass
            except OSError:
                if not self._closed:
                    logger.exception("Failed to notify MP server response loop")

    def _on_request_done(self, future: Future[ServerResponse]) -> None:
        if future.cancelled():
            return

        try:
            response = future.result()
        except Exception:
            logger.exception("Failed to process MP server request")
            return

        self._output_queue.put(response)
        self._notify_response_ready()

    def _submit_request(self, identity: bytes, request_frames: list[bytes]) -> None:
        future = self._executor.submit(self._execute_request, identity, request_frames)
        future.add_done_callback(self._on_request_done)

    def _receive_request(self) -> None:
        frames = self._socket.recv_multipart()
        if len(frames) < 3:
            logger.warning(
                "Discarding malformed request: expected "
                "[identity, request_id, method, *payloads], got %d frames",
                len(frames),
            )
            return

        identity, *request_frames = frames
        self._submit_request(identity, request_frames)

    def _send_responses(self) -> None:
        self._notify_reader.recv(4096)

        try:
            while True:
                self._socket.send_multipart(self._output_queue.get_nowait())
        except queue.Empty:
            pass

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        poller.register(self._notify_reader.fileno(), zmq.POLLIN)

        while True:
            events = dict(poller.poll())

            if self._socket in events:
                self._receive_request()

            if self._notify_reader.fileno() in events:
                self._send_responses()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return

            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._socket.close(linger=0)
            self._notify_reader.close()
            self._notify_writer.close()
            self._context.term()
