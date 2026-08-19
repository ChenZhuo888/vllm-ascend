"""Multiprocess RPC server.

The ROUTER socket is owned exclusively by the thread running ``run``.
Request handlers publish completed responses through the notification socket.
"""

import logging
import queue
import socket
import threading
from collections.abc import Callable, Hashable, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass

import zmq

from .error import MPServerBusyError
from .executor import AffinityExecutor, BoundedThreadPoolExecutor, ExecutionMode
from .protocol import (
    MultipartMessage,
    ResponseStatus,
    SystemMethod,
    decode_request,
    encode_response,
    normalize_method,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_PENDING_REQUESTS = 64

RequestHandler = Callable[[tuple[bytes, ...]], Iterable[bytes]]
AffinityKeyFactory = Callable[[bytes, tuple[bytes, ...]], Hashable]
ServerResponse = MultipartMessage


@dataclass(frozen=True)
class HandlerSpec:
    """Describe an RPC handler and its execution policy.

    ``INLINE`` is intended only for short, non-blocking control handlers. An
    affinity key factory also runs in the I/O thread and must remain lightweight.
    ``AFFINITY`` uses the client identity unless a key factory is provided.
    """

    handler: RequestHandler
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    affinity_key: AffinityKeyFactory | None = None

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise TypeError("execution_mode must be an ExecutionMode")
        if self.affinity_key is not None and not callable(self.affinity_key):
            raise TypeError("affinity_key must be callable")
        if self.affinity_key is not None and self.execution_mode is not ExecutionMode.AFFINITY:
            raise ValueError("affinity_key is only valid for AFFINITY handlers")


class MPServer:
    """Route requests to inline, parallel, or keyed-affinity handlers."""

    def __init__(
            self,
            bind_url: str,
            max_workers: int = 4,
            handlers: Mapping[str, RequestHandler | HandlerSpec] | None = None,
            *,
            max_pending_requests: int = DEFAULT_MAX_PENDING_REQUESTS,
            affinity_workers: int | None = None,
    ):
        self._handlers = self._build_handlers(handlers)
        self._parallel_executor = BoundedThreadPoolExecutor(max_workers, max_pending_requests, "ascend-store-mp-server")
        affinity_worker_count = max_workers if affinity_workers is None else affinity_workers
        self._affinity_executor = self._create_affinity_executor(affinity_worker_count, max_pending_requests)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        try:
            self.endpoint = self._bind(bind_url)
        except BaseException:
            self._socket.close(linger=0)
            self._context.term()
            if self._affinity_executor is not None:
                self._affinity_executor.shutdown(wait=True, cancel_futures=True)
            self._parallel_executor.shutdown(wait=True, cancel_futures=True)
            raise
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

    @staticmethod
    def _build_handlers(handlers: Mapping[str, RequestHandler | HandlerSpec] | None) -> dict[str, HandlerSpec]:
        handler_specs = {
            SystemMethod.PING.value: HandlerSpec(MPServer._handle_ping, ExecutionMode.INLINE),
            SystemMethod.ECHO.value: HandlerSpec(MPServer._handle_echo, ExecutionMode.INLINE),
        }
        if handlers is None:
            return handler_specs

        for method, handler in handlers.items():
            handler_specs[normalize_method(method)] = (
                handler if isinstance(handler, HandlerSpec) else HandlerSpec(handler)
            )
        return handler_specs

    def _create_affinity_executor(self, max_workers: int, max_pending_requests: int) -> AffinityExecutor | None:
        if not any(spec.execution_mode is ExecutionMode.AFFINITY for spec in self._handlers.values()):
            return None
        return AffinityExecutor(max_workers, max_pending_requests, "ascend-store-mp-affinity")

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
        handler_spec = self._handlers.get(method)
        if handler_spec is None:
            self._publish_response(
                self._encode_error_response(identity, request_id, method, ValueError(f"Unsupported method: {method}"))
            )
            return

        if handler_spec.execution_mode is ExecutionMode.INLINE:
            self._publish_response(self._execute_handler(identity, request_id, method, payloads, handler_spec.handler))
            return

        try:
            if handler_spec.execution_mode is ExecutionMode.AFFINITY:
                affinity_key = (
                    identity if handler_spec.affinity_key is None else handler_spec.affinity_key(identity, payloads)
                )
                if self._affinity_executor is None:
                    raise RuntimeError("Affinity executor is not configured")
                future = self._affinity_executor.submit(
                    affinity_key,
                    self._execute_handler,
                    identity,
                    request_id,
                    method,
                    payloads,
                    handler_spec.handler,
                )
            else:
                future = self._parallel_executor.submit(
                    self._execute_handler,
                    identity,
                    request_id,
                    method,
                    payloads,
                    handler_spec.handler,
                )
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
            if self._affinity_executor is not None:
                self._affinity_executor.shutdown(wait=True, cancel_futures=True)
            self._parallel_executor.shutdown(wait=True, cancel_futures=True)
            self._socket.close(linger=0)
            self._notify_reader.close()
            self._notify_writer.close()
            self._context.term()
