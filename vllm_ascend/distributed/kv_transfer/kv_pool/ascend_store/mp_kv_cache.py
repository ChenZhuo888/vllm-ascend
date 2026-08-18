import itertools
import logging
import math
import queue
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import zmq
from zmq.utils.monitor import recv_monitor_message

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp_protocol import (
    RequestType,
    ResponseStatus,
    decode_request_type,
    decode_response_status,
    encode_request_type,
    encode_response_status,
)

logger = logging.getLogger(__name__)


class AscendStoreKVCacheRemoteError(RuntimeError):
    pass


class AscendStoreKVCacheClientClosedError(RuntimeError):
    pass


class AscendStoreKVCacheServerUnavailableError(ConnectionError):
    pass


class AscendStoreKVCacheRequestTimeoutError(TimeoutError):
    pass


ServerResponse = list[bytes]


@dataclass(frozen=True)
class _OutboundRequest:
    request_id: bytes
    request_type: RequestType
    request_type_bytes: bytes
    payloads: list[bytes]
    future: Future[list[bytes]]
    deadline: float | None


@dataclass(frozen=True)
class _PendingRequest:
    request_type: RequestType
    request_type_bytes: bytes
    future: Future[list[bytes]]
    deadline: float | None


class AscendStoreKVCacheServer:
    def __init__(self, bind_url: str, max_workers: int = 4):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        if bind_url.endswith(":*"):
            base_url = bind_url[:-2]
            port = self.socket.bind_to_random_port(base_url)
            self.endpoint = f"{base_url}:{port}"
        else:
            self.socket.bind(bind_url)
            self.endpoint = bind_url

        self.handlers: dict[RequestType, Callable[[list[bytes]], list[bytes]]] = {
            RequestType.PING: self._handle_ping,
            RequestType.ECHO: self._handle_echo,
        }

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ascend-store-kv-server")
        self._output_queue: queue.Queue[ServerResponse] = queue.Queue()
        self._notify_reader, self._notify_writer = socket.socketpair()
        self._notify_lock = threading.Lock()

    def _handle_ping(self, payloads: list[bytes]) -> list[bytes]:
        if payloads:
            raise ValueError("PING does not accept payloads")
        return [b"OK"]

    def _handle_echo(self, payloads: list[bytes]) -> list[bytes]:
        if len(payloads) != 1:
            raise ValueError(f"ECHO expects 1 payload, got {len(payloads)}")
        return payloads

    def _execute_request(
        self, identity: bytes, request_id: bytes, request_type_bytes: bytes, payloads: list[bytes]
    ) -> ServerResponse:
        try:
            request_type = decode_request_type(request_type_bytes)
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(f"Unsupported request type: {request_type}")

            responses = handler(payloads)
            status = ResponseStatus.OK
        except Exception as exc:
            responses = [f"{type(exc).__name__}: {exc}".encode()]
            status = ResponseStatus.ERROR

        return [identity, request_id, request_type_bytes, encode_response_status(status), *responses]

    def _on_request_done(self, future: Future[ServerResponse]) -> None:
        self._output_queue.put(future.result())

        with self._notify_lock:
            self._notify_writer.send(b"\x01")

    def _submit_request(
        self, identity: bytes, request_id: bytes, request_type_bytes: bytes, payloads: list[bytes]
    ) -> None:
        future = self._executor.submit(self._execute_request, identity, request_id, request_type_bytes, payloads)
        future.add_done_callback(self._on_request_done)

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        poller.register(self._notify_reader.fileno(), zmq.POLLIN)

        while True:
            events = dict(poller.poll())

            if self.socket in events:
                frames = self.socket.recv_multipart()
                if len(frames) < 3:
                    raise ValueError(
                        f"Expected [identity, request_id, request_type, *payloads], got {len(frames)} frames"
                    )

                identity, request_id, request_type_bytes, *payloads = frames
                self._submit_request(identity, request_id, request_type_bytes, payloads)

            if self._notify_reader.fileno() in events:
                self._notify_reader.recv(4096)

                try:
                    while True:
                        self.socket.send_multipart(self._output_queue.get_nowait())
                except queue.Empty:
                    pass

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.socket.close(linger=0)
        self._notify_reader.close()
        self._notify_writer.close()
        self.context.term()


class AscendStoreKVCacheClient:
    def __init__(self, server_url: str):
        self.context = zmq.Context()
        self.server_url = server_url
        self._request_ids = itertools.count()

        self._outbound_queue: queue.Queue[_OutboundRequest] = queue.Queue()
        self._pending_futures: dict[bytes, _PendingRequest] = {}

        self._client_close_requested = threading.Event()
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
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True, name="ascend-store-kv-client")
        self._io_thread.start()

        self._io_ready.wait()
        if self._io_error is not None:
            raise RuntimeError("Failed to start KV cache client I/O thread") from self._io_error

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

    @staticmethod
    def _deadline_from_timeout(timeout_ms: int | None) -> float | None:
        if timeout_ms is None:
            return None
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be greater than 0, got {timeout_ms}")
        return time.monotonic() + timeout_ms / 1000

    def submit_request(
        self,
        request_type: RequestType,
        payloads: list[bytes] | None = None,
        timeout_ms: int | None = None,
    ) -> Future[list[bytes]]:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or self._resources_released:
                raise AscendStoreKVCacheClientClosedError("KV cache client is closed")

            if not self._transport_connected.is_set():
                raise AscendStoreKVCacheServerUnavailableError("KV cache server is unavailable")

            request_id = str(next(self._request_ids)).encode()
            request_type_bytes = encode_request_type(request_type)
            future: Future[list[bytes]] = Future()
            deadline = self._deadline_from_timeout(timeout_ms)

            self._outbound_queue.put(
                _OutboundRequest(request_id, request_type, request_type_bytes, payloads or [], future, deadline)
            )
            self._notify_writer.send(b"\x01")

        return future

    @staticmethod
    def _set_request_timeout(request_type: RequestType, future: Future[list[bytes]]) -> None:
        if not future.done():
            future.set_exception(
                AscendStoreKVCacheRequestTimeoutError(f"Timed out waiting for response to {request_type}")
            )

    def _process_outbound(self, zmq_socket: zmq.Socket) -> None:
        try:
            while True:
                request = self._outbound_queue.get_nowait()
                if request.future.done():
                    continue
                if request.deadline is not None and request.deadline <= time.monotonic():
                    self._set_request_timeout(request.request_type, request.future)
                    continue

                self._pending_futures[request.request_id] = _PendingRequest(
                    request.request_type, request.request_type_bytes, request.future, request.deadline
                )
                zmq_socket.send_multipart([request.request_id, request.request_type_bytes, *request.payloads])
        except queue.Empty:
            pass

    def _process_inbound(self, zmq_socket: zmq.Socket) -> None:
        frames = zmq_socket.recv_multipart()
        if len(frames) < 3:
            logger.error(
                "Malformed response: expected [request_id, request_type, status, *responses], got %d frames",
                len(frames),
            )
            return

        request_id, response_request_type, status_bytes, *responses = frames
        pending = self._pending_futures.pop(request_id, None)
        if pending is None:
            logger.debug("Discarding response for inactive request ID %r", request_id)
            return

        if pending.future.done():
            return

        if response_request_type != pending.request_type_bytes:
            pending.future.set_exception(
                ValueError(
                    f"Response request type mismatch: expected {pending.request_type_bytes!r}, "
                    f"got {response_request_type!r}"
                )
            )
            return

        try:
            status = decode_response_status(status_bytes)
        except ValueError:
            pending.future.set_exception(ValueError(f"Invalid response status: {status_bytes!r}"))
            logger.debug("Failed to decode response status", exc_info=True)
            return

        if status is ResponseStatus.ERROR:
            message = responses[0].decode() if responses else "Unknown server error"
            pending.future.set_exception(AscendStoreKVCacheRemoteError(message))
            return

        pending.future.set_result(responses)

    def _drain_inbound(self, zmq_socket: zmq.Socket) -> None:
        while zmq_socket.poll(timeout=0, flags=zmq.POLLIN):
            self._process_inbound(zmq_socket)

    def _handle_transport_disconnected(self, zmq_socket: zmq.Socket) -> None:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or not self._transport_connected.is_set():
                return

            self._transport_connected.clear()
            self._server_responsive.clear()

        self._drain_inbound(zmq_socket)
        self._fail_pending(
            AscendStoreKVCacheServerUnavailableError(f"KV cache server disconnected: {self.server_url}")
        )

    def _handle_transport_connected(self) -> None:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or self._resources_released:
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
        deadlines = [request.deadline for request in self._pending_futures.values() if request.deadline is not None]
        if not deadlines:
            return None

        remaining_seconds = min(deadlines) - time.monotonic()
        return max(0, math.ceil(remaining_seconds * 1000))

    def _expire_pending_requests(self) -> None:
        now = time.monotonic()
        expired_request_ids = [
            request_id
            for request_id, request in self._pending_futures.items()
            if request.deadline is not None and request.deadline <= now
        ]

        for request_id in expired_request_ids:
            request = self._pending_futures.pop(request_id)
            self._set_request_timeout(request.request_type, request.future)

    def _io_loop(self) -> None:
        zmq_socket = None
        monitor_socket = None

        try:
            zmq_socket = self.context.socket(zmq.DEALER)
            monitor_socket = zmq_socket.get_monitor_socket(events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
            zmq_socket.connect(self.server_url)

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

                    if self._client_close_requested.is_set():
                        self._fail_pending(AscendStoreKVCacheClientClosedError("KV cache client was closed"))
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
                if not request.future.done():
                    request.future.set_exception(exc)
            except queue.Empty:
                break

        for request in self._pending_futures.values():
            if not request.future.done():
                request.future.set_exception(exc)

        self._pending_futures.clear()

    def _request(
        self,
        request_type: RequestType,
        payloads: list[bytes] | None = None,
        timeout_ms: int = 5000,
    ) -> list[bytes]:
        return self.submit_request(request_type, payloads, timeout_ms=timeout_ms).result()

    def ping(self, timeout_ms: int = 5000) -> str:
        return self._request(RequestType.PING, timeout_ms=timeout_ms)[0].decode()

    def echo(self, payload: bytes, timeout_ms: int = 5000) -> bytes:
        return self._request(RequestType.ECHO, [payload], timeout_ms)[0]

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
            if self._client_close_requested.is_set() or self._resources_released:
                raise AscendStoreKVCacheClientClosedError("KV cache client is closed")

            with self._heartbeat_lock:
                if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                    return

                self._heartbeat_interval_ms = interval_ms
                self._heartbeat_timeout_ms = heartbeat_timeout_ms
                self._recovery_callback = recovery_callback
                self._heartbeat_stop.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True, name="ascend-store-kv-heartbeat"
                )
                self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            was_responsive = self._server_responsive.is_set()

            try:
                responsive = self.ping(timeout_ms=self._heartbeat_timeout_ms) == "OK"
            except Exception:
                logger.debug("KV cache heartbeat failed", exc_info=True)
                responsive = False

            if self._heartbeat_stop.is_set():
                break

            if responsive and not was_responsive and self._recovery_callback is not None:
                try:
                    responsive = bool(self._recovery_callback())
                except Exception:
                    logger.exception("KV cache heartbeat recovery callback failed")
                    responsive = False

            if self._heartbeat_stop.is_set():
                break

            if responsive:
                self._server_responsive.set()
                if not was_responsive:
                    logger.info("KV cache server is responsive again")
            else:
                self._server_responsive.clear()
                if was_responsive:
                    logger.warning("KV cache server is unresponsive")

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

        if not self._client_close_requested.is_set() and self._transport_connected.is_set():
            self._server_responsive.set()

    def close(self) -> None:
        heartbeat_thread: threading.Thread | None = None

        with self._lifecycle_lock:
            if self._resources_released:
                return

            with self._heartbeat_lock:
                heartbeat_thread = self._heartbeat_thread
                self._heartbeat_stop.set()

            if not self._client_close_requested.is_set():
                self._client_close_requested.set()
                self._transport_connected.clear()
                self._server_responsive.clear()

                if self._io_thread.is_alive():
                    try:
                        self._notify_writer.send(b"\x01")
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
            self.context.term()
            self._resources_released = True