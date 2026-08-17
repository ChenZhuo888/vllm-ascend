import itertools
import queue
import socket
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

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


class AscendStoreKVCacheRemoteError(RuntimeError):
    pass


class AscendStoreKVCacheClientClosedError(RuntimeError):
    pass


class AscendStoreKVCacheServerUnavailableError(ConnectionError):
    pass


ServerResponse = tuple[list[bytes], bool]
OutboundRequest = tuple[bytes, bytes, list[bytes], Future[list[bytes]]]


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
            RequestType.SHUTDOWN: self._handle_shutdown,
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

    def _handle_shutdown(self, payloads: list[bytes]) -> list[bytes]:
        if payloads:
            raise ValueError("SHUTDOWN does not accept payloads")
        return [b"OK"]

    def _handle_echo(self, payloads: list[bytes]) -> list[bytes]:
        if len(payloads) != 1:
            raise ValueError(f"ECHO expects 1 payload, got {len(payloads)}")
        return payloads

    def _execute_request(
        self, identity: bytes, request_id: bytes, request_type_bytes: bytes, payloads: list[bytes]
    ) -> ServerResponse:
        request_type = None

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

        frames = [identity, request_id, request_type_bytes, encode_response_status(status), *responses]
        should_shutdown = request_type is RequestType.SHUTDOWN and status is ResponseStatus.OK
        return frames, should_shutdown

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

        inflight = 0
        shutdown_response: list[bytes] | None = None
        accepting_requests = True

        while True:
            events = dict(poller.poll())

            if accepting_requests and self.socket in events:
                frames = self.socket.recv_multipart()
                if len(frames) < 3:
                    raise ValueError(
                        f"Expected [identity, request_id, request_type, *payloads], got {len(frames)} frames"
                    )

                identity, request_id, request_type_bytes, *payloads = frames
                self._submit_request(identity, request_id, request_type_bytes, payloads)
                inflight += 1

                try:
                    request_type = decode_request_type(request_type_bytes)
                except ValueError:
                    request_type = None

                if request_type is RequestType.SHUTDOWN and not payloads:
                    accepting_requests = False
                    poller.unregister(self.socket)

            if self._notify_reader.fileno() in events:
                self._notify_reader.recv(4096)

                try:
                    while True:
                        response_frames, should_shutdown = self._output_queue.get_nowait()
                        inflight -= 1

                        if should_shutdown:
                            shutdown_response = response_frames
                        else:
                            self.socket.send_multipart(response_frames)
                except queue.Empty:
                    pass

                if shutdown_response is not None and inflight == 0:
                    self.socket.send_multipart(shutdown_response)
                    return

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

        self._outbound_queue: queue.Queue[OutboundRequest] = queue.Queue()
        self._pending_futures: dict[bytes, tuple[bytes, Future[list[bytes]]]] = {}

        self._client_close_requested = threading.Event()
        self._server_healthy = threading.Event()
        self._server_healthy.set()
        self._lifecycle_lock = threading.Lock()
        self._resources_released = False

        self._io_ready = threading.Event()
        self._io_error: Exception | None = None

        self._notify_reader, self._notify_writer = socket.socketpair()
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True, name="ascend-store-kv-client")
        self._io_thread.start()

        self._io_ready.wait()
        if self._io_error is not None:
            raise RuntimeError("Failed to start KV cache client I/O thread") from self._io_error

    @property
    def is_healthy(self) -> bool:
        return self._server_healthy.is_set()

    def submit_request(
        self, request_type: RequestType, payloads: list[bytes] | None = None
    ) -> Future[list[bytes]]:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or self._resources_released:
                raise AscendStoreKVCacheClientClosedError("KV cache client is closed")

            if not self._server_healthy.is_set():
                raise AscendStoreKVCacheServerUnavailableError("KV cache server is unavailable")

            request_id = str(next(self._request_ids)).encode()
            request_type_bytes = encode_request_type(request_type)
            future: Future[list[bytes]] = Future()

            self._outbound_queue.put((request_id, request_type_bytes, payloads or [], future))
            self._notify_writer.send(b"\x01")

        return future

    def _process_outbound(self, zmq_socket: zmq.Socket) -> None:
        try:
            while True:
                request_id, request_type_bytes, payloads, future = self._outbound_queue.get_nowait()
                self._pending_futures[request_id] = (request_type_bytes, future)
                zmq_socket.send_multipart([request_id, request_type_bytes, *payloads])
        except queue.Empty:
            pass

    def _process_inbound(self, zmq_socket: zmq.Socket) -> None:
        frames = zmq_socket.recv_multipart()
        if len(frames) < 3:
            raise ValueError(f"Expected [request_id, request_type, status, *responses], got {len(frames)} frames")

        request_id, response_request_type, status_bytes, *responses = frames
        pending = self._pending_futures.pop(request_id, None)
        if pending is None:
            raise ValueError(f"Received response for unknown request ID: {request_id!r}")

        request_type_bytes, future = pending

        if response_request_type != request_type_bytes:
            future.set_exception(
                ValueError(
                    f"Response request type mismatch: expected {request_type_bytes!r}, got {response_request_type!r}"
                )
            )
            return

        status = decode_response_status(status_bytes)

        if status is ResponseStatus.ERROR:
            message = responses[0].decode() if responses else "Unknown server error"
            future.set_exception(AscendStoreKVCacheRemoteError(message))
            return

        future.set_result(responses)

    def _drain_inbound(self, zmq_socket: zmq.Socket) -> None:
        while zmq_socket.poll(timeout=0, flags=zmq.POLLIN):
            self._process_inbound(zmq_socket)

    def _handle_server_disconnect(self, zmq_socket: zmq.Socket) -> None:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or not self._server_healthy.is_set():
                return

            self._server_healthy.clear()

        self._drain_inbound(zmq_socket)
        self._fail_pending(
            AscendStoreKVCacheServerUnavailableError(f"KV cache server disconnected: {self.server_url}")
        )

    def _handle_server_connected(self) -> None:
        with self._lifecycle_lock:
            if self._client_close_requested.is_set() or self._resources_released:
                return

            self._server_healthy.set()

    def _process_monitor_event(self, zmq_socket: zmq.Socket, monitor_socket: zmq.Socket) -> None:
        monitor_event = recv_monitor_message(monitor_socket)
        event = monitor_event["event"]

        if event == zmq.EVENT_DISCONNECTED:
            self._handle_server_disconnect(zmq_socket)
        elif event == zmq.EVENT_CONNECTED:
            self._handle_server_connected()

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
                events = dict(poller.poll())

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
        except Exception as exc:
            self._io_error = exc

            with self._lifecycle_lock:
                self._server_healthy.clear()

            self._fail_pending(exc)
            self._io_ready.set()
        finally:
            if monitor_socket is not None:
                monitor_socket.close(linger=0)

            if zmq_socket is not None:
                zmq_socket.close(linger=0)

            self._notify_reader.close()

    def _fail_pending(self, exc: Exception) -> None:
        while True:
            try:
                _, _, _, future = self._outbound_queue.get_nowait()
                if not future.done():
                    future.set_exception(exc)
            except queue.Empty:
                break

        for _, future in self._pending_futures.values():
            if not future.done():
                future.set_exception(exc)

        self._pending_futures.clear()

    def _request(
        self,
        request_type: RequestType,
        payloads: list[bytes] | None = None,
        timeout_ms: int = 5000,
    ) -> list[bytes]:
        future = self.submit_request(request_type, payloads)

        try:
            return future.result(timeout=timeout_ms / 1000)
        except TimeoutError as exc:
            raise TimeoutError(f"Timed out waiting for response to {request_type}") from exc

    def ping(self, timeout_ms: int = 5000) -> str:
        return self._request(RequestType.PING, timeout_ms=timeout_ms)[0].decode()

    def echo(self, payload: bytes, timeout_ms: int = 5000) -> bytes:
        return self._request(RequestType.ECHO, [payload], timeout_ms)[0]

    def shutdown(self, timeout_ms: int = 5000) -> str:
        return self._request(RequestType.SHUTDOWN, timeout_ms=timeout_ms)[0].decode()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._resources_released:
                return

            if not self._client_close_requested.is_set():
                self._client_close_requested.set()

                if self._io_thread.is_alive():
                    try:
                        self._notify_writer.send(b"\x01")
                    except OSError:
                        pass

        self._io_thread.join()

        with self._lifecycle_lock:
            if self._resources_released:
                return

            self._notify_writer.close()
            self.context.term()
            self._resources_released = True