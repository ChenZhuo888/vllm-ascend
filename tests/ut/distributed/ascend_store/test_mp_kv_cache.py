import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp_kv_cache import (
    AscendStoreKVCacheClient,
    AscendStoreKVCacheClientClosedError,
    AscendStoreKVCacheRemoteError,
    AscendStoreKVCacheRequestTimeoutError,
    AscendStoreKVCacheServer,
    AscendStoreKVCacheServerUnavailableError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp_protocol import (
    RequestType,
    ResponseStatus,
    encode_response_status,
)


def _wait_until(predicate: Callable[[], bool], message: str, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), message


def _wait_until_connected(client: AscendStoreKVCacheClient, timeout: float = 5) -> None:
    _wait_until(
        lambda: client.is_transport_connected,
        "Client did not connect to server in time",
        timeout,
    )


def _send_ok_response(socket: zmq.Socket, request: list[bytes], responses: list[bytes] | None = None) -> None:
    identity, request_id, request_type, *payloads = request
    response_payloads = payloads if responses is None else responses
    socket.send_multipart(
        [identity, request_id, request_type, encode_response_status(ResponseStatus.OK), *response_payloads]
    )


def _run_server(conn):
    server = AscendStoreKVCacheServer("tcp://127.0.0.1:*")

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_server_at_endpoint(endpoint):
    server = AscendStoreKVCacheServer(endpoint)

    try:
        server.run()
    finally:
        server.close()


def _run_reordering_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")
        conn.close()

        requests = [socket.recv_multipart() for _ in range(3)]

        for identity, request_id, request_type, *payloads in reversed(requests):
            socket.send_multipart(
                [identity, request_id, request_type, encode_response_status(ResponseStatus.OK), *payloads]
            )
    finally:
        socket.close(linger=0)
        context.term()


class _ConcurrentTestServer(AscendStoreKVCacheServer):
    def __init__(self, bind_url: str):
        self._release_slow_request = threading.Event()
        super().__init__(bind_url, max_workers=2)

    def _handle_echo(self, payloads: list[bytes]) -> list[bytes]:
        if len(payloads) != 1:
            raise ValueError(f"ECHO expects 1 payload, got {len(payloads)}")

        if payloads[0] == b"slow":
            if not self._release_slow_request.wait(timeout=5):
                raise TimeoutError("Fast request did not execute")

            time.sleep(0.1)
            return [b"slow"]

        if payloads[0] == b"fast":
            self._release_slow_request.set()
            return [b"fast"]

        return payloads


def _run_concurrent_server(conn):
    server = _ConcurrentTestServer("tcp://127.0.0.1:*")

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_delayed_response_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        delayed_request = socket.recv_multipart()
        conn.send("request_received")

        if conn.recv() != "send_late_response":
            raise ValueError("Unexpected delayed response server command")

        _send_ok_response(socket, delayed_request)
        conn.send("late_response_sent")

        ping_request = socket.recv_multipart()
        _send_ok_response(socket, ping_request, [b"OK"])
        conn.send("completed")

        if conn.recv() != "stop":
            raise ValueError("Unexpected delayed response server stop command")
    finally:
        socket.close(linger=0)
        context.term()
        conn.close()


def _run_mixed_latency_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        requests = [socket.recv_multipart() for _ in range(2)]
        requests_by_payload = {request[-1]: request for request in requests}
        _send_ok_response(socket, requests_by_payload[b"fast"])
        conn.send("fast_response_sent")

        if conn.recv() != "send_late_response":
            raise ValueError("Unexpected mixed latency server command")

        _send_ok_response(socket, requests_by_payload[b"slow"])
        conn.send("late_response_sent")

        ping_request = socket.recv_multipart()
        _send_ok_response(socket, ping_request, [b"OK"])
        conn.send("completed")

        if conn.recv() != "stop":
            raise ValueError("Unexpected mixed latency server stop command")
    finally:
        socket.close(linger=0)
        context.term()
        conn.close()


def _run_controlled_heartbeat_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        blocked_heartbeat = socket.recv_multipart()
        conn.send("heartbeat_blocked")

        if conn.recv() != "recover":
            raise ValueError("Unexpected heartbeat server recovery command")

        _send_ok_response(socket, blocked_heartbeat, [b"OK"])
        first_live_response_sent = False

        while True:
            if conn.poll():
                if conn.recv() != "stop":
                    raise ValueError("Unexpected heartbeat server stop command")
                break

            if not socket.poll(timeout=50, flags=zmq.POLLIN):
                continue

            heartbeat_request = socket.recv_multipart()
            _send_ok_response(socket, heartbeat_request, [b"OK"])

            if not first_live_response_sent:
                conn.send("heartbeat_response_sent")
                first_live_response_sent = True
    finally:
        socket.close(linger=0)
        context.term()
        conn.close()


def _run_hanging_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        socket.recv_multipart()
        conn.send("request_received")

        time.sleep(30)
    finally:
        socket.close(linger=0)
        context.term()
        conn.close()


def _run_crashing_server(conn):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    conn.send(f"tcp://127.0.0.1:{port}")

    socket.recv_multipart()
    conn.send("request_received")

    os._exit(1)


def test_client_server_round_trip():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        assert client.is_transport_connected
        assert client.is_server_responsive
        assert not client.is_heartbeat_running
        assert client.ping() == "OK"
        assert client.echo(b"hello ascend store") == b"hello ascend store"

        with pytest.raises(AscendStoreKVCacheRemoteError, match="ECHO expects 1 payload"):
            client._request(RequestType.ECHO, [])

        assert server_process.is_alive()
        assert client.ping() == "OK"
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_async_requests():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        future_0 = client.submit_request(RequestType.ECHO, [b"0"])
        future_1 = client.submit_request(RequestType.ECHO, [b"1"])
        future_2 = client.submit_request(RequestType.ECHO, [b"2"])

        assert future_0.result(timeout=5) == [b"0"]
        assert future_1.result(timeout=5) == [b"1"]
        assert future_2.result(timeout=5) == [b"2"]
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_async_out_of_order_responses():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_reordering_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        future_0 = client.submit_request(RequestType.ECHO, [b"0"])
        future_1 = client.submit_request(RequestType.ECHO, [b"1"])
        future_2 = client.submit_request(RequestType.ECHO, [b"2"])

        assert future_0.result(timeout=5) == [b"0"]
        assert future_1.result(timeout=5) == [b"1"]
        assert future_2.result(timeout=5) == [b"2"]

        server_process.join(timeout=5)
        assert not server_process.is_alive()
        assert server_process.exitcode == 0
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_server_handles_requests_concurrently():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_concurrent_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        slow_future = client.submit_request(RequestType.ECHO, [b"slow"])
        fast_future = client.submit_request(RequestType.ECHO, [b"fast"])

        assert fast_future.result(timeout=5) == [b"fast"]
        assert not slow_future.done()
        assert slow_future.result(timeout=5) == [b"slow"]
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_request_timeout_cleans_pending_and_discards_late_response():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_delayed_response_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        future = client.submit_request(RequestType.ECHO, [b"late"], timeout_ms=500)

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"

        with pytest.raises(AscendStoreKVCacheRequestTimeoutError, match="Timed out waiting for response"):
            future.result(timeout=5)

        assert future.done()
        assert not client._pending_futures

        parent_conn.send("send_late_response")
        assert parent_conn.poll(5), "Server did not send late response in time"
        assert parent_conn.recv() == "late_response_sent"

        assert client.ping(timeout_ms=2000) == "OK"

        with pytest.raises(AscendStoreKVCacheRequestTimeoutError, match="Timed out waiting for response"):
            future.result()

        assert client.is_transport_connected
        assert parent_conn.poll(5), "Server did not finish requests in time"
        assert parent_conn.recv() == "completed"

        parent_conn.send("stop")
        server_process.join(timeout=5)
        assert not server_process.is_alive()
        assert server_process.exitcode == 0
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_request_timeout_does_not_affect_concurrent_request():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_mixed_latency_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        slow_future = client.submit_request(RequestType.ECHO, [b"slow"], timeout_ms=500)
        fast_future = client.submit_request(RequestType.ECHO, [b"fast"], timeout_ms=2000)

        assert parent_conn.poll(5), "Server did not send fast response in time"
        assert parent_conn.recv() == "fast_response_sent"
        assert fast_future.result(timeout=5) == [b"fast"]

        with pytest.raises(AscendStoreKVCacheRequestTimeoutError, match="Timed out waiting for response"):
            slow_future.result(timeout=5)

        parent_conn.send("send_late_response")
        assert parent_conn.poll(5), "Server did not send late response in time"
        assert parent_conn.recv() == "late_response_sent"

        assert client.ping(timeout_ms=2000) == "OK"
        assert client.is_transport_connected
        assert parent_conn.poll(5), "Server did not finish requests in time"
        assert parent_conn.recv() == "completed"

        parent_conn.send("stop")
        server_process.join(timeout=5)
        assert not server_process.is_alive()
        assert server_process.exitcode == 0
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_heartbeat_tracks_unresponsive_server_and_recovery():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_controlled_heartbeat_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None
    responsiveness_during_recovery: list[bool] = []
    recovery_called = threading.Event()

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        def recover() -> bool:
            responsiveness_during_recovery.append(client.is_server_responsive)
            recovery_called.set()
            return True

        client.start_heartbeat(interval_ms=50, timeout_ms=200, recovery_callback=recover)

        assert parent_conn.poll(5), "Server did not receive heartbeat in time"
        assert parent_conn.recv() == "heartbeat_blocked"
        _wait_until(
            lambda: not client.is_server_responsive,
            "Client did not mark server as unresponsive",
        )

        assert client.is_transport_connected
        assert client.is_heartbeat_running

        parent_conn.send("recover")
        assert parent_conn.poll(5), "Server did not send recovery heartbeat response in time"
        assert parent_conn.recv() == "heartbeat_response_sent"
        _wait_until(recovery_called.is_set, "Recovery callback was not called")
        _wait_until(
            lambda: client.is_server_responsive,
            "Client did not mark server as responsive after recovery",
        )

        assert responsiveness_during_recovery == [False]

        client.stop_heartbeat()

        assert not client.is_heartbeat_running
        assert client.is_transport_connected
        assert client.is_server_responsive

        parent_conn.send("stop")
        server_process.join(timeout=5)
        assert not server_process.is_alive()
        assert server_process.exitcode == 0
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_client_close_stops_heartbeat():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        client.start_heartbeat(interval_ms=50, timeout_ms=500)
        _wait_until(lambda: client.is_heartbeat_running, "Heartbeat thread did not start")

        client.close()

        assert not client.is_heartbeat_running
        assert not client.is_transport_connected
        assert not client.is_server_responsive

        with pytest.raises(AscendStoreKVCacheClientClosedError, match="KV cache client is closed"):
            client.submit_request(RequestType.PING)
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_client_close_fails_pending_request():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_hanging_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        future = client.submit_request(RequestType.ECHO, [b"never-return"])

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"
        assert not future.done()

        client.close()

        assert future.done()

        with pytest.raises(AscendStoreKVCacheClientClosedError, match="KV cache client was closed"):
            future.result()

        with pytest.raises(AscendStoreKVCacheClientClosedError, match="KV cache client is closed"):
            client.submit_request(RequestType.PING)
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_server_crash_fails_pending_request():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_crashing_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())
        _wait_until_connected(client)

        future = client.submit_request(RequestType.ECHO, [b"never-return"])

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"

        server_process.join(timeout=5)
        assert not server_process.is_alive()
        assert server_process.exitcode != 0

        with pytest.raises(AscendStoreKVCacheServerUnavailableError, match="disconnected"):
            future.result(timeout=5)

        assert future.done()
        assert not client.is_transport_connected

        with pytest.raises(AscendStoreKVCacheServerUnavailableError, match="unavailable"):
            client.submit_request(RequestType.PING)
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()


def test_client_reconnects_after_external_server_restart():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    first_server_process = ctx.Process(target=_run_crashing_server, args=(child_conn,))
    first_server_process.start()
    child_conn.close()

    second_server_process = None
    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        endpoint = parent_conn.recv()
        client = AscendStoreKVCacheClient(endpoint)
        _wait_until_connected(client)

        future = client.submit_request(RequestType.ECHO, [b"never-return"])

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"

        first_server_process.join(timeout=5)
        assert not first_server_process.is_alive()
        assert first_server_process.exitcode != 0

        with pytest.raises(AscendStoreKVCacheServerUnavailableError, match="disconnected"):
            future.result(timeout=5)

        assert future.done()
        assert not client.is_transport_connected

        with pytest.raises(AscendStoreKVCacheServerUnavailableError, match="unavailable"):
            client.submit_request(RequestType.PING)

        second_server_process = ctx.Process(target=_run_server_at_endpoint, args=(endpoint,))
        second_server_process.start()
        _wait_until_connected(client)

        assert client.ping() == "OK"
        assert client.echo(b"after-restart") == b"after-restart"
    finally:
        if client is not None:
            client.close()

        parent_conn.close()

        if first_server_process.is_alive():
            first_server_process.terminate()
            first_server_process.join()

        if second_server_process is not None and second_server_process.is_alive():
            second_server_process.terminate()
            second_server_process.join()