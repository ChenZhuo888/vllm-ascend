import multiprocessing as mp
import threading
import time

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp_kv_cache import (
    AscendStoreKVCacheClient,
    AscendStoreKVCacheClientClosedError,
    AscendStoreKVCacheRemoteError,
    AscendStoreKVCacheServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp_protocol import (
    RequestType,
    ResponseStatus,
    encode_response_status,
)


def _run_server(conn):
    server = AscendStoreKVCacheServer("tcp://127.0.0.1:*")

    try:
        conn.send(server.endpoint)
        conn.close()
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


def test_client_server_lifecycle():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    server_process = ctx.Process(target=_run_server, args=(child_conn,))
    server_process.start()
    child_conn.close()

    client = None

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        client = AscendStoreKVCacheClient(parent_conn.recv())

        assert client.ping() == "OK"
        assert client.echo(b"hello ascend store") == b"hello ascend store"

        with pytest.raises(AscendStoreKVCacheRemoteError, match="ECHO expects 1 payload"):
            client._request(RequestType.ECHO, [])

        assert server_process.is_alive()
        assert client.ping() == "OK"
        assert client.shutdown() == "OK"

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

        future_0 = client.submit_request(RequestType.ECHO, [b"0"])
        future_1 = client.submit_request(RequestType.ECHO, [b"1"])
        future_2 = client.submit_request(RequestType.ECHO, [b"2"])

        assert future_0.result(timeout=5) == [b"0"]
        assert future_1.result(timeout=5) == [b"1"]
        assert future_2.result(timeout=5) == [b"2"]

        assert client.shutdown() == "OK"

        server_process.join(timeout=5)
        assert server_process.exitcode == 0
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

        future_0 = client.submit_request(RequestType.ECHO, [b"0"])
        future_1 = client.submit_request(RequestType.ECHO, [b"1"])
        future_2 = client.submit_request(RequestType.ECHO, [b"2"])

        assert future_0.result(timeout=5) == [b"0"]
        assert future_1.result(timeout=5) == [b"1"]
        assert future_2.result(timeout=5) == [b"2"]

        server_process.join(timeout=5)
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

        slow_future = client.submit_request(RequestType.ECHO, [b"slow"])
        fast_future = client.submit_request(RequestType.ECHO, [b"fast"])

        assert fast_future.result(timeout=5) == [b"fast"]
        assert not slow_future.done()
        assert slow_future.result(timeout=5) == [b"slow"]

        assert client.shutdown() == "OK"

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