import multiprocessing as mp

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import (
    MPClient,
    MPClientClosedError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServer,
    SystemMethod,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import (
    ResponseStatus,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)

UPPERCASE_METHOD = "TEST_UPPERCASE"
INVALID_RESPONSE_METHOD = "TEST_INVALID_RESPONSE"


def _start_server(target):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=target, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
        process.join()
        raise

    return process, parent_conn, endpoint


def _cleanup(client: MPClient | None, conn, process) -> None:
    if client is not None:
        client.close()

    conn.close()

    if process.is_alive():
        process.terminate()
        process.join()


def _send_ok_response(
        router: zmq.Socket,
        request: list[bytes],
        responses: tuple[bytes, ...] | None = None,
) -> None:
    identity, *request_frames = request
    request_id, method, payloads = decode_request(request_frames)
    response_payloads = payloads if responses is None else responses
    router.send_multipart(
        [identity, *encode_response(request_id, method, ResponseStatus.OK, response_payloads)]
    )


def _run_server(conn) -> None:
    server = MPServer("tcp://127.0.0.1:*")

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _uppercase_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if len(payloads) != 1:
        raise ValueError(f"{UPPERCASE_METHOD} expects 1 payload, got {len(payloads)}")
    return (payloads[0].upper(),)


def _invalid_response_handler(_payloads: tuple[bytes, ...]):
    return None


def _run_server_with_injected_handlers(conn) -> None:
    server = MPServer(
        "tcp://127.0.0.1:*",
        handlers={
            UPPERCASE_METHOD: _uppercase_handler,
            INVALID_RESPONSE_METHOD: _invalid_response_handler,
        },
    )

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_reordering_server(conn) -> None:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")
        requests = [router.recv_multipart() for _ in range(3)]

        for request in reversed(requests):
            _send_ok_response(router, request)

        if conn.recv() != "stop":
            raise ValueError("Unexpected reordering server command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def _run_delayed_response_server(conn) -> None:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        delayed_request = router.recv_multipart()
        conn.send("request_received")

        if conn.recv() != "send_late_response":
            raise ValueError("Unexpected delayed response server command")

        _send_ok_response(router, delayed_request)
        conn.send("late_response_sent")

        ping_request = router.recv_multipart()
        _send_ok_response(router, ping_request, (b"OK",))
        conn.send("completed")

        if conn.recv() != "stop":
            raise ValueError("Unexpected delayed response server stop command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def _run_hanging_server(conn) -> None:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")
        router.recv_multipart()
        conn.send("request_received")

        if conn.recv() != "stop":
            raise ValueError("Unexpected hanging server command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def test_protocol_round_trip():
    request_frames = encode_request(b"request-1", SystemMethod.ECHO, (b"payload",))
    assert decode_request(request_frames) == (b"request-1", "ECHO", (b"payload",))

    response_frames = encode_response(
        b"request-1",
        SystemMethod.ECHO,
        ResponseStatus.OK,
        (b"response",),
    )
    assert decode_response(response_frames) == (
        b"request-1",
        "ECHO",
        ResponseStatus.OK,
        (b"response",),
    )


def test_client_server_round_trip():
    process, parent_conn, endpoint = _start_server(_run_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        assert client.is_transport_connected
        assert client.is_server_responsive
        assert not client.is_heartbeat_running
        assert client.ping() == "OK"
        assert client.echo(b"hello ascend store") == b"hello ascend store"

        with pytest.raises(MPRemoteError, match="ECHO expects 1 payload"):
            client.request(SystemMethod.ECHO, [])

        assert process.is_alive()
        assert client.ping() == "OK"
    finally:
        _cleanup(client, parent_conn, process)


def test_server_uses_injected_handlers():
    process, parent_conn, endpoint = _start_server(_run_server_with_injected_handlers)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        assert client.request(UPPERCASE_METHOD, [b"hello ascend store"]) == [b"HELLO ASCEND STORE"]

        with pytest.raises(MPRemoteError, match="Payloads must be an iterable of bytes"):
            client.request(INVALID_RESPONSE_METHOD)

        assert process.is_alive()
        assert client.ping() == "OK"
    finally:
        _cleanup(client, parent_conn, process)


def test_async_out_of_order_responses():
    process, parent_conn, endpoint = _start_server(_run_reordering_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        futures = [
            client.submit_request(SystemMethod.ECHO, [b"0"]),
            client.submit_request(SystemMethod.ECHO, [b"1"]),
            client.submit_request(SystemMethod.ECHO, [b"2"]),
        ]

        assert [future.result(timeout=5) for future in futures] == [[b"0"], [b"1"], [b"2"]]

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)


def test_request_timeout_discards_late_response():
    process, parent_conn, endpoint = _start_server(_run_delayed_response_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        future = client.submit_request(SystemMethod.ECHO, [b"late"], timeout_ms=500)

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"

        with pytest.raises(MPRequestTimeoutError, match="Timed out waiting for response"):
            future.result(timeout=5)

        parent_conn.send("send_late_response")
        assert parent_conn.poll(5), "Server did not send late response in time"
        assert parent_conn.recv() == "late_response_sent"

        assert client.ping(timeout_ms=2000) == "OK"

        with pytest.raises(MPRequestTimeoutError, match="Timed out waiting for response"):
            future.result()

        assert parent_conn.poll(5), "Server did not finish requests in time"
        assert parent_conn.recv() == "completed"

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)


def test_dispatched_request_cannot_be_cancelled():
    process, parent_conn, endpoint = _start_server(_run_hanging_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        future = client.submit_request(SystemMethod.ECHO, [b"never-return"])

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"
        assert future.running()
        assert not future.cancel()

        client.close()

        with pytest.raises(MPClientClosedError, match="MP client was closed"):
            future.result()

        with pytest.raises(MPClientClosedError, match="MP client is closed"):
            client.submit_request(SystemMethod.PING)

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)
