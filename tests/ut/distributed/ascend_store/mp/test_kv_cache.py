import multiprocessing as mp
import socket
import time
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import (
    KVCacheClient,
    KVCacheMethod,
    KVCacheServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import LookupHandler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPClient,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerUnavailableError,
)

KV_CACHE_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache"
_DEFAULT_URL = "tcp://127.0.0.1:*"
_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


def _lookup_first_16_tokens(
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
) -> int:
    assert block_hashes == [block_hash.hex() for block_hash in _BLOCK_HASHES]
    assert kv_cache_group_ids == [0, 1]
    assert not use_layerwise
    assert hbm_hit_tokens == 4
    return min(token_len, 16)


def _run_server(bind_url: str, conn, handler: LookupHandler | None) -> None:
    server = KVCacheServer(bind_url, max_workers=2, lookup_handler=handler)
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_server(bind_url: str = _DEFAULT_URL, handler: LookupHandler | None = None) -> tuple[mp.Process, str]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_server, args=(bind_url, child_conn, handler))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "KV cache server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        raise
    finally:
        parent_conn.close()

    return process, endpoint


def _stop_server(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)


def _get_unused_tcp_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _wait_for_connection_state(client: KVCacheClient, expected: bool, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while client.is_connected != expected:
        if time.monotonic() >= deadline:
            pytest.fail(f"Client connection state did not become {expected}")
        time.sleep(0.01)


@pytest.fixture
def kv_cache_server_url() -> Iterator[str]:
    process, endpoint = _start_server()
    try:
        yield endpoint
    finally:
        _stop_server(process)


def test_client_creation_does_not_wait_for_server() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        with KVCacheClient("tcp://127.0.0.1:12345"):
            pass

        rpc_client_class.assert_called_once_with("tcp://127.0.0.1:12345")
        rpc_client_class.return_value.wait_until_connected.assert_not_called()
        rpc_client_class.return_value.close.assert_called_once_with()


@pytest.mark.parametrize(
    "error",
    [
        MPServerUnavailableError("Server unavailable"),
        MPRequestTimeoutError("Request timeout"),
    ],
)
def test_lookup_returns_cache_miss_on_transport_failure(error: Exception) -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.request.side_effect = error

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            result = client.lookup(32, _BLOCK_HASHES, [0, 1], hbm_hit_tokens=4, timeout_ms=100)

        assert result == 0
        expected_payloads = (
            (32).to_bytes(8, byteorder="big"),
            (4).to_bytes(8, byteorder="big"),
            (2).to_bytes(8, byteorder="big"),
            (0).to_bytes(8, byteorder="big"),
            (1).to_bytes(8, byteorder="big"),
            b"01" * 32,
            b"02" * 32,
        )
        rpc_client.request.assert_called_once_with(KVCacheMethod.LOOKUP, expected_payloads, timeout_ms=100)


def test_lookup_recovers_when_server_starts_later() -> None:
    server_url = _get_unused_tcp_url()
    process = None

    with KVCacheClient(server_url) as client:
        assert not client.is_connected
        assert client.lookup(32, _BLOCK_HASHES, timeout_ms=100) == 0

        try:
            process, endpoint = _start_server(server_url)
            assert endpoint == server_url
            _wait_for_connection_state(client, expected=True)
            assert client.lookup(32, _BLOCK_HASHES) == 0

            _stop_server(process)
            _wait_for_connection_state(client, expected=False)
            assert client.lookup(32, _BLOCK_HASHES, timeout_ms=100) == 0
        finally:
            if process is not None and process.is_alive():
                _stop_server(process)


def test_lookup_uses_injected_handler() -> None:
    process, endpoint = _start_server(handler=_lookup_first_16_tokens)
    try:
        with KVCacheClient(endpoint) as client:
            _wait_for_connection_state(client, expected=True)
            assert client.lookup(32, _BLOCK_HASHES, [0, 1], hbm_hit_tokens=4) == 16
            assert client.lookup(8, _BLOCK_HASHES, [0, 1], hbm_hit_tokens=4) == 8
    finally:
        _stop_server(process)


def test_lookup_returns_cache_miss(kv_cache_server_url: str) -> None:
    with KVCacheClient(kv_cache_server_url) as client:
        _wait_for_connection_state(client, expected=True)
        assert client.lookup(32, _BLOCK_HASHES) == 0


def test_lookup_rejects_negative_token_count(kv_cache_server_url: str) -> None:
    with KVCacheClient(kv_cache_server_url) as client:
        _wait_for_connection_state(client, expected=True)
        with pytest.raises(ValueError, match="token_len must not be negative"):
            client.lookup(-1, _BLOCK_HASHES)
        assert client.lookup(0, _BLOCK_HASHES) == 0


def test_lookup_rejects_invalid_block_hash_type() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            with pytest.raises(TypeError, match="block_hash must be bytes"):
                client.lookup(32, ["invalid"])  # type: ignore[list-item]

        rpc_client_class.return_value.request.assert_not_called()


def test_server_rejects_malformed_lookup(kv_cache_server_url: str) -> None:
    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()
        with pytest.raises(MPRemoteError, match="LOOKUP expects at least 3 payloads"):
            client.request(KVCacheMethod.LOOKUP)
        assert client.ping() == "OK"
