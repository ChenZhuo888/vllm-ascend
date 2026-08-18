import multiprocessing as mp
from collections.abc import Iterator

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import (
    KVCacheClient,
    KVCacheMethod,
    KVCacheServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPClient,
    MPRemoteError,
)


def _run_kv_cache_server(conn) -> None:
    server = KVCacheServer("tcp://127.0.0.1:*", max_workers=2)

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_kv_cache_server() -> tuple[mp.Process, str]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_kv_cache_server, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "KV cache server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        parent_conn.close()

    return process, endpoint


@pytest.fixture
def kv_cache_server_url() -> Iterator[str]:
    process, endpoint = _start_kv_cache_server()

    try:
        yield endpoint
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)


def test_lookup_returns_cache_miss(kv_cache_server_url: str) -> None:
    with KVCacheClient(kv_cache_server_url) as client:
        assert client.is_connected
        assert client.lookup(num_computed_tokens=32) == 0


def test_lookup_rejects_negative_token_count(kv_cache_server_url: str) -> None:
    with KVCacheClient(kv_cache_server_url) as client:
        with pytest.raises(ValueError, match="token_count must not be negative"):
            client.lookup(num_computed_tokens=-1)

        assert client.lookup(num_computed_tokens=0) == 0


def test_server_rejects_malformed_lookup(kv_cache_server_url: str) -> None:
    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()

        with pytest.raises(MPRemoteError, match="LOOKUP expects 1 payload"):
            client.request(KVCacheMethod.LOOKUP)

        assert client.ping() == "OK"
