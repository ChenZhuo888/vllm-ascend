import multiprocessing as mp
import socket
import time
from collections.abc import Iterator
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient, KVCacheMethod, KVCacheServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import WorkerLookupHandler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
    encode_registration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPClient,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerUnavailableError,
)

KV_CACHE_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache"
_DEFAULT_URL = "tcp://127.0.0.1:*"
_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeWorker:
    def __init__(self, matched_tokens: int):
        self._matched_tokens = matched_tokens

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return min(token_len, self._matched_tokens)


class _FakeScheduler:
    def __init__(self, identity: SchedulerIdentity, lookup_handler: WorkerLookupHandler):
        self._identity = identity
        self._lookup_handler = lookup_handler

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        matched_tokens = self._lookup_handler(
            self._identity,
            len(request.prompt_token_ids),
            request.block_hashes,
            [0],
            False,
            num_computed_tokens,
        )
        return max(matched_tokens - num_computed_tokens, 0), False


def _make_vllm_config(
    engine_id: str = "engine-0",
    rank: int = 0,
    marker: str = "",
):
    parallel_config = SimpleNamespace(rank=rank)
    kv_transfer_config = SimpleNamespace(engine_id=engine_id)
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        marker=marker,
        parallel_config=parallel_config,
    )


def _make_request():
    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(32)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=32,
    )


def _create_scheduler(
    registration: SchedulerRegistration,
    lookup_handler: WorkerLookupHandler,
) -> _FakeScheduler:
    return _FakeScheduler(registration.identity, lookup_handler)


def _create_worker(registration: WorkerRegistration, worker_hits: dict[int, int]) -> _FakeWorker:
    return _FakeWorker(worker_hits[registration.identity.rank])


def _run_server(bind_url: str, conn, worker_hits: dict[int, int]) -> None:
    server = KVCacheServer(
        bind_url,
        max_workers=4,
        scheduler_factory=_create_scheduler,
        worker_factory=partial(_create_worker, worker_hits=worker_hits),
    )
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_server(
    bind_url: str = _DEFAULT_URL,
    worker_hits: dict[int, int] | None = None,
) -> tuple[mp.Process, str]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_server, args=(bind_url, child_conn, worker_hits or {0: 16}))
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


def _wait_until_connected(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_connected:
        if time.monotonic() >= deadline:
            pytest.fail("KV cache client did not connect in time")
        time.sleep(0.01)


def _wait_until_registered(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_registered:
        if time.monotonic() >= deadline:
            pytest.fail("KV cache client did not register in time")
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


def test_registration_checks_application_readiness() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.ping.side_effect = MPRequestTimeoutError("PING timeout")

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            registered = client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)

        assert not registered
        rpc_client.request.assert_not_called()
        rpc_client.start_heartbeat.assert_called_once()


def test_lookup_returns_cache_miss_on_transport_failure() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.is_server_responsive = True
        rpc_client.ping.return_value = "OK"
        rpc_client.request.side_effect = [[b"OK"], MPServerUnavailableError("Server unavailable")]

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            assert client.lookup(_make_request(), 0) == (0, False)
            assert not client.is_registered


def test_lookup_validates_request_before_degrading() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client_class.return_value.is_transport_connected = False

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert not client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            with pytest.raises(ValueError, match="num_computed_tokens must not be negative"):
                client.lookup(_make_request(), -1)


def test_multiple_workers_are_registered_and_rank_zero_serves_lookup() -> None:
    process, endpoint = _start_server(worker_hits={0: 16, 1: 32})
    clients: list[KVCacheClient] = []

    try:
        for rank in (1, 0):
            client = KVCacheClient(endpoint)
            clients.append(client)
            _wait_until_connected(client)
            assert client.register_worker(_make_vllm_config(rank=rank), kv_cache_config=None)

        scheduler_client = KVCacheClient(endpoint)
        clients.append(scheduler_client)
        _wait_until_connected(scheduler_client)
        assert scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), 0) == (16, False)
    finally:
        for client in clients:
            client.close()
        _stop_server(process)


def test_registration_identity_uses_engine_id_and_worker_rank() -> None:
    config = _make_vllm_config(engine_id="engine-1", rank=3)

    scheduler_registration = SchedulerRegistration.create(config, kv_cache_config=None, page_size_bytes=0)
    worker_registration = WorkerRegistration.create(config, kv_cache_config=None)

    assert scheduler_registration.identity == SchedulerIdentity("engine-1")
    assert worker_registration.identity == WorkerIdentity("engine-1", rank=3)


def test_registration_is_idempotent_and_rejects_conflicts(kv_cache_server_url: str) -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(marker="first"),
        kv_cache_config=None,
        page_size_bytes=0,
    )
    payload = encode_registration(registration)
    conflicting_registration = SchedulerRegistration.create(
        _make_vllm_config(marker="second"),
        kv_cache_config=None,
        page_size_bytes=0,
    )
    conflicting_payload = encode_registration(conflicting_registration)

    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()
        assert client.request(KVCacheMethod.REGISTER_SCHEDULER, (payload,)) == [b"OK"]
        assert client.request(KVCacheMethod.REGISTER_SCHEDULER, (payload,)) == [b"OK"]

        with pytest.raises(MPRemoteError, match="different configuration"):
            client.request(KVCacheMethod.REGISTER_SCHEDULER, (conflicting_payload,))


def test_lookup_recovers_when_server_starts_later() -> None:
    server_url = _get_unused_tcp_url()
    process = None

    with KVCacheClient(server_url) as worker_client, KVCacheClient(server_url) as scheduler_client:
        assert not worker_client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert not scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), 0) == (0, False)

        try:
            process, endpoint = _start_server(server_url, worker_hits={0: 16})
            assert endpoint == server_url
            _wait_until_connected(worker_client)
            _wait_until_connected(scheduler_client)
            _wait_until_registered(worker_client)
            _wait_until_registered(scheduler_client)
            assert scheduler_client.lookup(_make_request(), 0) == (16, False)
        finally:
            if process is not None:
                _stop_server(process)


def test_server_rejects_malformed_lookup(kv_cache_server_url: str) -> None:
    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()
        with pytest.raises(MPRemoteError, match="LOOKUP expects at least 5 payloads"):
            client.request(KVCacheMethod.LOOKUP)
        assert client.ping() == "OK"
