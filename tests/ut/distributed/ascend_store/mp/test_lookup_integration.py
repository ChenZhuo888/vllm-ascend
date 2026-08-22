import multiprocessing as mp
import time
from functools import partial
from unittest.mock import MagicMock, patch

# isort: off
# Import real pyzmq before _mock_deps to prevent it from being mocked.
import zmq.asyncio  # noqa: F401

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient, KVCacheServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.lookup_worker import (
    LookupKVPoolWorker,
    MPKVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerRegistration,
    WorkerLookupHandler,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

# isort: on

POOL_SCHEDULER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
_DEFAULT_URL = "tcp://127.0.0.1:*"
_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeStore:
    def __init__(self, exists_result: list[int]):
        self._exists_result = exists_result

    def exists(self, keys: list[str]) -> list[int]:
        return self._exists_result[: len(keys)]


def _make_vllm_config(tp_size: int = 1, rank: int = 0) -> MagicMock:
    config = MagicMock()

    hf_config = MagicMock(spec=[])
    config.model_config.model = "org/llama-7b"
    config.model_config.hf_text_config = hf_config
    config.model_config.hf_config = hf_config
    config.model_config.use_mla = False
    config.model_config.max_model_len = 1024
    config.model_config.get_num_layers.return_value = 2
    config.model_config.get_total_num_kv_heads.return_value = tp_size

    config.parallel_config.rank = rank
    config.parallel_config.world_size = tp_size
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
    config.kv_transfer_config.get_from_extra_config.return_value = True

    config.cache_config.block_size = 16
    config.cache_config.prefix_match_unit = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    return config


def _create_scheduler(
    registration: SchedulerRegistration,
    lookup_handler: WorkerLookupHandler,
) -> MPKVPoolScheduler:
    with patch(f"{POOL_SCHEDULER_MODULE}.importlib") as importlib_mock:
        importlib_mock.import_module.return_value = MagicMock()
        return MPKVPoolScheduler(registration, lookup_handler)


def _create_worker(registration: WorkerRegistration, exists_result: list[int]) -> LookupKVPoolWorker:
    return LookupKVPoolWorker(
        registration.vllm_config,
        store=_FakeStore(exists_result),
        kv_cache_config=registration.kv_cache_config,
        rank=registration.identity.rank,
    )


def _run_lookup_server(bind_url: str, conn, exists_result: list[int]) -> None:
    server = KVCacheServer(
        bind_url,
        max_workers=4,
        scheduler_factory=_create_scheduler,
        worker_factory=partial(_create_worker, exists_result=exists_result),
    )
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_lookup_server(exists_result: list[int]) -> tuple[mp.Process, str]:
    context = mp.get_context("fork")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_lookup_server, args=(_DEFAULT_URL, child_conn, exists_result))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "Lookup server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        raise
    finally:
        parent_conn.close()

    return process, endpoint


def _stop_lookup_server(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)


def _wait_until_connected(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_connected:
        if time.monotonic() >= deadline:
            raise AssertionError("KV cache client did not connect in time")
        time.sleep(0.01)


def test_scheduler_lookup_round_trip_uses_original_logic() -> None:
    process, endpoint = _start_lookup_server([1, 1, 1, 0])
    clients: list[KVCacheClient] = []

    try:
        for rank in range(2):
            worker_client = KVCacheClient(endpoint)
            clients.append(worker_client)
            _wait_until_connected(worker_client)
            assert worker_client.register_worker(_make_vllm_config(tp_size=2, rank=rank), kv_cache_config=None)

        scheduler_client = KVCacheClient(endpoint)
        clients.append(scheduler_client)
        _wait_until_connected(scheduler_client)
        assert scheduler_client.register_scheduler(
            _make_vllm_config(tp_size=2),
            kv_cache_config=None,
            page_size_bytes=0,
        )

        request = MagicMock()
        request.request_id = "request-0"
        request.prompt_token_ids = list(range(32))
        request.block_hashes = _BLOCK_HASHES
        request.num_tokens = 32

        assert scheduler_client.lookup(request, num_computed_tokens=0) == (16, False)
    finally:
        for client in clients:
            client.close()
        _stop_lookup_server(process)


def test_mp_classes_reuse_original_business_methods() -> None:
    assert MPKVPoolScheduler.get_num_new_matched_tokens is KVPoolScheduler.get_num_new_matched_tokens
