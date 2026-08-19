from unittest.mock import MagicMock

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.lookup_worker import LookupKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


# isort: on


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

    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = rank
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
    config.cache_config.block_size = 16
    config.cache_config.prefix_match_unit = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    return config


def _make_worker(exists_result: list[int], tp_size: int = 1, rank: int = 0) -> LookupKVPoolWorker:
    store = MagicMock()
    store.exists.return_value = exists_result
    return LookupKVPoolWorker(_make_vllm_config(tp_size, rank), store=store, rank=rank)


def test_lookup_worker_reuses_original_lookup_implementation() -> None:
    assert LookupKVPoolWorker.lookup_scheduler is KVPoolWorker.lookup_scheduler


def test_lookup_worker_uses_registered_rank() -> None:
    worker = _make_worker([1, 1, 1, 1], tp_size=2, rank=1)

    assert worker.tp_rank == 1
    assert worker.pp_rank == 0


@pytest.mark.parametrize(
    ("exists_result", "expected"),
    [
        ([1, 1], 32),
        ([1, 0], 16),
        ([0, 1], 0),
    ],
)
def test_lookup_worker_single_tp(exists_result: list[int], expected: int) -> None:
    worker = _make_worker(exists_result)
    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == expected


@pytest.mark.parametrize(
    ("exists_result", "expected"),
    [
        ([1, 1, 1, 1], 32),
        ([1, 1, 1, 0], 16),
        ([1, 1, 0, 1], 0),
    ],
)
def test_lookup_worker_requires_all_tp_ranks(exists_result: list[int], expected: int) -> None:
    worker = _make_worker(exists_result, tp_size=2)
    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == expected


def test_lookup_worker_returns_miss_when_store_fails() -> None:
    store = MagicMock()
    store.exists.side_effect = RuntimeError("store unavailable")
    worker = LookupKVPoolWorker(_make_vllm_config(), store=store)

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0


def test_lookup_worker_returns_miss_before_store_is_bound() -> None:
    worker = LookupKVPoolWorker(_make_vllm_config())

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0
