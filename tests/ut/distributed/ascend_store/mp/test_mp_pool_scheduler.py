from unittest.mock import MagicMock, patch

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mp_pool_scheduler import MPKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.request_view import BlocksView, RequestView
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

# isort: on

POOL_SCHEDULER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """KVPoolScheduler.__init__ loads the backend module dynamically; point
    importlib at a MagicMock so no real backend is imported."""
    with patch(f"{POOL_SCHEDULER_MODULE}.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


def _make_config(kv_role="kv_producer", extra_config=None, block_size=16):
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = block_size
    config.cache_config.hash_block_size = block_size
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    return config


def _make_scheduler(extra_config=None, kv_role="kv_producer", block_size=16) -> tuple[MPKVPoolScheduler, MagicMock]:
    lookup_handler = MagicMock(return_value=0)
    registration = SchedulerRegistration.create(_make_config(kv_role, extra_config, block_size), None, 0)
    return MPKVPoolScheduler(registration, lookup_handler), lookup_handler


def _make_request(num_tokens=64, num_computed=0) -> MagicMock:
    request = MagicMock()
    request.prompt_token_ids = list(range(num_tokens))
    request.num_tokens = num_tokens
    request.request_id = "r1"
    request.block_hashes = [bytes([index]) * 32 for index in range(num_tokens // 16)]
    return request


def test_mp_scheduler_reuses_original_business_method() -> None:
    assert MPKVPoolScheduler.get_num_new_matched_tokens is KVPoolScheduler.get_num_new_matched_tokens


def test_mp_scheduler_reads_use_layerwise_from_extra_config() -> None:
    scheduler, _ = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    assert scheduler.use_layerwise is True
    assert scheduler.use_gva_layerwise is False


def test_mp_scheduler_defaults_to_non_layerwise() -> None:
    scheduler, _ = _make_scheduler()
    assert scheduler.use_layerwise is False


def test_mp_scheduler_consumer_no_load_skips_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler(kv_role="kv_consumer")
    assert scheduler.get_num_new_matched_tokens(_make_request(), 0) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_too_short_prompt_skips_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler(block_size=64)
    request = _make_request(num_tokens=32)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_full_hbm_hit_skips_external_lookup() -> None:
    scheduler, lookup_handler = _make_scheduler()
    request = _make_request()
    assert scheduler.get_num_new_matched_tokens(request, 64) == (0, False)
    lookup_handler.assert_not_called()


def test_mp_scheduler_hit_returns_need_to_allocate_and_records_load_spec() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()

    need, is_async = scheduler.get_num_new_matched_tokens(request, 16)

    assert (need, is_async) == (32, False)
    load_spec = scheduler.load_specs["r1"]
    assert load_spec.vllm_cached_tokens == 16
    assert load_spec.kvpool_cached_tokens == 48
    assert load_spec.kvpool_store_skip_tokens == 48
    # The bridge hides the zmq client: the original client interface is served
    # by the in-process lookup handler with the same arguments.
    lookup_handler.assert_called_once_with(
        SchedulerIdentity("engine-0", 0),
        64,
        request.block_hashes,
        [0],
        False,
        16,
    )


def test_mp_scheduler_full_external_hit_returns_all_but_one_token() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 64

    need, _ = scheduler.get_num_new_matched_tokens(_make_request(), 0)

    assert need == 63
    assert scheduler.load_specs["r1"].kvpool_cached_tokens == 63
    assert scheduler.load_specs["r1"].kvpool_store_skip_tokens == 64


def test_mp_scheduler_hit_below_computed_tokens_allocates_nothing() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 8

    assert scheduler.get_num_new_matched_tokens(_make_request(), 16) == (0, False)


def test_mp_scheduler_async_hit_reports_async_load() -> None:
    scheduler, lookup_handler = _make_scheduler(extra_config={"backend": "mooncake", "load_async": True})
    lookup_handler.return_value = 48

    need, is_async = scheduler.get_num_new_matched_tokens(_make_request(), 16)

    assert (need, is_async) == (32, True)


def test_mp_scheduler_layerwise_queries_store_scheduler_directly() -> None:
    scheduler, lookup_handler = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    scheduler.store_scheduler.batch_is_exist = MagicMock(side_effect=lambda keys: [1] * len(keys))
    request = _make_request()

    need, _ = scheduler.get_num_new_matched_tokens(request, 0)

    # Every block hits across all layers: 64 tokens, reduced by one for scheduling.
    assert need == 63
    scheduler.store_scheduler.batch_is_exist.assert_called_once()
    lookup_handler.assert_not_called()


def test_mp_scheduler_layerwise_partial_layer_miss_stops_at_last_full_block() -> None:
    scheduler, _ = _make_scheduler(extra_config={"backend": "mooncake", "use_layerwise": True})
    # Each block spreads over 2 layer keys (num_layers=2); dropping the last
    # key misses one layer of the final block, leaving 3 full blocks = 48 tokens.
    scheduler.store_scheduler.batch_is_exist = MagicMock(side_effect=lambda keys: [1] * (len(keys) - 1) + [0])
    request = _make_request()

    need, _ = scheduler.get_num_new_matched_tokens(request, 0)

    assert need == 48


def _make_view(request: MagicMock) -> RequestView:
    return RequestView(
        request_id=request.request_id,
        prompt_token_ids=list(request.prompt_token_ids),
        block_hashes=list(request.block_hashes),
        num_prompt_tokens=len(request.prompt_token_ids),
        num_tokens=request.num_tokens,
    )


def test_mp_scheduler_update_state_after_alloc_flips_can_load_and_registers_view() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    view = _make_view(request)
    scheduler.update_state_after_alloc(view, BlocksView(block_ids_by_group=[[7, 8]]), 32)

    assert scheduler.load_specs["r1"].can_load is True
    stored_request, stored_blocks = scheduler._unfinished_requests["r1"]
    assert stored_request is view
    assert stored_blocks == [[7, 8]]


def test_mp_scheduler_update_state_after_alloc_without_load_spec_only_registers() -> None:
    scheduler, _ = _make_scheduler()
    request = _make_request()

    view = _make_view(request)
    scheduler.update_state_after_alloc(view, BlocksView(block_ids_by_group=[[7]]), 16)

    assert "r1" not in scheduler.load_specs
    assert scheduler._unfinished_requests["r1"] == (view, [[7]])


def test_mp_scheduler_update_state_after_alloc_zero_external_keeps_load_unloadable() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[]), 0)

    # Non-layerwise requests with zero allocated blocks cannot load.
    assert scheduler.load_specs["r1"].can_load is False
    assert scheduler._unfinished_requests["r1"][1] == [[]]


def test_mp_scheduler_update_state_after_alloc_rejects_mismatched_allocation() -> None:
    scheduler, lookup_handler = _make_scheduler()
    lookup_handler.return_value = 48
    request = _make_request()
    scheduler.get_num_new_matched_tokens(request, 16)

    with pytest.raises(AssertionError, match="Mismatch in number of tokens"):
        scheduler.update_state_after_alloc(_make_view(request), BlocksView(block_ids_by_group=[[7]]), 31)
