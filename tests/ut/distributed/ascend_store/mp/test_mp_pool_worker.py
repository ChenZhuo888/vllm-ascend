from unittest.mock import MagicMock

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mp_pool_worker import MPKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
    export_worker_kv_caches,
    import_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.request_view import KVCacheStorageSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on


class _CPUMemoryAdapter:
    def __init__(self):
        self.storages: list[torch.Tensor] = []

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        index = len(self.storages)
        self.storages.append(storage)
        return KVCacheStorageSpec(
            size_bytes=storage.untyped_storage().nbytes(),
            device_type="cpu",
            device_uuid="cpu",
            handle_type="test_cpu",
            handle_version=1,
            handle=index.to_bytes(4),
        )

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int]:
        return self.storages[int.from_bytes(spec.handle)], 3


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


def _make_worker(exists_result: list[int], tp_size: int = 1, rank: int = 0) -> MPKVPoolWorker:
    store = MagicMock()
    store.exists.return_value = exists_result
    return MPKVPoolWorker(_make_vllm_config(tp_size, rank), store=store, rank=rank)


def test_mp_worker_reuses_original_lookup_implementation() -> None:
    assert MPKVPoolWorker.lookup_scheduler is KVPoolWorker.lookup_scheduler


def test_mp_worker_uses_registered_rank() -> None:
    worker = _make_worker([1, 1, 1, 1], tp_size=2, rank=1)

    assert worker.tp_rank == 1
    assert worker.pp_rank == 0


def test_mp_worker_initializes_parent_cpu_state() -> None:
    worker = _make_worker([1, 1])

    assert worker.kv_send_thread is None
    assert worker.kv_recv_thread is None
    assert worker.physical_layer_to_group_layers == {}


def test_mp_worker_maps_cache_once_and_releases_it_on_close() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.arange(8)}, 1, adapter)
    importer = MagicMock(side_effect=lambda spec: import_worker_kv_caches(spec, adapter))
    worker = MPKVPoolWorker(_make_vllm_config(), store=MagicMock(), cache_importer=importer)

    worker.configure_kv_caches(exported.spec)
    worker.configure_kv_caches(exported.spec)

    assert worker.kv_cache_spec == exported.spec
    assert torch.equal(worker.kv_caches["layer.0"][0], torch.arange(8))
    assert worker.local_rank == 3
    importer.assert_called_once_with(exported.spec)

    worker.close()

    assert worker.kv_cache_spec is None
    assert worker.kv_caches == {}


def test_mp_worker_replaces_newer_generation_and_ignores_stale_replay() -> None:
    adapter = _CPUMemoryAdapter()
    first = export_worker_kv_caches({"layer.0": torch.zeros(8)}, 1, adapter)
    second = export_worker_kv_caches({"layer.0": torch.ones(8)}, 2, adapter)
    worker = MPKVPoolWorker(
        _make_vllm_config(),
        store=MagicMock(),
        cache_importer=lambda spec: import_worker_kv_caches(spec, adapter),
    )

    worker.configure_kv_caches(first.spec)
    first_mapping = worker.kv_caches
    worker.configure_kv_caches(second.spec)
    worker.configure_kv_caches(first.spec)

    assert worker.kv_cache_spec == second.spec
    assert torch.equal(worker.kv_caches["layer.0"][0], torch.ones(8))
    assert first_mapping == {}


def test_mp_worker_rejects_conflicting_spec_for_same_generation() -> None:
    adapter = _CPUMemoryAdapter()
    first = export_worker_kv_caches({"layer.0": torch.zeros(8)}, 1, adapter)
    conflicting = export_worker_kv_caches({"layer.0": torch.ones(8)}, 1, adapter)
    worker = MPKVPoolWorker(
        _make_vllm_config(),
        store=MagicMock(),
        cache_importer=lambda spec: import_worker_kv_caches(spec, adapter),
    )

    worker.configure_kv_caches(first.spec)

    with pytest.raises(RuntimeError, match="conflicting specifications"):
        worker.configure_kv_caches(conflicting.spec)


def test_mp_worker_keeps_current_mapping_when_new_import_fails() -> None:
    adapter = _CPUMemoryAdapter()
    first = export_worker_kv_caches({"layer.0": torch.zeros(8)}, 1, adapter)
    second = export_worker_kv_caches({"layer.0": torch.ones(8)}, 2, adapter)

    def import_cache(spec):
        if spec.generation == 2:
            raise RuntimeError("import failed")
        return import_worker_kv_caches(spec, adapter)

    worker = MPKVPoolWorker(_make_vllm_config(), store=MagicMock(), cache_importer=import_cache)
    worker.configure_kv_caches(first.spec)
    current_mapping = worker.kv_caches

    with pytest.raises(RuntimeError, match="import failed"):
        worker.configure_kv_caches(second.spec)

    assert worker.kv_cache_spec == first.spec
    assert worker.kv_caches is current_mapping
    assert torch.equal(worker.kv_caches["layer.0"][0], torch.zeros(8))


@pytest.mark.parametrize(
    ("exists_result", "expected"),
    [
        ([1, 1], 32),
        ([1, 0], 16),
        ([0, 1], 0),
    ],
)
def test_mp_worker_single_tp(exists_result: list[int], expected: int) -> None:
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
def test_mp_worker_requires_all_tp_ranks(exists_result: list[int], expected: int) -> None:
    worker = _make_worker(exists_result, tp_size=2)
    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == expected


def test_mp_worker_returns_miss_when_store_fails() -> None:
    store = MagicMock()
    store.exists.side_effect = RuntimeError("store unavailable")
    worker = MPKVPoolWorker(_make_vllm_config(), store=store)

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0


def test_mp_worker_returns_miss_before_backend_is_initialized() -> None:
    worker = MPKVPoolWorker(_make_vllm_config())

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0


def test_mp_worker_initializes_own_backend_after_cache_mapping() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.arange(8)}, 1, adapter)
    store = MagicMock()
    store.exists.return_value = [1, 1]
    backend_factory = MagicMock(return_value=store)
    config = _make_vllm_config()
    worker = MPKVPoolWorker(
        config,
        cache_importer=lambda spec: import_worker_kv_caches(spec, adapter),
        backend_factory=backend_factory,
    )

    worker.configure_kv_caches(exported.spec)

    assert worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False) == 32
    backend_factory.assert_called_once_with(config.parallel_config, 3, False)
