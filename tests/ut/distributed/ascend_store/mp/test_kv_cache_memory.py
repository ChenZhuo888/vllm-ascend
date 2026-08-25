from dataclasses import replace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
    export_worker_kv_caches,
    import_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.request_view import KVCacheStorageSpec


class _CPUMemoryAdapter:
    def __init__(self):
        self.storages: list[torch.Tensor] = []
        self.import_count = 0

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

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int | None]:
        self.import_count += 1
        index = int.from_bytes(spec.handle)
        return self.storages[index], None


def test_shared_storage_is_exported_once_and_views_are_rebuilt() -> None:
    adapter = _CPUMemoryAdapter()
    storage = torch.arange(32, dtype=torch.float32).view(4, 8)
    expected_slice = storage[1:, ::2]

    exported = export_worker_kv_caches(
        {"layer.0": storage, "layer.1": expected_slice},
        generation=1,
        adapter=adapter,
    )
    imported = import_worker_kv_caches(exported.spec, adapter)

    assert len(exported.spec.storages) == 1
    assert adapter.import_count == 1
    assert torch.equal(imported.tensors["layer.0"][0], storage)
    assert torch.equal(imported.tensors["layer.1"][0], expected_slice)
    assert imported.tensors["layer.0"][0].untyped_storage().data_ptr() == (
        imported.tensors["layer.1"][0].untyped_storage().data_ptr()
    )


def test_invalid_tensor_layout_is_rejected_before_import() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.zeros(8)}, 1, adapter)
    tensor = replace(exported.spec.caches["layer.0"][0], storage_index=1)
    invalid_spec = replace(exported.spec, caches={"layer.0": (tensor,)})

    with pytest.raises(ValueError, match="unknown storage"):
        import_worker_kv_caches(invalid_spec, adapter)

    assert adapter.import_count == 0


def test_exported_and_imported_cache_release_owned_references() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.zeros(8)}, 1, adapter)
    imported = import_worker_kv_caches(exported.spec, adapter)

    imported.close()
    exported.close()

    assert imported.tensors == {}
    assert imported._storages == ()
    assert exported._storages == ()
