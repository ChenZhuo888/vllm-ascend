import os
import sys
from unittest.mock import MagicMock, patch

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeStoreConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mooncake_backend import MPMooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import create_transfer_backend


def test_mooncake_ssd_setup_uses_worker_identity_without_distributed_groups(tmp_path):
    backend = MPMooncakeBackend.__new__(MPMooncakeBackend)
    backend._device_index = 2
    backend._global_rank = 7
    backend.parallel_config = None
    backend.config = MooncakeStoreConfig(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1 << 30,
        local_buffer_size=1 << 30,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        preferred_segment=False,
        prefer_alloc_in_same_node=True,
        enable_ssd_offload=True,
        ssd_offload_path=str(tmp_path),
    )
    backend.local_seg = None
    backend._use_fabric_mem = True
    backend._use_store_independent_te = False
    backend._contribute_memory = True
    store = MagicMock()
    store.setup.return_value = 0
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend"
    with (
        patch.object(sys.modules["mooncake.store"], "MooncakeDistributedStore", return_value=store, create=True),
        patch(f"{module}.get_ip", return_value="127.0.0.1"),
        patch(f"{module}.get_global_rank", side_effect=AssertionError("distributed rank lookup")),
        patch(f"{module}._mooncake_setup_supports_ssd_offload", return_value=True),
    ):
        backend._setup_store()

    assert store.setup.call_args.kwargs["ssd_offload_path"] == os.path.join(str(tmp_path), "rank_7")
    with patch.object(torch.npu, "set_device") as set_device:
        backend.set_device()
    set_device.assert_called_once_with(2)


def test_transfer_backend_builds_mooncake_process_backend():
    target = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mooncake_backend.MPMooncakeBackend"
    with patch(target) as factory:
        backend = create_transfer_backend("mooncake", device_index=2, global_rank=7, lazy_init=True)

    factory.assert_called_once_with(2, 7, lazy_init=True)
    assert backend.backend is factory.return_value
