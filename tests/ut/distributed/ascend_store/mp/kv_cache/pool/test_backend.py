from unittest.mock import MagicMock, patch

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.backend.memcache import (
    MPMemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.backend.mooncake import (
    MPMooncakeBackend,
    global_te,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.backend.yuanrong import (
    MPYuanrongBackend,
)

# isort: on


def test_mp_memcache_backend_unregisters_owned_buffers_before_close() -> None:
    store = MagicMock()
    store.register_buffer.return_value = 0
    store.unregister_buffer.return_value = 0
    with patch.object(MemcacheBackend, "_setup_store", return_value=store):
        backend = MPMemcacheBackend(MagicMock(), local_rank=2)

    backend.register_buffer([10, 20], [4, 8])
    backend.close()

    assert store.register_buffer.call_args_list == [((10, 4),), ((20, 8),)]
    assert store.unregister_buffer.call_args_list == [((20, 8),), ((10, 4),)]
    store.close.assert_called_once_with()


def test_mp_memcache_backend_retries_only_failed_buffer_unregistration() -> None:
    store = MagicMock()
    store.register_buffer.return_value = 0
    store.unregister_buffer.side_effect = [0, -1, 0]
    with patch.object(MemcacheBackend, "_setup_store", return_value=store):
        backend = MPMemcacheBackend(MagicMock(), local_rank=2)

    backend.register_buffer([10, 20], [4, 8])
    with pytest.raises(RuntimeError, match="unregistration failed"):
        backend.unregister_buffer()
    backend.unregister_buffer()

    assert store.unregister_buffer.call_args_list == [((20, 8),), ((10, 4),), ((10, 4),)]


def test_mp_mooncake_backend_registers_and_unregisters_each_worker_region() -> None:
    transfer_engine = MagicMock()
    transfer_engine.register_memory.return_value = 0
    transfer_engine.unregister_memory.return_value = 0
    global_te.get_transfer_engine.return_value = transfer_engine
    backend = object.__new__(MPMooncakeBackend)
    backend.local_rank = 1
    backend._use_fabric_mem = False
    backend._mp_registered_ptrs = []
    backend.store = MagicMock()

    backend.register_buffer([10, 20], [4, 8])
    backend.unregister_buffer()

    assert transfer_engine.register_memory.call_args_list == [((10, 4),), ((20, 8),)]
    assert transfer_engine.unregister_memory.call_args_list == [((20,),), ((10,),)]


def test_mp_mooncake_backend_retries_only_failed_memory_unregistration() -> None:
    transfer_engine = MagicMock()
    transfer_engine.register_memory.return_value = 0
    transfer_engine.unregister_memory.side_effect = [0, -1, 0]
    global_te.get_transfer_engine.return_value = transfer_engine
    backend = object.__new__(MPMooncakeBackend)
    backend.local_rank = 1
    backend._use_fabric_mem = False
    backend._mp_registered_ptrs = []
    backend.store = MagicMock()

    backend.register_buffer([10, 20], [4, 8])
    with pytest.raises(RuntimeError, match="unregistration failed"):
        backend.unregister_buffer()
    backend.unregister_buffer()

    assert transfer_engine.unregister_memory.call_args_list == [((20,),), ((10,),), ((10,),)]


def test_mp_yuanrong_backend_rejects_unreleasable_device_registration() -> None:
    backend = object.__new__(MPYuanrongBackend)
    backend._needs_dev_mem_pregister = True

    with pytest.raises(NotImplementedError, match="cannot be safely released"):
        backend.register_buffer([10], [4])
