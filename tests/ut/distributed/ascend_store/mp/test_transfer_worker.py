import ctypes
import threading
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.mp.test_transfer_npu_ipc import _CPUMemoryAdapter
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import npu_ipc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.adapter import (
    KVCacheStoreKeyLayerSendingProcessAdapter,
    KVCacheStoreLayerRecvingProcessAdapter,
    KVCacheStoreRecvingProcessAdapter,
    KVCacheStoreSendingProcessAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import TransferService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import KVTransferProcess
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import TransferBackend


class MemoryBackend:
    requires_exists_before_put = True

    def __init__(self):
        self.values = {}
        self.writes = []

    def set_device(self):
        pass

    def register_buffer(self, pointers, lengths):
        self.registrations = list(zip(pointers, lengths))

    def exists(self, keys):
        return [int(key in self.values) for key in keys]

    def put(self, keys, addresses, sizes):
        for key, row, row_sizes in zip(keys, addresses, sizes):
            value = [ctypes.string_at(address, size) for address, size in zip(row, row_sizes)]
            self.values[key] = value
            self.writes.append((key, value))

    def get(self, keys, addresses, sizes):
        result = []
        for key, row, row_sizes in zip(keys, addresses, sizes):
            if key not in self.values:
                result.append(-1)
                continue
            for address, size, data in zip(row, row_sizes, self.values[key]):
                assert len(data) == size
                ctypes.memmove(address, data, size)
            result.append(0)
        return result

    def close(self):
        pass


class GVAMemoryBackend(MemoryBackend):
    def __init__(self):
        super().__init__()
        self.store = self
        self.copies = []
        self.finishes = []

    def batch_copy(self, gvas, addresses, sizes, direction):
        for gva, address, size in zip(gvas, addresses, sizes):
            source, target = (address, gva) if direction == 0 else (gva, address)
            ctypes.memmove(target, source, size)
        self.copies.append((gvas, addresses, sizes, direction))
        return 0

    def batch_write_finish(self, keys, results):
        self.finishes.append((keys, results))
        return [0] * len(keys)

    def batch_remove_lease(self, keys):
        return 0


def database():
    db = ChunkedTokenDatabase([KeyMetadata("test", 1, 2, 1, 3)], [2], None, hash_block_size=2)
    db.set_group_buffers({0: [100]}, {0: [2]}, {0: [2]})
    return db


def request(req_id="request"):
    return ReqMeta(req_id, 4, [[1, 3]], [b"a" * 32, b"b" * 32], can_save=True)


class EventImportRecorder:
    """Child-side NPU event import spy: one imported object per IPC handle."""

    def __init__(self):
        self.calls = 0
        self.by_handle: dict[bytes, MagicMock] = {}

    def __call__(self, spec):
        self.calls += 1
        if spec.handle not in self.by_handle:
            self.by_handle[spec.handle] = MagicMock()
        return self.by_handle[spec.handle]


def registered_runtime(monkeypatch, config, backend, worker, caches, pointers, lengths):
    adapter = _CPUMemoryAdapter()
    export = npu_ipc.export_worker_kv_caches
    import_cache = npu_ipc.import_worker_kv_caches
    monkeypatch.setattr(npu_ipc, "export_worker_kv_caches", lambda values: export(values, adapter))
    monkeypatch.setattr(npu_ipc, "import_worker_kv_caches", lambda spec: import_cache(spec, adapter))
    event_imports = EventImportRecorder()
    monkeypatch.setattr(npu_ipc, "import_npu_event", event_imports)
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend.create_transfer_backend",
        return_value=backend,
    ):
        runtime = TransferService(config)
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        parent = KVTransferProcess(config)
    parent.register_kv_caches(worker, caches, pointers, lengths)
    payload = parent.client.call.call_args.args[1]
    runtime.execute("register", msgspec.msgpack.decode(msgspec.msgpack.encode(payload)))
    runtime.event_imports = event_imports
    return runtime, parent


def run_transfer(runtime, parent, operation, req):
    parent.submit_request(operation, req)
    payload = parent.client.submit.call_args.args[1]
    return runtime.submit(operation, msgspec.msgpack.decode(msgspec.msgpack.encode(payload))).result(2)


def layerwise_worker(db, *, use_gva, num_layers=1):
    group_ids = sorted(db.group_kv_caches_base_addr)
    save_events = []
    for layer in range(num_layers):
        event = MagicMock()
        event.ipc_handle.return_value = f"layer-{layer}-event".encode()
        save_events.append(event)
    return SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len=db.group_block_len,
        group_block_stride=db.group_block_stride,
        group_kv_cache_families={group_id: "default" for group_id in group_ids},
        group_num_layers={group_id: num_layers for group_id in group_ids},
        group_layer_cache_entry_offsets={
            group_id: [0, len(db.group_kv_caches_base_addr[group_id])] for group_id in group_ids
        },
        group_uses_align_state=[False] * len(group_ids),
        use_layerwise=True,
        use_layerwise_transfer=use_gva,
        block_size=db.block_size[0],
        num_layers=num_layers,
        sync_save_events=save_events,
        page_size_bytes=sum(db.group_block_len[0]),
        consumer_is_to_put=False,
        h2d_stagger_us=0,
        layerwise_max_transfer_blocks=0,
        layerwise_max_transfer_bytes=0,
        tp_mismatch=False,
    )


def run_layer_transfer(runtime, parent, operation, tasks):
    parent.submit_layer_request(operation, tasks, tasks[0].layer_id)
    payload = parent.client.submit.call_args.args[1]
    return runtime.submit(operation, msgspec.msgpack.decode(msgspec.msgpack.encode(payload))).result(2)


def test_child_handlers_match_thread_keys_and_roundtrip_buffer_contents(monkeypatch):
    caches = {"layer.0": torch.arange(8, dtype=torch.uint8).view(4, 2)}
    tensor = caches["layer.0"]
    db = database()
    db.set_group_buffers({0: [tensor.data_ptr()]}, {0: [2]}, {0: [2]})
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len={0: [2]},
        group_block_stride={0: [2]},
        group_kv_cache_families={0: "default"},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0]},
        group_uses_align_state=[False],
    )
    config = dict(
        device_index=None,
        global_rank=0,
        tp_rank=1,
        tp_size=2,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=True,
        lazy_init=False,
    )
    backend = MemoryBackend()
    config = {"backend": "mooncake", **config}
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, caches, [tensor.data_ptr()], [tensor.nbytes]
    )
    try:
        original_backend = MemoryBackend()
        original = KVCacheStoreSendingThread(original_backend, db, [2], 1, 2, 1, enable_kv_event=True)
        req = request()
        req.token_ids = [1, 2, 3, 4]
        req.original_block_size = 2
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes
        assert len(backend.writes) == 2
        assert "@pcp:2@dcp:1@head_or_tp_rank:1@pp_rank:3@" in backend.writes[0][0]
        assert result["events"] == original.get_kv_events()

        req.block_ids_by_group = [[0, 2]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert tensor[0].tolist() == tensor[1].tolist()
        assert tensor[2].tolist() == tensor[3].tolist()

        backend.values.clear()
        result = run_transfer(runtime, parent, "load", req)
        assert sorted(result["invalid_blocks"]) == [0, 2]
        with pytest.raises(ValueError, match="exceeds"):
            runtime.execute("get_ranges", (["key"], [[(0, 7, 2)]]))
    finally:
        runtime.close()
        parent.close()


def test_child_tp_mismatch_handler_reuses_worker_business_logic(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    tensor = torch.arange(16, dtype=torch.uint8).view(4, 2, 2)
    db = ChunkedTokenDatabase([KeyMetadata("test", 0, 0, 0, 0)], [2], None, hash_block_size=2)
    db.set_group_buffers({0: [tensor.data_ptr()]}, {0: [4]}, {0: [4]})
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len={0: [4]},
        group_block_stride={0: [4]},
        group_kv_cache_families={0: "default"},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0]},
        group_uses_align_state=[False],
        tp_mismatch=True,
        block_size=2,
        num_sub_keys=2,
        sub_size_bytes=1,
    )
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=2,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    try:
        original_backend = MemoryBackend()
        original_worker = KVPoolWorker.__new__(KVPoolWorker)
        original_worker.tp_mismatch = True
        original_worker.m_store = original_backend
        original_worker.token_database = db
        original_worker.group_kv_caches_base_addr = db.group_kv_caches_base_addr
        original_worker.group_block_len = worker.group_block_len
        original_worker.group_block_stride = worker.group_block_stride
        original_worker.block_size = worker.block_size
        original_worker.num_sub_keys = worker.num_sub_keys
        original_worker.sub_size_bytes = worker.sub_size_bytes
        original_worker.tp_rank = config["tp_rank"]
        original_worker.enable_kv_events = False
        original = KVCacheStoreSendingThread(original_backend, db, [2], 0, 2, 1, worker=original_worker)
        original_worker.kv_send_thread = original
        req = request()
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes

        req.block_ids_by_group = [[0, 2]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert tensor[0].tolist() == tensor[1].tolist()
        assert tensor[2].tolist() == tensor[3].tolist()
    finally:
        runtime.close()
        parent.close()


def test_child_handlers_preserve_hybrid_group_keys_and_buffers(monkeypatch):
    kv = torch.arange(8, dtype=torch.uint8).view(4, 2)
    state = torch.arange(16, 24, dtype=torch.uint8).view(4, 2)
    db = ChunkedTokenDatabase(
        [KeyMetadata("test", 0, 0, 0, 0, 0), KeyMetadata("test", 0, 0, 0, 0, 1)],
        [2, 2],
        None,
        hash_block_size=2,
    )
    group_addresses = {0: [kv.data_ptr()], 1: [state.data_ptr()]}
    group_lengths = {0: [2], 1: [2]}
    group_strides = {0: [2], 1: [2]}
    families = {0: "default", 1: "state"}
    db.set_group_buffers(
        group_addresses,
        group_lengths,
        group_strides,
        group_cache_families=families,
        group_num_layers={0: 1, 1: 1},
    )
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=group_addresses,
        group_block_len=group_lengths,
        group_block_stride=group_strides,
        group_kv_cache_families=families,
        group_num_layers={0: 1, 1: 1},
        group_layer_cache_entry_offsets={0: [0], 1: [0]},
        group_uses_align_state=[False, True],
        tp_mismatch=False,
    )
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch,
        config,
        backend,
        worker,
        {"layer.0.kv": kv, "layer.0.state": state},
        [kv.data_ptr(), state.data_ptr()],
        [kv.nbytes, state.nbytes],
    )
    try:
        original_backend = MemoryBackend()
        original = KVCacheStoreSendingThread(original_backend, db, [2, 2], 0, group_uses_align_state=[False, True])
        req = ReqMeta(
            "hybrid",
            4,
            [[1, 3], [0, 2]],
            [b"a" * 32, b"b" * 32],
            can_save=True,
            kv_cache_group_ids=[0, 1],
            skip_null_blocks_by_group=[False, True],
        )
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes
        assert any("@group:0@" in key for key, _ in backend.writes)
        assert any("@group:1@" in key for key, _ in backend.writes)

        req.block_ids_by_group = [[0, 2], [1, 3]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert kv[0].tolist() == kv[1].tolist() and kv[2].tolist() == kv[3].tolist()
        assert state[1].tolist() == [18, 19] and state[3].tolist() == state[2].tolist()
    finally:
        runtime.close()
        parent.close()


def test_child_key_layer_handlers_roundtrip_buffer_contents(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8).view(4, 2)
    db = database()
    db.set_group_buffers(
        {0: [tensor.data_ptr()]},
        {0: [2]},
        {0: [2]},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=False)
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    req = request()
    req.is_last_chunk = True
    task = LayerTransferTask(0, [LayerBlockRange(req, 0, 2)])
    expected = torch.zeros_like(tensor)
    expected[[1, 3]] = tensor[[1, 3]]
    try:
        result = run_layer_transfer(runtime, parent, "store", [task])
        assert result["finished_req_ids"] == [req.req_id]
        assert len(backend.writes) == 2
        imported_event = runtime.event_imports.by_handle[b"layer-0-event"]
        imported_event.synchronize.assert_called_once_with()

        tensor.zero_()
        result = run_layer_transfer(runtime, parent, "load", [task])
        assert result["finished_req_ids"] == [req.req_id]
        assert tensor.tolist() == expected.tolist()
    finally:
        runtime.close()
        parent.close()


def test_child_gva_layer_handlers_roundtrip_multiple_cache_groups(monkeypatch):
    caches = {
        "layer.0.kv": torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
        "layer.0.state": torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8),
    }
    remotes = [torch.zeros_like(cache) for cache in caches.values()]
    db = ChunkedTokenDatabase(
        [KeyMetadata("test", 0, 0, 0, 0, 0), KeyMetadata("test", 0, 0, 0, 0, 1)],
        [2, 2],
        None,
        hash_block_size=2,
    )
    addresses = {group_id: [cache.data_ptr()] for group_id, cache in enumerate(caches.values())}
    db.set_group_buffers(
        addresses,
        {0: [2], 1: [2]},
        {0: [2], 1: [2]},
        group_num_layers={0: 1, 1: 1},
        group_layer_cache_entry_offsets={0: [0, 1], 1: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=True)
    config = dict(
        backend="memcache",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = GVAMemoryBackend()
    pointers = [cache.data_ptr() for cache in caches.values()]
    lengths = [cache.nbytes for cache in caches.values()]
    runtime, parent = registered_runtime(monkeypatch, config, backend, worker, caches, pointers, lengths)
    tasks = []
    for group_id, remote in enumerate(remotes):
        shared = SharedBlockData(
            block_ids_arr=torch.tensor([0, 1]).numpy(),
            block_gvas_arr=torch.tensor([remote.data_ptr(), remote.data_ptr() + 2]).numpy(),
            req_ids=[f"request-{group_id}"],
            is_last_chunks=[True],
            save_keys=[f"key-{group_id}"],
            load_keys=[f"key-{group_id}"],
        )
        tasks.append(
            LayerTransferTask(
                0,
                [],
                shared_block_data=shared,
                group_id=group_id,
                write_finish_keys=["key-0", "key-1"] if group_id == 1 else [],
            )
        )
    try:
        result = run_layer_transfer(runtime, parent, "store", tasks)
        assert set(result["finished_req_ids"]) == {"request-0", "request-1"}
        assert [remote.tolist() for remote in remotes] == [cache.tolist() for cache in caches.values()]
        assert backend.finishes == [(["key-0", "key-1"], [0, 0])]
        imported_event = runtime.event_imports.by_handle[b"layer-0-event"]
        imported_event.synchronize.assert_called_once_with()

        for cache in caches.values():
            cache.zero_()
        result = run_layer_transfer(runtime, parent, "load", tasks)
        assert set(result["finished_req_ids"]) == {"request-0", "request-1"}
        assert [cache.tolist() for cache in caches.values()] == [remote.tolist() for remote in remotes]
        assert [len(copy[0]) for copy in backend.copies] == [4, 4]
        assert [copy[-1] for copy in backend.copies] == [0, 1]
    finally:
        runtime.close()
        parent.close()


def two_layer_key_runtime(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8).view(4, 2)
    db = database()
    db.set_group_buffers(
        {0: [tensor.data_ptr()]},
        {0: [2]},
        {0: [2]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=False, num_layers=2)
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    return runtime, parent, backend


def test_layerwise_events_are_imported_once_per_layer_at_registration(monkeypatch):
    runtime, parent, _ = two_layer_key_runtime(monkeypatch)
    try:
        assert runtime.event_imports.calls == 2
        assert set(runtime.event_imports.by_handle) == {b"layer-0-event", b"layer-1-event"}
        assert runtime.sender.sync_save_events == [
            runtime.event_imports.by_handle[b"layer-0-event"],
            runtime.event_imports.by_handle[b"layer-1-event"],
        ]
    finally:
        runtime.close()
        parent.close()


def test_repeated_layer_stores_reuse_the_imported_event(monkeypatch):
    runtime, parent, backend = two_layer_key_runtime(monkeypatch)
    imported_event = runtime.event_imports.by_handle[b"layer-1-event"]
    req = request()
    req.is_last_chunk = True
    task = LayerTransferTask(1, [LayerBlockRange(req, 0, 2)])
    try:
        for round_id in range(2):
            if round_id > 0:
                # A hot request stores its new suffix blocks, which the pool
                # has not seen yet.
                backend.values.clear()
            result = run_layer_transfer(runtime, parent, "store", [task])
            assert result["finished_req_ids"] == [req.req_id]
            # The layer request carries only the layer identity, never an event.
            assert "current_event" not in parent.client.submit.call_args.args[1]

        assert imported_event.synchronize.call_count == 2
        # Stores must not re-import handles: the count stays at the two
        # registration imports even after a second (hot) store round.
        assert runtime.event_imports.calls == 2
        assert runtime.sender.sync_save_events[1] is imported_event
    finally:
        runtime.close()
        parent.close()


def process_endpoint(cls):
    process = MagicMock()
    process.client.timeout = 1
    process.submit_request.side_effect = lambda *args: Future()
    endpoint = cls(MagicMock(), database(), [2], 0, process=process)
    return process, endpoint


def completed(future, *, invalid_blocks=()):
    future.set_result({"finished": True, "events": [], "invalid_blocks": list(invalid_blocks)})


def test_sending_completion_counts_all_submissions_and_ignores_preempted_generation():
    process, sender = process_endpoint(KVCacheStoreSendingProcessAdapter)
    futures: list[Future] = [Future(), Future(), Future()]
    process.submit_request.side_effect = futures
    req = request()
    for _ in range(2):
        sender.add_stored_request(req.req_id)
        sender.add_request(req)
    completed(futures[0])
    assert sender.get_and_clear_finished_requests() == set()
    sender.delete_finished_stored_request(req.req_id)
    sender.discard_finished_requests({req.req_id})
    sender.add_stored_request(req.req_id)
    sender.add_request(req)
    completed(futures[1])
    assert sender.get_stored_request_count(req.req_id) == 1
    completed(futures[2])
    sender.wait_for_pending()
    assert sender.get_and_clear_finished_requests() == {req.req_id}
    assert not sender._generations


def test_preempted_load_does_not_publish_stale_failure_or_completion():
    process, receiver = process_endpoint(KVCacheStoreRecvingProcessAdapter)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    receiver.add_request(request())
    receiver.discard_finished_requests({"request"})
    completed(future, invalid_blocks=[1])
    assert receiver.get_and_clear_finished_requests() == set()
    assert not receiver._invalid_block_ids


def test_failed_async_transfer_is_raised_by_waiter():
    process, sender = process_endpoint(KVCacheStoreSendingProcessAdapter)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    sender.add_request(request())
    future.set_exception(RuntimeError("child died"))
    with pytest.raises(RuntimeError, match="asynchronous transfer"):
        sender.wait_for_pending()


def test_registration_rollback_retains_only_failed_unregistrations():
    backend = MagicMock()
    backend.store.register_buffer.side_effect = [0, 0, -1]
    backend.store.unregister_buffer.side_effect = [-2, 0, 0]
    adapter = TransferBackend("memcache", backend, 0)
    with pytest.raises(RuntimeError, match="unregistration failed"):
        adapter.register_buffer([100, 200, 300], [10, 10, 10])
    assert adapter._registered == [(200, 10)]
    adapter.close()
    assert not adapter._registered
    assert backend.store.unregister_buffer.call_args_list[-1].args == (200, 10)


def test_backend_exists_before_put_capability_is_forwarded():
    backend = MagicMock(requires_exists_before_put=False)
    assert TransferBackend("yuanrong", backend, 0).requires_exists_before_put is False


def test_lazy_memcache_registration_keeps_backend_initialization_semantics():
    backend = MagicMock()
    backend._lazy_init = True
    backend._store_initialized = False
    backend.store = None
    adapter = TransferBackend("memcache", backend, 0)

    adapter.register_buffer([100], [10])

    backend.register_buffer.assert_called_once_with([100], [10])
    assert adapter._registered == [(100, 10)]

    backend._store_initialized = True
    backend.store = MagicMock()
    backend.store.unregister_buffer.return_value = 0
    adapter.close()
    backend.store.unregister_buffer.assert_called_once_with(100, 10)


def test_yuanrong_registration_uses_existing_backend_path():
    backend = MagicMock()
    adapter = TransferBackend("yuanrong", backend, 0)

    adapter.register_buffer([100], [10])

    backend.register_buffer.assert_called_once_with([100], [10])
    adapter.close()


@pytest.mark.parametrize(
    "use_hybrid,use_compress,tp_mismatch", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_worker_selects_process_without_changing_ordinary_transfer_modes(use_hybrid, use_compress, tp_mismatch):
    case = unittest.TestCase()
    try:
        worker = make_worker(case)
        # The existing fixture's hf_config is a MagicMock; provide the actual
        # ordinary-model capability before exercising backend selection.
        worker.use_hybrid = use_hybrid
        worker.use_compress = use_compress
        worker.tp_mismatch = tp_mismatch
        worker.use_multiprocess = True
        parallel_config = SimpleNamespace(
            rank=5,
            data_parallel_index=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            prefill_context_parallel_size=1,
        )
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess") as factory:
            worker._init_backend(parallel_config, worker._extra_config)
            assert worker.m_store is factory.return_value
            assert factory.call_args.args[0]["tp_rank"] == worker.tp_rank
            assert factory.call_args.args[0]["kv_role"] == "kv_producer"
            assert factory.call_args.args[0]["lazy_init"] is use_compress
            assert factory.call_args.args[0]["global_rank"] == 9
    finally:
        case.doCleanups()


def test_mooncake_uses_transfer_process_with_worker_rank():
    case = unittest.TestCase()
    try:
        worker = make_worker(case)
        worker.use_compress = False
        worker.use_multiprocess = True
        parallel_config = SimpleNamespace(
            rank=3,
            data_parallel_index=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=2,
        )
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess") as factory:
            worker._init_backend(parallel_config, worker._extra_config)
        assert worker.transfer_process is factory.return_value
        assert factory.call_args.args[0]["global_rank"] == 7
    finally:
        case.doCleanups()


def test_worker_selects_process_for_layerwise_mode():
    case = unittest.TestCase()
    try:
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess") as factory:
            worker = make_worker(case, use_layerwise=True, extra_config={"use_multiprocess": True})
        assert worker.transfer_process is factory.return_value
        assert worker.m_store is factory.return_value
    finally:
        case.doCleanups()


@pytest.mark.parametrize(
    "use_gva,sender_name,receiver_name",
    [
        (False, "KVCacheStoreKeyLayerSendingProcessAdapter", "KVCacheStoreKeyLayerRecvingProcessAdapter"),
        (True, "KVCacheStoreLayerSendingProcessAdapter", "KVCacheStoreLayerRecvingProcessAdapter"),
    ],
)
def test_layerwise_worker_selects_matching_process_adapters(use_gva, sender_name, receiver_name):
    case = unittest.TestCase()
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
    try:
        worker = make_worker(case, kv_role="kv_both", use_layerwise=True)
        worker.transfer_process = MagicMock()
        worker.m_store = worker.transfer_process
        worker.use_layerwise_transfer = use_gva
        worker._transfer_threads_started = False
        worker.page_size_bytes = 2
        with (
            patch(f"{module}.{sender_name}") as sender,
            patch(f"{module}.{receiver_name}") as receiver,
            patch.object(torch.npu, "Event"),
            patch.object(worker, "_build_group_layer_builders", return_value=[MagicMock()]),
        ):
            worker._start_kv_transfer_threads()
        assert worker.kv_send_thread is sender.return_value
        assert worker.kv_recv_thread is receiver.return_value
        worker.transfer_process.bind_adapters.assert_called_once_with((sender.return_value, receiver.return_value))
        sender.return_value.start.assert_not_called()
        receiver.return_value.start.assert_not_called()
    finally:
        case.doCleanups()


def test_layer_receive_process_preserves_parent_ordering():
    order = []

    def record(item, result):
        order.append(item)
        return result

    process = MagicMock()
    process.client.timeout = 1
    process.client.wait.side_effect = lambda _future: record(
        "wait_child", {"completed_req_ids": ["request"], "finished_req_ids": [], "events": []}
    )
    process.submit_layer_request.side_effect = lambda *_args: record("submit_child", Future())
    save_finished = MagicMock()
    save_finished.wait.side_effect = lambda timeout: record("wait_save", True)
    save_finished.clear.side_effect = lambda: order.append("clear_save")
    sync_event = MagicMock()
    sync_event.synchronize.side_effect = lambda: order.append("sync_save")
    gate = MagicMock()
    gate.wait.side_effect = lambda timeout: record("wait_attention", True)
    receiver = KVCacheStoreLayerRecvingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        page_size_bytes=2,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=[threading.Event()],
        layer_save_finished_events=[save_finished],
        sync_save_events=[sync_event],
        num_layers=1,
        group_builders=[MagicMock()],
        external_slot_release_waiter=lambda _layer_id: order.append("release_slot"),
        process=process,
    )
    receiver._stagger_h2d_submit = lambda _layer_id: order.append("stagger_load")
    task = LayerTransferTask(0, [LayerBlockRange(request(), 0, 1)])
    data = LayerLoadTask(0, [task], 0, gate)
    try:
        receiver._coordinate_load(data)
    finally:
        receiver.close()
    assert order == [
        "wait_save",
        "sync_save",
        "clear_save",
        "wait_attention",
        "stagger_load",
        "release_slot",
        "submit_child",
        "wait_child",
    ]


def test_key_layer_sending_process_completes_parent_request_state():
    process = MagicMock()
    process.client.timeout = 1
    future: Future[dict[str, object]] = Future()
    process.submit_layer_request.return_value = future
    save_finished = threading.Event()
    sender = KVCacheStoreKeyLayerSendingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=1,
        layer_save_finished_events=[save_finished],
        sync_save_events=[MagicMock()],
        process=process,
    )
    task = LayerTransferTask(0, [LayerBlockRange(request(), 0, 1)])
    tasks = [task]
    sender.add_stored_request("request")
    sender.add_request(tasks)
    future.set_result(
        {
            "completed_req_ids": ["request"],
            "finished_req_ids": ["request"],
            "events": [],
        }
    )
    sender.wait_for_pending()
    assert sender.get_and_clear_finished_requests() == {"request"}
    assert "request" not in sender.stored_requests
    assert save_finished.is_set()
    assert tasks == []


def test_shutdown_accepts_scheduler_role_without_a_worker():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector

    scheduler = AscendStoreConnector.__new__(AscendStoreConnector)
    scheduler.shutdown()
    worker = AscendStoreConnector.__new__(AscendStoreConnector)
    worker.connector_worker = MagicMock()
    worker.shutdown()
    worker.connector_worker.close.assert_called_once()


@pytest.mark.parametrize("failed", [False, True])
def test_source_event_is_retained_until_completion_or_child_reaping(failed):
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        parent = KVTransferProcess({})
    parent.cache = MagicMock()
    parent._device_uuid = "device-0"
    req = request()
    req.current_event = MagicMock()
    req.current_event.ipc_handle.return_value = b"source-event"
    future: Future = Future()
    parent.client.submit.return_value = future
    parent.submit_request("store", req)
    assert parent._events[future] is req.current_event
    payload = parent.client.submit.call_args.args[1]
    assert payload["current_event"].handle == b"source-event"
    if failed:
        future.set_exception(RuntimeError("child stopped responding"))
        assert parent._events[future] is req.current_event
    else:
        future.set_result({})
        assert not parent._events
    parent._lifecycle.stopped = True
    parent.close()
    assert not parent._events
