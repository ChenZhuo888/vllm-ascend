"""KV-specific client facade for the worker-owned transfer process."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .process import TransferProcess


class _ProcessAdapter(Protocol):
    """Parent-side state that must drain before child shutdown."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class _TransferKeyInfo:
    size_bytes: int
    gvas: list[int]

    def size(self) -> int:
        return self.size_bytes

    def gva_list(self) -> list[int]:
        return self.gvas


def _snapshot_layer_request(request: Any) -> dict[str, Any]:
    return dict(
        req_id=request.req_id,
        token_len_chunk=request.token_len_chunk,
        save_end_token=request.save_end_token,
        target_token_len=request.target_token_len,
        save_start_token=request.save_start_token,
        block_ids_by_group=request.block_ids_by_group,
        block_hashes=request.block_hashes,
        is_last_chunk=request.is_last_chunk,
    )


def _snapshot_layer_task(task: Any) -> dict[str, Any]:
    shared = task.shared_block_data
    return dict(
        layer_id=task.layer_id,
        group_id=task.group_id,
        layer_idx_in_group=task.layer_idx_in_group,
        write_finish_keys=task.write_finish_keys,
        shared=(
            None
            if shared is None
            else dict(
                block_ids=shared.block_ids_arr.tolist(),
                block_gvas=shared.block_gvas_arr.tolist(),
                req_ids=shared.req_ids,
                is_last_chunks=shared.is_last_chunks,
                save_keys=shared.save_keys,
                load_keys=shared.load_keys,
            )
        ),
        block_ranges=[
            dict(
                request=_snapshot_layer_request(block_range.request),
                start_block=block_range.start_block,
                end_block=block_range.end_block,
                partial_block_index=block_range.partial_block_index,
            )
            for block_range in task.block_ranges
        ],
    )


class KVTransferProcess:
    """Translate KV operations while a separate owner manages child lifetime.

    Exported NPU resources stay alive here until the child has been reaped.
    """

    def __init__(self, config: dict[str, Any]):
        self._lifecycle = TransferProcess()
        self.client = self._lifecycle.client
        self.cache: Any = None
        self._events: dict[Future, Any] = {}
        self._event_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._device_uuid: str | None = None
        self._adapters: tuple[_ProcessAdapter, ...] = ()
        self._adapters_bound = False
        try:
            self.client.call("init", config)
        except BaseException:
            self._lifecycle.close()
            raise

    def bind_adapters(self, adapters: Iterable[_ProcessAdapter | None]) -> None:
        """Give the process facade one close path for its parent-side adapters."""
        if self._adapters_bound:
            raise RuntimeError("KV transfer process adapters are already bound")
        self._adapters = tuple(adapter for adapter in adapters if adapter is not None)
        self._adapters_bound = True

    def register_kv_caches(self, worker, kv_caches, pointers, lengths) -> None:
        from .npu_ipc import export_worker_kv_caches

        if self.cache is not None:
            raise RuntimeError("KV caches are already registered")
        self.cache = export_worker_kv_caches(kv_caches)
        self._device_uuid = self.cache.spec.storages[0].device_uuid
        database = worker.token_database
        payload = dict(
            cache=self.cache.spec,
            registered_ranges=[self.cache.describe_range(p, n) for p, n in zip(pointers, lengths)],
            group_ranges={
                group: [self.cache.describe_range(address, 0) for address in addresses]
                for group, addresses in worker.group_kv_caches_base_addr.items()
            },
            metadata=[asdict(item) for item in database.metadata],
            block_sizes=database.block_size,
            partitions=database.partitions,
            hash_block_size=database.hash_block_size,
            block_lengths=worker.group_block_len,
            block_strides=worker.group_block_stride,
            families=worker.group_kv_cache_families,
            num_layers=worker.group_num_layers,
            entry_offsets=worker.group_layer_cache_entry_offsets,
            align_state=worker.group_uses_align_state,
            layerwise=(
                dict(
                    use_gva=worker.use_layerwise_transfer,
                    block_size=worker.block_size,
                    num_layers=worker.num_layers,
                    page_size_bytes=worker.page_size_bytes,
                    consumer_is_to_put=worker.consumer_is_to_put,
                    h2d_stagger_us=worker.h2d_stagger_us,
                    max_transfer_blocks=worker.layerwise_max_transfer_blocks,
                    max_transfer_bytes=worker.layerwise_max_transfer_bytes,
                )
                if getattr(worker, "use_layerwise", False)
                else None
            ),
            tp_mismatch=(
                dict(
                    block_size=worker.block_size,
                    num_sub_keys=worker.num_sub_keys,
                    sub_size_bytes=worker.sub_size_bytes,
                )
                if getattr(worker, "tp_mismatch", False)
                else None
            ),
        )
        try:
            self.client.call("register", payload)
        except BaseException:
            self.close()
            raise

    def submit_request(self, operation: str, request) -> Future:
        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        event = request.current_event
        # Snapshot only fields consumed by the ordinary transfer handlers.
        # ReqMeta also contains layerwise arrays/GVAs and a live device event;
        # serializing the whole object would leak unrelated process state.
        payload = dict(
            req_id=request.req_id,
            save_end_token=request.save_end_token,
            target_token_len=request.target_token_len,
            save_start_token=request.save_start_token,
            block_ids_by_group=request.block_ids_by_group,
            block_hashes=request.block_hashes,
            can_save=request.can_save,
            load_spec=None if request.load_spec is None else asdict(request.load_spec),
            is_last_chunk=request.is_last_chunk,
            current_event=self._export_event(event),
            kv_cache_group_ids=request.kv_cache_group_ids,
            skip_null_blocks_by_group=request.skip_null_blocks_by_group,
            num_prompt_tokens=request.num_prompt_tokens,
            token_ids=request.token_ids,
            original_block_size=request.original_block_size,
            event_id=request.event_id,
        )
        future = self.client.submit(operation, payload)
        self._retain_event(future, event)
        return future

    def submit_layer_request(self, operation: str, tasks: list[Any], layer_id: int, event: Any = None) -> Future:
        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        payload = dict(
            layer_id=layer_id,
            tasks=[_snapshot_layer_task(task) for task in tasks],
            current_event=self._export_event(event),
        )
        future = self.client.submit(operation, payload)
        self._retain_event(future, event)
        return future

    # These synchronous proxies are the Backend surface used by KVPoolWorker.
    def exists(self, keys):
        return self.client.call("exists", keys)

    def batch_is_exist(self, keys):
        return self.client.call("batch_is_exist", keys)

    def batch_get_key_info(self, keys):
        rows = self.client.call("batch_get_key_info", keys)
        return [_TransferKeyInfo(size, list(gvas)) for size, gvas in rows]

    def batch_alloc(self, keys, sizes):
        return self.client.call("batch_alloc", (keys, sizes))

    def batch_add_lease(self, keys, lease_ttl_ms=0):
        return self.client.call("batch_add_lease", (keys, lease_ttl_ms))

    def batch_remove_lease(self, keys):
        return self.client.call("batch_remove_lease", keys)

    def ensure_initialized(self) -> None:
        self.client.call("ensure_ready")

    def get(self, keys, addresses, sizes):
        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        if len(keys) != len(addresses) or len(keys) != len(sizes):
            raise ValueError("Keys, addresses and sizes must have equal lengths")
        ranges = []
        for row, row_sizes in zip(addresses, sizes):
            if len(row) != len(row_sizes):
                raise ValueError("Each buffer address needs one size")
            ranges.append([self.cache.describe_range(pointer, size) for pointer, size in zip(row, row_sizes)])
        return self.client.call("get_ranges", (keys, ranges))

    def close(self) -> None:
        """Drain adapters and child work before releasing exported resources."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            error: BaseException | None = None
            for adapter in self._adapters:
                try:
                    adapter.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
            try:
                self._lifecycle.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                # TransferProcess reaps the child before returning, even after
                # a failed drain. No child can access these handles now.
                if self._lifecycle.stopped:
                    with self._event_lock:
                        self._events.clear()
                    if self.cache is not None:
                        self.cache.close()
            if error is not None:
                raise error

    def _export_event(self, event: Any):
        if event is None:
            return None
        from .npu_ipc import NPUEventSpec

        assert self._device_uuid is not None
        return NPUEventSpec(self._device_uuid, event.ipc_handle())

    def _retain_event(self, future: Future, event: Any) -> None:
        if event is None:
            return
        with self._event_lock:
            self._events[future] = event
        future.add_done_callback(self._release_event)

    def _release_event(self, future: Future) -> None:
        # An error/timeout is not proof that device IO stopped. In that case
        # keep the event until close has reaped the child interpreter.
        if future.exception() is None:
            with self._event_lock:
                self._events.pop(future, None)
