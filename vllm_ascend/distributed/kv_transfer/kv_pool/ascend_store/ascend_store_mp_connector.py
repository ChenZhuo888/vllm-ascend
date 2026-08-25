import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreKVEvents
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendStoreKVConnectorWorkerMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
    ExportedKVCache,
    export_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.request_view import WorkerKVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

_KV_CACHE_SERVER_URL_KEY = "kv_cache_server_url"


class AscendStoreMPConnectorMetadata(KVConnectorMetadata):
    pass


def _get_kv_cache_server_url(vllm_config: VllmConfig) -> str:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        raise ValueError("kv_transfer_config must be set for AscendStoreMPConnector")

    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    server_url = extra_config.get(_KV_CACHE_SERVER_URL_KEY)
    if not isinstance(server_url, str) or not server_url:
        raise ValueError(f"kv_connector_extra_config[{_KV_CACHE_SERVER_URL_KEY!r}] must be a non-empty string")
    return server_url


class AscendStoreMPConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        if getattr(vllm_config.model_config, "enable_sleep_mode", False) is True:
            raise ValueError("AscendStoreMPConnector does not support sleep mode")

        self._kv_cache_export_lock = threading.Lock()
        self._next_kv_cache_generation = 1
        self._active_kv_cache_export: ExportedKVCache | None = None
        self._pending_kv_cache_exports: dict[int, ExportedKVCache] = {}
        self._kv_cache_client = KVCacheClient(_get_kv_cache_server_url(vllm_config))
        # Scheduler-process-local state: the live Request references feeding
        # the all_token_ids increments, the real BlockPool the server's
        # touch/free commands are replayed on, and the KV event aggregation
        # fed from worker outputs.
        self._local_requests: dict[str, Request] = {}
        self._synced_token_len: dict[str, int] = {}
        self._gpu_block_pool = None
        self._kv_cache_events: AscendStoreKVEvents | None = None

        try:
            if role == KVConnectorRole.SCHEDULER:
                if kv_cache_config is None:
                    raise ValueError("kv_cache_config must be set for the scheduler connector")

                page_size_bytes = kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
                self._kv_cache_client.register_scheduler(
                    vllm_config,
                    kv_cache_config,
                    page_size_bytes,
                )
            else:
                self._kv_cache_client.register_worker(
                    vllm_config,
                    kv_cache_config,
                )
        except Exception:
            self._kv_cache_client.close()
            raise

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        self._require_scheduler_role("get_num_new_matched_tokens")
        return self._kv_cache_client.lookup(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        self._require_scheduler_role("update_state_after_alloc")
        # The server registers a snapshot; the live reference stays here to
        # supply all_token_ids increments in later build_connector_meta calls.
        self._local_requests[request.request_id] = request
        self._synced_token_len[request.request_id] = len(request.prompt_token_ids)
        self._kv_cache_client.update_state_after_alloc(request, blocks, num_external_tokens)

    def _require_scheduler_role(self, action: str) -> None:
        if self.role != KVConnectorRole.SCHEDULER:
            raise RuntimeError(f"{action} is only available on the scheduler connector")

    def _require_worker_role(self, action: str) -> None:
        if self.role != KVConnectorRole.WORKER:
            raise RuntimeError(f"{action} is only available on the worker connector")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        self._require_scheduler_role("build_connector_meta")
        new_token_ids = self._collect_token_id_increments(scheduler_output)
        result = self._kv_cache_client.build_connector_meta(scheduler_output, new_token_ids)
        for req_id in scheduler_output.finished_req_ids:
            self._local_requests.pop(req_id, None)
            self._synced_token_len.pop(req_id, None)
        if result is None:
            return AscendStoreMPConnectorMetadata()
        metadata, touch_block_ids = result
        if touch_block_ids and self._gpu_block_pool is not None:
            pool = self._gpu_block_pool
            pool.touch([pool.blocks[block_id] for block_id in touch_block_ids])
        return metadata

    def _collect_token_id_increments(self, scheduler_output: SchedulerOutput) -> dict[str, list[int]]:
        """Tokens appended to each scheduled cached request since the last step."""
        increments: dict[str, list[int]] = {}
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            request = self._local_requests.get(req_id)
            if request is None:
                continue
            all_token_ids = request.all_token_ids
            start = self._synced_token_len.get(req_id, len(all_token_ids))
            if len(all_token_ids) > start:
                increments[req_id] = list(all_token_ids[start:])
            self._synced_token_len[req_id] = len(all_token_ids)
        return increments

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        # The BlockPool never crosses the process border; the server's mamba
        # bookkeeping returns block-id commands that are replayed on it here.
        self._gpu_block_pool = gpu_block_pool

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self._finish_request(request, block_ids, all_groups=False)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._finish_request(request, block_ids, all_groups=True)

    def _finish_request(self, request: Request, block_ids, all_groups: bool) -> tuple[bool, dict[str, Any] | None]:
        self._require_scheduler_role("request_finished")
        delay_free, extra = self._kv_cache_client.request_finished(request.request_id, block_ids, all_groups)
        self._local_requests.pop(request.request_id, None)
        self._synced_token_len.pop(request.request_id, None)
        return delay_free, extra

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._require_scheduler_role("update_connector_output")
        worker_meta = connector_output.kv_connector_worker_meta
        completed_events = (
            worker_meta.completed_events if isinstance(worker_meta, AscendStoreKVConnectorWorkerMetadata) else {}
        )
        free_block_ids = self._kv_cache_client.update_connector_output(completed_events)
        if free_block_ids and self._gpu_block_pool is not None:
            pool = self._gpu_block_pool
            pool.free_blocks([pool.blocks[block_id] for block_id in free_block_ids])

        # KV events are a pure data stream: aggregate locally instead of
        # round-tripping them through the server.
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
            return
        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(kv_cache_events.get_number_of_workers())

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._require_worker_role("register_kv_caches")
        with self._kv_cache_export_lock:
            generation = self._next_kv_cache_generation
            self._next_kv_cache_generation += 1

        exported = export_worker_kv_caches(kv_caches, generation)
        with self._kv_cache_export_lock:
            self._pending_kv_cache_exports[generation] = exported
        try:
            self._kv_cache_client.register_kv_caches(exported.spec, on_registered=self._confirm_kv_cache_export)
        except Exception:
            with self._kv_cache_export_lock:
                failed = self._pending_kv_cache_exports.pop(generation, None)
            if failed is not None:
                failed.close()
            raise

    def _confirm_kv_cache_export(self, spec: WorkerKVCacheSpec) -> None:
        to_close: list[ExportedKVCache] = []
        with self._kv_cache_export_lock:
            confirmed = self._pending_kv_cache_exports.pop(spec.generation, None)
            if confirmed is None:
                return

            active = self._active_kv_cache_export
            if active is not None and active.spec.generation >= spec.generation:
                to_close.append(confirmed)
            else:
                self._active_kv_cache_export = confirmed
                if active is not None:
                    to_close.append(active)
                stale_generations = [
                    generation for generation in self._pending_kv_cache_exports if generation < spec.generation
                ]
                to_close.extend(self._pending_kv_cache_exports.pop(generation) for generation in stale_generations)

        for cache_export in to_close:
            cache_export.close()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def shutdown(self) -> None:
        try:
            self._kv_cache_client.close()
        finally:
            with self._kv_cache_export_lock:
                exports = list(self._pending_kv_cache_exports.values())
                self._pending_kv_cache_exports.clear()
                if self._active_kv_cache_export is not None:
                    exports.append(self._active_kv_cache_export)
                    self._active_kv_cache_export = None
            for cache_export in exports:
                cache_export.close()
