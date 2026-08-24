from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
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
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient

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
        self._kv_cache_client = KVCacheClient(_get_kv_cache_server_url(vllm_config))
        # Scheduler-process-local state: the live Request references feeding
        # the all_token_ids increments, and the real BlockPool the server's
        # touch/free commands are replayed on.
        self._local_requests: dict[str, Request] = {}
        self._synced_token_len: dict[str, int] = {}
        self._gpu_block_pool = None

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
        if self.role != KVConnectorRole.SCHEDULER:
            raise RuntimeError("get_num_new_matched_tokens is only available on the scheduler connector")
        return self._kv_cache_client.lookup(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if self.role != KVConnectorRole.SCHEDULER:
            raise RuntimeError("update_state_after_alloc is only available on the scheduler connector")
        # The server registers a snapshot; the live reference stays here to
        # supply all_token_ids increments in later build_connector_meta calls.
        self._local_requests[request.request_id] = request
        self._synced_token_len[request.request_id] = len(request.prompt_token_ids)
        self._kv_cache_client.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        if self.role != KVConnectorRole.SCHEDULER:
            raise RuntimeError("build_connector_meta is only available on the scheduler connector")
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
        self._kv_cache_client.close()
