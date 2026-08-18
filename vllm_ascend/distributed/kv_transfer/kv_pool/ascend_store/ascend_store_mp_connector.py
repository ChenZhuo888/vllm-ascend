from typing import Any

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
        raise ValueError(
            f"kv_connector_extra_config[{_KV_CACHE_SERVER_URL_KEY!r}] must be a non-empty string"
        )
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

    def get_num_new_matched_tokens(
            self,
            request: Request,
            num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.role is not KVConnectorRole.SCHEDULER:
            raise RuntimeError("get_num_new_matched_tokens is only available on the scheduler connector")

        matched_tokens = self._kv_cache_client.lookup(num_computed_tokens)
        return matched_tokens, False

    def update_state_after_alloc(
            self,
            request: Request,
            blocks: KVCacheBlocks,
            num_external_tokens: int,
    ) -> None:
        return None

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return AscendStoreMPConnectorMetadata()

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
