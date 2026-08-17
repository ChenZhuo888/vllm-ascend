from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request


class AscendStoreMPConnector(KVConnectorBase_V1):
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        pass

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int):
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        pass

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs: Any
    ) -> None:
        pass

    def wait_for_save(self):
        pass
