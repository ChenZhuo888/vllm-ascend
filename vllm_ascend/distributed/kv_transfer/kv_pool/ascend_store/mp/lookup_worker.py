"""Lookup-only KVPoolWorker for AscendStore multiprocessing mode."""

from typing import Protocol

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from ..pool_worker import KVPoolWorker


class LookupStore(Protocol):
    def exists(self, keys: list[str]) -> list[int]:
        ...


class LookupKVPoolWorker(KVPoolWorker):
    """Initialize only the CPU-side state required by lookup_scheduler."""

    def __init__(self, vllm_config: VllmConfig, store: LookupStore, kv_cache_config: KVCacheConfig | None = None):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config or {}

        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)

        self.use_compress = self.compress_ratios is not None
        self.max_model_len = model_config.max_model_len
        self.dp_rank = parallel_config.data_parallel_rank
        self.local_rank = 0

        use_mla = getattr(model_config, "use_mla", False)
        self.use_mla = isinstance(use_mla, bool) and use_mla
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")

        self.tp_rank = 0
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_rank = 0
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pcp_rank = 0
        self.pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_rank = 0
        self.dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        self.model_name = model_config.model.split("/")[-1]

        self._init_kv_transfer_config(
            vllm_config, extra_config, use_layerwise=False, kv_cache_config=kv_cache_config
        )
        self._init_key_head_config(model_config, parallel_config)
        self._init_metadata(model_config, vllm_config, extra_config)

        self.token_database.group_cache_families["kv"] = {
            group_id: family for group_id, family in enumerate(self.kv_cache_group_families)
        }
        self.m_store = store
