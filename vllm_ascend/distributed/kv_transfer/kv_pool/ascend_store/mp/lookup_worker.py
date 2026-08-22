"""Lookup-side PoolScheduler and PoolWorker adaptations for MP mode."""

from collections.abc import Sequence
from typing import Protocol

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig

from ..pool_scheduler import KVPoolScheduler
from ..pool_worker import KVPoolWorker
from .registration import SchedulerIdentity, SchedulerRegistration, WorkerLookupHandler


class LookupStore(Protocol):
    def exists(self, keys: list[str]) -> list[int]: ...


class _MissingLookupStore:
    @staticmethod
    def exists(keys: list[str]) -> list[int]:
        return [0] * len(keys)


class _WorkerLookupBridge:
    """Expose the registered Worker through KVPoolScheduler's existing client interface."""

    def __init__(self, identity: SchedulerIdentity, lookup_handler: WorkerLookupHandler):
        self._identity = identity
        self._lookup_handler = lookup_handler

    def lookup(
        self,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return self._lookup_handler(
            self._identity,
            token_len,
            block_hashes,
            kv_cache_group_ids,
            False,
            hbm_hit_tokens,
        )


class MPKVPoolScheduler(KVPoolScheduler):
    """Run the original KVPoolScheduler inside KVCacheServer."""

    def __init__(self, registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler):
        super().__init__(
            registration.vllm_config,
            use_layerwise=False,
            kv_cache_config=registration.kv_cache_config,
            page_size_bytes=registration.page_size_bytes,
        )
        self.client = _WorkerLookupBridge(  # type: ignore[assignment]
            registration.identity,
            lookup_handler,
        )

    def close(self) -> None:
        close = getattr(self.store_scheduler, "close", None)
        if callable(close):
            close()


class LookupKVPoolWorker(KVPoolWorker):
    """Initialize CPU-side state and use the Scheduler metadata store for Lookup."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        store: LookupStore | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        rank: int | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config or {}
        worker_rank = parallel_config.rank if rank is None else rank

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

        self.tp_size = parallel_config.tensor_parallel_size
        self.tp_rank = worker_rank % self.tp_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (worker_rank // self.tp_size) % self.pp_size
        self.pcp_rank = 0
        self.pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_rank = 0
        self.dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        self.model_name = model_config.model.split("/")[-1]

        self._init_kv_transfer_config(
            vllm_config,
            extra_config,
            use_layerwise=False,
            kv_cache_config=kv_cache_config,
        )
        self._init_key_head_config(model_config, parallel_config)
        self._init_metadata(model_config, vllm_config, extra_config)

        self.token_database.group_cache_families["kv"] = {
            group_id: family for group_id, family in enumerate(self.kv_cache_group_families)
        }
        self._store_is_external = store is not None
        self.m_store: LookupStore = store or _MissingLookupStore()

    def bind_lookup_store(self, store: LookupStore) -> None:
        if self._store_is_external:
            return
        self.m_store = store
