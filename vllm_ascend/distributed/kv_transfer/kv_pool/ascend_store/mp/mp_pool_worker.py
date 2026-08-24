"""Worker-side lookup logic reused inside the KVCacheServer process."""

from typing import Protocol

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from ..pool_worker import KVPoolWorker


class LookupStore(Protocol):
    """Metadata-only store interface the lookup paths rely on."""

    def exists(self, keys: list[str]) -> list[int]: ...


class _MissingLookupStore:
    """Stand-in until the scheduler's store client is bound; every key counts as missing."""

    @staticmethod
    def exists(keys: list[str]) -> list[int]:
        return [0] * len(keys)


class MPKVPoolWorker(KVPoolWorker):
    """Reuse KVPoolWorker's lookup inside the KVCacheServer process.

    The real KVPoolWorker lives in the worker process next to the NPU caches
    and the transfer backend. The server process has neither: no
    torch.distributed group exists there, so the rank fields are derived from
    the registered rank, and m_store is not a transfer backend but the
    scheduler's backend metadata client, bound once the scheduler registers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        store: LookupStore | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        rank: int | None = None,
    ):
        self._registered_rank = vllm_config.parallel_config.rank if rank is None else rank
        self._store_is_external = store is not None
        self.m_store: LookupStore = store if store is not None else _MissingLookupStore()
        use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        super().__init__(vllm_config, use_layerwise, kv_cache_config=kv_cache_config)

        # register_kv_caches never runs in the server process, but the lookup
        # paths read the per-group cache families from the token database.
        self.token_database.group_cache_families["kv"] = {
            group_id: self._get_group_family(self.kv_cache_group_families, group_id)
            for group_id in range(self.num_kv_cache_groups)
        }

    def _init_parallelism_info(self, model_config, parallel_config) -> None:
        # The server process has no distributed group to query, so the same
        # rank fields are derived arithmetically from the registered rank.
        self.local_rank = 0
        use_mla = getattr(model_config, "use_mla", False)
        self.use_mla = isinstance(use_mla, bool) and use_mla
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")

        self.tp_size = parallel_config.tensor_parallel_size
        self.tp_rank = self._registered_rank % self.tp_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (self._registered_rank // self.tp_size) % self.pp_size
        self.pcp_rank = 0
        self.pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_rank = 0
        self.dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        self.model_name = model_config.model.split("/")[-1]

    def _init_backend(self, _parallel_config, _extra_config) -> None:
        """The transfer backend stays in the worker process; m_store is the
        scheduler's metadata client, bound later by bind_lookup_store."""
        pass

    def bind_lookup_store(self, store: LookupStore) -> None:
        if self._store_is_external:
            return
        self.m_store = store
