"""Worker-side KV cache logic reused inside the KVCacheServer process."""

from collections.abc import Callable
from typing import Protocol

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from ..pool_worker import KVPoolWorker
from .kv_cache_memory import ImportedKVCache, import_worker_kv_caches
from .request_view import WorkerKVCacheSpec


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

    The server process has no torch.distributed group, so rank fields are
    derived from registration. Its lookup store is bound by the Scheduler;
    NPU cache mappings are attached later by the Worker registration path.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        store: LookupStore | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        rank: int | None = None,
        cache_importer: Callable[[WorkerKVCacheSpec], ImportedKVCache] = import_worker_kv_caches,
    ):
        self._registered_rank = vllm_config.parallel_config.rank if rank is None else rank
        self._store_is_external = store is not None
        self._cache_importer = cache_importer
        self._imported_kv_cache: ImportedKVCache | None = None
        self.kv_cache_spec: WorkerKVCacheSpec | None = None
        self.m_store: LookupStore = store if store is not None else _MissingLookupStore()
        use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        super().__init__(vllm_config, use_layerwise, kv_cache_config=kv_cache_config)

        # Lookup needs these families before the later cache registration.
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
        """Backend activation follows cache mapping instead of construction."""
        pass

    def bind_lookup_store(self, store: LookupStore) -> None:
        if self._store_is_external:
            return
        self.m_store = store

    def configure_kv_caches(self, spec: WorkerKVCacheSpec) -> None:
        """Install a newer cache generation, tolerating stale RPC replay."""
        current_spec = self.kv_cache_spec
        if current_spec is not None:
            if spec.generation < current_spec.generation:
                return
            if spec.generation == current_spec.generation:
                if spec == current_spec:
                    return
                raise RuntimeError(f"KV cache generation {spec.generation} has conflicting specifications")

        imported = self._cache_importer(spec)
        previous = self._imported_kv_cache
        self._imported_kv_cache = imported
        self.kv_cache_spec = spec
        self.kv_caches = imported.tensors
        if imported.device_index is not None:
            self.local_rank = imported.device_index
        if previous is not None:
            previous.close()

    def close(self) -> None:
        if self._imported_kv_cache is None:
            return
        self._imported_kv_cache.close()
        self._imported_kv_cache = None
        self.kv_cache_spec = None
        self.kv_caches = {}
