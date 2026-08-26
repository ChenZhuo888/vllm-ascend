"""Worker-side KV cache logic reused inside the KVCacheServer process."""

import importlib
import inspect
from collections.abc import Callable
from typing import Any, Protocol

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from ....pool_worker import KVPoolWorker
from ..memory import ImportedKVCache, import_worker_kv_caches
from ..view import WorkerKVCacheSpec


class LookupStore(Protocol):
    """Metadata-only store interface the lookup paths rely on."""

    def exists(self, keys: list[str]) -> list[int]: ...


class _MissingLookupStore:
    """Stand-in until this Worker can initialize its own backend."""

    @staticmethod
    def exists(keys: list[str]) -> list[int]:
        return [0] * len(keys)


class _DeviceBoundBackend:
    """Keep backend device selection independent of torch.distributed."""

    def __init__(self, backend: Any, device_index: int | None):
        self._backend = backend
        self._device_index = device_index

    def set_device(self) -> None:
        if self._device_index is not None:
            torch.npu.set_device(self._device_index)
            return
        self._backend.set_device()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


WorkerBackendFactory = Callable[[object, int | None, bool], LookupStore]


class MPKVPoolWorker(KVPoolWorker):
    """Reuse KVPoolWorker's lookup inside the KVCacheServer process.

    The server process has no torch.distributed group, so rank fields are
    derived from registration. NPU cache mappings and the Worker backend are
    attached later by the Worker registration path.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        store: LookupStore | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        rank: int | None = None,
        cache_importer: Callable[[WorkerKVCacheSpec], ImportedKVCache] = import_worker_kv_caches,
        backend_factory: WorkerBackendFactory | None = None,
    ):
        self._registered_rank = vllm_config.parallel_config.rank if rank is None else rank
        self._store_is_external = store is not None
        self._cache_importer = cache_importer
        self._backend_factory = backend_factory or self._create_backend
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

    def _init_backend(self, parallel_config, _extra_config) -> None:
        """Defer backend creation until the IPC mapping identifies its NPU."""
        self._parallel_config = parallel_config

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
        try:
            self._activate_backend(imported.device_index)
        except Exception:
            imported.close()
            raise

        previous = self._imported_kv_cache
        self._imported_kv_cache = imported
        self.kv_cache_spec = spec
        self.kv_caches = imported.tensors
        if imported.device_index is not None:
            self.local_rank = imported.device_index
        if previous is not None:
            previous.close()

    def _activate_backend(self, device_index: int | None) -> None:
        if self._store_is_external or not isinstance(self.m_store, _MissingLookupStore):
            return
        backend = self._backend_factory(self._parallel_config, device_index, self.use_compress)
        self.m_store = _DeviceBoundBackend(backend, device_index)

    def _create_backend(self, parallel_config, device_index: int | None, lazy_init: bool) -> LookupStore:
        from ..backend import backend_map

        backend_config = backend_map.get(self.backend.lower())
        if backend_config is None:
            raise ValueError(f"Unsupported AscendStore backend {self.backend!r}")

        backend_module = importlib.import_module(backend_config["path"])
        backend_class = getattr(backend_module, backend_config["name"])
        parameters = inspect.signature(backend_class).parameters
        backend_kwargs = {}
        if "lazy_init" in parameters:
            backend_kwargs["lazy_init"] = lazy_init
        if device_index is not None and "local_rank" in parameters:
            backend_kwargs["local_rank"] = device_index
        if device_index is not None:
            torch.npu.set_device(device_index)
        return backend_class(parallel_config, **backend_kwargs)

    def close(self) -> None:
        if self._imported_kv_cache is None:
            return
        self._imported_kv_cache.close()
        self._imported_kv_cache = None
        self.kv_cache_spec = None
        self.kv_caches = {}
