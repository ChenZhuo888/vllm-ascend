"""Worker-side KV cache logic reused inside the KVCacheServer process."""

import threading
from collections.abc import Callable
from typing import Any, Protocol

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from ....metadata import AscendConnectorMetadata
from ....pool_worker import KVPoolWorker
from ..memory import ImportedKVCache, import_worker_kv_caches
from ..synchronization import NPUEventSpec, import_npu_event
from ..view import WorkerKVCacheSpec
from .backend import create_mp_backend
from .transfer import (
    MPKVCacheStoreKeyLayerRecvingThread,
    MPKVCacheStoreKeyLayerSendingThread,
    MPKVCacheStoreLayerRecvingThread,
    MPKVCacheStoreLayerSendingThread,
    MPKVCacheStoreRecvingThread,
    MPKVCacheStoreSendingThread,
)


class WorkerBackend(Protocol):
    """Backend operations reused by the Worker service."""

    def exists(self, keys: list[str]) -> list[int]: ...

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None: ...

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> None: ...

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None: ...

    def unregister_buffer(self) -> None: ...

    def set_device(self) -> None: ...

    def close(self) -> None: ...


class _MissingWorkerBackend:
    """Lookup-only stand-in used before the Worker runtime is active."""

    @staticmethod
    def exists(keys: list[str]) -> list[int]:
        return [0] * len(keys)


class _ImportedNPUEvent:
    """Preserve an event recorded by the vLLM Worker process."""

    def __init__(self, event: Any):
        self._event = event

    def record(self) -> None:
        # The source Worker already recorded this event on its attention
        # stream. Recording it again in KVCacheServer would lose that order.
        return None

    def synchronize(self) -> None:
        self._event.synchronize()


WorkerBackendFactory = Callable[[object, int | None, bool], WorkerBackend]


class MPKVPoolWorker(KVPoolWorker):
    """Reuse KVPoolWorker's lookup inside the KVCacheServer process.

    The server process has no torch.distributed group, so rank fields are
    derived from registration. NPU cache mappings and the Worker backend are
    attached later by the Worker registration path.
    """

    def __init__(
        self,
        vllm_config: Any,
        store: WorkerBackend | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        rank: int | None = None,
        cache_importer: Callable[[WorkerKVCacheSpec], ImportedKVCache] = import_worker_kv_caches,
        backend_factory: WorkerBackendFactory | None = None,
    ):
        self._registered_rank = vllm_config.parallel_config.rank if rank is None else rank
        self._store_is_external = store is not None
        self._cache_importer = cache_importer
        self._backend_factory = backend_factory or self._create_backend
        self._backend_device_index: int | None = None
        self._imported_kv_cache: ImportedKVCache | None = None
        self._failed_imported_kv_cache: ImportedKVCache | None = None
        self._runtime_failure: BaseException | None = None
        self._runtime_active = False
        self._current_connector_metadata: AscendConnectorMetadata | None = None
        self.kv_cache_spec: WorkerKVCacheSpec | None = None
        self.m_store: WorkerBackend = store if store is not None else _MissingWorkerBackend()
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
        self.device_index: int | None = None
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
        if self._runtime_failure is not None:
            raise RuntimeError("Worker KV cache runtime is unavailable; re-register the Worker service") from (
                self._runtime_failure
            )

        current_spec = self.kv_cache_spec
        if current_spec is not None:
            if spec.generation < current_spec.generation:
                return
            if spec.generation == current_spec.generation:
                if spec == current_spec:
                    return
                raise RuntimeError(f"KV cache generation {spec.generation} has conflicting specifications")

        imported = self._cache_importer(spec)
        if self._imported_kv_cache is None:
            self._configure_first_generation(spec, imported)
            return
        self._replace_generation(spec, imported)

    def _configure_first_generation(self, spec: WorkerKVCacheSpec, imported: ImportedKVCache) -> None:
        try:
            self._activate_backend(imported.device_index)
            self._register_runtime(imported)
        except BaseException:
            try:
                self._deactivate_runtime()
            except BaseException as cleanup_error:
                self._failed_imported_kv_cache = imported
                self._runtime_failure = cleanup_error
                raise RuntimeError("Failed to clean up the first KV cache generation") from cleanup_error
            try:
                self._close_backend()
            finally:
                imported.close()
            raise
        self._publish_generation(spec, imported)

    def _replace_generation(self, spec: WorkerKVCacheSpec, imported: ImportedKVCache) -> None:
        previous = self._imported_kv_cache
        previous_spec = self.kv_cache_spec
        assert previous is not None and previous_spec is not None

        if imported.device_index != self._backend_device_index and not self._store_is_external:
            imported.close()
            raise RuntimeError(
                f"KV cache generation moved from NPU {self._backend_device_index} to {imported.device_index}"
            )

        try:
            self._deactivate_runtime()
        except BaseException:
            imported.close()
            raise

        try:
            self._register_runtime(imported)
        except BaseException as registration_error:
            try:
                self._deactivate_runtime()
            except BaseException as cleanup_error:
                self._failed_imported_kv_cache = imported
                self._runtime_failure = cleanup_error
                raise RuntimeError(
                    f"Failed to clean up KV cache generation {spec.generation} after activation failed"
                ) from cleanup_error
            try:
                self._register_runtime(previous)
            except BaseException as rollback_error:
                imported.close()
                self._runtime_failure = rollback_error
                raise RuntimeError(
                    f"Failed to activate KV cache generation {spec.generation} and restore "
                    f"generation {previous_spec.generation}: {type(registration_error).__name__}: {registration_error}"
                ) from rollback_error
            imported.close()
            raise

        self._publish_generation(spec, imported)
        previous.close()

    def _register_runtime(self, imported: ImportedKVCache) -> None:
        self.m_store.set_device()
        self._runtime_active = True
        super().register_kv_caches(imported.tensors)

    def _publish_generation(self, spec: WorkerKVCacheSpec, imported: ImportedKVCache) -> None:
        self._imported_kv_cache = imported
        self.kv_cache_spec = spec
        self.kv_caches = imported.tensors
        self.device_index = imported.device_index

    def _clear_generation(self) -> None:
        self._imported_kv_cache = None
        self.kv_cache_spec = None
        self.kv_caches = {}
        self.device_index = None

    def _activate_backend(self, device_index: int | None) -> None:
        if self._store_is_external:
            return
        if not isinstance(self.m_store, _MissingWorkerBackend):
            if device_index != self._backend_device_index:
                raise RuntimeError(
                    f"Worker backend is bound to NPU {self._backend_device_index}, got cache on NPU {device_index}"
                )
            return
        self.m_store = self._backend_factory(self._parallel_config, device_index, self.use_compress)
        self._backend_device_index = device_index

    def _create_backend(self, parallel_config, device_index: int | None, lazy_init: bool) -> WorkerBackend:
        return create_mp_backend(self.backend, parallel_config, device_index, lazy_init)

    def start_load_kv(self, metadata: AscendConnectorMetadata) -> None:
        self._current_connector_metadata = metadata
        super().start_load_kv(metadata)

    def save_kv_layer_from_event(self, event_spec: NPUEventSpec) -> None:
        if self._current_connector_metadata is None:
            raise RuntimeError("Layer Store requires start_load_kv for the current step")
        if self.kv_cache_spec is None:
            raise RuntimeError("Worker KV caches must be configured before layer Store")
        device_uuids = {storage.device_uuid for storage in self.kv_cache_spec.storages}
        if event_spec.device_uuid not in device_uuids:
            raise ValueError(f"NPU event device {event_spec.device_uuid!r} does not match Worker KV caches")
        if self.sync_save_events is None or self.current_layer >= len(self.sync_save_events):
            raise RuntimeError(f"Invalid Layerwise Store position {self.current_layer}")

        event = _ImportedNPUEvent(import_npu_event(event_spec))
        self.sync_save_events[self.current_layer] = event  # type: ignore[assignment]
        super().save_kv_layer(self._current_connector_metadata)

    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return

        if self.use_layerwise:
            self.get_event = threading.Event()
            self.layer_load_finished_events = [threading.Event() for _ in range(self.num_layers)]
            self.layer_save_finished_events = [threading.Event() for _ in range(self.num_layers)]
            self.sync_save_events = [torch.npu.Event() for _ in range(self.num_layers)]
            can_save = self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            if self.use_gva_layerwise and can_save:
                ready_event_sending = threading.Event()
                self.kv_send_thread = MPKVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.put_step,
                    self.my_key_index,
                    self.num_ranks_per_layer,
                    self.page_size_bytes,
                    ready_event_sending,
                    self.num_layers,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                    self.layerwise_max_transfer_blocks,
                    self.layerwise_max_transfer_bytes,
                    group_builders=self._build_group_layer_builders(),
                )
                self._start_transfer_thread(self.kv_send_thread, ready_event_sending)
            elif can_save:
                ready_event_sending = threading.Event()
                self.kv_send_thread = MPKVCacheStoreKeyLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.put_step,
                    ready_event_sending,
                    self.num_layers,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                )
                self._start_transfer_thread(self.kv_send_thread, ready_event_sending)

            ready_event = threading.Event()
            if self.use_gva_layerwise:
                self.kv_recv_thread = MPKVCacheStoreLayerRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.my_key_index,
                    self.num_ranks_per_layer,
                    self.page_size_bytes,
                    ready_event,
                    self.get_event,
                    self.layer_load_finished_events,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                    self.num_layers,
                    self.h2d_stagger_us,
                    self.layerwise_max_transfer_blocks,
                    self.layerwise_max_transfer_bytes,
                    group_builders=self._build_group_layer_builders(),
                )
            else:
                self.kv_recv_thread = MPKVCacheStoreKeyLayerRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    ready_event,
                    self.get_event,
                    self.layer_load_finished_events,
                    self.layer_save_finished_events,
                    self.num_layers,
                )
            self._start_transfer_thread(self.kv_recv_thread, ready_event)
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = MPKVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.group_uses_align_state,
                    self.enable_kv_events,
                )
                self._start_transfer_thread(self.kv_send_thread, ready_event_sending)
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = MPKVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    ready_event,
                    invalid_block_ids=self._invalid_block_ids,
                    invalid_block_ids_lock=self._invalid_block_ids_lock,
                )
                self._start_transfer_thread(self.kv_recv_thread, ready_event)
        self._transfer_threads_started = True

    @staticmethod
    def _start_transfer_thread(thread, ready_event: threading.Event) -> None:
        thread.start()
        ready_event.wait()
        thread.raise_if_failed()

    def _stop_kv_transfer_threads(self) -> None:
        threads = [thread for thread in (self.kv_send_thread, self.kv_recv_thread) if thread is not None]
        for thread in threads:
            thread.stop(wait=False)
        for thread in threads:
            thread.stop()
        self.kv_send_thread = None
        self.kv_recv_thread = None
        self._transfer_threads_started = False

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata, event_spec: NPUEventSpec) -> None:
        if self.kv_send_thread is None:
            raise RuntimeError("Worker Store runtime is not initialized")
        send_thread = self.kv_send_thread
        send_thread.raise_if_failed()
        save_requests = [request for request in connector_metadata.requests if request.can_save]
        if not save_requests:
            return

        if self.kv_cache_spec is None:
            raise RuntimeError("Worker KV caches must be configured before wait_for_save")
        device_uuids = {storage.device_uuid for storage in self.kv_cache_spec.storages}
        if event_spec.device_uuid not in device_uuids:
            raise ValueError(f"NPU event device {event_spec.device_uuid!r} does not match Worker KV caches")

        current_event = import_npu_event(event_spec)
        for request in save_requests:
            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            send_thread.add_stored_request(request.req_id)
            send_thread.add_request(request)
        send_thread.request_queue.join()

    def close(self) -> None:
        imported = self._imported_kv_cache
        failed_imported = self._failed_imported_kv_cache
        self._deactivate_runtime()
        try:
            self._close_backend()
        finally:
            if imported is not None:
                imported.close()
            if failed_imported is not None:
                failed_imported.close()
            self._failed_imported_kv_cache = None
            self._runtime_failure = None
            self._current_connector_metadata = None
            self._clear_generation()

    def _deactivate_runtime(self) -> None:
        if not self._runtime_active:
            return
        self._stop_kv_transfer_threads()
        self.m_store.unregister_buffer()
        self._runtime_active = False

    def _close_backend(self) -> None:
        if self._store_is_external or isinstance(self.m_store, _MissingWorkerBackend):
            return
        try:
            self.m_store.close()
        finally:
            self.m_store = _MissingWorkerBackend()
            self._backend_device_index = None
