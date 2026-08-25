"""NPU IPC export and import of Worker KV cache allocations."""

from dataclasses import dataclass
from typing import Protocol

import cloudpickle
import torch

from .request_view import KVCacheStorageSpec, KVCacheTensorSpec, WorkerKVCacheSpec

TORCH_NPU_IPC_HANDLE = "torch_npu_ipc"
TORCH_NPU_IPC_VERSION = 1


class KVCacheStorageAdapter(Protocol):
    """Device-specific storage sharing hidden from KV cache registration."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec: ...

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int | None]: ...


@dataclass
class ExportedKVCache:
    """Wire specification plus references keeping exported allocations alive."""

    spec: WorkerKVCacheSpec
    _storages: tuple[torch.Tensor, ...]

    def close(self) -> None:
        self._storages = ()


@dataclass
class ImportedKVCache:
    """Reconstructed cache tensors plus references keeping IPC mappings alive."""

    tensors: dict[str, tuple[torch.Tensor, ...]]
    device_index: int | None
    _storages: tuple[torch.Tensor, ...]

    def close(self) -> None:
        self.tensors.clear()
        self._storages = ()


class TorchNPUIPCAdapter:
    """Share NPU allocations through torch-npu multiprocessing handles."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        if storage.device.type != "npu":
            raise ValueError(f"TorchNPUIPCAdapter only supports NPU storage, got {storage.device}")

        from torch.multiprocessing.reductions import reduce_tensor

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        _, ipc_args = reduce_tensor(storage)
        return KVCacheStorageSpec(
            size_bytes=_storage_size_bytes(storage),
            device_type=storage.device.type,
            device_uuid=npu_generate_uuid(storage.device.index),
            handle_type=TORCH_NPU_IPC_HANDLE,
            handle_version=TORCH_NPU_IPC_VERSION,
            handle=cloudpickle.dumps(tuple(ipc_args)),
        )

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int]:
        if spec.handle_type != TORCH_NPU_IPC_HANDLE or spec.handle_version != TORCH_NPU_IPC_VERSION:
            raise ValueError(f"Unsupported KV cache handle {spec.handle_type!r} version {spec.handle_version}")

        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        device_index = self._resolve_device(spec.device_uuid)
        torch.npu.set_device(device_index)
        ipc_args = list(cloudpickle.loads(spec.handle))
        if len(ipc_args) <= 6:
            raise ValueError("Malformed torch-npu IPC handle")

        # Logical device indices may differ between the Worker and server.
        ipc_args[6] = device_index
        return rebuild_npu_tensor(*ipc_args), device_index

    @staticmethod
    def _resolve_device(device_uuid: str) -> int:
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        for device_index in range(torch.npu.device_count()):
            if npu_generate_uuid(device_index) == device_uuid:
                return device_index
        raise ValueError(f"No local NPU matches KV cache device UUID {device_uuid!r}")


def export_worker_kv_caches(
    kv_caches: dict[str, torch.Tensor],
    generation: int,
    adapter: KVCacheStorageAdapter | None = None,
) -> ExportedKVCache:
    """Export each allocation once and describe every tensor view over it."""
    if generation <= 0:
        raise ValueError(f"KV cache generation must be greater than 0, got {generation}")
    if not kv_caches:
        raise ValueError("kv_caches must not be empty")

    adapter = TorchNPUIPCAdapter() if adapter is None else adapter
    storage_indices: dict[tuple[str, int | None, int], int] = {}
    storages: list[torch.Tensor] = []
    storage_specs: list[KVCacheStorageSpec] = []
    caches: dict[str, tuple[KVCacheTensorSpec, ...]] = {}

    for name, cache_or_caches in kv_caches.items():
        tensors = _normalize_cache_tensors(name, cache_or_caches)
        tensor_specs: list[KVCacheTensorSpec] = []
        for tensor in tensors:
            storage = _untyped_storage(tensor)
            storage_key = (tensor.device.type, tensor.device.index, storage.data_ptr())
            storage_index = storage_indices.get(storage_key)
            if storage_index is None:
                storage_index = len(storages)
                storage_indices[storage_key] = storage_index
                storage_tensor = _storage_as_bytes(tensor)
                storage_spec = adapter.export_storage(storage_tensor)
                if storage_spec.size_bytes != _storage_size_bytes(storage_tensor):
                    raise ValueError("KV cache adapter returned an incorrect storage size")
                if storage_spec.device_type != tensor.device.type:
                    raise ValueError("KV cache adapter returned an incorrect device type")
                storages.append(storage_tensor)
                storage_specs.append(storage_spec)

            tensor_specs.append(
                KVCacheTensorSpec(
                    storage_index=storage_index,
                    storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=str(tensor.dtype),
                )
            )
        caches[name] = tuple(tensor_specs)

    spec = WorkerKVCacheSpec(
        generation=generation,
        caches=caches,
        storages=tuple(storage_specs),
    )
    _validate_worker_spec(spec)
    return ExportedKVCache(spec, tuple(storages))


def import_worker_kv_caches(
    spec: WorkerKVCacheSpec,
    adapter: KVCacheStorageAdapter | None = None,
) -> ImportedKVCache:
    """Import each allocation once and rebuild the registered tensor views."""
    _validate_worker_spec(spec)
    adapter = TorchNPUIPCAdapter() if adapter is None else adapter
    imported_storages: list[tuple[torch.Tensor, int | None]] = []
    try:
        for storage_spec in spec.storages:
            storage, device_index = adapter.import_storage(storage_spec)
            if _storage_size_bytes(storage) < storage_spec.size_bytes:
                raise ValueError("Imported KV cache storage is smaller than its specification")
            if storage.device.type != storage_spec.device_type:
                raise ValueError(
                    f"Imported KV cache storage is on {storage.device.type}, expected {storage_spec.device_type}"
                )
            imported_storages.append((storage, device_index))

        device_indices = {device_index for _, device_index in imported_storages}
        if len(device_indices) > 1:
            raise ValueError("One Worker registration cannot span multiple server devices")
        device_index = next(iter(device_indices), None)
        caches = _rebuild_cache_tensors(spec, imported_storages)
        return ImportedKVCache(caches, device_index, tuple(storage for storage, _ in imported_storages))
    except Exception:
        imported_storages.clear()
        raise


def _rebuild_cache_tensors(
    spec: WorkerKVCacheSpec,
    storages: list[tuple[torch.Tensor, int | None]],
) -> dict[str, tuple[torch.Tensor, ...]]:
    caches: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, tensor_specs in spec.caches.items():
        tensors = []
        for tensor_spec in tensor_specs:
            storage = storages[tensor_spec.storage_index][0]
            dtype = _decode_dtype(tensor_spec.dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            tensor = torch.empty(0, dtype=dtype, device=storage.device)
            tensor.set_(
                storage.untyped_storage(),
                tensor_spec.storage_offset_bytes // element_size,
                tensor_spec.shape,
                tensor_spec.stride,
            )
            tensors.append(tensor)
        caches[name] = tuple(tensors)
    return caches


def _validate_worker_spec(spec: WorkerKVCacheSpec) -> None:
    if spec.generation <= 0:
        raise ValueError(f"KV cache generation must be greater than 0, got {spec.generation}")
    if not spec.storages:
        raise ValueError("KV cache storage handles are required")
    if not spec.caches:
        raise ValueError("KV cache tensor layouts are required")
    for storage in spec.storages:
        if storage.size_bytes <= 0:
            raise ValueError("KV cache storage size must be greater than 0")
        if not storage.device_type or not storage.device_uuid:
            raise ValueError("KV cache storage device identity is required")
        if not storage.handle_type or storage.handle_version <= 0 or not storage.handle:
            raise ValueError("KV cache storage handle is invalid")
    for name, tensors in spec.caches.items():
        if not isinstance(name, str) or not name or not tensors:
            raise ValueError("KV cache tensor layouts must have non-empty names and values")
        for tensor in tensors:
            if not 0 <= tensor.storage_index < len(spec.storages):
                raise ValueError(f"KV cache {name!r} references an unknown storage")
            dtype = _decode_dtype(tensor.dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            _validate_tensor_spec(name, tensor, spec.storages[tensor.storage_index], element_size)


def _validate_tensor_spec(
    name: str,
    tensor: KVCacheTensorSpec,
    storage: KVCacheStorageSpec,
    element_size: int,
) -> None:
    if tensor.storage_offset_bytes < 0 or tensor.storage_offset_bytes % element_size:
        raise ValueError(f"KV cache {name!r} has an invalid storage offset")
    if len(tensor.shape) != len(tensor.stride) or any(size < 0 for size in tensor.shape):
        raise ValueError(f"KV cache {name!r} has an invalid shape or stride")
    if any(stride < 0 for stride in tensor.stride):
        raise ValueError(f"KV cache {name!r} has a negative stride")

    required_bytes = tensor.storage_offset_bytes
    if all(tensor.shape):
        last_element = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride))
        required_bytes += (last_element + 1) * element_size
    if required_bytes > storage.size_bytes:
        raise ValueError(f"KV cache {name!r} exceeds its storage allocation")


def _normalize_cache_tensors(name: str, cache_or_caches) -> tuple[torch.Tensor, ...]:
    if not isinstance(name, str) or not name:
        raise ValueError("KV cache names must be non-empty strings")
    tensors = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else tuple(cache_or_caches)
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(f"KV cache {name!r} must contain one or more tensors")
    return tensors


def _storage_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = _untyped_storage(tensor)
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))


def _untyped_storage(tensor: torch.Tensor):
    try:
        return tensor.untyped_storage()
    except AttributeError:
        return tensor.storage()


def _storage_size_bytes(tensor: torch.Tensor) -> int:
    return _untyped_storage(tensor).nbytes()


def _decode_dtype(value: str) -> torch.dtype:
    if not value.startswith("torch."):
        raise ValueError(f"Invalid KV cache dtype {value!r}")
    dtype = getattr(torch, value.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported KV cache dtype {value!r}")
    return dtype
