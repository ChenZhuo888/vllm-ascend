"""Cross-process NPU synchronization for KV cache operations."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NPUEventSpec:
    """Process-neutral identity and IPC handle for one NPU event."""

    device_uuid: str
    handle: bytes


@dataclass
class ExportedNPUEvent:
    """Keep the source event alive while another process imports it."""

    spec: NPUEventSpec
    _event: Any | None

    def close(self) -> None:
        self._event = None


def record_npu_event(stream: Any | None = None) -> ExportedNPUEvent:
    """Record and export an event on the current logical NPU device."""
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    device_index = torch.npu.current_device()
    event = torch.npu.Event(interprocess=True)
    if stream is None:
        event.record()
    else:
        event.record(stream)
    handle = event.ipc_handle()
    if not isinstance(handle, bytes) or not handle:
        raise RuntimeError("torch-npu returned an invalid NPU event IPC handle")
    return ExportedNPUEvent(NPUEventSpec(npu_generate_uuid(device_index), handle), event)


def import_npu_event(spec: NPUEventSpec) -> Any:
    """Rebuild an event on the local logical device matching its UUID."""
    if not isinstance(spec, NPUEventSpec):
        raise TypeError(f"spec must be NPUEventSpec, got {type(spec).__name__}")
    if not spec.device_uuid or not spec.handle:
        raise ValueError("NPU event specification is incomplete")

    device_index = _resolve_device(spec.device_uuid)
    torch.npu.set_device(device_index)
    return torch.npu.Event.from_ipc_handle(device_index, spec.handle)


def _resolve_device(device_uuid: str) -> int:
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    for device_index in range(torch.npu.device_count()):
        if npu_generate_uuid(device_index) == device_uuid:
            return device_index
    raise ValueError(f"No local NPU matches event device UUID {device_uuid!r}")


__all__ = ["ExportedNPUEvent", "NPUEventSpec", "import_npu_event", "record_npu_event"]
