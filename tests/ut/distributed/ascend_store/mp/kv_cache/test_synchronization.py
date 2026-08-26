from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.synchronization import (
    NPUEventSpec,
    import_npu_event,
    record_npu_event,
)

# isort: on

_UUID_TARGET = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine.npu_generate_uuid"


def test_record_npu_event_exports_current_device_event() -> None:
    event = MagicMock()
    event.ipc_handle.return_value = b"event-handle"
    event_type = MagicMock(return_value=event)

    with (
        patch.object(torch.npu, "current_device", return_value=2),
        patch.object(torch.npu, "Event", event_type),
        patch(_UUID_TARGET, return_value="device-2"),
    ):
        exported = record_npu_event()

    assert exported.spec == NPUEventSpec("device-2", b"event-handle")
    event_type.assert_called_once_with(interprocess=True)
    event.record.assert_called_once_with()
    event.ipc_handle.assert_called_once_with()

    exported.close()
    assert exported._event is None


def test_import_npu_event_resolves_local_device_by_uuid() -> None:
    imported_event = MagicMock()
    event_type = MagicMock()
    event_type.from_ipc_handle.return_value = imported_event

    with (
        patch.object(torch.npu, "device_count", return_value=3),
        patch.object(torch.npu, "set_device") as set_device,
        patch.object(torch.npu, "Event", event_type),
        patch(_UUID_TARGET, side_effect=lambda index: f"device-{index}"),
    ):
        result = import_npu_event(NPUEventSpec("device-2", b"event-handle"))

    assert result is imported_event
    set_device.assert_called_once_with(2)
    event_type.from_ipc_handle.assert_called_once_with(2, b"event-handle")


def test_import_npu_event_rejects_unknown_device() -> None:
    with (
        patch.object(torch.npu, "device_count", return_value=2),
        patch(_UUID_TARGET, side_effect=lambda index: f"device-{index}"),
        pytest.raises(ValueError, match="No local NPU matches"),
    ):
        import_npu_event(NPUEventSpec("missing-device", b"event-handle"))
