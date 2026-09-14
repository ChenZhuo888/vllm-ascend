# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Diagnose Mooncake registration and D2D copy of child NPU IPC mappings."""

import importlib
import multiprocessing as mp
import subprocess
import sys
import traceback
from contextlib import suppress
from importlib import metadata
from multiprocessing.connection import Connection

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
    KVCacheStorageSpec,
    TorchNPUIPCAdapter,
)

BUFFER_SIZE_BYTES = 4 << 20
COPY_OFFSET_BYTES = 1 << 20
COPY_SIZE_BYTES = 2 << 20
CHILD_TIMEOUT_SECONDS = 120
INITIAL_VALUE = 17
UPDATED_VALUE = 29
DEVICE_TO_DEVICE = 2
MOONCAKE_PACKAGE = "mooncake-transfer-engine-npu"
MOONCAKE_VERSION = "0.3.12.post1"
MOONCAKE_INDEX = "https://mirrors.aliyun.com/pypi/simple/"


def _ensure_mooncake_version() -> None:
    """Bridge the /e2e command's default-branch workflow until this PR merges."""
    try:
        installed = metadata.version(MOONCAKE_PACKAGE)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != MOONCAKE_VERSION:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--index-url",
                MOONCAKE_INDEX,
                f"{MOONCAKE_PACKAGE}=={MOONCAKE_VERSION}",
            ],
            check=True,
        )
    actual = metadata.version(MOONCAKE_PACKAGE)
    if actual != MOONCAKE_VERSION:
        raise RuntimeError(f"Expected {MOONCAKE_PACKAGE} {MOONCAKE_VERSION}, got {actual}")


def _assert_filled(tensor: torch.Tensor, value: int, stage: str) -> None:
    if not torch.all(tensor == value).item():
        raise AssertionError(f"Data mismatch after {stage}")


def _raw_d2d_copy(source: torch.Tensor, destination: torch.Tensor, stream: torch.npu.Stream) -> None:
    pointers_source = torch.tensor([source.data_ptr()], dtype=torch.int64)
    pointers_destination = torch.tensor([destination.data_ptr()], dtype=torch.int64)
    sizes = torch.tensor([source.numel() * source.element_size()], dtype=torch.int64)
    with torch.npu.stream(stream):
        torch.ops._C_ascend.swap_blocks_batch(
            pointers_source,
            pointers_destination,
            sizes,
            DEVICE_TO_DEVICE,
        )
    stream.synchronize()


def _copy_in_child(spec: KVCacheStorageSpec, device_index: int, result_conn: Connection) -> None:
    stage = "initialize"
    engine = None
    registered = False
    try:
        torch.npu.set_device(device_index)
        importlib.import_module("vllm_ascend.vllm_ascend_C")

        from mooncake.engine import TransferEngine  # type: ignore[import-not-found]
        from vllm.utils.network_utils import get_ip

        engine = TransferEngine()
        initialize_result = engine.initialize(get_ip(), "P2PHANDSHAKE", "ascend", "")
        assert initialize_result == 0

        imported, imported_device, imported_size = TorchNPUIPCAdapter().import_storage(spec, device_index)
        assert imported_device == device_index
        assert imported_size == BUFFER_SIZE_BYTES

        stream = torch.npu.Stream()
        imported_view = imported[COPY_OFFSET_BYTES : COPY_OFFSET_BYTES + COPY_SIZE_BYTES]
        native = torch.empty_like(imported_view)

        stage = "torch copy imported-to-native"
        with torch.npu.stream(stream):
            native.copy_(imported_view)
        stream.synchronize()
        _assert_filled(native, INITIAL_VALUE, stage)

        stage = "raw aclrtMemcpyAsync before Mooncake registration"
        native.zero_()
        torch.npu.synchronize()
        _raw_d2d_copy(imported_view, native, stream)
        _assert_filled(native, INITIAL_VALUE, stage)

        stage = "Mooncake registration of imported storage"
        register_result = engine.register_memory(
            imported.data_ptr(),
            imported_size,
            f"npu:{device_index}",
        )
        assert register_result in (None, 0)
        registered = True

        stage = "raw aclrtMemcpyAsync after Mooncake registration"
        native.zero_()
        torch.npu.synchronize()
        _raw_d2d_copy(imported_view, native, stream)
        _assert_filled(native, INITIAL_VALUE, stage)

        stage = "raw aclrtMemcpyAsync native-to-imported after registration"
        native.fill_(UPDATED_VALUE)
        torch.npu.synchronize()
        _raw_d2d_copy(native, imported_view, stream)

        stage = "Mooncake unregistration of imported storage"
        unregister_result = engine.unregister_memory(imported.data_ptr())
        assert unregister_result in (None, 0)
        registered = False
        result_conn.send({"ok": True, "stage": stage})
    except BaseException:
        result_conn.send({"ok": False, "stage": stage, "traceback": traceback.format_exc()})
    finally:
        if registered and engine is not None:
            with suppress(BaseException):
                engine.unregister_memory(imported.data_ptr())
        result_conn.close()


@pytest.mark.skipif(torch.npu.device_count() < 1, reason="NPU IPC copy test requires an NPU")
def test_child_can_copy_to_and_from_parent_npu_ipc_mapping() -> None:
    _ensure_mooncake_version()
    device_index = 0
    torch.npu.set_device(device_index)
    source = torch.full(
        (BUFFER_SIZE_BYTES,),
        INITIAL_VALUE,
        dtype=torch.uint8,
        device=f"npu:{device_index}",
    )
    torch.npu.synchronize()
    spec = TorchNPUIPCAdapter().export_storage(source)

    context = mp.get_context("spawn")
    result_reader, result_writer = context.Pipe(duplex=False)
    child = context.Process(target=_copy_in_child, args=(spec, device_index, result_writer))
    child.start()
    result_writer.close()
    child.join(CHILD_TIMEOUT_SECONDS)
    if child.is_alive():
        child.terminate()
        child.join()
        pytest.fail(f"Child copy did not finish within {CHILD_TIMEOUT_SECONDS} seconds")

    result = result_reader.recv() if result_reader.poll() else None
    result_reader.close()
    assert child.exitcode == 0, f"Child exited with {child.exitcode}; result={result}"
    assert result is not None and result["ok"], result

    torch.npu.synchronize()
    updated = source[COPY_OFFSET_BYTES : COPY_OFFSET_BYTES + COPY_SIZE_BYTES]
    _assert_filled(updated, UPDATED_VALUE, "parent verification of child write")
    _assert_filled(source[:COPY_OFFSET_BYTES], INITIAL_VALUE, "parent prefix preservation")
    _assert_filled(source[COPY_OFFSET_BYTES + COPY_SIZE_BYTES :], INITIAL_VALUE, "parent suffix preservation")
