# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

"""Exercise a minimal TP8 Mooncake transfer through the subprocess path."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import MooncakeLauncher, RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.nightly.multi_node.scripts.utils import get_cur_ip, get_net_interface

MODEL = "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp"
# The model's hybrid KV groups have a 16,384-token transfer granularity.
# Keep one token in the trailing MTP block so the first full block is reusable.
PREFIX_LENGTH = 16_385
OUTPUT_LENGTH = 1
MAX_MODEL_LENGTH = 16_512
MAX_NUM_BATCHED_TOKENS = 16_384
TRANSFER_TIMEOUT_SECONDS = 300
MOONCAKE_PACKAGE = "mooncake-transfer-engine-npu"
MOONCAKE_VERSION = "0.3.12.post1"
MOONCAKE_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
MOONCAKE_REPOSITORY = "https://github.com/kvcache-ai/Mooncake.git"
LOCAL_COPY_FIX_COMMIT = "fa115fd6b76eebbc4e7faf94bff1773ad4640936"
LOCAL_COPY_FIX_SHA256 = "bd232ccdc0df95dc7987d812619a52eb328b99c7b0468388092aee04d2608c21"


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


def _install_local_copy_fix(build_parent: Path) -> None:
    """Backport Mooncake PR #4026 onto the pinned v0.3.12 plugin."""
    source_dir = build_parent / "mooncake-0.3.12-local-copy-fix"
    build_dir = build_parent / "mooncake-transfer-engine-build"
    prefix_dir = build_parent / "mooncake-build-prefix"

    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "libasio-dev",
            "libgflags-dev",
            "libgoogle-glog-dev",
            "libibverbs-dev",
            "libjsoncpp-dev",
            "librdmacm-dev",
            "libyaml-cpp-dev",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{MOONCAKE_VERSION}",
            "--recurse-submodules",
            "--shallow-submodules",
            MOONCAKE_REPOSITORY,
            str(source_dir),
        ],
        check=True,
    )

    patch_url = f"{MOONCAKE_REPOSITORY.removesuffix('.git')}/commit/{LOCAL_COPY_FIX_COMMIT}.patch"
    response = requests.get(patch_url, timeout=60)
    response.raise_for_status()
    patch = response.content
    actual_hash = hashlib.sha256(patch).hexdigest()
    if actual_hash != LOCAL_COPY_FIX_SHA256:
        raise RuntimeError(f"Unexpected Mooncake PR #4026 patch SHA256: {actual_hash}")
    subprocess.run(["git", "apply", "-"], cwd=source_dir, input=patch, check=True)

    jobs = str(min(os.cpu_count() or 1, 16))
    yalanting_source = source_dir / "extern" / "yalantinglibs"
    yalanting_build = build_parent / "yalantinglibs-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(yalanting_source),
            "-B",
            str(yalanting_build),
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DBUILD_UNIT_TESTS=OFF",
            f"-DCMAKE_INSTALL_PREFIX={prefix_dir}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(yalanting_build), "-j", jobs], check=True)
    subprocess.run(["cmake", "--install", str(yalanting_build)], check=True)

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir / "mooncake-transfer-engine"),
            "-B",
            str(build_dir),
            "-DUSE_ASCEND_DIRECT=ON",
            "-DUSE_TCP=OFF",
            "-DUSE_HTTP=OFF",
            "-DWITH_METRICS=ON",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_UNIT_TESTS=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DENABLE_DEBUG_SYMBOLS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={prefix_dir}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "ascend_transport", "-j", jobs],
        check=True,
    )

    plugin = build_dir / "src" / "transport" / "ascend_transport" / "ascend_transport.so"
    distribution = metadata.distribution(MOONCAKE_PACKAGE)
    installed_plugins = [
        distribution.locate_file(path) for path in distribution.files or () if path.name == "ascend_transport.so"
    ]
    if not plugin.is_file() or len(installed_plugins) != 1:
        raise RuntimeError(f"Cannot replace Mooncake Ascend plugin: built={plugin}, installed={installed_plugins}")
    shutil.copy2(plugin, str(installed_plugins[0]))
    print(
        f"[ascend-store-mp-smoke] installed Mooncake {MOONCAKE_VERSION} with local-copy fix {LOCAL_COPY_FIX_COMMIT}",
        flush=True,
    )


def _wait_for_local_cache_reset(server: RemoteOpenAIServer) -> None:
    deadline = time.monotonic() + TRANSFER_TIMEOUT_SECONDS
    while True:
        response = requests.post(
            server.url_for("reset_prefix_cache"),
            params={"reset_external": "false"},
            timeout=60,
        )
        response.raise_for_status()
        if response.json().get("success") is True:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the subprocess store to finish")
        time.sleep(1)


def _complete(server: RemoteOpenAIServer, prompt: list[int]) -> dict:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": OUTPUT_LENGTH,
            "ignore_eos": True,
            "temperature": 0,
            "return_token_ids": True,
        },
        timeout=TRANSFER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    assert result["usage"]["completion_tokens"] == OUTPUT_LENGTH
    assert len(result["choices"][0]["token_ids"]) == OUTPUT_LENGTH
    return result


def _server_args(server_port: int) -> list[str]:
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": "0",
            "use_layerwise": False,
            "use_multiprocess": True,
        },
    }
    return [
        "--tensor-parallel-size",
        "8",
        "--distributed-executor-backend",
        "mp",
        "--port",
        str(server_port),
        "--max-model-len",
        str(MAX_MODEL_LENGTH),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--block-size",
        "128",
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--quantization",
        "ascend",
        "--tokenizer-mode",
        "deepseek_v4",
        "--safetensors-load-strategy",
        "prefetch",
        "--enable-prompt-tokens-details",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=600)
def test_ascend_store_multiprocess_mooncake_tp8(tmp_path) -> None:
    _ensure_mooncake_version()
    _install_local_copy_fix(tmp_path)
    mooncake_port = get_open_port()
    mooncake_metrics_port = get_open_port()
    server_port = get_open_port()
    local_ip = get_cur_ip()
    nic_name = get_net_interface(local_ip)
    mooncake_config_path = tmp_path / "mooncake.json"
    mooncake_config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{mooncake_port}",
                "global_segment_size": 1 << 30,
                "local_buffer_size": 1 << 30,
            }
        ),
        encoding="utf-8",
    )
    env = {
        "MOONCAKE_CONFIG_PATH": str(mooncake_config_path),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
        "HCCL_INTRA_ROCE_ENABLE": "1",
        "HCCL_IF_IP": local_ip,
        "HCCL_SOCKET_IFNAME": nic_name,
        "GLOO_SOCKET_IFNAME": nic_name,
        "TP_SOCKET_IFNAME": nic_name,
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    prompt = [10 + index % 1000 for index in range(PREFIX_LENGTH)]

    with (
        MooncakeLauncher(mooncake_port, mooncake_metrics_port),
        RemoteOpenAIServer(
            MODEL,
            _server_args(server_port),
            server_host="127.0.0.1",
            server_port=server_port,
            env_dict=env,
            auto_port=False,
            max_wait_seconds=1800,
        ) as server,
    ):
        print("[ascend-store-mp-smoke] cold request", flush=True)
        cold = _complete(server, prompt)
        print("[ascend-store-mp-smoke] wait for store and clear local cache", flush=True)
        _wait_for_local_cache_reset(server)
        print("[ascend-store-mp-smoke] load request", flush=True)
        loaded = _complete(server, prompt)

        cached_tokens = loaded["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached_tokens > 0, "Mooncake subprocess load missed the stored prefix"
        assert loaded["choices"][0]["token_ids"] == cold["choices"][0]["token_ids"]
        print(f"[ascend-store-mp-smoke] passed with cached_tokens={cached_tokens}", flush=True)
