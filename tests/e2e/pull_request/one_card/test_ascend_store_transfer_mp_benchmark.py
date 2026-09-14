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

"""Find the concurrency crossover for subprocess Mooncake transfers.

The benchmark fills Mooncake with one deterministic serving workload, clears
only vLLM's local prefix cache, and then repeats the exact workload. It rejects
a run unless vLLM's external-cache counters prove that the measured requests
actually loaded their prefixes from Mooncake.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import regex as re
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import MooncakeLauncher, RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
MOONCAKE_PACKAGE = "mooncake-transfer-engine-npu"
MOONCAKE_VERSION = "0.3.12.post1"
MOONCAKE_INDEX = "https://mirrors.aliyun.com/pypi/simple/"

PREFIX_LENGTH = 768
SUFFIX_LENGTH = 128
OUTPUT_LENGTH = 64
NUM_PROMPTS = 96
MEASURED_ROUNDS = 3
CONCURRENCY_LEVELS = (1, 8, 16, 32, 64, 96)
MODE_SCHEDULE = (True, False)
DATASET_SEED = 20260914

MAX_MODEL_LENGTH = 1024
MAX_NUM_BATCHED_TOKENS = 32768
SERVER_START_TIMEOUT_SECONDS = 1200
BENCHMARK_TIMEOUT_SECONDS = 600
TRANSFER_TIMEOUT_SECONDS = 300
METRIC_UPDATE_TIMEOUT_SECONDS = 20
MINIMUM_EXTERNAL_HIT_RATE = 0.8

EXTERNAL_QUERY_METRIC = "vllm:external_prefix_cache_queries"
EXTERNAL_HIT_METRIC = "vllm:external_prefix_cache_hits"


def _ensure_mooncake_version() -> str:
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
    return actual


def _reset_local_cache(server: RemoteOpenAIServer) -> float:
    """Wait for pending stores, then drop only vLLM's local prefix cache."""
    started = time.perf_counter()
    deadline = time.monotonic() + TRANSFER_TIMEOUT_SECONDS
    while True:
        response = requests.post(
            server.url_for("reset_prefix_cache"),
            params={"reset_external": "false"},
            timeout=60,
        )
        response.raise_for_status()
        if response.json().get("success") is True:
            return time.perf_counter() - started
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for AscendStore stores to drain")
        time.sleep(1)


def _complete(server: RemoteOpenAIServer, prompt: list[int]) -> dict[str, Any]:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "return_token_ids": True,
        },
        timeout=TRANSFER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    assert result["usage"]["completion_tokens"] == 1
    assert len(result["choices"][0]["token_ids"]) == 1
    return result


def _prove_external_roundtrip(server: RemoteOpenAIServer, mode: str) -> None:
    """Fail before the benchmark if this server cannot store and load KV."""
    prompt = [10 + index % 1000 for index in range(PREFIX_LENGTH + 1)]
    cold = _complete(server, prompt)
    _reset_local_cache(server)
    loaded = _complete(server, prompt)
    cached_tokens = loaded["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens >= PREFIX_LENGTH, (
        f"{mode} Mooncake smoke missed the stored prefix: cached_tokens={cached_tokens}"
    )
    assert loaded["choices"][0]["token_ids"] == cold["choices"][0]["token_ids"]
    print(
        f"[ascend-store-mp-benchmark-smoke] mode={mode} cached_tokens={cached_tokens}",
        flush=True,
    )


def _metric_total(metrics_text: str, metric: str) -> float:
    pattern = re.compile(
        rf"^{re.escape(metric)}(?:_total)?(?:\{{[^}}]*\}})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$",
        re.MULTILINE,
    )
    return sum(float(value) for value in pattern.findall(metrics_text))


def _external_cache_counters(server: RemoteOpenAIServer) -> tuple[float, float]:
    response = requests.get(server.url_for("metrics"), timeout=30)
    response.raise_for_status()
    return (
        _metric_total(response.text, EXTERNAL_QUERY_METRIC),
        _metric_total(response.text, EXTERNAL_HIT_METRIC),
    )


def _wait_for_external_cache_counters(
    server: RemoteOpenAIServer,
    queries_before: float,
) -> tuple[float, float]:
    deadline = time.monotonic() + METRIC_UPDATE_TIMEOUT_SECONDS
    counters = _external_cache_counters(server)
    while counters[0] <= queries_before and time.monotonic() < deadline:
        time.sleep(1)
        counters = _external_cache_counters(server)
    return counters


def _run_benchmark(
    server: RemoteOpenAIServer,
    result_dir: Path,
    label: str,
    *,
    max_concurrency: int,
) -> dict[str, Any]:
    result_path = result_dir / f"{label}.json"
    log_path = result_dir / f"{label}.log"
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        server.url_root,
        "--endpoint",
        "/v1/completions",
        "--model",
        MODEL,
        "--tokenizer",
        MODEL,
        "--dataset-name",
        "prefix_repetition",
        "--prefix-repetition-prefix-len",
        str(PREFIX_LENGTH),
        "--prefix-repetition-suffix-len",
        str(SUFFIX_LENGTH),
        "--prefix-repetition-num-prefixes",
        str(NUM_PROMPTS),
        "--prefix-repetition-output-len",
        str(OUTPUT_LENGTH),
        "--num-prompts",
        str(NUM_PROMPTS),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(DATASET_SEED),
        "--temperature",
        "0",
        "--ignore-eos",
        "--disable-tqdm",
        "--ready-check-timeout-sec",
        "0",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_path.name,
        "--metadata",
        f"phase={label}",
    ]
    env = os.environ.copy()
    env.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "VLLM_USE_MODELSCOPE": "true",
        }
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=BENCHMARK_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log_path.write_text(output, encoding="utf-8")
        tail = "\n".join(output.splitlines()[-80:])
        raise TimeoutError(f"vllm bench serve timed out: {label}\n{tail}") from exc

    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"vllm bench serve failed with exit code {completed.returncode}: {label}\n{tail}")

    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    if result["completed"] != NUM_PROMPTS or result["failed"] != 0:
        raise RuntimeError(
            f"Incomplete benchmark {label}: completed={result['completed']} "
            f"failed={result['failed']} expected={NUM_PROMPTS}"
        )
    return result


def _run_hot_benchmark(
    server: RemoteOpenAIServer,
    result_dir: Path,
    label: str,
    *,
    max_concurrency: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    queries_before, hits_before = _external_cache_counters(server)
    result = _run_benchmark(
        server,
        result_dir,
        label,
        max_concurrency=max_concurrency,
    )
    queries_after, hits_after = _wait_for_external_cache_counters(server, queries_before)
    query_delta = queries_after - queries_before
    hit_delta = hits_after - hits_before
    if query_delta <= 0:
        raise RuntimeError(f"No external-cache queries were recorded for {label}")
    hit_rate = hit_delta / query_delta
    cache_evidence = {
        "external_query_tokens": query_delta,
        "external_hit_tokens": hit_delta,
        "external_hit_rate": hit_rate,
    }
    print(
        "[ascend-store-mp-benchmark-cache] " + json.dumps({"label": label, **cache_evidence}, sort_keys=True),
        flush=True,
    )
    if hit_rate < MINIMUM_EXTERNAL_HIT_RATE:
        raise RuntimeError(
            f"External hit rate for {label} was {hit_rate:.3f}, expected at least {MINIMUM_EXTERNAL_HIT_RATE:.3f}"
        )
    return result, cache_evidence


def _server_args(use_multiprocess: bool, server_port: int) -> list[str]:
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": "0",
            "use_layerwise": False,
            "use_multiprocess": use_multiprocess,
        },
    }
    return [
        "--port",
        str(server_port),
        "--max-model-len",
        str(MAX_MODEL_LENGTH),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(max(CONCURRENCY_LEVELS)),
        "--gpu-memory-utilization",
        "0.5",
        "--block-size",
        "128",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--enforce-eager",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]


def _compact_result(
    result: dict[str, Any],
    cache_evidence: dict[str, float],
    drain_seconds: float,
) -> dict[str, float | int]:
    duration = float(result["duration"])
    completed = int(result["completed"])
    compact: dict[str, float | int] = {
        "completed": completed,
        "duration": duration,
        "drain_seconds": drain_seconds,
        "request_throughput": float(result["request_throughput"]),
        "request_plus_drain_throughput": completed / (duration + drain_seconds),
        "total_token_throughput": float(result["total_token_throughput"]),
        "mean_ttft_ms": float(result["mean_ttft_ms"]),
        "p95_ttft_ms": float(result["p95_ttft_ms"]),
        "mean_tpot_ms": float(result["mean_tpot_ms"]),
        "p95_tpot_ms": float(result["p95_tpot_ms"]),
    }
    compact.update(cache_evidence)
    return compact


def _mooncake_config(path: Path, mooncake_port: int) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{mooncake_port}",
                "global_segment_size": 12 << 30,
                "local_buffer_size": 1 << 30,
            }
        ),
        encoding="utf-8",
    )


def _run_mode(artifact_dir: Path, use_multiprocess: bool) -> dict[str, Any]:
    mode = "multiprocess" if use_multiprocess else "inprocess"
    run_dir = artifact_dir / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    mooncake_port = get_open_port()
    mooncake_metrics_port = get_open_port()
    server_port = get_open_port()
    mooncake_config_path = run_dir / "mooncake.json"
    _mooncake_config(mooncake_config_path, mooncake_port)
    env = {
        "MOONCAKE_CONFIG_PATH": str(mooncake_config_path),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "1",
        "VLLM_USE_MODELSCOPE": "true",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    mode_result: dict[str, Any] = {"mode": mode, "rounds": {}}
    with (
        MooncakeLauncher(
            mooncake_port,
            mooncake_metrics_port,
            eviction_high_watermark_ratio=0.8,
        ),
        RemoteOpenAIServer(
            MODEL,
            _server_args(use_multiprocess, server_port),
            server_host="127.0.0.1",
            server_port=server_port,
            env_dict=env,
            auto_port=False,
            max_wait_seconds=SERVER_START_TIMEOUT_SECONDS,
        ) as server,
    ):
        _prove_external_roundtrip(server, mode)
        print(f"[ascend-store-mp-benchmark] filling Mooncake for mode={mode}", flush=True)
        _run_benchmark(
            server,
            run_dir,
            "cold-fill",
            max_concurrency=16,
        )
        mode_result["cold_fill_drain_seconds"] = _reset_local_cache(server)

        for concurrency in CONCURRENCY_LEVELS:
            rounds = []
            for round_index in range(1, MEASURED_ROUNDS + 1):
                label = f"hit-c{concurrency}-r{round_index}"
                result, cache_evidence = _run_hot_benchmark(
                    server,
                    run_dir,
                    label,
                    max_concurrency=concurrency,
                )
                drain_seconds = _reset_local_cache(server)
                rounds.append(_compact_result(result, cache_evidence, drain_seconds))
                mode_result["rounds"][str(concurrency)] = rounds
                (run_dir / "partial-summary.json").write_text(
                    json.dumps(mode_result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    return mode_result


def _summarize_rounds(
    rounds: list[dict[str, float | int]],
) -> dict[str, float]:
    fields = (
        "request_throughput",
        "request_plus_drain_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "mean_tpot_ms",
        "p95_tpot_ms",
        "external_hit_rate",
    )
    return {f"median_{field}": statistics.median(float(item[field]) for item in rounds) for field in fields}


@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=300)
def _wait_for_npu_release() -> None:
    pass


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="prefix_caching",
    parallel="",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=300)
def test_ascend_store_multiprocess_serving_benchmark(tmp_path: Path) -> None:
    actual_mooncake_version = _ensure_mooncake_version()
    artifact_dir = (
        Path(os.environ.get("RUNNER_TEMP", tmp_path))
        / "selected-tests-a2-1card"
        / "ascend-store-multiprocess-benchmark"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    experiment = {
        "model": MODEL,
        "backend": "mooncake",
        "mooncake_version": actual_mooncake_version,
        "fabric_memory": False,
        "tensor_parallel_size": 1,
        "prefix_length": PREFIX_LENGTH,
        "suffix_length": SUFFIX_LENGTH,
        "output_length": OUTPUT_LENGTH,
        "num_prompts_per_round": NUM_PROMPTS,
        "measured_rounds": MEASURED_ROUNDS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "dataset_seed": DATASET_SEED,
        "minimum_external_hit_rate": MINIMUM_EXTERNAL_HIT_RATE,
        "mode_schedule": ["multiprocess" if item else "inprocess" for item in MODE_SCHEDULE],
    }
    (artifact_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    mode_results: list[dict[str, Any]] = []
    try:
        for use_multiprocess in MODE_SCHEDULE:
            mode_results.append(_run_mode(artifact_dir, use_multiprocess))
            _wait_for_npu_release()
    finally:
        (artifact_dir / "modes.partial.json").write_text(
            json.dumps(mode_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    by_mode = {item["mode"]: item for item in mode_results}
    summary: dict[str, Any] = {"comparison": {}}
    for mode, mode_result in by_mode.items():
        summary[mode] = {
            concurrency: _summarize_rounds(rounds) for concurrency, rounds in mode_result["rounds"].items()
        }
    for concurrency in CONCURRENCY_LEVELS:
        key = str(concurrency)
        inprocess = summary["inprocess"][key]
        multiprocess = summary["multiprocess"][key]
        summary["comparison"][key] = {
            "request_throughput_change_pct": (
                multiprocess["median_request_throughput"] / inprocess["median_request_throughput"] - 1
            )
            * 100,
            "request_plus_drain_throughput_change_pct": (
                multiprocess["median_request_plus_drain_throughput"] / inprocess["median_request_plus_drain_throughput"]
                - 1
            )
            * 100,
            "mean_ttft_change_pct": (multiprocess["median_mean_ttft_ms"] / inprocess["median_mean_ttft_ms"] - 1) * 100,
            "mean_tpot_change_pct": (multiprocess["median_mean_tpot_ms"] / inprocess["median_mean_tpot_ms"] - 1) * 100,
        }

    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[ascend-store-mp-benchmark-final] {json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
