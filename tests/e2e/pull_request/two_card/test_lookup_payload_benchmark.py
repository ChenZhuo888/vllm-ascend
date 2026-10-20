# SPDX-License-Identifier: Apache-2.0
"""Temporary transport benchmark; removed from the PR after evidence collection."""

import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import LookupKeyClient
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

PRODUCTION_COMMIT = "a5ea57f60"
WARMUP_PAIRS = 80
SAMPLE_PAIRS = 1000
ROUNDS = 5
MICROSECONDS_PER_SECOND = 1_000_000
# The 312-hash case matches the historical A3 IPC experiment. The other
# cases exercise the RFC's 16K/64K prompts with 16-token request hashes.
SCENARIOS = ((16384, 16), (65536, 16), (39936, 128))


class LookupStub:
    """Keep Worker cost fixed so measurements isolate production RPC transport."""

    def __init__(self) -> None:
        self.hash_counts: Counter[tuple[LookupHashMode, int]] = Counter()

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int],
        use_layerwise: bool,
        hbm_hit_tokens: int,
        lookup_hash_mode: LookupHashMode,
    ) -> int:
        self.hash_counts[lookup_hash_mode, len(block_hashes)] += 1
        return token_len


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "p50": statistics.median(values),
        "p90": ordered[math.ceil(len(values) * 0.9) - 1],
        "p99": ordered[math.ceil(len(values) * 0.99) - 1],
        "mean_ci95_normal_low": mean - 1.96 * stderr,
        "mean_ci95_normal_high": mean + 1.96 * stderr,
    }


def encoded_sizes(hashes: list[BlockHash], mode: LookupHashMode) -> dict[str, int]:
    encoder = MsgpackEncoder()
    hash_bytes = sum(len(frame) for frame in encoder.encode([value.hex() for value in hashes]))
    group_bytes = sum(len(frame) for frame in encoder.encode([0]))
    mode_bytes = sum(len(frame) for frame in encoder.encode(mode.value))
    return {
        "hash_count": len(hashes),
        "encoded_hash_bytes": hash_bytes,
        # Two four-byte token offsets; excludes ZeroMQ transport framing.
        "request_frame_bytes": hash_bytes + group_bytes + mode_bytes + 8,
    }


def measure_scenario(token_len: int, hash_block_size: int) -> dict[str, Any]:
    """Run alternating Full/Suffix pairs with identical fixed Worker replies."""
    total_hashes = token_len // hash_block_size
    omitted_hashes = total_hashes * 3 // 4
    full_hashes = [cast(BlockHash, index.to_bytes(32, "big")) for index in range(total_hashes)]
    payloads = {
        LookupHashMode.FULL: full_hashes,
        LookupHashMode.SUFFIX: full_hashes[omitted_hashes:],
    }
    config = cast(
        VllmConfig,
        SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0),
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"lookup_rpc_port": get_open_port()}),
        ),
    )
    stub = LookupStub()
    server = LookupKeyServer(cast(KVPoolWorker, stub), config)
    client = LookupKeyClient(config)
    full_samples: list[float] = []
    suffix_samples: list[float] = []
    round_results: list[dict[str, Any]] = []
    try:
        for round_index in range(ROUNDS):
            samples: dict[LookupHashMode, list[float]] = {mode: [] for mode in LookupHashMode}
            for pair_index in range(WARMUP_PAIRS + SAMPLE_PAIRS):
                order = (LookupHashMode.FULL, LookupHashMode.SUFFIX)
                if (pair_index + round_index) % 2:
                    order = tuple(reversed(order))
                for mode in order:
                    # Match the old IPC timer: scheduler-side slicing is excluded.
                    started = time.perf_counter()
                    result = client.lookup(
                        token_len,
                        payloads[mode],
                        hbm_hit_tokens=omitted_hashes * hash_block_size,
                        lookup_hash_mode=mode,
                    )
                    elapsed = (time.perf_counter() - started) * MICROSECONDS_PER_SECOND
                    assert result == token_len, "Lookup failed or the transport returned the wrong response"
                    if pair_index >= WARMUP_PAIRS:
                        samples[mode].append(elapsed)
            full = samples[LookupHashMode.FULL]
            suffix = samples[LookupHashMode.SUFFIX]
            full_samples.extend(full)
            suffix_samples.extend(suffix)
            round_results.append(
                {
                    "round": round_index,
                    "full_us": distribution(full),
                    "suffix_us": distribution(suffix),
                    "mean_reduction_percent": (1 - statistics.fmean(suffix) / statistics.fmean(full)) * 100,
                }
            )
    finally:
        client.close()
        server.close()

    for mode, payload in payloads.items():
        assert stub.hash_counts[mode, len(payload)] == ROUNDS * (WARMUP_PAIRS + SAMPLE_PAIRS)
    assert len(stub.hash_counts) == 2
    sizes = {mode.value: encoded_sizes(payload, mode) for mode, payload in payloads.items()}
    assert sizes["suffix"]["encoded_hash_bytes"] < sizes["full"]["encoded_hash_bytes"]
    # Report timing observations without a flaky performance pass/fail threshold.
    return {
        "token_len": token_len,
        "hash_block_size": hash_block_size,
        "hbm_hit_tokens": omitted_hashes * hash_block_size,
        "sizes": sizes,
        "full_us": distribution(full_samples),
        "suffix_us": distribution(suffix_samples),
        "paired_saved_us": distribution([full - suffix for full, suffix in zip(full_samples, suffix_samples)]),
        "mean_reduction_percent": (1 - statistics.fmean(suffix_samples) / statistics.fmean(full_samples)) * 100,
        "rounds": round_results,
        "raw_full_us": full_samples,
        "raw_suffix_us": suffix_samples,
    }


def test_lookup_payload_benchmark(tmp_path: Path) -> None:
    """Collect current transport evidence without modifying production code."""
    subprocess.run(["git", "diff", "--exit-code", PRODUCTION_COMMIT, "--", "vllm_ascend"], check=True)
    report: dict[str, Any] = {
        "scope": "production Lookup RPC; same-process server thread; stub Worker; no model/backend",
        "timing": "client hex encoding through RPC reply; excludes scheduler suffix slicing",
        "production_commit": subprocess.check_output(["git", "rev-parse", PRODUCTION_COMMIT], text=True).strip(),
        "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "vllm_verified_commit": Path(".github/vllm-main-verified.commit").read_text().strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("vllm", "pyzmq", "msgspec")},
        "warmup_pairs_per_round": WARMUP_PAIRS,
        "sample_pairs_per_round": SAMPLE_PAIRS,
        "round_count": ROUNDS,
        "scenarios": [measure_scenario(token_len, unit) for token_len, unit in SCENARIOS],
    }
    # The reusable workflow archives selected-tests-* directories. Preserve
    # every raw pair there, while keeping the console summary small.
    evidence_dir = Path(os.environ.get("RUNNER_TEMP", str(tmp_path))) / "selected-tests-lookup-benchmark-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "lookup_payload_benchmark.json").write_text(json.dumps(report, indent=2))
    (evidence_dir / Path(__file__).name).write_text(Path(__file__).read_text())
    for scenario in report["scenarios"]:
        del scenario["raw_full_us"]
        del scenario["raw_suffix_us"]
    print("\nLOOKUP_PAYLOAD_BENCHMARK=" + json.dumps(report, sort_keys=True))
