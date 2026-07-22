#!/usr/bin/env python3
"""
C++ backend microbenchmark via prewarm_experts().

Uses HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1 to disable parallel pread dispatches
*within* each expert load (see moe_expert_io.inl).  Compare against default
(parallel intra-expert reads).

Note: prewarm_experts() loads experts sequentially across slots and layers —
it does NOT test inter-expert parallelism.  Use microbench_raw_io.py for that.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


def _stats(latencies: Sequence[float]) -> Dict[str, float]:
    n = len(latencies)
    avg = statistics.mean(latencies)
    med = statistics.median(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0.0
    p95 = sorted(latencies)[min(int(0.95 * n), n - 1)]
    return {
        "n": n,
        "avg_ms": avg,
        "median_ms": med,
        "std_ms": std,
        "p95_ms": p95,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def build_model(model_name: str, cache_size: int, device: str, expert_weights_dir: str):
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent / "unified_llm_w4a16"))

    if model_name in ("qwen3", "qwen3_cached"):
        import importlib
        mod = importlib.import_module("qwen3_30B-A3B_w4a16_model")
        return mod.Qwen3_30BA3BW4A16Model(
            backend="cached",
            max_cached_experts_per_layer=cache_size,
            device=device,
            expert_weights_dir=expert_weights_dir,
        )
    if model_name in ("mixtral", "mixtral_cached"):
        from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model
        return Mixtral8x7BW4A16Model(
            backend="cached",
            max_cached_experts_per_layer=cache_size,
            device=device,
            expert_weights_dir=expert_weights_dir,
        )
    raise ValueError(f"Unknown model: {model_name}")


def is_packed_dir(bin_dir: str) -> bool:
    return os.path.exists(f"{bin_dir}/layer_0_expert_0.bin")


def run_prewarm_benchmark(
    *,
    model_name: str,
    bin_dir: str,
    cache_size: int,
    num_layers: int,
    sequential_intra: bool,
    num_rounds: int,
    warmup: int,
    device: str,
    verbose: bool,
) -> Dict:
    env_key = "HETEROPREDICT_SEQUENTIAL_EXPERT_IO"
    prev = os.environ.get(env_key)
    if sequential_intra:
        os.environ[env_key] = "1"
    else:
        os.environ.pop(env_key, None)

    try:
        model = build_model(model_name, cache_size, device, bin_dir)
        m = model.model

        # Untimed slot allocation
        m.prewarm_experts(cache_size)
        torch.cuda.synchronize()

        latencies: List[float] = []
        for r in range(warmup + num_rounds):
            m.reset_cache_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m.prewarm_experts(cache_size)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            if r >= warmup:
                latencies.append(ms)
            if verbose:
                hits, misses = m.get_cache_stats()
                print(f"  round {r+1}: {ms:.1f} ms  hits={hits} misses={misses}")

        n_experts = cache_size * num_layers
        st = _stats(latencies)
        fmt = "packed" if is_packed_dir(bin_dir) else "unpacked"
        return {
            "backend": "cpp_prewarm",
            "model": model_name,
            "format": fmt,
            "cache_size": cache_size,
            "num_layers": num_layers,
            "num_experts_per_round": n_experts,
            "sequential_intra": sequential_intra,
            "ms_per_expert": st["avg_ms"] / n_experts,
            **st,
        }
    finally:
        if prev is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = prev


def main() -> None:
    p = argparse.ArgumentParser(description="C++ prewarm_experts microbenchmark")
    p.add_argument("--model", default="qwen3")
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--cache-size", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rows: List[Dict] = []
    for seq in (False, True):
        label = "sequential_intra" if seq else "parallel_intra"
        print(f"\n[cpp_prewarm] {label}")
        row = run_prewarm_benchmark(
            model_name=args.model,
            bin_dir=args.bin_dir,
            cache_size=args.cache_size,
            num_layers=args.num_layers,
            sequential_intra=seq,
            num_rounds=args.num_rounds,
            warmup=args.warmup,
            device=args.device,
            verbose=args.verbose,
        )
        rows.append(row)
        print(f"  avg {row['avg_ms']:.1f} ms/round  ({row['ms_per_expert']:.2f} ms/expert)")

    if len(rows) == 2:
        seq, par = rows if rows[0]["sequential_intra"] else rows[::-1]
        speedup = seq["ms_per_expert"] / par["ms_per_expert"]
        print(f"\n  Intra-expert parallel speedup (SSD+H2D): {speedup:.2f}x")

    if args.out_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fmt = rows[0]["format"]
        out = Path(args.out_dir) / f"cpp_prewarm_{fmt}_{ts}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
