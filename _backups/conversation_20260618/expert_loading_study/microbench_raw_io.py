#!/usr/bin/env python3
"""
Pure SSD microbenchmark — no GPU, no C++ extension.

Answers: is expert-load latency dominated by (a) file layout / packing or
(b) how many concurrent O_DIRECT pread requests we issue?

Scenarios
---------
1. intra_expert   — one expert, sequential vs parallel tensor reads
2. inter_expert   — N experts, sequential vs parallel expert loads
                    (each expert uses parallel intra reads by default)
3. factorial      — 2×2 grid: {packed, unpacked} × {sequential, parallel} intra reads

All reads use O_DIRECT (same as the C++ backend).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from io_common import (
    expert_paths,
    expert_total_bytes,
    is_packed_dir,
    load_experts,
    load_one_expert,
    timed_load,
)


def _stats(latencies: Sequence[float]) -> Dict[str, float]:
    n = len(latencies)
    if n == 0:
        return {}
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


def _pick_expert_specs(
    bin_dir: str,
    num_layers: int,
    num_experts: int,
    count: int,
    seed: int,
    layer: int | None,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    if layer is not None:
        return [(layer, rng.randrange(num_experts)) for _ in range(count)]
    return [(rng.randrange(num_layers), rng.randrange(num_experts)) for _ in range(count)]


def run_intra_expert(
    bin_dir: str,
    layer: int,
    expert: int,
    num_experts: int,
    num_rounds: int,
    warmup: int,
    seed: int,
) -> List[Dict]:
    nbytes = expert_total_bytes(bin_dir, layer, expert)
    fmt = expert_paths(bin_dir, layer, expert).format
    rows: List[Dict] = []
    rng = random.Random(seed)

    for parallel in (False, True):
        latencies: List[float] = []
        for r in range(warmup + num_rounds):
            # Rotate expert id each round to avoid SSD/controller caching the same LBA.
            eid = (expert + r) % num_experts
            ms, _ = timed_load(
                lambda p=parallel, e=eid: load_one_expert(bin_dir, layer, e, parallel_intra=p)
            )
            if r >= warmup:
                latencies.append(ms)

        st = _stats(latencies)
        rows.append({
            "scenario": "intra_expert",
            "format": fmt,
            "layer": layer,
            "expert": expert,
            "num_experts": 1,
            "parallel_intra": parallel,
            "parallel_inter": False,
            "bytes": nbytes,
            "ms_per_expert": st["avg_ms"],
            **st,
        })
    return rows


def run_inter_expert(
    bin_dir: str,
    specs: List[Tuple[int, int]],
    num_rounds: int,
    warmup: int,
    parallel_intra: bool,
    seed: int,
) -> List[Dict]:
    n = len(specs)
    nbytes_one = expert_total_bytes(bin_dir, specs[0][0], specs[0][1])
    fmt = expert_paths(bin_dir, specs[0][0], specs[0][1]).format
    rows: List[Dict] = []

    for parallel_inter in (False, True):
        latencies: List[float] = []
        for r in range(warmup + num_rounds):
            layer = specs[0][0]
            round_specs = [(layer, (specs[0][1] + r * n + i) % 128) for i in range(n)]
            ms, _ = timed_load(
                lambda rs=round_specs, p=parallel_inter: load_experts(
                    bin_dir,
                    rs,
                    parallel_intra=parallel_intra,
                    parallel_inter=p,
                    max_workers=n,
                )
            )
            if r >= warmup:
                latencies.append(ms)

        st = _stats(latencies)
        rows.append({
            "scenario": "inter_expert",
            "format": fmt,
            "layer": specs[0][0],
            "expert": specs[0][1],
            "num_experts": n,
            "parallel_intra": parallel_intra,
            "parallel_inter": parallel_inter,
            "bytes": nbytes_one * n,
            "ms_per_expert": st["avg_ms"] / n,
            **st,
        })
    return rows


def run_factorial(
    bin_dir: str,
    layer: int,
    expert: int,
    num_rounds: int,
    warmup: int,
) -> List[Dict]:
    nbytes = expert_total_bytes(bin_dir, layer, expert)
    fmt = expert_paths(bin_dir, layer, expert).format
    rows: List[Dict] = []

    for parallel_intra in (False, True):
        latencies: List[float] = []
        for r in range(warmup + num_rounds):
            ms, _ = timed_load(
                lambda p=parallel_intra: load_one_expert(bin_dir, layer, expert, parallel_intra=p)
            )
            if r >= warmup:
                latencies.append(ms)
        st = _stats(latencies)
        rows.append({
            "scenario": "factorial",
            "format": fmt,
            "layer": layer,
            "expert": expert,
            "num_experts": 1,
            "parallel_intra": parallel_intra,
            "parallel_inter": False,
            "bytes": nbytes,
            "ms_per_expert": st["avg_ms"],
            **st,
        })
    return rows


def run_sweep(
    bin_dir: str,
    num_layers: int,
    num_experts: int,
    expert_counts: Sequence[int],
    num_rounds: int,
    warmup: int,
    seed: int,
    layer: int | None,
) -> List[Dict]:
    all_rows: List[Dict] = []
    ref_layer = layer if layer is not None else 0
    ref_expert = 0

    print(f"\n[intra_expert] layer={ref_layer} expert={ref_expert}")
    all_rows.extend(
        run_intra_expert(bin_dir, ref_layer, ref_expert, num_experts, num_rounds, warmup, seed)
    )

    for n in expert_counts:
        specs = _pick_expert_specs(bin_dir, num_layers, num_experts, n, seed + n, layer)
        print(f"\n[inter_expert] N={n}  (parallel intra reads ON)")
        all_rows.extend(
            run_inter_expert(bin_dir, specs, num_rounds, warmup, parallel_intra=True, seed=seed + n)
        )

    return all_rows


def print_summary(rows: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print("  RAW SSD MICROBENCHMARK SUMMARY")
    print("=" * 72)

    intra = [r for r in rows if r["scenario"] == "intra_expert"]
    if len(intra) == 2:
        seq, par = sorted(intra, key=lambda r: r["parallel_intra"])
        speedup = seq["ms_per_expert"] / par["ms_per_expert"]
        print(f"\n  Intra-expert ({seq['format']}):")
        print(f"    sequential tensors : {seq['ms_per_expert']:.2f} ms/expert")
        print(f"    parallel tensors   : {par['ms_per_expert']:.2f} ms/expert")
        print(f"    speedup            : {speedup:.2f}x")

    by_n: Dict[int, List[Dict]] = {}
    for r in rows:
        if r["scenario"] == "inter_expert":
            by_n.setdefault(r["num_experts"], []).append(r)

    for n in sorted(by_n):
        seq, par = sorted(by_n[n], key=lambda r: r["parallel_inter"])
        speedup = seq["avg_ms"] / par["avg_ms"]
        print(f"\n  Inter-expert N={n} ({seq['format']}, parallel intra=ON):")
        print(f"    sequential experts : {seq['avg_ms']:.1f} ms total  ({seq['ms_per_expert']:.2f} ms/expert)")
        print(f"    parallel experts   : {par['avg_ms']:.1f} ms total  ({par['ms_per_expert']:.2f} ms/expert)")
        print(f"    speedup            : {speedup:.2f}x")

    print("\n  Interpretation:")
    print("    • Large intra-expert speedup → SSD benefits from parallel preads within one expert.")
    print("    • packed ≈ unpacked → file packing is NOT the bottleneck.")
    print("    • Large inter-expert speedup → backend should load multiple experts in parallel.")
    print("    • inter speedup ≈ 1x     → SSD/controller is saturated; packing may help more.")
    print("=" * 72)


def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "format", "layer", "expert", "num_experts",
        "parallel_intra", "parallel_inter", "bytes", "ms_per_expert",
        "n", "avg_ms", "median_ms", "std_ms", "p95_ms", "min_ms", "max_ms",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Raw SSD expert-loading microbenchmark")
    p.add_argument("--bin-dir", type=str, required=True)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--layer", type=int, default=None, help="Fix layer index (default: random)")
    p.add_argument("--expert-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    if not Path(args.bin_dir).is_dir():
        sys.exit(f"--bin-dir not found: {args.bin_dir}")

    fmt = "packed" if is_packed_dir(args.bin_dir) else "unpacked"
    print(f"Format: {fmt}  dir={args.bin_dir}")

    rows = run_sweep(
        args.bin_dir,
        args.num_layers,
        args.num_experts,
        args.expert_counts,
        args.num_rounds,
        args.warmup,
        args.seed,
        args.layer,
    )
    print_summary(rows)

    if args.out_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.out_dir) / f"raw_io_{fmt}_{ts}.csv"
        write_csv(rows, out)
        meta = {
            "bin_dir": args.bin_dir,
            "format": fmt,
            "timestamp": ts,
            "expert_counts": args.expert_counts,
            "num_rounds": args.num_rounds,
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
