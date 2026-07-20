"""
bench_expert_loading.py — Compare SSD read throughput for unpacked vs packed expert formats.

Tests two scenarios for a random subset of (layer, expert) pairs:
  UNPACKED: open 9 separate files, read each sequentially → concatenate
  PACKED:   open 1 combined file, read in one shot

Metrics reported:
  - Wall-clock time (cold/warm cache)
  - Throughput (GB/s)
  - Per-expert average latency (ms)

Usage:
    # Drop the page cache first for a cold-cache test (requires sudo):
    #   sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

    python bench_expert_loading.py \
        --unpacked model_weights/Qwen3-30B-A3B-AWQ_unpacked \
        --packed   model_weights/Qwen3-30B-A3B-AWQ_packed \
        --num-experts 128 --num-layers 48 \
        --sample 256 --seed 42 \
        [--cold]   # add --cold to attempt cache drop (needs sudo)
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import sys
import time
from pathlib import Path

MAGIC = b"EXPK"
NUM_TENSORS = 9
DESC_ENTRY_SIZE = 48
HEADER_SIZE = 8 + NUM_TENSORS * DESC_ENTRY_SIZE

TENSOR_NAMES = [
    "gate.qweight",
    "gate.scales",
    "gate.zeros",
    "up.qweight",
    "up.scales",
    "up.zeros",
    "down.qweight",
    "down.scales",
    "down.zeros",
]

UNPACKED_SUFFIXES = [
    ("gate", "qweight"), ("gate", "scales"), ("gate", "zeros"),
    ("up",   "qweight"), ("up",   "scales"), ("up",   "zeros"),
    ("down", "qweight"), ("down", "scales"), ("down", "zeros"),
]


# ──────────────────────────────────────────────────────────────
#  Low-level loaders
# ──────────────────────────────────────────────────────────────

def load_unpacked(src_dir: Path, layer: int, expert: int) -> bytes:
    """Read the 9 individual files and concatenate their bytes."""
    parts = []
    for proj, kind in UNPACKED_SUFFIXES:
        path = src_dir / f"layer_{layer}_expert_{expert}_{proj}.{kind}.bin"
        with open(path, "rb") as f:
            parts.append(f.read())
    return b"".join(parts)


def load_packed(packed_dir: Path, layer: int, expert: int) -> bytes:
    """Read the single packed file for one expert."""
    path = packed_dir / f"layer_{layer}_expert_{expert}.bin"
    with open(path, "rb") as f:
        return f.read()


def load_packed_payload_only(packed_dir: Path, layer: int, expert: int) -> bytes:
    """Read only the payload bytes (skip header) — matches what a real loader would do."""
    path = packed_dir / f"layer_{layer}_expert_{expert}.bin"
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        return f.read()


# ──────────────────────────────────────────────────────────────
#  Benchmark
# ──────────────────────────────────────────────────────────────

def drop_page_cache() -> bool:
    """Try to drop the Linux page cache.  Returns True on success."""
    import subprocess
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:
        print(f"  [warn] Could not drop page cache: {e}", file=sys.stderr)
        return False


def run_benchmark(
    unpacked_dir: Path,
    packed_dir: Path,
    pairs: list[tuple[int, int]],
    drop_cache: bool,
    label: str,
) -> dict:
    """Run both loaders over `pairs`, return timing/throughput results."""
    n = len(pairs)

    # ── UNPACKED ──────────────────────────────────────────────
    if drop_cache:
        print(f"  [{label}] Dropping page cache for UNPACKED pass…")
        drop_page_cache()

    t0 = time.perf_counter()
    total_bytes_unpacked = 0
    latencies_unpacked = []
    for layer, expert in pairs:
        t_start = time.perf_counter()
        data = load_unpacked(unpacked_dir, layer, expert)
        t_end = time.perf_counter()
        total_bytes_unpacked += len(data)
        latencies_unpacked.append((t_end - t_start) * 1e3)
    t1 = time.perf_counter()
    elapsed_unpacked = t1 - t0

    # ── PACKED ────────────────────────────────────────────────
    if drop_cache:
        print(f"  [{label}] Dropping page cache for PACKED pass…")
        drop_page_cache()

    t0 = time.perf_counter()
    total_bytes_packed = 0
    latencies_packed = []
    for layer, expert in pairs:
        t_start = time.perf_counter()
        data = load_packed_payload_only(packed_dir, layer, expert)
        t_end = time.perf_counter()
        total_bytes_packed += len(data)
        latencies_packed.append((t_end - t_start) * 1e3)
    t1 = time.perf_counter()
    elapsed_packed = t1 - t0

    gb_unpacked = total_bytes_unpacked / 1e9
    gb_packed   = total_bytes_packed   / 1e9

    return {
        "label": label,
        "n": n,
        "unpacked": {
            "total_s":   elapsed_unpacked,
            "bytes":     total_bytes_unpacked,
            "gb_s":      gb_unpacked / elapsed_unpacked,
            "lat_avg_ms": sum(latencies_unpacked) / n,
            "lat_p50_ms": sorted(latencies_unpacked)[n // 2],
            "lat_p95_ms": sorted(latencies_unpacked)[int(n * 0.95)],
        },
        "packed": {
            "total_s":   elapsed_packed,
            "bytes":     total_bytes_packed,
            "gb_s":      gb_packed / elapsed_packed,
            "lat_avg_ms": sum(latencies_packed) / n,
            "lat_p50_ms": sorted(latencies_packed)[n // 2],
            "lat_p95_ms": sorted(latencies_packed)[int(n * 0.95)],
        },
    }


def print_result(r: dict) -> None:
    u = r["unpacked"]
    p = r["packed"]
    speedup = u["total_s"] / p["total_s"]

    w = 58
    bar = "─" * w
    print(f"\n┌{bar}┐")
    print(f"│  {r['label']:^{w - 2}}│")
    print(f"│  {r['n']} experts sampled{' ' * (w - 2 - 18 - len(str(r['n'])))}│")
    print(f"├{bar}┤")
    print(f"│  {'Metric':<22} {'UNPACKED (9 files)':>16}   {'PACKED (1 file)':>14} │")
    print(f"├{bar}┤")

    def row(name, uval, pval, fmt=".3f"):
        uf = f"{uval:{fmt}}"
        pf = f"{pval:{fmt}}"
        print(f"│  {name:<22} {uf:>16}   {pf:>14} │")

    row("Total time (s)",     u["total_s"],    p["total_s"])
    row("Throughput (GB/s)",  u["gb_s"],       p["gb_s"])
    row("Avg latency (ms)",   u["lat_avg_ms"], p["lat_avg_ms"])
    row("P50 latency (ms)",   u["lat_p50_ms"], p["lat_p50_ms"])
    row("P95 latency (ms)",   u["lat_p95_ms"], p["lat_p95_ms"])

    print(f"├{bar}┤")
    direction = "faster" if speedup >= 1.0 else "slower"
    ratio = speedup if speedup >= 1.0 else 1.0 / speedup
    print(f"│  Packed is {ratio:.2f}× {direction} than unpacked{' ' * (w - 2 - 12 - len(f'{ratio:.2f}') - len(direction))}│")
    print(f"└{bar}┘")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main() -> int:
    pa = argparse.ArgumentParser(description="Benchmark unpacked vs packed expert loading from SSD.")
    pa.add_argument("--unpacked",    default="model_weights/Qwen3-30B-A3B-AWQ_unpacked")
    pa.add_argument("--packed",      default="model_weights/Qwen3-30B-A3B-AWQ_packed")
    pa.add_argument("--num-layers",  type=int, default=48)
    pa.add_argument("--num-experts", type=int, default=128)
    pa.add_argument("--sample",      type=int, default=256,
                    help="Number of (layer, expert) pairs to load per pass.")
    pa.add_argument("--seed",        type=int, default=42)
    pa.add_argument("--cold",        action="store_true",
                    help="Drop page cache before each pass (requires sudo).")
    pa.add_argument("--sequential",  action="store_true",
                    help="Also benchmark sequential (all experts in layer order) access pattern.")
    args = pa.parse_args()

    unpacked_dir = Path(args.unpacked)
    packed_dir   = Path(args.packed)

    for d, name in [(unpacked_dir, "unpacked"), (packed_dir, "packed")]:
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}", file=sys.stderr)
            return 1

    rng = random.Random(args.seed)
    all_pairs = [
        (layer, expert)
        for layer in range(args.num_layers)
        for expert in range(args.num_experts)
    ]

    # Random access pattern
    sample_random = rng.sample(all_pairs, min(args.sample, len(all_pairs)))

    print("=" * 60)
    print("  Expert Loading Benchmark: UNPACKED vs PACKED")
    print("=" * 60)
    print(f"  Unpacked dir : {unpacked_dir}")
    print(f"  Packed dir   : {packed_dir}")
    print(f"  Layers       : {args.num_layers}")
    print(f"  Experts/layer: {args.num_experts}")
    print(f"  Sample size  : {args.sample}")
    print(f"  Cache drop   : {'yes (sudo)' if args.cold else 'no (warm cache)'}")
    print()

    # ── Pass 1: warm cache, random access ─────────────────────
    print("Pass 1: WARM CACHE, RANDOM access")
    r1 = run_benchmark(unpacked_dir, packed_dir, sample_random, drop_cache=False, label="Warm cache / Random")
    print_result(r1)

    # ── Pass 2: warm cache, sequential access ─────────────────
    if args.sequential:
        sample_seq = all_pairs[:min(args.sample, len(all_pairs))]
        print("\nPass 2: WARM CACHE, SEQUENTIAL access")
        r2 = run_benchmark(unpacked_dir, packed_dir, sample_seq, drop_cache=False, label="Warm cache / Sequential")
        print_result(r2)

    # ── Pass 3: cold cache (optional) ─────────────────────────
    if args.cold:
        print("\nPass 3: COLD CACHE, RANDOM access (sudo required)")
        r3 = run_benchmark(unpacked_dir, packed_dir, sample_random, drop_cache=True, label="Cold cache / Random")
        print_result(r3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
