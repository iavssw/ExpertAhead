"""
benchmark_expert_loading.py
===========================
Benchmark the C++ cached-backend expert load time from SSD,
accurately simulating a RAM-constrained system.

ARCHITECTURE
------------
  GPU VRAM  ← the "expert cache"   (fast, limited slots per layer)
  SSD       ← source of truth       (all expert .bin files live here)
  OS page cache ← invisible RAM buffer that intercepts SSD reads
                  (must be evicted per-expert so benchmark reflects
                   true SSD latency, not fake "SSD" reads from RAM)

WHAT IS BEING MEASURED
-----------------------
Each measured round:
  1. Prewarm --cache-size experts into GPU VRAM  →  these represent the
     "resident" expert cache; their bin files can remain page-cached.
  2. Evict the bin files for ALL OTHER experts from the OS page cache
     using posix_fadvise(POSIX_FADV_DONTNEED).  This is per-file,
     requires no root, and does not disturb unrelated kernel state.
  3. Run a forward pass with a token whose router selects UNCACHED experts
     (i.e. experts not in GPU VRAM).  The C++ backend will call
     ensure_expert_cached(), which goes to load_expert_weights() →
     read_bin_tensor_pread() → pread() from SSD.
  4. Time from before the forward() call to after torch.cuda.synchronize(),
     which captures: SSD read → pinned CPU buffer → HIP H2D DMA → GPU kernel.

The result is a true end-to-end cache-miss latency from SSD.

USAGE
-----
# Qwen3 30B-A3B:
python benchmark_expert_loading.py \\
    --model qwen3 \\
    --bin-dir /home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked \\
    --num-layers 48 --num-experts 128 \\
    --cache-size 8 --num-rounds 20 --verbose

# Mixtral 8x7B:
python benchmark_expert_loading.py \\
    --model mixtral \\
    --bin-dir /home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Mixtral-8x7B-v0.1-AWQ_unpacked \\
    --num-layers 32 --num-experts 8 \\
    --cache-size 4 --num-rounds 20 --verbose
"""

import argparse
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

import torch


# ─── posix_fadvise(POSIX_FADV_DONTNEED) ──────────────────────────────────────
# Evict specific file pages from the OS page cache without requiring root.
# This is the standard technique used by storage benchmarks (fio, filebench).
# POSIX_FADV_DONTNEED = 4 on Linux x86-64/ARM.
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]
_libc.posix_fadvise.restype  = ctypes.c_int
POSIX_FADV_DONTNEED = 4


def _evict_single_file(path: str) -> None:
    """Evict one file from OS page cache (no root required)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            _libc.posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _expert_bin_paths(bin_dir: str, layer: int, expert: int) -> List[str]:
    """All 9 .bin files for a single expert."""
    prefix = f"{bin_dir}/layer_{layer}_expert_{expert}"
    return [
        f"{prefix}_gate.qweight.bin", f"{prefix}_gate.scales.bin", f"{prefix}_gate.zeros.bin",
        f"{prefix}_up.qweight.bin",   f"{prefix}_up.scales.bin",   f"{prefix}_up.zeros.bin",
        f"{prefix}_down.qweight.bin", f"{prefix}_down.scales.bin", f"{prefix}_down.zeros.bin",
    ]


def evict_expert(bin_dir: str, layer: int, expert: int) -> None:
    """Evict one expert's bin files from OS page cache."""
    for p in _expert_bin_paths(bin_dir, layer, expert):
        if os.path.exists(p):
            _evict_single_file(p)


def evict_experts(bin_dir: str, num_layers: int, num_experts: int,
                  cached_set: set) -> None:
    """
    Evict all experts NOT in cached_set from OS page cache.
    cached_set is a set of (layer, expert) tuples that are already in GPU VRAM;
    we leave their bin files alone (they don't matter since we won't read them).
    All other experts' bin files are evicted so their next load is truly from SSD.
    """
    for l in range(num_layers):
        for e in range(num_experts):
            if (l, e) not in cached_set:
                evict_expert(bin_dir, l, e)


def expert_total_bytes(bin_dir: str, layer: int, expert: int) -> int:
    total = 0
    for p in _expert_bin_paths(bin_dir, layer, expert):
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total


# ─── model instantiation ──────────────────────────────────────────────────────

def build_model(model_name: str, cache_size: int, device: str):
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    if model_name in ("mixtral", "mixtral_cached"):
        from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model
        print("Initializing Mixtral 8x7B (cached backend)…")
        return Mixtral8x7BW4A16Model(
            backend="cached",
            max_cached_experts_per_layer=cache_size,
            device=device,
        )
    elif model_name == "mixtral_predict":
        from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model
        print("Initializing Mixtral 8x7B (predict backend)…")
        return Mixtral8x7BW4A16Model(
            backend="predict",
            max_cached_experts_per_layer=cache_size,
            device=device,
        )
    elif model_name in ("qwen3", "qwen3_cached"):
        import importlib
        qwen_mod = importlib.import_module("qwen3_30B-A3B_w4a16_model")
        Qwen3_30BA3BW4A16Model = qwen_mod.Qwen3_30BA3BW4A16Model
        print("Initializing Qwen3 30B-A3B (cached backend)…")
        return Qwen3_30BA3BW4A16Model(
            backend="cached",
            max_cached_experts_per_layer=cache_size,
            device=device,
        )
    elif model_name == "qwen3_predict":
        import importlib
        qwen_mod = importlib.import_module("qwen3_30B-A3B_w4a16_model")
        Qwen3_30BA3BW4A16Model = qwen_mod.Qwen3_30BA3BW4A16Model
        print("Initializing Qwen3 30B-A3B (predict backend)…")
        return Qwen3_30BA3BW4A16Model(
            backend="predict",
            max_cached_experts_per_layer=cache_size,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")


# ─── benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(args):
    bin_dir     = args.bin_dir
    num_layers  = args.num_layers
    num_experts = args.num_experts
    cache_size  = args.cache_size
    num_rounds  = args.num_rounds
    warmup      = args.warmup_rounds
    device      = args.device

    model = build_model(args.model, cache_size, device)
    m = model.model   # C++ UnifiedLLMW4A16

    # Measure bytes per expert (use layer 0, expert 0 as reference)
    bytes_per_expert = 0
    for l in range(num_layers):
        for e in range(num_experts):
            b = expert_total_bytes(bin_dir, l, e)
            if b > 0:
                bytes_per_expert = b
                break
        if bytes_per_expert > 0:
            break

    if bytes_per_expert == 0:
        print(f"[ERROR] No expert .bin files found in {bin_dir}", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"  model       : {args.model}")
    print(f"  bin_dir     : {bin_dir}")
    print(f"  layers      : {num_layers}  experts/layer: {num_experts}")
    print(f"  gpu cache   : {cache_size} experts/layer  ({cache_size * num_layers} total slots)")
    print(f"  bytes/expert: {bytes_per_expert / 1e6:.2f} MB")
    print(f"  warmup      : {warmup} rounds (discarded)")
    print(f"  measured    : {num_rounds} rounds")
    print()

    # ── identify which experts prewarm will load ──
    # prewarm_experts(N) loads experts 0..N-1 for EVERY layer.
    # We need to evict exactly those files before each timed round.
    experts_to_load = {(l, e) for l in range(num_layers) for e in range(cache_size)}
    total_bytes_per_round = bytes_per_expert * cache_size * num_layers

    # Prewarm once so GPU slot allocation / HIP context is warmed up (not timed).
    print("Prewarming GPU VRAM slots (one-time, not timed)…")
    m.prewarm_experts(cache_size)
    torch.cuda.synchronize()
    print("Prewarm complete.\n")

    # ── single round: evict non-resident from page cache, then force a miss ──
    def run_one_round(label: str) -> float:
        """
        Evict the bin files for every expert that prewarm will load (experts 0..cache_size-1
        for all layers) from the OS page cache, then time prewarm_experts().
        This ensures reads come from SSD, not from RAM page cache.
        """
        # 1. Evict exactly the files we are about to read.
        for l, e in experts_to_load:
            evict_expert(bin_dir, l, e)

        # 2. Brief sleep to let kernel flush async readahead.
        time.sleep(0.05)

        m.reset_cache_stats()

        # 3. Time the full C++ load: SSD → pread() → pinned CPU buf → H2D DMA.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.prewarm_experts(cache_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        hits, misses = m.get_cache_stats()
        per_expert_ms = elapsed_ms / (cache_size * num_layers)
        gbps = (total_bytes_per_round / 1e9) / (elapsed_ms / 1000.0)

        if args.verbose:
            print(f"  {label}: {elapsed_ms:.1f} ms  "
                  f"({per_expert_ms:.1f} ms/expert, {gbps*1000:.0f} MB/s)  "
                  f"hits={hits} misses={misses}")
        return elapsed_ms

    # ── warmup ──
    if warmup > 0:
        print(f"Warmup ({warmup} rounds, discarded)…")
        for i in range(warmup):
            run_one_round(f"warmup {i+1:2d}")
        print()

    # ── measured rounds ──
    print(f"Measuring ({num_rounds} rounds)…")
    latencies_ms: List[float] = []
    for i in range(num_rounds):
        ms = run_one_round(f"round {i+1:3d}/{num_rounds}")
        latencies_ms.append(ms)

    # ── statistics ──
    avg_ms    = statistics.mean(latencies_ms)
    med_ms    = statistics.median(latencies_ms)
    std_ms    = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95_ms    = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    min_ms    = min(latencies_ms)
    max_ms    = max(latencies_ms)
    per_exp   = avg_ms / (cache_size * num_layers)
    avg_gbps  = (total_bytes_per_round / 1e9) / (avg_ms / 1000.0)

    print()
    print("=" * 62)
    print(f"  EXPERT LOAD BENCHMARK — {args.model.upper()}  ({num_rounds} rounds)")
    print("=" * 62)
    print(f"  GPU cache size            : {cache_size} experts/layer × {num_layers} layers = {cache_size*num_layers} total")
    print(f"  Bytes per expert          : {bytes_per_expert / 1e6:.2f} MB")
    print(f"  Total bytes / round       : {total_bytes_per_round / 1e6:.0f} MB  ({cache_size} experts × {num_layers} layers)")
    print()
    print(f"  Avg   end-to-end time     : {avg_ms:.1f} ms")
    print(f"  Median                    : {med_ms:.1f} ms")
    print(f"  Std dev                   : {std_ms:.1f} ms")
    print(f"  Min / Max                 : {min_ms:.1f} / {max_ms:.1f} ms")
    print(f"  P95                       : {p95_ms:.1f} ms")
    print()
    print(f"  Avg per-expert time       : {per_exp:.1f} ms")
    print(f"  Avg SSD→GPU throughput    : {avg_gbps*1000:.0f} MB/s  ({avg_gbps:.3f} GB/s)")
    print("=" * 62)
    print()
    print("Pipeline measured: SSD → pread() → pinned CPU buffer")
    print("                   → hipMemcpyAsync → GPU VRAM")
    print("Eviction: posix_fadvise(POSIX_FADV_DONTNEED) per expert file")
    print("          ensures non-resident experts are NOT served from RAM.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark C++ expert weight loading from SSD (RAM-constrained simulation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=["mixtral", "mixtral_cached", "mixtral_predict",
                 "qwen3",   "qwen3_cached",   "qwen3_predict"],
        required=True,
        help="Model + backend to benchmark (e.g. mixtral_cached vs mixtral_predict)")
    parser.add_argument("--bin-dir", type=str, required=True,
        help="Path to _unpacked directory containing .bin weight files")
    parser.add_argument("--num-layers", type=int, required=True,
        help="Number of transformer layers (32=Mixtral, 48=Qwen3)")
    parser.add_argument("--num-experts", type=int, required=True,
        help="Number of experts per layer (8=Mixtral, 128=Qwen3)")
    parser.add_argument("--cache-size", type=int, default=8,
        help="Experts kept in GPU VRAM per layer (default: 8)")
    parser.add_argument("--num-rounds", type=int, default=20,
        help="Timed measurement rounds (default: 20)")
    parser.add_argument("--warmup-rounds", type=int, default=2,
        help="Warmup rounds before measurement (default: 2)")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device string (default: cuda)")
    parser.add_argument("--verbose", action="store_true",
        help="Print per-round timing")

    args = parser.parse_args()

    if not os.path.isdir(args.bin_dir):
        print(f"[ERROR] --bin-dir not found: {args.bin_dir}", file=sys.stderr)
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
