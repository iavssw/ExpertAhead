"""
benchmark_expert_loading.py
===========================
Benchmark three components of MoE expert loading latency on an edge AMD Ryzen SoC:

  Mode 1: ssd     — SSD → O_DIRECT pread() → pinned CPU buffer → H2D DMA
                    The C++ backend always opens weight files with O_DIRECT, which
                    bypasses the OS page cache entirely. No eviction is needed —
                    every load goes straight to the SSD controller.

  Mode 2: ram     — System RAM (DRAM) → H2D DMA
                    Because the C++ backend uses O_DIRECT there is no "page cache
                    warm" path through prewarm_experts(). This mode instead times
                    a plain buffered read() (no O_DIRECT) of the same bytes into a
                    CPU buffer after first warming the page cache, isolating raw
                    DRAM bandwidth. The H2D DMA component is measured separately
                    via a pinned-memory → GPU cudaMemcpy of the same size.

  Mode 3: compute — Pure GPU compute for one expert FFN (weights already in VRAM)

  Mode 4: all     — Run all three and print a full breakdown table

Latency decomposition:
  SSD total   = SSD read (O_DIRECT) + H2D DMA + compute
  RAM total   = DRAM read (buffered) + H2D DMA + compute
  H2D DMA     = measured directly (pinned CPU → GPU memcpy)
  SSD read    = ssd_total - h2d - compute
  DRAM read   = ram_total - h2d - compute

EXPERT DIMENSIONS (Qwen3-30B-A3B)
-----------------------------------
  hidden_size        = 2048
  intermediate_size  = 2048  (A3B variant — much smaller than full 30B)
  active_k           = 4 experts/token/layer
  num_moe_layers     = 48
  ⇒ 4 × 48 = 192 expert invocations per decode token

USAGE
-----
  # Unpacked format (9 files/expert, O_DIRECT):
  python benchmark_expert_loading.py \\
      --model qwen3 \\
      --bin-dir py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked \\
      --num-layers 48 --num-experts 128 \\
      --cache-size 8 --num-rounds 20 --mode all --verbose

  # Packed format (1 file/expert, auto-detected from presence of layer_0_expert_0.bin):
  python benchmark_expert_loading.py \\
      --model qwen3 \\
      --bin-dir py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed \\
      --num-layers 48 --num-experts 128 \\
      --cache-size 8 --num-rounds 20 --mode ssd --verbose

  # SSD only:
  python benchmark_expert_loading.py --model qwen3 ... --mode ssd

  # RAM only (no page eviction):
  python benchmark_expert_loading.py --model qwen3 ... --mode ram

  # Compute only (GPU GEMM with correct expert shape):
  python benchmark_expert_loading.py --model qwen3 ... --mode compute \\
      --hidden-size 2048 --intermediate-size 2048 --active-k 4
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch


# NOTE: posix_fadvise / page-cache eviction is NOT used here.
# The C++ backend opens all expert .bin files with O_DIRECT | O_RDONLY, which
# bypasses the OS page cache entirely.  Evicting files with POSIX_FADV_DONTNEED
# before a prewarm_experts() call has zero effect on what is measured — reads
# always go straight to the SSD controller regardless of page cache state.


def _is_packed_dir(bin_dir: str) -> bool:
    """Return True if bin_dir contains packed expert files (layer_0_expert_0.bin)."""
    return os.path.exists(f"{bin_dir}/layer_0_expert_0.bin")


def _expert_bin_paths(bin_dir: str, layer: int, expert: int) -> List[str]:
    """All .bin files for a single expert (unpacked format)."""
    prefix = f"{bin_dir}/layer_{layer}_expert_{expert}"
    return [
        f"{prefix}_gate.qweight.bin", f"{prefix}_gate.scales.bin", f"{prefix}_gate.zeros.bin",
        f"{prefix}_up.qweight.bin",   f"{prefix}_up.scales.bin",   f"{prefix}_up.zeros.bin",
        f"{prefix}_down.qweight.bin", f"{prefix}_down.scales.bin", f"{prefix}_down.zeros.bin",
    ]


def expert_total_bytes(bin_dir: str, layer: int, expert: int) -> int:
    """Return bytes for one expert, auto-detecting packed vs unpacked format."""
    if _is_packed_dir(bin_dir):
        packed_path = f"{bin_dir}/layer_{layer}_expert_{expert}.bin"
        return os.path.getsize(packed_path) if os.path.exists(packed_path) else 0
    total = 0
    for p in _expert_bin_paths(bin_dir, layer, expert):
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total


# ─── model instantiation ──────────────────────────────────────────────────────

def build_model(model_name: str, cache_size: int, device: str):
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent / "unified_llm_w4a16"))

    if model_name in ("mixtral", "mixtral_cached"):
        from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model
        return Mixtral8x7BW4A16Model(backend="cached",
                                     max_cached_experts_per_layer=cache_size, device=device)
    elif model_name in ("qwen3", "qwen3_cached"):
        import importlib
        mod = importlib.import_module("qwen3_30B-A3B_w4a16_model")
        return mod.Qwen3_30BA3BW4A16Model(backend="cached",
                                          max_cached_experts_per_layer=cache_size, device=device)
    else:
        raise ValueError(f"Unknown model '{model_name}'")


# ─── SSD benchmark (O_DIRECT path through prewarm_experts) ───────────────────

def _run_ssd_benchmark(args) -> Tuple[List[float], int]:
    """
    Time expert loading via prewarm_experts().

    The C++ backend always uses O_DIRECT, so the page cache is never consulted.
    Every call to prewarm_experts() goes straight to the SSD — no eviction is
    needed to guarantee this.  The measured time is:
        SSD_total = SSD_read (O_DIRECT pread) + H2D_DMA (pinned → VRAM)

    Returns (latencies_ms, bytes_per_expert).
    """
    model = build_model(args.model, args.cache_size, args.device)
    m = model.model

    bytes_per_expert = 0
    for l in range(args.num_layers):
        for e in range(args.num_experts):
            b = expert_total_bytes(args.bin_dir, l, e)
            if b > 0:
                bytes_per_expert = b
                break
        if bytes_per_expert > 0:
            break

    if bytes_per_expert == 0:
        print(f"[ERROR] No .bin files found in {args.bin_dir}", file=sys.stderr)
        sys.exit(1)

    expert_fmt = "packed (1 file/expert)" if _is_packed_dir(args.bin_dir) else "unpacked (9 files/expert)"
    total_bytes = bytes_per_expert * args.cache_size * args.num_layers
    print(f"\n[SSD] bytes/expert={bytes_per_expert/1e6:.2f} MB  "
          f"total/round={total_bytes/1e6:.0f} MB  "
          f"cache={args.cache_size}×{args.num_layers} layers  [{expert_fmt}]")

    # Untimed slot warmup so VRAM slots are allocated
    m.prewarm_experts(args.cache_size)
    torch.cuda.synchronize()

    def one_round(label: str) -> float:
        m.reset_cache_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.prewarm_experts(args.cache_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0
        per_exp = ms / (args.cache_size * args.num_layers)
        gbps = (total_bytes / 1e9) / (ms / 1000.0)
        if args.verbose:
            hits, misses = m.get_cache_stats()
            print(f"  {label}: {ms:.1f} ms  ({per_exp:.2f} ms/expert, "
                  f"{gbps*1000:.0f} MB/s)  hits={hits} misses={misses}")
        return ms

    if args.warmup_rounds > 0:
        print(f"[SSD] Warmup ({args.warmup_rounds} rounds)…")
        for i in range(args.warmup_rounds):
            one_round(f"warmup {i+1}")

    print(f"[SSD] Measuring ({args.num_rounds} rounds)…")
    latencies: List[float] = []
    for i in range(args.num_rounds):
        latencies.append(one_round(f"round {i+1:3d}/{args.num_rounds}"))

    return latencies, bytes_per_expert


# ─── H2D DMA benchmark (pinned CPU → VRAM) ───────────────────────────────────

def _run_h2d_benchmark(args, bytes_per_expert: int) -> List[float]:
    """
    Measure the Host-to-Device DMA cost in isolation.

    The C++ backend's I/O path is:
        pread (O_DIRECT, SSD → CPU pinned buffer) → cudaMemcpy (pinned → VRAM)

    prewarm_experts() bundles both steps.  To isolate the H2D DMA component we
    allocate a pinned CPU tensor and a GPU tensor of the same size as one
    expert's weights, then time repeated .cuda() copies.

    This is the "RAM → VRAM" transfer cost that would apply if the expert weights
    were already in DRAM (e.g. a hypothetical 3-tier DRAM buffer).
    """
    device = torch.device(args.device)
    n_experts = args.cache_size * args.num_layers
    total_bytes = bytes_per_expert * n_experts

    # Pinned (page-locked) CPU buffer — same data layout as the real weights
    # Use bytes (uint8) since we're benchmarking raw transfer bandwidth
    cpu_buf = torch.zeros(total_bytes, dtype=torch.uint8).pin_memory()
    gpu_buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    print(f"\n[H2D DMA] {total_bytes/1e6:.0f} MB ({bytes_per_expert/1e6:.2f} MB × {n_experts} experts)  "
          f"pinned CPU → VRAM")

    def one_round(label: str) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_buf.copy_(cpu_buf, non_blocking=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        per_exp = ms / n_experts
        gbps = (total_bytes / 1e9) / (ms / 1000.0)
        if args.verbose:
            print(f"  {label}: {ms:.1f} ms  ({per_exp:.2f} ms/expert, {gbps*1000:.0f} MB/s)")
        return ms

    if args.warmup_rounds > 0:
        print(f"[H2D DMA] Warmup ({args.warmup_rounds} rounds)…")
        for i in range(args.warmup_rounds):
            one_round(f"warmup {i+1}")

    print(f"[H2D DMA] Measuring ({args.num_rounds} rounds)…")
    latencies: List[float] = []
    for i in range(args.num_rounds):
        latencies.append(one_round(f"round {i+1:3d}/{args.num_rounds}"))

    del cpu_buf, gpu_buf
    return latencies


# ─── Compute benchmark (pure GPU) ────────────────────────────────────────────

def _run_compute_benchmark(args) -> List[float]:
    """
    Time the GPU compute for one expert's FFN (W4A16 → fp16 output) using
    torch matmuls of the correct expert shape.

    Each Qwen3 MoE expert (A3B variant) is a 3-layer FFN:
        gate_proj:  [hidden, inter]  W4A16
        up_proj:    [hidden, inter]  W4A16
        down_proj:  [inter,  hidden] W4A16
    Active during decode: batch=1, seq=1

    We benchmark with fp16 weights (conservative lower bound on latency —
    actual W4A16 dequantize+GEMM is slightly slower but same memory footprint).
    """
    device = torch.device(args.device)
    H = args.hidden_size
    I = args.intermediate_size
    K = args.active_k
    L = args.num_layers

    print(f"\n[Compute] hidden={H}  intermediate={I}  active_k={K}  moe_layers={L}")
    print(f"[Compute] Expert invocations per token: {K*L}")
    print(f"[Compute] Benchmarking with fp16 matmuls (shape matches W4A16 expert)")

    # One token, fp16
    x = torch.randn(1, H, dtype=torch.float16, device=device)

    # Expert weight tensors: one set represents one expert
    w_gate = torch.randn(I, H, dtype=torch.float16, device=device)
    w_up   = torch.randn(I, H, dtype=torch.float16, device=device)
    w_down = torch.randn(H, I, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(args.warmup_rounds + 2):
        gate = torch.nn.functional.silu(x @ w_gate.t())
        up   = x @ w_up.t()
        _    = (gate * up) @ w_down.t()
    torch.cuda.synchronize()

    def one_token_all_experts() -> float:
        """Simulate K active experts across L MoE layers for one decode token."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(K * L):
            gate = torch.nn.functional.silu(x @ w_gate.t())
            up   = x @ w_up.t()
            _    = (gate * up) @ w_down.t()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # One round = time to compute all expert invocations for 1 decode token
    print(f"[Compute] Measuring ({args.num_rounds} rounds, each = {K*L} expert FFNs)…")
    latencies: List[float] = []
    for i in range(args.num_rounds):
        ms = one_token_all_experts()
        latencies.append(ms)
        if args.verbose:
            per_exp = ms / (K * L)
            print(f"  round {i+1:3d}/{args.num_rounds}: {ms:.2f} ms/token  "
                  f"({per_exp:.3f} ms/expert)")

    return latencies


# ─── Statistics & report ─────────────────────────────────────────────────────

def _stats(latencies: List[float]):
    n = len(latencies)
    avg = statistics.mean(latencies)
    med = statistics.median(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0.0
    p95 = sorted(latencies)[int(0.95 * n)]
    return avg, med, std, p95, min(latencies), max(latencies)


def _print_io_stats(label: str, latencies: List[float],
                    n_experts: int, bytes_per_expert: int) -> float:
    """Print stats for prewarm_experts() (SSD) or H2D DMA; return avg ms/expert."""
    avg, med, std, p95, lo, hi = _stats(latencies)
    per_exp = avg / n_experts
    total_bytes = bytes_per_expert * n_experts
    gbps = (total_bytes / 1e9) / (avg / 1000.0)

    print(f"\n  ── {label} ──")
    print(f"  Avg  total time      : {avg:.1f} ms   (std {std:.1f}, p95 {p95:.1f})")
    print(f"  Avg  per-expert      : {per_exp:.2f} ms/expert")
    print(f"  Avg  throughput      : {gbps*1000:.0f} MB/s  ({gbps:.3f} GB/s)")
    print(f"  Min / Max            : {lo:.1f} / {hi:.1f} ms")
    return per_exp


def _print_compute_stats(latencies: List[float], active_k: int, num_layers: int) -> float:
    avg, med, std, p95, lo, hi = _stats(latencies)
    per_exp = avg / (active_k * num_layers)

    print(f"\n  ── GPU Compute ──")
    print(f"  Avg  time/token      : {avg:.2f} ms/token   (std {std:.2f}, p95 {p95:.2f})")
    print(f"  Avg  per-expert      : {per_exp:.3f} ms/expert  ({active_k*num_layers} experts/token)")
    print(f"  Min / Max            : {lo:.2f} / {hi:.2f} ms/token")
    return per_exp


def _print_breakdown(ssd_per_exp: Optional[float],
                     h2d_per_exp: Optional[float],
                     compute_per_exp: Optional[float],
                     active_k: int, num_layers: int,
                     bytes_per_expert: int) -> None:
    """
    Decompose latency.  The C++ backend's per-expert load path is:
        pread O_DIRECT (SSD → pinned CPU)  +  cudaMemcpy (pinned CPU → VRAM)
        ─────────────────────────────────     ──────────────────────────────
             = ssd_io_per_exp                       = h2d_per_exp
    prewarm_experts() measures ssd_io + h2d combined (ssd_per_exp).
    h2d_per_exp is measured independently via the synthetic pinned-memcpy benchmark.
    """
    print()
    print("=" * 66)
    print("  LATENCY BREAKDOWN (per expert)")
    print("=" * 66)

    ssd_io_per_exp: Optional[float] = None

    if compute_per_exp is not None:
        print(f"  GPU compute          : {compute_per_exp*1000:.1f} µs/expert")

    if h2d_per_exp is not None:
        n_experts = args_holder["cache_size"] * args_holder["num_layers"]
        bw = bytes_per_expert / 1e6 / max(h2d_per_exp / 1000.0, 1e-9)
        print(f"  H2D DMA (pinned→VRAM): {h2d_per_exp*1000:.1f} µs/expert  "
              f"({bytes_per_expert/1e6:.2f} MB @ {bw:.0f} MB/s)")

    if ssd_per_exp is not None and h2d_per_exp is not None:
        ssd_io_per_exp = ssd_per_exp - h2d_per_exp
        bw = bytes_per_expert / 1e6 / max(ssd_io_per_exp / 1000.0, 1e-9)
        print(f"  SSD read (O_DIRECT)  : {ssd_io_per_exp*1000:.1f} µs/expert  "
              f"({bytes_per_expert/1e6:.2f} MB @ {bw:.0f} MB/s)  [= SSD_total − H2D]")

    if ssd_per_exp is not None:
        print(f"  SSD load total       : {ssd_per_exp*1000:.1f} µs/expert  (O_DIRECT pread + H2D)")

    print()

    if ssd_per_exp is not None and compute_per_exp is not None:
        ratio = ssd_per_exp / compute_per_exp
        print(f"  SSD load / compute   : {ratio:.1f}×  "
              f"(one SSD miss ≈ {ratio:.1f} compute-only experts)")
    if h2d_per_exp is not None and compute_per_exp is not None:
        ratio = h2d_per_exp / compute_per_exp
        print(f"  H2D DMA / compute    : {ratio:.1f}×")

    print()
    print("  Per-token impact — varying cache hit rate:")
    experts_per_tok = active_k * num_layers
    if compute_per_exp is not None:
        compute_ms = experts_per_tok * compute_per_exp
        print(f"    100% hit rate  : {compute_ms:.1f} ms/token  ({experts_per_tok} GEMMs)")
    for hit_pct in (90, 80, 60, 40, 20, 0):
        miss_pct = 100 - hit_pct
        if ssd_per_exp is not None and compute_per_exp is not None:
            misses = int((miss_pct / 100) * experts_per_tok)
            hits   = experts_per_tok - misses
            total  = hits * compute_per_exp + misses * ssd_per_exp
            tps    = 1000.0 / total if total > 0 else float("inf")
            print(f"    {hit_pct:3d}% hit rate  : {total:.1f} ms/token  "
                  f"({misses} SSD loads)  → {tps:.1f} TPS")

    print("=" * 66)


# Mutable holder so _print_breakdown can access args without full refactor
args_holder: dict = {}


# ─── main ─────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    global args_holder
    args_holder = {"cache_size": args.cache_size, "num_layers": args.num_layers}

    ssd_latencies: Optional[List[float]] = None
    h2d_latencies: Optional[List[float]] = None
    compute_latencies: Optional[List[float]] = None

    run_ssd     = args.mode in ("ssd",     "all")
    run_h2d     = args.mode in ("h2d",     "all")
    run_compute = args.mode in ("compute", "all")

    bytes_per_expert = 0

    if run_ssd:
        ssd_latencies, bytes_per_expert = _run_ssd_benchmark(args)

    if run_h2d:
        if bytes_per_expert == 0:
            # Need bytes_per_expert even if SSD mode wasn't run
            for l in range(args.num_layers):
                for e in range(args.num_experts):
                    b = expert_total_bytes(args.bin_dir, l, e)
                    if b > 0:
                        bytes_per_expert = b
                        break
                if bytes_per_expert:
                    break
            if bytes_per_expert == 0:
                print(f"[ERROR] No .bin files in {args.bin_dir}", file=sys.stderr)
                sys.exit(1)
        h2d_latencies = _run_h2d_benchmark(args, bytes_per_expert)

    if run_compute:
        compute_latencies = _run_compute_benchmark(args)

    # ── Print results ──
    n_experts = args.cache_size * args.num_layers
    print()
    print("=" * 66)
    print(f"  EXPERT LOADING BENCHMARK — {args.model.upper()}")
    print("=" * 66)

    ssd_per_exp = h2d_per_exp = compute_per_exp = None

    if ssd_latencies:
        ssd_per_exp = _print_io_stats(
            "SSD → O_DIRECT pread → H2D DMA → VRAM  (prewarm_experts)",
            ssd_latencies, n_experts, bytes_per_expert)

    if h2d_latencies:
        h2d_per_exp = _print_io_stats(
            "H2D DMA only  (pinned CPU → VRAM, synthetic)",
            h2d_latencies, n_experts, bytes_per_expert)

    if compute_latencies:
        compute_per_exp = _print_compute_stats(
            compute_latencies, args.active_k, args.num_layers)

    if args.mode == "all":
        _print_breakdown(ssd_per_exp, h2d_per_exp, compute_per_exp,
                         args.active_k, args.num_layers, bytes_per_expert)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark SSD / RAM / compute latency for MoE expert loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Mode
    p.add_argument("--mode", choices=["ssd", "h2d", "compute", "all"], default="ssd",
                   help=("What to benchmark: "
                         "ssd=O_DIRECT pread+H2D via prewarm_experts(), "
                         "h2d=H2D DMA only (pinned memcpy, synthetic), "
                         "compute=GPU GEMM only, "
                         "all=all three + breakdown"))

    # I/O benchmark args
    p.add_argument("--model", choices=["mixtral", "mixtral_cached", "qwen3", "qwen3_cached"],
                   default="qwen3")
    p.add_argument("--bin-dir", type=str, default="",
                   help="Path to _unpacked .bin weight directory (required for ssd/ram modes)")
    p.add_argument("--num-layers",  type=int, default=48, help="Number of transformer layers")
    p.add_argument("--num-experts", type=int, default=128, help="Experts per layer")
    p.add_argument("--cache-size",  type=int, default=8, help="Experts kept in GPU VRAM per layer")

    # Compute benchmark args
    p.add_argument("--hidden-size",       type=int, default=2048,
                   help="Expert hidden dimension (default: 2048 for Qwen3-30B-A3B)")
    p.add_argument("--intermediate-size", type=int, default=2048,
                   help="Expert intermediate dimension (default: 2048 for Qwen3-30B-A3B)")
    p.add_argument("--active-k",          type=int, default=4,
                   help="Active experts per token per MoE layer (default: 4 for Qwen3-30B-A3B)")

    # Common
    p.add_argument("--num-rounds",    type=int, default=20)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--device",        type=str, default="cuda")
    p.add_argument("--verbose",       action="store_true")

    args = p.parse_args()

    if args.mode in ("ssd", "h2d", "all") and not args.bin_dir:
        p.error("--bin-dir is required for ssd/h2d/all modes")
    if args.mode in ("ssd", "h2d", "all") and not os.path.isdir(args.bin_dir):
        p.error(f"--bin-dir not found: {args.bin_dir}")

    run_benchmark(args)


if __name__ == "__main__":
    main()
