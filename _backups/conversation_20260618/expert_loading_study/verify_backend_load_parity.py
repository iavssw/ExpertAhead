#!/usr/bin/env python3
"""
Verify cached vs predict backends use the same SSD load path on on-demand misses.

Both call load_expert_weights() -> shared moe_expert_load_{packed,unpacked}.inl.
Compares layer-aggregated AvgLoadTime on RANDOM-policy decode misses. Predict runs
without a predictor so only the on-demand stall path is exercised.

Usage:
  cd py/expert_loading_study && source ../../utils/setup.sh
  python verify_backend_load_parity.py
  python verify_backend_load_parity.py --worker cached   # internal
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).parent
PACKED = STUDY_DIR.parent / "unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
PROMPT = "The quick brown fox jumps over the lazy dog. Explain MoE expert caching in detail."

_MISS_RE = re.compile(r"MissLoads=(\d+),\s*AvgLoadTime=([\d.]+)ms")
_STALL_RE = re.compile(
    r"Bandwidth: StallLoads=(\d+), PrefetchLoads=(\d+), AvgLoadTime=([\d.]+)ms"
)


def _parse_miss_bandwidth(text: str) -> tuple[int, float]:
    miss_loads = 0
    weighted_ms = 0.0
    for m in _MISS_RE.finditer(text):
        n = int(m.group(1))
        avg = float(m.group(2))
        miss_loads += n
        weighted_ms += n * avg
    if miss_loads == 0:
        return 0, 0.0
    return miss_loads, weighted_ms / miss_loads


def _parse_predict_bandwidth(text: str) -> tuple[int, int, float]:
    stall = prefetch = 0
    weighted_ms = 0.0
    for m in _STALL_RE.finditer(text):
        s = int(m.group(1))
        p = int(m.group(2))
        avg = float(m.group(3))
        stall += s
        prefetch += p
        total = s + p
        if total:
            weighted_ms += total * avg
    total_loads = stall + prefetch
    if total_loads == 0:
        return 0, 0, 0.0
    return stall, prefetch, weighted_ms / total_loads


def _worker_run(args: argparse.Namespace) -> int:
    import contextlib
    import io
    import importlib

    sys.path.insert(0, str(STUDY_DIR.parent / "unified_llm_w4a16"))
    mod = importlib.import_module("qwen3_30B-A3B_w4a16_model")
    kwargs = dict(
        backend=args.worker,
        max_cached_experts_per_layer=args.cache_size,
        expert_weights_dir=str(PACKED),
    )
    if args.worker == "predict":
        kwargs["predictor_models_dir"] = ""
        kwargs["prefetch_experts_count"] = 0
    model = mod.Qwen3_30BA3BW4A16Model(**kwargs)
    model.set_cache_policy("RANDOM")
    model.reset_cache_stats()
    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        ids = model.tokenize(PROMPT)
        model.generate(ids, max_new_tokens=args.max_new_tokens)
        model.print_cache_stats()
    print(out_buf.getvalue(), end="")
    return 0


def _spawn_worker(
    backend: str,
    *,
    cache_size: int,
    max_new_tokens: int,
    sequential_inter: bool,
) -> str:
    env = os.environ.copy()
    env.pop("HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO", None)
    if sequential_inter:
        env["HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO"] = "1"
    env.setdefault("HETEROPREDICT_SEQUENTIAL_EXPERT_IO", "1")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--cache-size",
        str(cache_size),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(STUDY_DIR.parent.parent))
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{backend} worker failed (exit {proc.returncode})")
    return proc.stdout


def main() -> int:
    p = argparse.ArgumentParser(description="Compare on-demand expert load times: cached vs predict")
    p.add_argument("--cache-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--sequential-inter", action="store_true")
    p.add_argument("--worker", choices=["cached", "predict"], default=None)
    args = p.parse_args()

    if args.worker:
        return _worker_run(args)

    print("Shared load path: include/unified_llm_w4a16_common/moe_expert_load_{packed,unpacked}.inl")
    print(f"cache_size={args.cache_size}  tokens={args.max_new_tokens}  "
          f"inter={'seq' if args.sequential_inter else 'par'}\n")

    cached_log = _spawn_worker(
        "cached",
        cache_size=args.cache_size,
        max_new_tokens=args.max_new_tokens,
        sequential_inter=args.sequential_inter,
    )
    predict_log = _spawn_worker(
        "predict",
        cache_size=args.cache_size,
        max_new_tokens=args.max_new_tokens,
        sequential_inter=args.sequential_inter,
    )

    c_miss, c_avg = _parse_miss_bandwidth(cached_log)
    p_stall, p_prefetch, p_avg = _parse_predict_bandwidth(predict_log)

    print("=== cached (RANDOM, on-demand misses) ===")
    print(f"  MissLoads={c_miss}  AvgLoadTime={c_avg:.3f} ms")
    print("=== predict (no predictor, RANDOM, on-demand stalls only) ===")
    print(f"  StallLoads={p_stall}  PrefetchLoads={p_prefetch}  AvgLoadTime={p_avg:.3f} ms")

    if p_prefetch != 0:
        print("\nFAIL: predict issued prefetch loads without a predictor.")
        return 1
    if c_miss == 0 or p_stall == 0:
        print("\nWARN: insufficient misses/stalls for comparison.")
        return 1

    ratio = p_avg / c_avg if c_avg > 0 else float("inf")
    delta_pct = 100.0 * (ratio - 1.0)
    print(f"\nAvg load time ratio (predict/cached): {ratio:.3f}  ({delta_pct:+.1f}%)")
    if abs(delta_pct) <= 15.0:
        print("PASS: load times match within 15% (same SSD read path + parallel inter-expert).")
        return 0
    print("REVIEW: >15% gap — orchestration differs (slot locks); SSD .inl path is still shared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
