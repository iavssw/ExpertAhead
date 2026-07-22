#!/usr/bin/env python3
"""
Reproduce C24_B12_postfix_smoke (prefetch win vs LRU) with sequential vs parallel inter-expert I/O.

Original smoke (Jun 17, sequential inter default):
  py/utils/final_results_runs/sec5_verify_fixed/C24_B12_postfix_smoke/sweep.csv
  Prefetch Only B=12: 6.21 TPS vs LRU 4.65 TPS (+33%)

Rebuild after C++ changes (repo root):
  source utils/setup.sh && cmake --build build -j

Usage:
  # Focused A/B: LRU + RANDOM + Prefetch B=12, parallel vs sequential inter
  python run_smoke_repro.py

  # Full 5-row smoke (RANDOM, LRU, Cache-Cond, Prefetch, Both λ=1)
  python run_smoke_repro.py --full-smoke

  # Keep measurement on for overlap / window metrics
  python run_smoke_repro.py --enable-measurement

  # Jun-17 pre-batch I/O (reproduce ~4.65 LRU / ~6.21 prefetch smoke)
  python run_smoke_repro.py --legacy-io
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

STUDY_DIR = Path(__file__).parent
REPO_ROOT = STUDY_DIR.parent.parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

SWEEP = REPO_ROOT / "py/utils/sweep_predict_cached_cache_metrics.py"
from io_common import (
    apply_strict_ssd_env,
    apply_legacy_expert_io_env,
    apply_modern_expert_io_env,
    strict_ssd_sweep_flags,
)

DEFAULT_PACKED = REPO_ROOT / "py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
SMOKE_PREDICTOR = (
    REPO_ROOT / "trainingData/qwen3_30b/transformer_final/transformer_eh4_h64_f1"
)
EXPERT_REUSE_CSV = REPO_ROOT / "py/expert_predictor/expert_reuse_qwen3_30b.csv"
ORIGINAL_SMOKE_CSV = (
    REPO_ROOT
    / "py/utils/final_results_runs/sec5_verify_fixed/C24_B12_postfix_smoke/sweep.csv"
)

KEY_LABELS = (
    "Neither (LRU)",
    "Prefetch Only B=12",
    "Both λ=1.0 B=12",
)


def _run_smoke_sweep(
    *,
    out_csv: Path,
    sequential_inter: bool,
    expert_weights_dir: str,
    predictor_base_dir: str,
    full_smoke: bool,
    disable_measurement: bool,
    strict_ssd: bool,
    legacy_io: bool,
) -> int:
    env = os.environ.copy()
    env.setdefault("HSA_ENABLE_SDMA", "0")
    apply_strict_ssd_env(env, enabled=strict_ssd)
    if legacy_io:
        apply_legacy_expert_io_env(env, enabled=True, sequential_inter=True)
    else:
        apply_modern_expert_io_env(env, sequential_inter=sequential_inter, sequential_intra=True)

    cmd = [
        sys.executable,
        str(SWEEP),
        "--sweep-question",
        "custom_1_16_no_ppl",
        "--model",
        "qwen",
        "--dataset",
        "wikitext",
        "--cache-sizes",
        "24",
        "--lookaheads",
        "1",
        "--budget-fractions",
        "0.5",
        "--routing-bias-top-n",
        "5",
        "--constraint-expert-reuse-csv",
        str(EXPERT_REUSE_CSV),
        "--predictor-base-dir",
        predictor_base_dir,
        "--num-prompts",
        "1",
        "--prompt-max-chars",
        "4096",
        "--max-new-tokens",
        "128",
        "--temperature",
        "0.0",
        "--non-baseline-cache-policy",
        "LRU",
        "--drop-page-cache-between-runs",
        "--expert-weights-dir",
        expert_weights_dir,
        "--csv-file",
        str(out_csv),
        "--out-dir",
        str(out_csv.parent),
        "--log-file",
        str(out_csv.with_suffix(".log")),
    ]

    if full_smoke:
        cmd.extend(["--lambdas", "0", "1", "--cache-cond-forced-top-ns", "5"])
    else:
        cmd.append("--prefetch-only")

    if disable_measurement:
        cmd.append("--disable-measurement")

    cmd.extend(strict_ssd_sweep_flags(enabled=strict_ssd))

    label = "legacy_io" if legacy_io else ("sequential_inter" if sequential_inter else "parallel_inter")
    print(f"\n{'=' * 72}\n  smoke_repro  inter_expert={label}\n  CSV: {out_csv}\n{'=' * 72}")
    (out_csv.parent / f"sweep_command_{label}.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))


def _read_tps_by_label(csv_path: Path, labels: Iterable[str]) -> dict[str, float]:
    if not csv_path.is_file():
        return {}
    wanted = set(labels)
    out: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            if label in wanted and label not in out:
                try:
                    out[label] = float(row["tokens_per_second"])
                except (KeyError, TypeError, ValueError):
                    pass
    return out


def _print_comparison(
    *,
    parallel_csv: Path,
    sequential_csv: Path,
    original_csv: Path,
    labels: tuple[str, ...],
) -> None:
    par = _read_tps_by_label(parallel_csv, labels)
    seq = _read_tps_by_label(sequential_csv, labels)
    orig = _read_tps_by_label(original_csv, labels)

    if not par and not seq:
        print("\n[smoke_repro] No results to compare yet.", flush=True)
        return

    print(f"\n{'=' * 72}\n  C24 B=12 smoke repro — tokens/sec\n{'=' * 72}")
    header = f"{'label':<28} {'orig':>8} {'parallel':>10} {'sequential':>10} {'par vs LRU':>12}"
    print(header)
    print("-" * len(header))

    lru_par = par.get("Neither (LRU)")
    lru_seq = seq.get("Neither (LRU)")

    for label in labels:
        o = orig.get(label)
        p = par.get(label)
        s = seq.get(label)
        delta = ""
        if p is not None and lru_par is not None and label != "Neither (LRU)":
            pct = 100.0 * (p - lru_par) / lru_par if lru_par else 0.0
            delta = f"{pct:+.1f}%"
        o_s = f"{o:>8.2f}" if o is not None else f"{'—':>8}"
        p_s = f"{p:>10.2f}" if p is not None else f"{'—':>10}"
        s_s = f"{s:>10.2f}" if s is not None else f"{'—':>10}"
        print(f"{label:<28} {o_s} {p_s} {s_s} {delta:>12}")

    if lru_par is not None and lru_seq is not None:
        lift = 100.0 * (lru_par - lru_seq) / lru_seq if lru_seq else 0.0
        print(f"\nLRU parallel vs sequential inter: {lift:+.1f}% (parallel faster if positive)")

    pf_par = par.get("Prefetch Only B=12")
    pf_seq = seq.get("Prefetch Only B=12")
    if pf_par is not None and lru_par is not None:
        print(f"Prefetch parallel vs LRU parallel: {100.0 * (pf_par - lru_par) / lru_par:+.1f}%")
    if pf_seq is not None and lru_seq is not None:
        print(f"Prefetch sequential vs LRU sequential: {100.0 * (pf_seq - lru_seq) / lru_seq:+.1f}%")
    if pf_par is not None and pf_seq is not None:
        print(f"Prefetch parallel vs sequential inter: {100.0 * (pf_par - pf_seq) / pf_seq:+.1f}%")


def main() -> int:
    p = argparse.ArgumentParser(
        description="C24_B12_postfix_smoke repro: sequential vs parallel inter-expert I/O"
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--expert-weights-dir", type=str, default=str(DEFAULT_PACKED))
    p.add_argument(
        "--predictor-base-dir",
        type=str,
        default=str(SMOKE_PREDICTOR),
        help="Default matches original smoke (transformer_final/transformer_eh4_h64_f1)",
    )
    p.add_argument(
        "--full-smoke",
        action="store_true",
        help="All 5 rows (RANDOM, LRU, Cache-Cond, Prefetch, Both). Default: prefetch-only + baselines.",
    )
    p.add_argument(
        "--enable-measurement",
        action="store_true",
        help="Keep predictor stats (original smoke used --disable-measurement).",
    )
    p.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Only run parallel inter (skip sequential baseline).",
    )
    p.add_argument(
        "--skip-parallel",
        action="store_true",
        help="Only run sequential inter.",
    )
    p.add_argument(
        "--legacy-io",
        action="store_true",
        help="Jun-17 pre-batch I/O (HETEROPREDICT_LEGACY_EXPERT_IO=1). Single run, skips parallel A/B.",
    )
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Print comparison from existing out-dir CSVs without running sweeps.",
    )
    p.add_argument(
        "--strict-ssd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="O_DIRECT loads + drop page cache before first run (original smoke did not use this).",
    )
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else STUDY_DIR / "results" / f"smoke_repro_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parallel_csv = out_dir / "sweep_smoke_parallel.csv"
    sequential_csv = out_dir / "sweep_smoke_sequential.csv"
    legacy_csv = out_dir / "sweep_smoke_legacy.csv"

    labels: tuple[str, ...]
    if args.full_smoke:
        labels = KEY_LABELS + ("Neither (RANDOM)", "Cache-Cond Only λ=1.0")
    else:
        labels = KEY_LABELS[:2]  # LRU + Prefetch Only B=12

    if args.compare_only:
        _print_comparison(
            parallel_csv=parallel_csv,
            sequential_csv=sequential_csv,
            original_csv=ORIGINAL_SMOKE_CSV,
            labels=labels,
        )
        return 0

    if args.legacy_io:
        modes = [("legacy", True, True)]
    else:
        modes = []
        if not args.skip_parallel:
            modes.append(("parallel", False, False))
        if not args.skip_sequential:
            modes.append(("sequential", True, False))

    if not modes:
        print("Nothing to run: both --skip-parallel and --skip-sequential set.", file=sys.stderr)
        return 2

    rc = 0
    for name, sequential, legacy in modes:
        if legacy:
            csv_path = legacy_csv
        elif sequential:
            csv_path = sequential_csv
        else:
            csv_path = parallel_csv
        code = _run_smoke_sweep(
            out_csv=csv_path,
            sequential_inter=sequential,
            expert_weights_dir=args.expert_weights_dir,
            predictor_base_dir=args.predictor_base_dir,
            full_smoke=args.full_smoke,
            disable_measurement=not args.enable_measurement,
            strict_ssd=args.strict_ssd,
            legacy_io=legacy,
        )
        if code != 0:
            rc = code

    if args.legacy_io:
        _print_comparison(
            parallel_csv=legacy_csv,
            sequential_csv=legacy_csv,
            original_csv=ORIGINAL_SMOKE_CSV,
            labels=labels,
        )
    else:
        _print_comparison(
            parallel_csv=parallel_csv,
            sequential_csv=sequential_csv,
            original_csv=ORIGINAL_SMOKE_CSV,
            labels=labels,
        )
    print(f"\nResults: {out_dir}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
