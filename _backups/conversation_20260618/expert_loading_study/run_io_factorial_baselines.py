#!/usr/bin/env python3
"""
Full 2×2×2 factorial: packed/unpacked × intra seq/par × inter seq/par.

Runs LRU + RANDOM cached baselines only (--lru-random-baselines-only).

Env flags per cell:
  HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1     → intra sequential (0 → intra parallel)
  HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1 → inter sequential (unset → inter parallel)

Usage:
  cd py/expert_loading_study
  python run_io_factorial_baselines.py --cache-sizes 8 --quick
  python run_io_factorial_baselines.py --cache-sizes 8 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

STUDY_DIR = Path(__file__).parent
REPO_ROOT = STUDY_DIR.parent.parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
SWEEP = REPO_ROOT / "py/utils/sweep_predict_cached_cache_metrics.py"
from io_common import apply_strict_ssd_env, strict_ssd_sweep_flags
PACKED = REPO_ROOT / "py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
UNPACKED = REPO_ROOT / "py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_unpacked"
DEFAULT_PREDICTOR = (
    REPO_ROOT
    / "trainingData/qwen3_30b/transformer_final_pfill_markov_emb/transformer_eh4_h64_f1"
)


def _io_label(sequential_intra: bool, sequential_inter: bool) -> str:
    i = "intra_seq" if sequential_intra else "intra_par"
    j = "inter_seq" if sequential_inter else "inter_par"
    return f"{i}_{j}"


def _run_one(
    *,
    out_csv: Path,
    cache_sizes: List[int],
    weights_dir: Path,
    weights_format: str,
    sequential_intra: bool,
    sequential_inter: bool,
    max_new_tokens: int,
    num_prompts: int,
    predictor_base_dir: str,
    strict_ssd: bool,
) -> int:
    env = os.environ.copy()
    env.setdefault("HSA_ENABLE_SDMA", "0")
    apply_strict_ssd_env(env, enabled=strict_ssd)

    if sequential_intra:
        env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "1"
    else:
        env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "0"

    env.pop("HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO", None)
    if sequential_inter:
        env["HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO"] = "1"

    cmd = [
        sys.executable,
        str(SWEEP),
        "--sweep-question",
        "custom_1_16_no_ppl",
        "--model",
        "qwen",
        "--cache-sizes",
        *[str(c) for c in cache_sizes],
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-prompts",
        str(num_prompts),
        "--csv-file",
        str(out_csv),
        "--out-dir",
        str(out_csv.parent),
        "--log-file",
        str(out_csv.with_suffix(".log")),
        "--expert-weights-dir",
        str(weights_dir),
        "--predictor-base-dir",
        predictor_base_dir,
        "--lru-random-baselines-only",
    ]
    cmd.extend(strict_ssd_sweep_flags(enabled=strict_ssd))

    label = _io_label(sequential_intra, sequential_inter)
    print(
        f"\n{'='*72}\n"
        f"  format={weights_format}  {label}\n"
        f"  weights={weights_dir}\n"
        f"  SEQUENTIAL_EXPERT_IO={env['HETEROPREDICT_SEQUENTIAL_EXPERT_IO']}\n"
        f"  SEQUENTIAL_INTER_EXPERT_IO={env.get('HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO', '(unset)')}\n"
        f"  CSV: {out_csv}\n"
        f"{'='*72}",
        flush=True,
    )
    return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))


def _merge_results(out_dir: Path, manifest: List[Dict[str, Any]]) -> Path:
    merged_path = out_dir / "factorial_merged.csv"
    rows_out: List[Dict[str, Any]] = []

    for entry in manifest:
        csv_path = Path(entry["csv"])
        if not csv_path.is_file():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                rows_out.append({
                    "weights_format": entry["weights_format"],
                    "sequential_intra": entry["sequential_intra"],
                    "sequential_inter": entry["sequential_inter"],
                    "io_label": entry["io_label"],
                    **row,
                })

    if not rows_out:
        return merged_path

    fieldnames: List[str] = []
    for r in rows_out:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with merged_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    return merged_path


def _print_summary(merged_path: Path) -> None:
    if not merged_path.is_file():
        print("No merged results to summarize.")
        return

    with merged_path.open() as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*72}\n  FACTORIAL BASELINE SUMMARY (TPS)\n{'='*72}")
    print(f"  {'format':8} {'io_mode':22} {'label':18} {'C':>3} {'TPS':>8} {'stall':>6} {'ms/load':>8}")
    print("  " + "-" * 68)

    for r in sorted(rows, key=lambda x: (
        x.get("weights_format", ""),
        x.get("io_label", ""),
        x.get("label", ""),
        int(x.get("cache_size") or 0),
    )):
        tps = r.get("tokens_per_second") or ""
        stall = r.get("stall_loads") or ""
        ms = r.get("avg_ms_per_expert_load") or ""
        print(
            f"  {r.get('weights_format',''):8} {r.get('io_label',''):22} "
            f"{str(r.get('label','')):18} {r.get('cache_size',''):>3} "
            f"{str(tps):>8} {str(stall):>6} {str(ms):>8}"
        )
    print(f"\n  Full table: {merged_path}\n{'='*72}")


def main() -> int:
    p = argparse.ArgumentParser(description="2×2×2 IO factorial LRU/RANDOM baseline sweep")
    p.add_argument("--cache-sizes", type=int, nargs="+", default=[8])
    p.add_argument("--formats", choices=["packed", "unpacked", "both"], default="both")
    p.add_argument("--predictor-base-dir", type=str, default=str(DEFAULT_PREDICTOR))
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="40 tokens, 1 prompt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip cells whose CSV already exists")
    p.add_argument(
        "--strict-ssd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="O_DIRECT-only loads + drop page cache between sweep configs (default: on)",
    )
    args = p.parse_args()

    format_dirs = []
    if args.formats in ("packed", "both"):
        format_dirs.append(("packed", PACKED))
    if args.formats in ("unpacked", "both"):
        format_dirs.append(("unpacked", UNPACKED))

    for name, path in format_dirs:
        if not path.is_dir():
            sys.exit(f"Missing weights dir for {name}: {path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else STUDY_DIR / "results" / f"io_factorial_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = 40 if args.quick else 80
    num_prompts = 1 if args.quick else 3

    manifest: List[Dict[str, Any]] = []
    rc = 0

    for (fmt, wdir), seq_intra, seq_inter in product(
        format_dirs,
        (False, True),
        (False, True),
    ):
        io_label = _io_label(seq_intra, seq_inter)
        csv_path = out_dir / f"sweep_{fmt}_{io_label}.csv"
        entry = {
            "weights_format": fmt,
            "weights_dir": str(wdir),
            "sequential_intra": seq_intra,
            "sequential_inter": seq_inter,
            "io_label": io_label,
            "csv": str(csv_path),
            "cache_sizes": args.cache_sizes,
            "max_new_tokens": max_new_tokens,
            "num_prompts": num_prompts,
        }
        manifest.append(entry)

        if args.dry_run:
            print(f"[dry-run] would run {fmt} {io_label} -> {csv_path}")
            continue

        if csv_path.is_file() and args.resume:
            print(f"[skip] {csv_path.name} exists (--resume)")
            continue

        step_rc = _run_one(
            out_csv=csv_path,
            cache_sizes=args.cache_sizes,
            weights_dir=wdir,
            weights_format=fmt,
            sequential_intra=seq_intra,
            sequential_inter=seq_inter,
            max_new_tokens=max_new_tokens,
            num_prompts=num_prompts,
            predictor_base_dir=args.predictor_base_dir,
            strict_ssd=args.strict_ssd,
        )
        rc = rc or step_rc

    manifest_path = out_dir / "factorial_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not args.dry_run:
        merged = _merge_results(out_dir, manifest)
        _print_summary(merged)
        print(f"\nManifest: {manifest_path}")
        print(f"Merged:   {merged}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
