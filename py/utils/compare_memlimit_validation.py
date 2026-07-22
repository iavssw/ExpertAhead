#!/usr/bin/env python3
"""Compare 2-prompt memlimit validation runs against canonical final_10prompt results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHOD_ORDER = [
    "Neither (RANDOM)",
    "Gating",
    "Cache-Cond Only",
    "Prefetch Only",
    "Both",
]


def _method_key(label: str) -> str:
    for prefix in METHOD_ORDER:
        if label.startswith(prefix):
            return prefix
    return label


def load_run_tps(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "sweep.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    if "tokens_per_second" not in df.columns:
        raise ValueError(f"No tokens_per_second in {csv}")
    out = df[["label", "tokens_per_second"]].copy()
    out["method"] = out["label"].map(_method_key)
    if "peak_hwm_mb" in df.columns:
        out["peak_hwm_mb"] = df["peak_hwm_mb"]
    if "memory_max" in df.columns:
        out["memory_max"] = df["memory_max"]
    if "oom_killed" in df.columns:
        out["oom_killed"] = df["oom_killed"]
    return out


def compare_cache(
    canonical_dir: Path,
    validate_dir: Path,
    cache_size: int,
    rtol: float,
) -> pd.DataFrame:
    base = load_run_tps(canonical_dir)
    test = load_run_tps(validate_dir)

    rows = []
    random_base = base.loc[base["method"] == "Neither (RANDOM)", "tokens_per_second"].iloc[0]
    random_test = test.loc[test["method"] == "Neither (RANDOM)", "tokens_per_second"].iloc[0]

    for method in METHOD_ORDER:
        b_row = base[base["method"] == method]
        t_row = test[test["method"] == method]
        if b_row.empty or t_row.empty:
            continue
        b_tps = float(b_row["tokens_per_second"].iloc[0])
        t_tps = float(t_row["tokens_per_second"].iloc[0])
        b_su = b_tps / random_base
        t_su = t_tps / random_test
        pct = 100.0 * (t_tps - b_tps) / b_tps if b_tps else float("nan")
        ok = abs(pct) <= 100.0 * rtol
        row = {
            "cache_size": cache_size,
            "method": method,
            "canonical_tps": b_tps,
            "validate_tps": t_tps,
            "canonical_speedup": b_su,
            "validate_speedup": t_su,
            "tps_pct_diff": pct,
            "ballpark": ok,
        }
        if "peak_hwm_mb" in t_row.columns:
            row["peak_hwm_mb"] = t_row["peak_hwm_mb"].iloc[0]
        if "memory_max" in t_row.columns:
            row["memory_max"] = t_row["memory_max"].iloc[0]
        if "oom_killed" in t_row.columns:
            row["oom_killed"] = t_row["oom_killed"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("py/utils/final_results_runs/final_10prompt"),
    )
    parser.add_argument(
        "--canonical-suffix",
        default="20260709_001606",
        help="Timestamp suffix for canonical run dirs",
    )
    parser.add_argument(
        "--validate-root",
        type=Path,
        required=True,
        help="Root dir containing C{16,32,48,64}_* validation runs",
    )
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.10,
        help="Ballpark if |TPS %% diff| <= 100*rtol (default 10%%)",
    )
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    validate_dirs = sorted(args.validate_root.glob("C*"))
    by_cache = {}
    for d in validate_dirs:
        try:
            c = int(d.name.split("_")[0][1:])
        except (IndexError, ValueError):
            continue
        by_cache[c] = d

    frames = []
    for c in args.cache_sizes:
        canon = args.canonical_root / f"C{c}_{args.canonical_suffix}"
        test = by_cache.get(c)
        if test is None:
            print(f"[skip] C={c}: no validation dir under {args.validate_root}")
            continue
        if not canon.exists():
            print(f"[skip] C={c}: missing canonical {canon}")
            continue
        frames.append(compare_cache(canon, test, c, args.rtol))

    if not frames:
        raise SystemExit("No comparisons produced")

    df = pd.concat(frames, ignore_index=True)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")

    print(f"\nValidation vs canonical (ballpark = within {100*args.rtol:.0f}% TPS)\n")
    for c in args.cache_sizes:
        sub = df[df["cache_size"] == c]
        if sub.empty:
            continue
        print(f"=== C={c} ===")
        for _, r in sub.iterrows():
            flag = "OK" if r["ballpark"] else "!!"
            oom = ""
            if "oom_killed" in r and pd.notna(r["oom_killed"]) and r["oom_killed"]:
                oom = " OOM"
            hwm = ""
            if "peak_hwm_mb" in r and pd.notna(r["peak_hwm_mb"]):
                cap = r.get("memory_max", "")
                hwm = f"  HWM={r['peak_hwm_mb']:.0f}MB cap={cap}"
            print(
                f"  [{flag}] {r['method']:<22s}  "
                f"canon={r['canonical_tps']:.2f}  test={r['validate_tps']:.2f}  "
                f"Δ={r['tps_pct_diff']:+.1f}%  "
                f"su canon={r['canonical_speedup']:.2f} test={r['validate_speedup']:.2f}"
                f"{hwm}{oom}"
            )
        print()

    n_ok = int(df["ballpark"].sum())
    print(f"Summary: {n_ok}/{len(df)} methods within {100*args.rtol:.0f}% of canonical TPS")
    if (df["oom_killed"] == True).any() if "oom_killed" in df.columns else False:
        print("WARNING: some runs hit OOM under memory cap")


if __name__ == "__main__":
    main()
