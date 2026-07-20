#!/usr/bin/env python3
"""Summarize Oracle Full Union lookahead sweep and pick best S per cache."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def _expand_run_dirs(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        elif os.path.isdir(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"No run dir match: {p}")
    return out


def _load_oracle_sweeps(run_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        csv_path = os.path.join(d, "sweep.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing sweep.csv in {d}")
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True, help="Sweep dirs or globs")
    p.add_argument("--baseline-tps", type=float, default=5.26, help="SSD streaming baseline for speedup")
    args = p.parse_args()

    run_dirs = _expand_run_dirs(args.run_dirs)
    df = _load_oracle_sweeps(run_dirs)
    oracle = df.loc[
        df["label"].astype(str).str.startswith("Oracle Full Union LA=")
        & df["tokens_per_second"].notna()
    ].copy()
    if oracle.empty:
        raise SystemExit("No Oracle Full Union rows with tokens_per_second.")

    oracle["lookahead"] = pd.to_numeric(oracle["lookahead"], errors="coerce").astype("Int64")
    oracle = oracle.sort_values(["cache_size", "lookahead"])

    print("Oracle Full Union TPS by cache and S:\n")
    for c, sub in oracle.groupby("cache_size"):
        print(f"C={int(c)}:")
        for _, r in sub.iterrows():
            tps = float(r["tokens_per_second"])
            s = int(r["lookahead"])
            print(f"  S={s:>2}: {tps:6.2f} tok/s  ({tps / args.baseline_tps:.2f}x vs {args.baseline_tps})")
        best = sub.loc[sub["tokens_per_second"].idxmax()]
        print(
            f"  → best S={int(best['lookahead'])}  "
            f"TPS={float(best['tokens_per_second']):.2f}  "
            f"({float(best['tokens_per_second']) / args.baseline_tps:.2f}x)\n"
        )

    best_rows = (
        oracle.loc[oracle.groupby("cache_size")["tokens_per_second"].idxmax()]
        .sort_values("cache_size")
    )
    print("LaTeX-friendly summary (best S per cache):")
    for _, r in best_rows.iterrows():
        c = int(r["cache_size"])
        s = int(r["lookahead"])
        tps = float(r["tokens_per_second"])
        print(f"  C={c}: {tps:.2f} TPS (S={s}), {tps / args.baseline_tps:.2f}x")


if __name__ == "__main__":
    main()
