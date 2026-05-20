#!/usr/bin/env python3
"""Pareto-optimal (cache_size, prefetch_budget) points per lookahead from a sweep CSV.

A prefetch row is *dominated* if another row has >= TPS, <= cache_size, <= prefetch_budget,
with at least one strict inequality. Non-dominated rows form the tradeoff frontier.

Usage:
  python analyze_cache_prefetch_tradeoff.py --csv path/to/sweep.csv [--plot] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd


def _prefetch_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    q = df.get("question", pd.Series("", index=df.index)).astype(str)
    backend = df.get("backend", "").astype(str)
    lab = df.get("label", pd.Series("", index=df.index)).astype(str)
    lam = pd.to_numeric(df.get("lambda_val", 0), errors="coerce").fillna(0.0)
    mask = (
        (backend == "predict")
        & (lam == 0.0)
        & lab.str.contains("Prefetch Only", na=False)
        & q.str.contains("CUSTOM_1_16", na=False)
    )
    out = df.loc[mask].copy()
    out["cache_size"] = pd.to_numeric(out["cache_size"], errors="coerce")
    out["lookahead"] = pd.to_numeric(out["lookahead"], errors="coerce")
    out["prefetch_budget"] = pd.to_numeric(out["prefetch_budget"], errors="coerce")
    out["tokens_per_second"] = pd.to_numeric(out["tokens_per_second"], errors="coerce")
    return out.dropna(subset=["cache_size", "lookahead", "prefetch_budget", "tokens_per_second"])


def pareto_indices(tps: np.ndarray, cache: np.ndarray, budget: np.ndarray) -> List[int]:
    """Return indices of non-dominated points (maximize TPS; minimize cache, budget)."""
    n = len(tps)
    keep: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (
                tps[j] >= tps[i]
                and cache[j] <= cache[i]
                and budget[j] <= budget[i]
                and (tps[j] > tps[i] or cache[j] < cache[i] or budget[j] < budget[i])
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    pred = _prefetch_only_rows(df)
    if pred.empty:
        return pred

    rows_out: List[pd.Series] = []
    for la in sorted(pred["lookahead"].unique()):
        sub = pred[pred["lookahead"] == la].reset_index(drop=True)
        tps = sub["tokens_per_second"].to_numpy(dtype=float)
        cache = sub["cache_size"].to_numpy(dtype=float)
        budget = sub["prefetch_budget"].to_numpy(dtype=float)
        for idx in pareto_indices(tps, cache, budget):
            r = sub.iloc[idx].copy()
            rows_out.append(r)
    if not rows_out:
        return pd.DataFrame()
    return pd.DataFrame(rows_out).sort_values(["lookahead", "tokens_per_second"], ascending=[True, False])


def maybe_plot(pareto: pd.DataFrame, raw: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    if pareto.empty:
        print("[tradeoff] No Pareto points to plot.", file=sys.stderr)
        return

    las = sorted(pareto["lookahead"].unique())
    n = max(1, len(las))
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    ax_list = np.ravel(axs).tolist()
    raw_p = _prefetch_only_rows(raw)
    for ax, la in zip(ax_list, las):
        sub = raw_p[raw_p["lookahead"] == la]
        par = pareto[pareto["lookahead"] == la]
        ax.scatter(sub["cache_size"], sub["prefetch_budget"], c="0.75", s=20, alpha=0.45, label="all prefetch")
        sc = ax.scatter(
            par["cache_size"],
            par["prefetch_budget"],
            c=par["tokens_per_second"],
            cmap="viridis",
            s=70,
            edgecolors="k",
            linewidths=0.5,
            label="Pareto",
        )
        fig.colorbar(sc, ax=ax, shrink=0.85, label="TPS")
        ax.set_xlabel("cache_size")
        ax.set_ylabel("prefetch_budget")
        ax.set_title(f"lookahead={int(la)}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pareto (cache, budget): maximize TPS, minimize cache & B", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="sweep_predict_cached_cache_metrics sweep.csv")
    ap.add_argument("--out-dir", default=None, help="Default: same directory as CSV")
    ap.add_argument("--plot", action="store_true", help="Write tradeoff_pareto_cache_budget.png")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    pareto = analyze(df)
    out_csv = os.path.join(out_dir, "tradeoff_pareto_by_lookahead.csv")
    pareto.to_csv(out_csv, index=False)
    print(f"[tradeoff] Wrote {out_csv} ({len(pareto)} rows)")

    if args.plot:
        png = os.path.join(out_dir, "tradeoff_pareto_cache_budget.png")
        maybe_plot(pareto, df, png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
