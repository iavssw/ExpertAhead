"""
Plot C=16 TPS vs window recall/precision from comparison sweep data.

Usage:
    python plot_sec5_2_C16.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_sec5_2 import BASELINE_LABEL, MARKER_CYCLE, plot_cache_panel, save_figure

CSV_PATH = (
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "C16_comparison_20260703_154927/sweep.csv"
)
OUTPUT_PATH = (
    "/home/michael/heteroPredict/py/utils/section_5_2/tps_vs_recall_precision_C16.png"
)
CACHE_SIZE = 16

PREFETCH_BUDGET_FILTER = None  # e.g. [4, 8, 12, 16]


def _baseline_tps(sweep: pd.DataFrame) -> float:
    row = sweep[
        (sweep["cache_size"] == CACHE_SIZE)
        & sweep["label"].astype(str).str.contains("RANDOM", na=False)
    ]
    if row.empty:
        raise ValueError(f"No RANDOM baseline for C={CACHE_SIZE} in {CSV_PATH}")
    return float(row["tokens_per_second"].iloc[0])


def _prefetch_rows(sweep: pd.DataFrame) -> pd.DataFrame:
    df = sweep[
        (sweep["cache_size"] == CACHE_SIZE)
        & (sweep["backend"] == "predict")
        & (sweep["lambda_val"] == 0.0)
        & sweep["label"].astype(str).str.startswith("Prefetch Only")
    ].copy()
    if df.empty:
        raise ValueError(f"No Prefetch Only rows for C={CACHE_SIZE} in {CSV_PATH}")
    return df


def main() -> None:
    sweep = pd.read_csv(CSV_PATH)
    baseline_tps = _baseline_tps(sweep)
    df = _prefetch_rows(sweep)

    if PREFETCH_BUDGET_FILTER is not None:
        allowed = {int(b) for b in PREFETCH_BUDGET_FILTER}
        df = df[df["prefetch_budget"].isin(allowed)].copy()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
    })

    needed_cols = [
        "lookahead", "prefetch_budget", "tokens_per_second",
        "pred_window_recall_pct", "pred_window_precision_pct",
    ]
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed_cols).copy()
    for col in ("lookahead", "prefetch_budget"):
        df[col] = df[col].astype(int)

    lookaheads = sorted(df["lookahead"].unique())
    marker_for = {la: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, la in enumerate(lookaheads)}
    norm = Normalize(
        vmin=df["pred_window_precision_pct"].min(),
        vmax=df["pred_window_precision_pct"].max(),
    )
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    plot_cache_panel(
        ax,
        cs=CACHE_SIZE,
        sub=df,
        lookaheads=lookaheads,
        marker_for=marker_for,
        norm=norm,
        cmap=cmap,
        baseline=baseline_tps,
        annotate_full_strides=set(lookaheads),
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Window Precision (%)")
    save_figure(fig, OUTPUT_PATH)
    plt.close(fig)

    print(f"Read {CSV_PATH}")
    print(f"Baseline: {BASELINE_LABEL} = {baseline_tps:g} TPS (1.0x)")
    print(f"Configs plotted: {len(df)}")
    best = df.loc[df["tokens_per_second"].idxmax()]
    print(
        f"Best: S={int(best['lookahead'])} B={int(best['prefetch_budget'])} "
        f"TPS={best['tokens_per_second']:.3f} "
        f"({best['tokens_per_second'] / baseline_tps:.2f}x)"
    )
    print("\nBest per stride:")
    for s in lookaheads:
        sub = df[df["lookahead"] == s]
        row = sub.loc[sub["tokens_per_second"].idxmax()]
        print(
            f"  S={s}: B={int(row['prefetch_budget'])} "
            f"TPS={row['tokens_per_second']:.3f} "
            f"recall={row['pred_window_recall_pct']:.1f}% "
            f"prec={row['pred_window_precision_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
