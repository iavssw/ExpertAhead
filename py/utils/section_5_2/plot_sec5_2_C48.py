"""
Plot C=48 TPS vs window recall/precision.

Usage:
    python plot_sec5_2_C48.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_sec5_2 import BASELINE_LABEL, MARKER_CYCLE, plot_cache_panel, save_figure

CSV_PATH = "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_C48.csv"
OUTPUT_PATH = (
    "/home/michael/heteroPredict/py/utils/section_5_2/tps_vs_recall_precision_C48.png"
)
CACHE_SIZE = 48
BASELINE_TPS = 4.61
TPS_DECIMALS = 2

# Set to None to plot all budgets present; e.g. [10, 16, 20, 24]
# PREFETCH_BUDGET_FILTER = None
PREFETCH_BUDGET_FILTER = [10, 16, 20, 24, 32]


def main() -> None:
    baseline_tps = BASELINE_TPS

    df = pd.read_csv(CSV_PATH)
    df = df[
        (df["cache_size"] == CACHE_SIZE)
        & df["label"].str.startswith("Prefetch Only", na=False)
    ].copy()

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
        cs=48,
        sub=df,
        lookaheads=lookaheads,
        marker_for=marker_for,
        norm=norm,
        cmap=cmap,
        baseline=baseline_tps,
        annotate_full_strides={1, 5},
        tps_decimals=TPS_DECIMALS,
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
        f"TPS={best['tokens_per_second']:.{TPS_DECIMALS}f} "
        f"({best['tokens_per_second'] / baseline_tps:.2f}x)"
    )
    print("\nBest per stride:")
    for s in lookaheads:
        sub = df[df["lookahead"] == s]
        row = sub.loc[sub["tokens_per_second"].idxmax()]
        print(
            f"  S={s}: B={int(row['prefetch_budget'])} "
            f"TPS={row['tokens_per_second']:.{TPS_DECIMALS}f} "
            f"recall={row['pred_window_recall_pct']:.1f}% "
            f"prec={row['pred_window_precision_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
