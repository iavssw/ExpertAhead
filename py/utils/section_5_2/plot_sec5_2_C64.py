"""
Plot C=64 TPS vs window recall/precision from merged comparison + refine sweeps.

Usage:
    python merge_c64_sweeps.py   # if sec_5_2_raw_C64.csv is missing/stale
    python plot_sec5_2_C64.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_sec5_2 import (
    BASELINE_LABEL,
    MARKER_CYCLE,
    plot_cache_panel,
    save_figure,
)

CSV_PATH = "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_C64.csv"
MERGE_SCRIPT = "/home/michael/heteroPredict/py/utils/section_5_2/merge_c64_sweeps.py"
OUTPUT_PATH = (
    "/home/michael/heteroPredict/py/utils/section_5_2/tps_vs_recall_precision_C64.png"
)

# RANDOM baseline from C64_comparison (S=1).
BASELINE_TPS = 5.756443


def ensure_csv() -> None:
    if os.path.exists(CSV_PATH):
        return
    subprocess.check_call([sys.executable, MERGE_SCRIPT])


def main() -> None:
    ensure_csv()

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

    df = pd.read_csv(CSV_PATH)
    df = df[df["backend"] == "predict"].copy()
    needed_cols = [
        "lookahead", "prefetch_budget", "tokens_per_second",
        "pred_window_recall_pct", "pred_window_precision_pct",
    ]
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
        cs=64,
        sub=df,
        lookaheads=lookaheads,
        marker_for=marker_for,
        norm=norm,
        cmap=cmap,
        baseline=BASELINE_TPS,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Window Precision (%)")
    save_figure(fig, OUTPUT_PATH)
    plt.close(fig)

    print(f"Baseline: {BASELINE_LABEL} = {BASELINE_TPS:g} TPS (1.0x)")
    print(f"Configs plotted: {len(df)}")
    best = df.loc[df["tokens_per_second"].idxmax()]
    print(
        f"Best: S={int(best['lookahead'])} B={int(best['prefetch_budget'])} "
        f"TPS={best['tokens_per_second']:.3f} "
        f"({best['tokens_per_second'] / BASELINE_TPS:.2f}x)"
    )


if __name__ == "__main__":
    main()
