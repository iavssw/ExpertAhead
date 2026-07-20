"""
Plot Prefetch Only TPS vs recall/precision from final_sec5_collection sweep.csv.

Filters to predictor-only rows (label starts with "Prefetch Only").
Writes one PNG per cache size under the collection directory.

Usage:
    python plot_final_sec5_collection.py
    python plot_final_sec5_collection.py --csv /path/to/sweep.csv --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_sec5_2 import BASELINE_LABEL, MARKER_CYCLE, plot_cache_panel, save_figure

DEFAULT_CSV = (
    "/home/michael/heteroPredict/py/utils/final_results_runs/"
    "final_sec5_collection/20260707_002948/sweep.csv"
)


def _random_baselines(df: pd.DataFrame) -> dict[int, float]:
    rand = df[df["label"].str.contains("RANDOM", na=False)]
    out: dict[int, float] = {}
    for cs in sorted(rand["cache_size"].dropna().unique()):
        row = rand[rand["cache_size"] == cs].iloc[0]
        out[int(cs)] = float(row["tokens_per_second"])
    return out


def _prefetch_only(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["backend"] == "predict")
        & df["label"].str.startswith("Prefetch Only", na=False)
    ].copy()
    needed = [
        "cache_size", "lookahead", "prefetch_budget", "tokens_per_second",
        "pred_window_recall_pct", "pred_window_precision_pct",
    ]
    sub = sub.dropna(subset=needed)
    for col in ("cache_size", "lookahead", "prefetch_budget"):
        sub[col] = sub[col].astype(int)
    return sub


def plot_collection(csv_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    baselines = _random_baselines(df)
    predict = _prefetch_only(df)
    if predict.empty:
        raise ValueError("No Prefetch Only rows found in sweep CSV")

    predict.to_csv(out / "prefetch_only.csv", index=False)

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

    cache_sizes = sorted(predict["cache_size"].unique())
    lookaheads = sorted(predict["lookahead"].unique())
    marker_for = {la: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, la in enumerate(lookaheads)}
    norm = Normalize(
        vmin=predict["pred_window_precision_pct"].min(),
        vmax=predict["pred_window_precision_pct"].max(),
    )
    cmap = plt.get_cmap("viridis")

    # Combined panel
    n_panels = len(cache_sizes)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.5 * n_panels, 5.5), squeeze=False, constrained_layout=True,
    )
    for ax, cs in zip(axes[0], cache_sizes):
        sub = predict[predict["cache_size"] == cs]
        baseline = baselines[cs]
        plot_cache_panel(
            ax, cs=cs, sub=sub, lookaheads=lookaheads, marker_for=marker_for,
            norm=norm, cmap=cmap, baseline=baseline,
        )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Window Precision (%)")
    combined = out / "tps_vs_recall_precision.png"
    save_figure(fig, str(combined))
    plt.close(fig)

    # Per-cache panels
    for cs in cache_sizes:
        sub = predict[predict["cache_size"] == cs]
        baseline = baselines[cs]
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        plot_cache_panel(
            ax, cs=cs, sub=sub, lookaheads=lookaheads, marker_for=marker_for,
            norm=norm, cmap=cmap, baseline=baseline,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Window Precision (%)")
        path = out / f"tps_vs_recall_precision_C{cs}.png"
        save_figure(fig, str(path))
        plt.close(fig)

        best = sub.loc[sub["tokens_per_second"].idxmax()]
        print(
            f"C={cs}: baseline RANDOM={baseline:.4f} TPS | "
            f"best S={int(best['lookahead'])} B={int(best['prefetch_budget'])} "
            f"TPS={best['tokens_per_second']:.3f} ({best['tokens_per_second']/baseline:.2f}x) | "
            f"saved {path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to the directory containing sweep.csv",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or str(Path(args.csv).parent)
    plot_collection(args.csv, out_dir)


if __name__ == "__main__":
    main()
