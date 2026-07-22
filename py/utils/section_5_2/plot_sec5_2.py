"""
Plot: Tokens/sec (Y) vs. Predictor Window Recall (X), colored by Window Precision.
Points sharing a stride value (with varying `prefetch_budget`) are connected
by a line and share a marker shape. Each point is annotated with its
stride/budget combo. A horizontal reference line marks baseline TPS.

Usage:
    Edit the CONFIG block below, then run:
        python plot_sec5_2.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

# =========================
# CONFIG
# =========================
CSV_PATHS = [
    "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_C16.csv",
    "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_C32.csv",
]

# Baseline (e.g. naive decode) TPS, per cache_size -- each cache_size gets its own
# baseline line since they're typically measured under different cache constraints.
BASELINE_TPS_BY_CACHE_SIZE = {
    # Parallel
    16: 2.54,
    32: 3.27,
    # Sequential
    # 16: 1.65,
    # 32: 2.78,
}

BASELINE_LABEL = "Baseline TPS"

# Optionally restrict to one cache_size; set to None to facet (one subplot per
# cache_size present in the data, each with its own baseline line)
CACHE_SIZE_FILTER = None    # e.g. 32

# Optionally restrict prefetch budgets; set to None to plot all budgets present.
PREFETCH_BUDGET_FILTER = None    # e.g. [10, 16, 20, 24]

OUTPUT_DIR = "/home/michael/heteroPredict/py/utils/section_5_2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tps_vs_recall_precision.png")
OUTPUT_PATH_BY_CACHE = os.path.join(OUTPUT_DIR, "tps_vs_recall_precision_C{cache_size}.png")

# Annotate every point when a panel has at most this many configs; otherwise
# label only min/max budget per stride (legend already encodes S).
ANNOTATE_ALL_MAX_POINTS = 16
# On dense panels, still label every budget for these strides (e.g. best performer).
ANNOTATE_FULL_STRIDES = {1}

# Marker shapes cycled per distinct stride value
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _budgets_to_annotate(grp: pd.DataFrame, annotate_all: bool) -> set[int]:
    budgets = sorted(grp["prefetch_budget"].unique())
    if annotate_all or len(budgets) <= 2:
        return {int(b) for b in budgets}
    return {int(budgets[0]), int(budgets[-1])}


# =========================
# Global Plot Style
# =========================
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


def plot_cache_panel(
    ax,
    *,
    cs: int,
    sub: pd.DataFrame,
    lookaheads,
    marker_for,
    norm,
    cmap,
    baseline: float,
    annotate_full_strides: set[int] | None = None,
    tps_decimals: int | None = None,
) -> None:
    if annotate_full_strides is None:
        annotate_full_strides = ANNOTATE_FULL_STRIDES
    sub = sub.copy()
    plot_baseline = baseline
    if tps_decimals is not None:
        display_baseline = round(baseline, tps_decimals)
    else:
        display_baseline = baseline
    annotate_all = len(sub) <= ANNOTATE_ALL_MAX_POINTS
    for stride in lookaheads:
        grp = sub[sub["lookahead"] == stride].sort_values("prefetch_budget")
        if grp.empty:
            continue
        marker = marker_for[stride]
        y = grp["tokens_per_second"] / plot_baseline
        stride_annotate_all = annotate_all or int(stride) in annotate_full_strides
        label_budgets = _budgets_to_annotate(grp, stride_annotate_all)

        ax.plot(
            grp["pred_window_recall_pct"], y,
            color="0.55", linewidth=1.2, zorder=1,
        )
        ax.scatter(
            grp["pred_window_recall_pct"], y,
            c=grp["pred_window_precision_pct"], cmap=cmap, norm=norm,
            marker=marker, s=70, edgecolor="black", linewidth=0.6,
            zorder=2, label=f"S={stride}",
        )
        for i, ((_, row), yv) in enumerate(zip(grp.iterrows(), y)):
            budget = int(row["prefetch_budget"])
            if budget not in label_budgets:
                continue
            if stride_annotate_all:
                text = f"S={int(row['lookahead'])},B={budget}"
            else:
                text = f"B={budget}"
            # Stagger labels slightly to reduce overlap on dense panels.
            yoff = 4 + (i % 2) * 3
            ax.annotate(
                text,
                (row["pred_window_recall_pct"], yv),
                textcoords="offset points", xytext=(5, yoff),
                fontsize=6 if stride_annotate_all else 5, color="0.25",
            )

    ax.axhline(1.0, color="firebrick", linestyle=":", linewidth=1.4, zorder=0)
    baseline_txt = (
        f"{display_baseline:.{tps_decimals}f}"
        if tps_decimals is not None
        else f"{plot_baseline:g}"
    )
    ax.text(
        ax.get_xlim()[0], 1.0,
        f"{BASELINE_LABEL} (1.0x, {baseline_txt} TPS) ", color="firebrick",
        fontsize=9, va="bottom", ha="left",
    )
    ax.set_xlabel("Window Recall (%)")
    ax.set_ylabel("Normalized TPS")
    if tps_decimals is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{tps_decimals}f"))
    # ax.set_title(f"cache_size = {cs}")

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), title="Stride", loc="best")


def save_figure(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot to {path}")


def main() -> None:
    # =========================
    # Load + filter data
    # =========================
    frames = []
    for p in CSV_PATHS:
        if not os.path.exists(p):
            print(f"Skipping missing CSV: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No CSV files found; nothing to plot.")
    df = pd.concat(frames, ignore_index=True)

    df = df[df["backend"] == "predict"].copy()

    if CACHE_SIZE_FILTER is not None:
        df = df[df["cache_size"] == CACHE_SIZE_FILTER].copy()

    if PREFETCH_BUDGET_FILTER is not None:
        allowed = {int(b) for b in PREFETCH_BUDGET_FILTER}
        df = df[df["prefetch_budget"].isin(allowed)].copy()

    needed_cols = [
        "lookahead", "prefetch_budget", "tokens_per_second",
        "pred_window_recall_pct", "pred_window_precision_pct",
    ]
    df = df.dropna(subset=needed_cols).copy()

    for c in ["lookahead", "prefetch_budget"]:
        df[c] = df[c].astype(int)

    # =========================
    # Plot (combined + one file per cache_size)
    # =========================
    cache_sizes = sorted(df["cache_size"].unique())
    lookaheads = sorted(df["lookahead"].unique())
    marker_for = {la: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, la in enumerate(lookaheads)}

    norm = Normalize(
        vmin=df["pred_window_precision_pct"].min(),
        vmax=df["pred_window_precision_pct"].max(),
    )
    cmap = plt.get_cmap("viridis")
    title = " "

    n_panels = len(cache_sizes)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.5 * n_panels, 5.5), squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    for ax, cs in zip(axes, cache_sizes):
        sub = df[df["cache_size"] == cs]
        baseline = BASELINE_TPS_BY_CACHE_SIZE.get(cs)
        if baseline is None:
            raise ValueError(f"No baseline TPS set for cache_size={cs} in BASELINE_TPS_BY_CACHE_SIZE")
        plot_cache_panel(
            ax, cs=cs, sub=sub, lookaheads=lookaheads, marker_for=marker_for,
            norm=norm, cmap=cmap, baseline=baseline,
        )

    fig.suptitle(title, y=1.02)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Window Precision (%)")
    save_figure(fig, OUTPUT_PATH)
    plt.close(fig)

    for cs in cache_sizes:
        sub = df[df["cache_size"] == cs]
        baseline = BASELINE_TPS_BY_CACHE_SIZE[cs]
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        plot_cache_panel(
            ax, cs=cs, sub=sub, lookaheads=lookaheads, marker_for=marker_for,
            norm=norm, cmap=cmap, baseline=baseline,
        )
        fig.suptitle(title, y=1.02)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Window Precision (%)")
        save_figure(fig, OUTPUT_PATH_BY_CACHE.format(cache_size=cs))
        plt.close(fig)


if __name__ == "__main__":
    main()
