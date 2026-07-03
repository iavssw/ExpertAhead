"""
Plot: Tokens/sec (Y) vs. Predictor Window Recall (X), colored by Window Precision.
Points sharing a `lookahead` value (with varying `prefetch_budget`) are connected
by a line and share a marker shape. Each point is annotated with its
lookahead/budget combo. A horizontal reference line marks baseline TPS.

Usage:
    Edit the CONFIG block below, then run:
        python plot_tps_recall_precision.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# =========================
# CONFIG
# =========================
CSV_PATHS = [
    "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw_32_take2.csv",   # <- path to first results CSV
    "/home/michael/heteroPredict/py/utils/section_5_2/sec_5_2_raw.csv",   # <- path to second results CSV
]

# Baseline (e.g. naive decode) TPS, per cache_size -- each cache_size gets its own
# baseline line since they're typically measured under different cache constraints.
BASELINE_TPS_BY_CACHE_SIZE = {
    # Parallel
    # 16: 2.54,   
    # 32: 3.44,   
    # Sequential
    16: 1.65,
    32: 2.78,   
}

BASELINE_LABEL = "Baseline TPS"
 
# Optionally restrict to one cache_size; set to None to facet (one subplot per
# cache_size present in the data, each with its own baseline line)
CACHE_SIZE_FILTER = None    # e.g. 32
 
OUTPUT_PATH = "tps_vs_recall_precision.png"
 
# Marker shapes cycled per distinct `lookahead` value
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
 
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
 
# =========================
# Load + filter data
# =========================
df = pd.concat([pd.read_csv(p) for p in CSV_PATHS], ignore_index=True)
 
# Only rows from the predictive backend have recall/precision/lookahead data
df = df[df["backend"] == "predict"].copy()
 
if CACHE_SIZE_FILTER is not None:
    df = df[df["cache_size"] == CACHE_SIZE_FILTER].copy()
 
needed_cols = [
    "lookahead", "prefetch_budget", "tokens_per_second",
    "pred_window_recall_pct", "pred_window_precision_pct",
]
df = df.dropna(subset=needed_cols).copy()
 
for c in ["lookahead", "prefetch_budget"]:
    df[c] = df[c].astype(int)
 
# =========================
# Plot (faceted by cache_size, since each has its own baseline TPS)
# =========================
cache_sizes = sorted(df["cache_size"].unique())
lookaheads = sorted(df["lookahead"].unique())
marker_for = {la: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, la in enumerate(lookaheads)}
 
norm = Normalize(vmin=df["pred_window_precision_pct"].min(),
                  vmax=df["pred_window_precision_pct"].max())
cmap = plt.get_cmap("viridis")
 
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
 
    for la in lookaheads:
        grp = sub[sub["lookahead"] == la].sort_values("prefetch_budget")
        if grp.empty:
            continue
        marker = marker_for[la]
        y = grp["tokens_per_second"] / baseline  # normalized to baseline
 
        # connecting line (neutral gray so it doesn't fight the color-coded points)
        ax.plot(
            grp["pred_window_recall_pct"], y,
            color="0.55", linewidth=1.2, zorder=1,
        )
 
        # color-coded scatter
        ax.scatter(
            grp["pred_window_recall_pct"], y,
            c=grp["pred_window_precision_pct"], cmap=cmap, norm=norm,
            marker=marker, s=70, edgecolor="black", linewidth=0.6,
            zorder=2, label=f"lookahead={la}",
        )
 
        # annotate each point with lookahead/budget
        for (_, row), yv in zip(grp.iterrows(), y):
            ax.annotate(
                f"L{int(row['lookahead'])},B{int(row['prefetch_budget'])}",
                (row["pred_window_recall_pct"], yv),
                textcoords="offset points", xytext=(5, 4),
                fontsize=7, color="0.25",
            )
 
    # Baseline reference line -- always at 1.0 once normalized
    ax.axhline(1.0, color="firebrick", linestyle=":", linewidth=1.4, zorder=0)
    ax.text(
        ax.get_xlim()[1], 1.0,
        f" {BASELINE_LABEL} (1.0x, {baseline:g} tok/s)", color="firebrick",
        fontsize=9, va="bottom", ha="right",
    )
 
    ax.set_xlabel("Window Recall (%)")
    ax.set_ylabel("Tokens/sec, normalized to baseline (x)")
    ax.set_title(f"cache_size = {cs}")
 
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), title="Lookahead", loc="best")
 
fig.suptitle("Throughput vs. Predictor Window Recall (colored by Precision)", y=1.02)
 
# Shared colorbar for precision
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85, pad=0.02)
cbar.set_label("Window Precision (%)")
 
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved plot to {OUTPUT_PATH}")