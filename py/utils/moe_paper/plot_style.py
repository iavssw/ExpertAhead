"""
plot_style.py — Shared publication-quality matplotlib style for the MoE SSD paper.

Keeps fonts, colors, tick/spine settings, and save helpers consistent
across every figure so they look like they belong in the same paper.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Font / RC
# ---------------------------------------------------------------------------
# Use a system serif/sans combination typical in IEEE/ACM papers.
# Falls back gracefully if DejaVu is all that's available.
_FONT_FAMILY = "sans-serif"
_FONT_NAMES  = ["Helvetica Neue", "Arial", "DejaVu Sans"]
_MONO_NAMES  = ["Courier New", "DejaVu Sans Mono"]

mpl.rcParams.update({
    "font.family":        _FONT_FAMILY,
    "font.sans-serif":    _FONT_NAMES,
    "font.monospace":     _MONO_NAMES,
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.7,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "patch.linewidth":    0.8,
})

# ---------------------------------------------------------------------------
# Color palette — hand-curated, colorblind-safe
# ---------------------------------------------------------------------------
COLORS = {
    # Lookahead-depth palette (4 curves)
    "la1":       "#2166ac",   # deep blue
    "la2":       "#f4a582",   # salmon
    "la4":       "#1b7837",   # forest green
    "la8":       "#762a83",   # purple

    # Strategy palette
    "sync":      "#d73027",   # red   — worst
    "async":     "#4393c3",   # steel blue
    "routing":   "#1a9641",   # green — best

    # Precision curves
    "p100":      "#1b7837",
    "p90":       "#4dac26",
    "p70":       "#f4a582",
    "p50":       "#d73027",

    # Reuse scenarios
    "high_reuse": "#2166ac",
    "low_reuse":  "#d73027",

    # Stacked bar components
    "compute":   "#4393c3",
    "hidden":    "#92c5de",
    "blocking":  "#d73027",

    # Annotation / reference
    "measured":  "#252525",
    "threshold": "#e08214",
}

LOOKAHEAD_COLORS = {1: COLORS["la1"], 2: COLORS["la2"],
                    4: COLORS["la4"], 8: COLORS["la8"]}
PRECISION_COLORS = {1.0: COLORS["p100"], 0.9: COLORS["p90"],
                    0.7: COLORS["p70"], 0.5: COLORS["p50"]}
STRATEGY_COLORS  = {"Synchronous": COLORS["sync"],
                    "Async Prefetch": COLORS["async"],
                    "Async + Cache-Aware Routing": COLORS["routing"]}


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def new_fig(width: float = 5.5, height: float = 3.8) -> Tuple[plt.Figure, plt.Axes]:
    """Return a pre-styled (fig, ax) pair."""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def annotate_measured(ax, x, y, label, color=COLORS["measured"],
                      offset=(6, 6), fontsize=8.5):
    """Plot a ★ for a measured benchmark point and label it."""
    ax.scatter([x], [y], marker="*", s=120, color=color, zorder=9,
               linewidths=0.6, edgecolors="white")
    ax.annotate(label, xy=(x, y), xytext=(x + offset[0] * 0.01,
                                           y + offset[1] * 0.01),
                fontsize=fontsize, color=color,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=0.8, relpos=(0, 0)))


def vline_threshold(ax, x, label="threshold", color=COLORS["threshold"],
                    linestyle="--", alpha=0.8, fontsize=8.5):
    """Draw a vertical threshold line."""
    ax.axvline(x, color=color, linestyle=linestyle, linewidth=1.4, alpha=alpha)
    yhi = ax.get_ylim()[1]
    ax.text(x + 0.01, yhi * 0.97, label, color=color,
            fontsize=fontsize, va="top", ha="left",
            rotation=90, rotation_mode="anchor")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig, out_dir: str, stem: str):
    """Save figure as both PNG and PDF into out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)
