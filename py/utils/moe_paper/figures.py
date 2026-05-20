"""
figures.py — Publication-quality figure generators for the MoE SSD-prefetch paper.

Four figures covering the thesis subsection on simulated tradeoffs:
  fig1_tps_vs_cache_hit_rate  — (i)   impact of cache hit rate
  fig2_recall_vs_tps          — (ii)  impact of prediction recall
                                (iii) impact of lookahead depth (as curves)
  fig3_tps_vs_cache_size      — (iv)  impact of cache capacity
  fig5_latency_breakdown      — supporting: latency decomposition
  fig8_strategy_comparison    — supporting: sync vs async strategy comparison

Each function: accepts a pre-computed DataFrame, draws one figure,
saves PNG + PDF, returns the figure object.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from plot_style import (
    new_fig, save_fig, annotate_measured, vline_threshold,
    COLORS, LOOKAHEAD_COLORS, PRECISION_COLORS, STRATEGY_COLORS,
)
from params import MODEL, HW, MEASURED_POINTS


# ---------------------------------------------------------------------------
# Fig 1: TPS vs Cache Hit Rate
# ---------------------------------------------------------------------------

def fig1_tps_vs_cache_hit_rate(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    TPS as a function of cache hit rate (no predictor, synchronous on-demand loads).
    Single y-axis. Only the measured all-in-VRAM ceiling is annotated.
    """
    fig, ax = new_fig()

    ax.plot(df["cache_hit_rate"] * 100, df["tps"],
            color=COLORS["async"], linewidth=2.2, label="Sync on-demand (model)")

    # Shade area below the curve to emphasize blocking penalty
    ax.fill_between(df["cache_hit_rate"] * 100, df["tps"],
                    alpha=0.12, color=COLORS["blocking"], label="_nolegend_")

    # Only annotate the all-in-VRAM measured ceiling (cache_hit_rate == 1.0)
    for mp in MEASURED_POINTS:
        if mp.cache_hit_rate == 1.0:
            annotate_measured(ax, mp.cache_hit_rate * 100, mp.tps,
                              "All experts in RAM", offset=(-45, 4))

    ax.set_xlabel("Cache Hit Rate (%)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Throughput vs. Cache Hit Rate\n"
                 "(synchronous on-demand SSD loads for misses)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_tps_vs_cache_hit_rate")
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Recall vs TPS (multiple lookahead depths)
# ---------------------------------------------------------------------------

def fig2_recall_vs_tps(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """TPS vs recall, one curve per lookahead depth.
    Covers thesis points (ii) prediction recall and (iii) lookahead depth.
    """
    fig, ax = new_fig()

    for la, grp in df.groupby("lookahead_depth"):
        grp = grp.sort_values("recall")
        c = LOOKAHEAD_COLORS.get(la, COLORS["la1"])
        ax.plot(grp["recall"] * 100, grp["tps"],
                color=c, label=f"Lookahead = {la}", linewidth=2.2)

    # Horizontal reference: base TPS ceiling
    ax.axhline(HW.tps_compute, color=COLORS["measured"], linewidth=1.2,
               linestyle=":", label=f"Base ceiling ({HW.tps_compute:.0f} TPS)")

    ax.set_xlabel("Predictor Recall (%)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Throughput vs. Predictor Recall\n"
                 "(by lookahead depth, fixed precision \u2248 85%)")
    ax.set_xlim(50, 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_recall_vs_tps")
    return fig


# ---------------------------------------------------------------------------
# Fig 3: TPS vs Cache Capacity  [thesis point (iv)]
# ---------------------------------------------------------------------------

def fig3_tps_vs_cache_size(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    TPS vs per-layer cache size (number of expert weight sets kept in VRAM).
    Shows both synchronous baseline and async-prefetch curves so the reader
    can see (a) how much cache capacity alone buys and (b) how much the
    predictor adds on top.

    Hit rate is derived from the calibrated exponential working-set model
    (natural_hit_rate in params.py), anchored to the measured point
    cache=24 -> hit_rate~72%.
    """
    fig, ax = new_fig()

    cache_sizes = df["cache_size_per_layer"].values
    hit_rates   = df["cache_hit_rate"].values * 100

    ax.plot(cache_sizes, df["sync_tps"],
            color=COLORS["sync"], linewidth=2.2, linestyle="-",
            label="Sync on-demand (no predictor)")
    ax.plot(cache_sizes, df["predict_tps"],
            color=COLORS["async"], linewidth=2.2, linestyle="--",
            label="Async prefetch (with predictor)")

    # Shade gap between strategies
    ax.fill_between(cache_sizes, df["sync_tps"], df["predict_tps"],
                    alpha=0.12, color=COLORS["async"], label="_nolegend_")

    # Annotate the measured calibration point (cache=24, hit_rate~72%)
    cal_row = df[df["cache_size_per_layer"] == 24]
    if not cal_row.empty:
        r = cal_row.iloc[0]
        ax.scatter([24], [r["sync_tps"]], marker="*", s=110,
                   color=COLORS["measured"], zorder=9, edgecolors="white",
                   linewidths=0.6)
        ax.annotate(f"Measured\n(cache=24, {r['cache_hit_rate']*100:.0f}% hit)",
                    xy=(24, r["sync_tps"]),
                    xytext=(26, r["sync_tps"] * 0.88),
                    fontsize=8, color=COLORS["measured"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["measured"], lw=0.8))

    # Ceiling reference
    ax.axhline(HW.tps_compute, color=COLORS["measured"], linewidth=1.1,
               linestyle=":", label=f"Base ceiling ({HW.tps_compute:.0f} TPS)")

    # Secondary x-axis showing implied hit rate
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(cache_sizes[::2])
    ax2.set_xticklabels([f"{h:.0f}%" for h in hit_rates[::2]], fontsize=8)
    ax2.set_xlabel("Implied Cache Hit Rate (%)", fontsize=9)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(False)

    ax.set_xlabel("Cache Size (experts / layer)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Throughput vs. Cache Capacity\n"
                 "(working-set model: hit rate = 1 - exp(-C / 18.8))")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_tps_vs_cache_size")
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Precision vs Recall Sensitivity
# ---------------------------------------------------------------------------

def fig3_precision_recall_sensitivity(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    Show that recall drives TPS far more strongly than precision.
    Multiple precision curves vs recall x-axis.
    """
    fig, ax = new_fig()

    for prec, grp in df.groupby("precision"):
        grp = grp.sort_values("recall")
        c = PRECISION_COLORS.get(round(prec, 2), "#555555")
        label = f"Precision = {int(prec*100)}%"
        ax.plot(grp["recall"] * 100, grp["tps"],
                color=c, label=label, linewidth=2.2)

    ax.axhline(HW.tps_compute, color=COLORS["measured"], linewidth=1.2,
               linestyle=":", label=f"Base ceiling ({HW.tps_compute:.0f} TPS)")

    ax.set_xlabel("Predictor Recall (%)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Recall vs. Precision Sensitivity\n"
                 "(recall dominates; precision matters less)")
    ax.set_xlim(50, 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper left")

    # Annotation arrow showing recall dominates
    recall_vals = df[df["precision"] == df["precision"].min()]["recall"].values * 100
    tps_low_rec = df[(df["precision"] == df["precision"].min()) &
                     (df["recall"] < 0.65)]["tps"].mean()
    tps_hi_rec  = df[(df["precision"] == df["precision"].min()) &
                     (df["recall"] > 0.95)]["tps"].mean()
    if not np.isnan(tps_low_rec) and not np.isnan(tps_hi_rec):
        ax.annotate("", xy=(97, tps_hi_rec), xytext=(97, tps_low_rec),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["blocking"],
                                    lw=1.4))
        ax.text(97.5, (tps_hi_rec + tps_low_rec) / 2,
                f"Recall\nimpact", fontsize=8, color=COLORS["blocking"],
                ha="left", va="center")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_precision_recall_sensitivity")
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Lookahead Depth vs Overlap Capacity
# ---------------------------------------------------------------------------

def fig4_lookahead_overlap(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    Overlap budget (ms) and max hidden loads vs lookahead depth.
    Shows diminishing returns.
    """
    fig, ax = new_fig()

    ax.plot(df["lookahead_depth"], df["max_hidden_loads"],
            color=COLORS["async"], linewidth=2.4, marker="o", markersize=6,
            label="Max hideable SSD loads")

    # Demand misses reference line
    experts = MODEL.active_k * MODEL.num_layers
    demand_misses_per_token = experts * (1 - 0.72)  # at typical LRU hit rate
    ax.axhline(demand_misses_per_token, color=COLORS["blocking"],
               linewidth=1.3, linestyle="--",
               label=f"Demand misses/token\n(LRU hit rate ≈ 72%, "
                     f"{demand_misses_per_token:.0f} experts)")

    ax2 = ax.twinx()
    ax2.plot(df["lookahead_depth"], df["overlap_budget_ms"],
             color=COLORS["la4"], linewidth=1.8, linestyle="-.",
             marker="s", markersize=5, label="Overlap budget (ms)")
    ax2.set_ylabel("Overlap Budget (ms)", color=COLORS["la4"])
    ax2.tick_params(axis="y", labelcolor=COLORS["la4"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("Lookahead Depth (tokens)")
    ax.set_ylabel("Max Hideable SSD Loads (experts)")
    ax.set_title("Overlap Budget & Max Hidden Loads vs. Lookahead Depth\n"
                 "(diminishing returns beyond depth ≈ 4)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_lookahead_overlap")
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Stacked Latency Breakdown
# ---------------------------------------------------------------------------

def fig5_latency_breakdown(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    Stacked area / bar breakdown of token latency:
      compute | hidden SSD | blocking SSD
    vs predictor recall.
    """
    fig, ax = new_fig()

    recalls_pct = df["recall"] * 100

    ax.stackplot(
        recalls_pct,
        df["compute_ms"],
        df["hidden_ssd_ms"],
        df["blocking_ssd_ms"],
        labels=["GPU Compute", "Hidden SSD (overlapped)", "Blocking SSD (stall)"],
        colors=[COLORS["compute"], COLORS["hidden"], COLORS["blocking"]],
        alpha=0.88,
    )

    # Overlay compute floor
    ax.axhline(HW.t_compute_ms, color=COLORS["measured"],
               linewidth=1.1, linestyle=":", label=f"Compute floor ({HW.t_compute_ms:.0f} ms)")

    ax.set_xlabel("Predictor Recall (%)")
    ax.set_ylabel("Token Latency (ms)")
    ax.set_title("Token Latency Decomposition vs. Predictor Recall\n"
                 "(stacked: compute / hidden SSD / blocking SSD)")
    ax.set_xlim(recalls_pct.min(), recalls_pct.max())
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_latency_breakdown")
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Pipeline Collapse Curve
# ---------------------------------------------------------------------------

def fig6_pipeline_collapse(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    Sharp TPS degradation when unique_future_misses > max_hidden_loads.
    X-axis normalised to max_hidden_loads (threshold = 1.0).
    """
    fig, ax = new_fig()

    max_hidden = df["max_hidden_loads"].iloc[0]
    norm = df["normalized_misses"]

    ax.plot(norm, df["tps"],
            color=COLORS["async"], linewidth=2.4)
    ax.fill_between(norm, df["tps"],
                    where=(norm > 1.0), alpha=0.20,
                    color=COLORS["blocking"],
                    label="Pipeline stall region")
    ax.fill_between(norm, df["tps"],
                    where=(norm <= 1.0), alpha=0.10,
                    color=COLORS["routing"],
                    label="Fully hidden region")

    # Threshold line
    ax.axvline(1.0, color=COLORS["threshold"], linewidth=1.6,
               linestyle="--", label=f"Overlap capacity threshold\n"
                                      f"(= {max_hidden:.0f} experts)")
    ax.set_xlabel("Unique Future Misses / Max Hidden Loads")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Pipeline Collapse: TPS vs. Unique Future Misses\n"
                 "(sharp knee at overlap capacity threshold)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

    # Annotate the knee
    knee_idx = (norm - 1.0).abs().argmin()
    knee_tps = df["tps"].iloc[knee_idx]
    ax.annotate("Collapse knee", xy=(1.0, knee_tps),
                xytext=(1.3, knee_tps * 1.15),
                arrowprops=dict(arrowstyle="->", color=COLORS["blocking"], lw=1.3),
                fontsize=9, color=COLORS["blocking"])

    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_pipeline_collapse")
    return fig


# ---------------------------------------------------------------------------
# Fig 7: Reuse Sensitivity Study
# ---------------------------------------------------------------------------

def fig7_reuse_sensitivity(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    High reuse (Qwen-like) vs low reuse (Mixtral-like):
    shows why Qwen-3's expert locality is more prefetch-friendly.
    """
    fig, ax = new_fig()

    scenario_style = {
        "High reuse (Qwen-like)":   dict(color=COLORS["high_reuse"], linestyle="-"),
        "Medium reuse":              dict(color=COLORS["async"],       linestyle="--"),
        "Low reuse (Mixtral-like)":  dict(color=COLORS["low_reuse"],   linestyle="-."),
    }

    for scenario, grp in df.groupby("reuse_scenario"):
        grp = grp.sort_values("recall")
        style = scenario_style.get(scenario,
                                   dict(color=COLORS["compute"], linestyle=":"))
        ax.plot(grp["recall"] * 100, grp["tps"],
                label=scenario, linewidth=2.2, **style)

    ax.axhline(HW.tps_compute, color=COLORS["measured"], linewidth=1.1,
               linestyle=":", label=f"Base ceiling ({HW.tps_compute:.0f} TPS)")

    ax.set_xlabel("Predictor Recall (%)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Expert Reuse Sensitivity\n"
                 "(high reuse → more loads hidden per prefetch)")
    ax.set_xlim(50, 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper left")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig7_reuse_sensitivity")
    return fig


# ---------------------------------------------------------------------------
# Fig 8: Compare Strategies
# ---------------------------------------------------------------------------

def fig8_strategy_comparison(df: pd.DataFrame, out_dir: str) -> plt.Figure:
    """
    Three strategies across cache hit rates:
      1. Synchronous on-demand
      2. Async prefetch (predictor)
      3. Async prefetch + cache-aware routing
    """
    fig, ax = new_fig()

    hr_pct = df["cache_hit_rate"] * 100

    strategies = [
        ("Synchronous\n(on-demand)", "sync_tps",    COLORS["sync"],    "-"),
        ("Async Prefetch",           "async_tps",   COLORS["async"],   "--"),
        ("Async + Cache-Aware\nRouting", "routing_tps", COLORS["routing"], "-."),
    ]
    for label, col, color, ls in strategies:
        ax.plot(hr_pct, df[col], color=color, linestyle=ls,
                linewidth=2.2, label=label)

    # Shade the gap between async and sync
    ax.fill_between(hr_pct, df["sync_tps"], df["async_tps"],
                    alpha=0.10, color=COLORS["async"],
                    label="_nolegend_")
    ax.fill_between(hr_pct, df["async_tps"], df["routing_tps"],
                    alpha=0.10, color=COLORS["routing"],
                    label="_nolegend_")

    # Measured points
    for mp in MEASURED_POINTS:
        annotate_measured(ax, mp.cache_hit_rate * 100, mp.tps, mp.label)

    ax.axhline(HW.tps_compute, color=COLORS["measured"], linewidth=1.1,
               linestyle=":", label=f"Base ceiling ({HW.tps_compute:.0f} TPS)")

    ax.set_xlabel("Cache Hit Rate (%)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Throughput vs. Cache Hit Rate: Strategy Comparison")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig8_strategy_comparison")
    return fig
