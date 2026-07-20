#!/usr/bin/env python3
"""Plots for sec1 expert cache policy sweeps (sweep_cache_policy.py output)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

POLICY_ORDER = ["LRU", "LFRU", "LFU", "PREFILL", "RANDOM", "MRU", "MFU"]
POLICY_COLORS = {
    "LRU": "#333333",
    "LFRU": "#0072B2",
    "LFU": "#55A868",
    "PREFILL": "#C44E52",
    "RANDOM": "#DD8452",
    "MRU": "#8172B3",
    "MFU": "#CCB974",
}
CACHE_SIZE_COLORS = {
    8: "#B3CDE3",
    16: "#8C96C6",
    24: "#8856A7",
    32: "#6BAED6",
    40: "#3182BD",
    48: "#08519C",
    56: "#004C6D",
    64: "#002A40",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="py/utils/final_results_runs/sec1_cache_policy/20260630_224555/sweep.csv",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: alongside CSV)")
    p.add_argument(
        "--ms-per-token",
        type=float,
        default=49.0,
        help="Hardware ceiling latency (ms/token); TPS = 1000 / ms_per_token",
    )
    return p.parse_args()


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["cache_size"] = pd.to_numeric(df["cache_size"], errors="coerce")
    df["tps"] = pd.to_numeric(df["tps"], errors="coerce")
    df["cache_hit_rate"] = pd.to_numeric(df["cache_hit_rate"], errors="coerce")
    if "lambda" in df.columns:
        df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    return df.dropna(subset=["cache_size", "policy"])


def policy_order(policies) -> list[str]:
    known = [p for p in POLICY_ORDER if p in policies]
    extra = sorted(set(policies) - set(known))
    return known + extra


def _grouped_bar_layout(n_groups: int, n_series: int) -> tuple[np.ndarray, float, float]:
    cluster_width = 0.82
    bar_width = cluster_width / n_series
    x_centers = np.arange(n_groups, dtype=float)
    return x_centers, cluster_width, bar_width


def _cache_size_color(cache_size: int) -> str:
    if cache_size in CACHE_SIZE_COLORS:
        return CACHE_SIZE_COLORS[cache_size]
    return plt.cm.Blues(0.3 + 0.6 * (cache_size / 64))


def _series_heights(
    df: pd.DataFrame,
    policies: list[str],
    cache_size: int,
    column: str,
) -> list[float]:
    sub = df[df["cache_size"] == cache_size].set_index("policy")
    return [float(sub.loc[p, column]) if p in sub.index else 0.0 for p in policies]


def plot_tps_bars_vs_cache(df: pd.DataFrame, out_path: Path, hw_tps: float) -> None:
    cache_sizes = [int(c) for c in sorted(df["cache_size"].unique())]
    policies = policy_order(df["policy"].unique())
    x_centers, cluster_width, bar_width = _grouped_bar_layout(len(policies), len(cache_sizes))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, cache_size in enumerate(cache_sizes):
        color = _cache_size_color(cache_size)
        offsets = x_centers - cluster_width / 2 + (i + 0.5) * bar_width
        heights = _series_heights(df, policies, cache_size, "tps")
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            label=f"C={cache_size}",
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, h in zip(bars, heights):
            if h <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.25,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=color,
            )

    ax.axhline(
        hw_tps,
        color="#C44E52",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label=f"No SSD Loads ({hw_tps:.1f} TPS)",
        zorder=5,
    )
    ax.set_xlabel("Eviction policy")
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title(
        "Decode Throughput by Expert Cache Size and Policy\n"
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(policies)
    ax.set_ylim(0, max(hw_tps * 1.08, df["tps"].max() * 1.12))
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)

    # fig.text(
    #     0.5, 0.01,
    #     "Dashed line = compute-only ceiling (no SSD stalls). 10 prompts × 256 decode tokens.",
    #     ha="center", fontsize=9, color="#555555",
    # )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hitrate_bars_vs_cache(df: pd.DataFrame, out_path: Path) -> None:
    cache_sizes = [int(c) for c in sorted(df["cache_size"].unique())]
    policies = policy_order(df["policy"].unique())
    x_centers, cluster_width, bar_width = _grouped_bar_layout(len(policies), len(cache_sizes))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, cache_size in enumerate(cache_sizes):
        color = _cache_size_color(cache_size)
        offsets = x_centers - cluster_width / 2 + (i + 0.5) * bar_width
        heights = _series_heights(df, policies, cache_size, "cache_hit_rate")
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            label=f"C={cache_size}",
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, h in zip(bars, heights):
            if h <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=color,
            )

    ax.set_xlabel("Eviction policy")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_title(
        "Expert Cache Hit Rate by Size and Policy"
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(policies)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tps_vs_cache(df: pd.DataFrame, out_path: Path, hw_tps: float) -> None:
    cache_sizes = sorted(df["cache_size"].unique())
    policies = policy_order(df["policy"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(
        hw_tps,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Hardware ceiling ({hw_tps:.1f} TPS)",
        zorder=0,
    )

    for policy in policies:
        sub = df[df["policy"] == policy].sort_values("cache_size")
        xs = sub["cache_size"].astype(int).tolist()
        ys = sub["tps"].tolist()
        color = POLICY_COLORS.get(policy, None)
        ax.plot(
            xs, ys,
            marker="o",
            linewidth=2,
            markersize=6,
            label=policy,
            color=color,
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7.5,
                color=color or "black",
            )

    ax.set_xlabel("Expert cache size (slots per layer)")
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title("Expert Cache Policy\nDecode TPS vs cache size")
    ax.set_xticks(cache_sizes)
    ax.set_xticklabels([str(int(c)) for c in cache_sizes])
    ax.legend(loc="upper right", ncol=2, framealpha=0.92)
    ax.set_ylim(bottom=0, top=max(hw_tps * 1.08, df["tps"].max() * 1.12))

    fig.text(
        0.5, 0.01,
        "Dashed line = compute-only ceiling (no SSD stalls). 10 prompts × 256 decode tokens.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hitrate_vs_cache(df: pd.DataFrame, out_path: Path) -> None:
    cache_sizes = sorted(df["cache_size"].unique())
    policies = policy_order(df["policy"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for policy in policies:
        sub = df[df["policy"] == policy].sort_values("cache_size")
        color = POLICY_COLORS.get(policy, None)
        ax.plot(
            sub["cache_size"].astype(int),
            sub["cache_hit_rate"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=policy,
            color=color,
        )

    ax.set_xlabel("Expert cache size (slots per layer)")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_title("Expert cache policy hit rate vs cache size\nλ=0, wikitext")
    ax.set_xticks(cache_sizes)
    ax.set_xticklabels([str(int(c)) for c in cache_sizes])
    ax.legend(loc="lower right", ncol=2, framealpha=0.92)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)
    if "lambda" in df.columns and df["lambda"].nunique() == 1:
        lam = df["lambda"].iloc[0]
        df = df[df["lambda"] == lam]

    stem = csv_path.stem
    hw_tps = 1000.0 / args.ms_per_token
    plot_tps_vs_cache(df, out_dir / f"{stem}_tps_vs_cache.png", hw_tps)
    plot_hitrate_vs_cache(df, out_dir / f"{stem}_hitrate_vs_cache.png")
    plot_tps_bars_vs_cache(df, out_dir / f"{stem}_tps_bars_vs_cache.png", hw_tps)
    plot_hitrate_bars_vs_cache(df, out_dir / f"{stem}_hitrate_bars_vs_cache.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
