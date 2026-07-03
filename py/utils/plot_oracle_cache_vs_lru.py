#!/usr/bin/env python3
"""Plots for oracle full-union lookahead sweeps vs LRU.

Modes:
  default   — grouped bars: LRU + perfect oracle per lookahead
  --tradeoff — line chart: 7/8 predictor @ C vs LRU @ C+32 (draft-model vs bigger cache)
"""

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

LA_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
    4: "#C44E52",
    6: "#8172B3",
}
LRU_COLOR = "#333333"
NOISY_COLOR = "#E69F00"
PERFECT_COLOR = "#0072B2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="py/utils/final_results_runs/oracle_full_union_la_sweep/"
        "oracle_full_union_20260702_204246.csv",
    )
    p.add_argument("--out", default=None, help="Output PNG (default: alongside CSV)")
    p.add_argument(
        "--ms-per-token",
        type=float,
        default=49.0,
        help="Hardware ceiling latency (ms/token); TPS = 1000 / ms_per_token",
    )
    p.add_argument(
        "--hardware-ceiling-tps",
        type=float,
        default=None,
        help="Override TPS ceiling (default: 1000 / --ms-per-token)",
    )
    p.add_argument(
        "--annotate-speedup",
        action="store_true",
        help="Print speedup vs LRU above oracle bars",
    )
    p.add_argument(
        "--tradeoff",
        action="store_true",
        help="Plot 7/8 predictor @ C vs LRU @ C+Δ (instead of grouped lookahead bars).",
    )
    p.add_argument(
        "--cache-delta",
        type=int,
        default=32,
        help="Extra cache slots to compare against (default: 32).",
    )
    p.add_argument(
        "--tradeoff-lookahead",
        type=int,
        default=1,
        help="Lookahead for noisy/perfect oracle in tradeoff plot (default: 1).",
    )
    p.add_argument(
        "--noisy-agreement",
        type=float,
        default=0.875,
        help="Routing agreement for draft-model proxy (default: 0.875 = 7/8).",
    )
    return p.parse_args()


def load_sweep_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["question"] == "ORACLE_BASELINE_SWEEP"].copy()
    df["tokens_per_second"] = pd.to_numeric(df["tokens_per_second"], errors="coerce")
    df["cache_size"] = pd.to_numeric(df["cache_size"], errors="coerce")
    df["lookahead"] = pd.to_numeric(df["lookahead"], errors="coerce")
    if "oracle_routing_agreement" in df.columns:
        df["oracle_routing_agreement"] = pd.to_numeric(
            df["oracle_routing_agreement"], errors="coerce"
        )
    return df


def hw_ceiling_tps(args: argparse.Namespace) -> float:
    if args.hardware_ceiling_tps is not None:
        return args.hardware_ceiling_tps
    return 1000.0 / args.ms_per_token


def tps_map(df: pd.DataFrame, label_mask: pd.Series) -> dict[int, float]:
    sub = df[label_mask].dropna(subset=["tokens_per_second", "cache_size"])
    sub = sub.drop_duplicates("cache_size")
    return dict(zip(sub["cache_size"].astype(int), sub["tokens_per_second"].astype(float)))


def oracle_rows(df: pd.DataFrame, *, perfect: bool, lookahead: int) -> pd.DataFrame:
    if perfect:
        mask = df["label"].str.startswith("Oracle Full Union", na=False)
    else:
        mask = df["label"].str.startswith("Oracle Noisy", na=False)
        if "oracle_routing_agreement" in df.columns:
            mask = mask | (
                df["oracle_routing_agreement"].notna()
                & (df["oracle_routing_agreement"] < 1.0)
            )
    out = df[mask & (df["lookahead"] == lookahead)].copy()
    return out.dropna(subset=["tokens_per_second"])


def plot_grouped_bars(df: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    lru = (
        df[df["label"] == "Neither (LRU)"]
        .drop_duplicates("cache_size")
        .sort_values("cache_size")
    )
    oracle = df[df["label"].str.startswith("Oracle Full Union", na=False)].copy()
    oracle = oracle.dropna(subset=["tokens_per_second", "lookahead"])
    oracle["lookahead"] = oracle["lookahead"].astype(int)

    cache_sizes = sorted(int(c) for c in df["cache_size"].dropna().unique())
    lookaheads = sorted(oracle["lookahead"].unique())
    lru_map = dict(zip(lru["cache_size"].astype(int), lru["tokens_per_second"]))
    ceiling = hw_ceiling_tps(args)

    series_specs = [("LRU", None, LRU_COLOR, "LRU (no prefetch)")]
    for la in lookaheads:
        series_specs.append(
            (f"LA={la}", la, LA_COLORS.get(la, "#888888"), f"Oracle LA={la}")
        )

    n_caches = len(cache_sizes)
    n_series = len(series_specs)
    cluster_width = 0.82
    bar_width = cluster_width / n_series
    x_centers = np.arange(n_caches, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, (key, la, color, label) in enumerate(series_specs):
        offsets = x_centers - cluster_width / 2 + (i + 0.5) * bar_width
        heights = []
        for c in cache_sizes:
            if key == "LRU":
                heights.append(float(lru_map[c]))
            else:
                row = oracle[(oracle["cache_size"] == c) & (oracle["lookahead"] == la)]
                heights.append(float(row.iloc[0]["tokens_per_second"]) if not row.empty else 0.0)

        bars = ax.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        for bar, h, c in zip(bars, heights, cache_sizes):
            if h <= 0:
                continue
            label_text = f"{h:.1f}"
            if key != "LRU" and args.annotate_speedup:
                speedup = h / lru_map[c]
                label_text = f"{h:.1f}\n({speedup:.2f}×)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.25,
                label_text,
                ha="center",
                va="bottom",
                fontsize=6.5 if key != "LRU" else 7,
                color=color if key != "LRU" else "black",
                fontweight="bold" if key == "LRU" else "normal",
            )

    ax.axhline(
        ceiling,
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label=f"Hardware ceiling ({ceiling:.1f} TPS, {args.ms_per_token:.0f} ms/tok)",
        zorder=5,
    )

    ax.set_xlabel("Expert cache size (slots per layer)")
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title(
        "Perfect prefetch (oracle) vs LRU across cache sizes\n"
        "Grouped by cache size; bars = lookahead window"
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(c) for c in cache_sizes])

    ymax = max(ceiling, oracle["tokens_per_second"].max()) + 2.5
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Oracle ≈ 100% routing predictor; long LA proxies draft-model horizon. "
        "Dashed line = compute-only ceiling (no SSD stalls).",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_predictor_vs_cache_tradeoff(
    df: pd.DataFrame, args: argparse.Namespace, out_path: Path
) -> None:
    la = args.tradeoff_lookahead
    delta = args.cache_delta
    agreement = args.noisy_agreement

    lru_map = tps_map(df, df["label"] == "Neither (LRU)")

    noisy = oracle_rows(df, perfect=False, lookahead=la)
    if "oracle_routing_agreement" in noisy.columns:
        noisy = noisy[
            noisy["oracle_routing_agreement"].isna()
            | (np.isclose(noisy["oracle_routing_agreement"], agreement, rtol=0, atol=1e-3))
        ]
    noisy_map = dict(zip(noisy["cache_size"].astype(int), noisy["tokens_per_second"].astype(float)))

    perfect = oracle_rows(df, perfect=True, lookahead=la)
    perfect_map = dict(
        zip(perfect["cache_size"].astype(int), perfect["tokens_per_second"].astype(float))
    )

    max_cache = max(int(c) for c in df["cache_size"].dropna().unique())
    cache_sizes = sorted(
        c for c in lru_map if c + delta <= max_cache and c in noisy_map
    )
    if not cache_sizes:
        raise SystemExit(
            f"No comparable cache sizes for tradeoff (need C and C+{delta} in sweep, plus noisy LA={la})."
        )

    ceiling = hw_ceiling_tps(args)

    lru_at_c = [lru_map[c] for c in cache_sizes]
    lru_at_c_plus = [lru_map[c + delta] for c in cache_sizes]
    noisy_at_c = [noisy_map[c] for c in cache_sizes]
    perfect_at_c = [perfect_map.get(c, np.nan) for c in cache_sizes]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.array(cache_sizes, dtype=float)

    ax.plot(x, lru_at_c, "o-", color=LRU_COLOR, linewidth=2.0, label=f"LRU @ C")
    ax.plot(
        x,
        lru_at_c_plus,
        "s--",
        color="#888888",
        linewidth=2.0,
        label=f"LRU @ C+{delta} (bigger cache, no predictor)",
    )
    ax.plot(
        x,
        noisy_at_c,
        "^-",
        color=NOISY_COLOR,
        linewidth=2.2,
        markersize=8,
        label=f"7/8 predictor @ C (LA={la}, A={agreement:g})",
    )
    if any(np.isfinite(v) for v in perfect_at_c):
        ax.plot(
            x,
            perfect_at_c,
            "D-",
            color=PERFECT_COLOR,
            linewidth=1.8,
            alpha=0.85,
            label=f"Perfect oracle @ C (LA={la})",
        )

    for c, noisy_tps, lru_plus_tps in zip(cache_sizes, noisy_at_c, lru_at_c_plus):
        winner = "predictor" if noisy_tps >= lru_plus_tps else f"+{delta} cache"
        ax.annotate(
            winner,
            (c, max(noisy_tps, lru_plus_tps)),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#444444",
        )

    ax.axhline(
        ceiling,
        color="#c0392b",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=f"Hardware ceiling ({ceiling:.1f} TPS)",
    )

    ax.set_xlabel("Predictor cache size C (slots per layer)")
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title(
        f"7/8 routing predictor @ C vs adding {delta} cache slots (LRU @ C+{delta})\n"
        f"Lookahead = {la}"
    )
    ax.set_xticks(cache_sizes)
    ax.set_xticklabels([str(c) for c in cache_sizes])

    ymax = max(
        ceiling,
        max(lru_at_c),
        max(lru_at_c_plus),
        max(noisy_at_c),
        max(v for v in perfect_at_c if np.isfinite(v)) if perfect_at_c else 0,
    ) + 2.0
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        f"Annotations mark whether 7/8 predictor @ C beats LRU @ C+{delta} at each cache size.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if args.out:
        out_path = Path(args.out)
    elif args.tradeoff:
        out_path = csv_path.with_name(csv_path.stem + "_predictor_vs_cache32.png")
    else:
        out_path = csv_path.with_name(csv_path.stem + "_tps_vs_cache.png")

    df = load_sweep_df(csv_path)
    if args.tradeoff:
        plot_predictor_vs_cache_tradeoff(df, args, out_path)
    else:
        plot_grouped_bars(df, args, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
