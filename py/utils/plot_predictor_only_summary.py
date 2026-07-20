#!/usr/bin/env python3
"""Plot predictor-only throughput summary vs cache size.

Shows:
1. RANDOM baseline TPS
2. Best predictor-only TPS at S=1 (best budget)
3. Best predictor-only TPS over all strides (annotated with winner)

Accepts a single --csv or multiple --run-dirs (merged per cache, richest source wins).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_expert_ahead_evaluation import THESIS_STYLE, load_merged_sweeps, summarize_all

plt.rcParams.update(THESIS_STYLE)
plt.rcParams["figure.dpi"] = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Single sweep.csv path")
    src.add_argument(
        "--run-dirs",
        nargs="+",
        help="Run directories containing sweep.csv (merged per cache)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: alongside csv or --out-dir)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory when using --run-dirs (default: first run dir parent)",
    )
    p.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        for col in ("tokens_per_second", "cache_size", "lookahead", "prefetch_budget"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["tokens_per_second", "cache_size"])
        df = df[df["tokens_per_second"] > 0].copy()
        out_path = Path(args.out) if args.out else Path(args.csv).parent / "predictor_only_summary.png"
        summaries = summarize_all(df)
    else:
        df = load_merged_sweeps(args.run_dirs)
        out_dir = Path(args.out_dir or Path(args.run_dirs[-1]).parent)
        out_path = Path(args.out) if args.out else out_dir / "predictor_only_summary.png"
        summaries = summarize_all(df)

    if not summaries:
        raise SystemExit("No cache summaries produced.")

    cache_sizes = [s["cache_size"] for s in summaries]
    random_tps = [s["random_tps"] for s in summaries]
    la1_tps = [s["tps_la1"] for s in summaries]
    best_tps = [s["tps_ea"] for s in summaries]
    la1_b = [s["b_la1"] for s in summaries]
    best_la = [s["x_ea"] for s in summaries]
    best_b = [s["b_ea"] for s in summaries]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(
        cache_sizes,
        random_tps,
        marker="^",
        color="#999999",
        linestyle="--",
        linewidth=2,
        markersize=7,
        label="Random replacement",
        zorder=2,
    )
    ax.plot(
        cache_sizes,
        la1_tps,
        marker="s",
        color="#4C72B0",
        linewidth=2,
        markersize=7,
        label="ExpertAhead, $S{=}1$ (best $B$)",
        zorder=3,
    )
    ax.plot(
        cache_sizes,
        best_tps,
        marker="o",
        color="#0072B2",
        linewidth=2.2,
        markersize=7,
        label="ExpertAhead, best $S$ (best $B$)",
        zorder=4,
    )

    for c, tps, la, b in zip(cache_sizes, best_tps, best_la, best_b):
        if pd.notna(b):
            ax.annotate(
                f"$S={int(la)}$",
                (c, tps),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#003366",
            )

    ax.set_xticks(cache_sizes)
    ax.set_xlabel("Expert cache size $C$ (experts per layer)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Prefetch-only throughput vs. cache size")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    table = pd.DataFrame(
        {
            "cache_size": cache_sizes,
            "random_tps": random_tps,
            "s1_best_tps": la1_tps,
            "s1_best_budget": la1_b,
            "best_tps": best_tps,
            "best_s": best_la,
            "best_budget": best_b,
        }
    )
    print("\nSummary:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
