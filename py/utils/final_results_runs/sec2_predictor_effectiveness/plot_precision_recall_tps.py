#!/usr/bin/env python3
"""
plot_precision_recall_tps.py — Section 2 thesis figure.

Scatter plot of Window Recall vs Relative TPS (normalized to LRU baseline)
for a given expert cache size.  Points are colored by Window Precision and
annotated with lookahead depth (LA) and prefetch budget (B).

Usage:
    python plot_precision_recall_tps.py \
        --csv 20260606_045623/sweep.csv \
        --cache-size 24 \
        --out precision_recall_tps_C24.png

    # Merge multiple CSV files (e.g. when C=48 run finishes):
    python plot_precision_recall_tps.py \
        --csv 20260606_045623/sweep.csv 20260607_XXXXXX/sweep.csv \
        --cache-size 48 \
        --out precision_recall_tps_C48.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Okabe-Ito markers for different lookahead depths (colorblind-safe)
# ---------------------------------------------------------------------------
LA_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 6: "v", 8: "P", 10: "X", 12: "*", 16: "h"}
LA_COLORS  = {1: "#0072B2", 2: "#009E73", 3: "#D55E00", 4: "#E69F00",
              6: "#CC79A7", 8: "#56B4E9", 10: "#117733", 12: "#882255", 16: "#332288"}

PLOT_STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "text.color": "#1a1a1a",
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
}


def load_data(csv_paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[warn] CSV not found: {p}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No valid CSVs found in: {csv_paths}")
    return pd.concat(frames, ignore_index=True)


def make_plot(df: pd.DataFrame, cache_size: int, out_path: str) -> None:
    df_c = df[df["cache_size"] == cache_size].copy()
    if df_c.empty:
        print(f"[error] No rows for cache_size={cache_size}.", file=sys.stderr)
        sys.exit(1)

    # --- LRU baseline: average TPS across all LRU rows for this cache size ---
    lru_rows = df_c[df_c["backend"] == "cached"]
    if lru_rows.empty or lru_rows["tokens_per_second"].isna().all():
        print("[error] No LRU (cached) rows with TPS found.", file=sys.stderr)
        sys.exit(1)
    lru_tps = lru_rows["tokens_per_second"].mean()
    print(f"[info] LRU baseline TPS (C={cache_size}): {lru_tps:.4f}")

    # --- Predictor rows ---
    pred_rows = df_c[df_c["backend"] == "predict"].copy()

    # Use window-aware recall/precision if available, else fall back to topk metrics.
    if "pred_window_recall_pct" in pred_rows.columns and pred_rows["pred_window_recall_pct"].notna().any():
        recall_col    = "pred_window_recall_pct"
        precision_col = "pred_window_precision_pct"
        recall_label    = "Window Recall, ρ  (%)"
        precision_label = "Window Precision, π  (%)"
    else:
        recall_col    = "pred_hit_rate_routed_topk_pct"
        precision_col = "pred_requested_rate_topk_pct"
        recall_label    = "Recall — Routed Experts Found in Predictions  (%)"
        precision_label = "Precision — Predicted Experts Used by Router  (%)"

    pred_rows = pred_rows.dropna(subset=[recall_col, precision_col, "tokens_per_second"])
    if pred_rows.empty:
        print("[error] No predictor rows with both recall/precision and TPS.", file=sys.stderr)
        sys.exit(1)

    # Ensure prefetch_budget is numeric for proper sorting
    pred_rows["prefetch_budget"] = pd.to_numeric(pred_rows["prefetch_budget"], errors="coerce")
    pred_rows["rel_tps"]  = pred_rows["tokens_per_second"] / lru_tps
    pred_rows["recall"]   = pred_rows[recall_col].astype(float)
    pred_rows["precision"]= pred_rows[precision_col].astype(float)

    lookaheads = sorted(pred_rows["lookahead"].dropna().unique(), key=int)

    # --- Color map: precision (%) → color ---
    prec_min = max(0.0, pred_rows["precision"].min() - 2)
    prec_max = min(100.0, pred_rows["precision"].max() + 2)
    norm = mcolors.Normalize(vmin=prec_min, vmax=prec_max)
    cmap = cm.plasma

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5.5))

        for la in lookaheads:
            sub = pred_rows[pred_rows["lookahead"] == la].sort_values("prefetch_budget")
            if sub.empty:
                continue
            marker = LA_MARKERS.get(int(la), "o")
            color = LA_COLORS.get(int(la), "#555555")

            # Draw line connecting same lookahead points sorted by budget
            ax.plot(
                sub["recall"], sub["rel_tps"],
                color=color, linestyle="-", linewidth=1.2,
                alpha=0.6, zorder=2
            )

            for _, row in sub.iterrows():
                sc = ax.scatter(
                    row["recall"], row["rel_tps"],
                    c=[[cmap(norm(row["precision"]))]],
                    s=90, marker=marker,
                    edgecolors="#222222", linewidths=0.6,
                    zorder=3,
                )
            # Invisible dummy plot for legend entry (line + marker in lookahead color)
            ax.plot([], [], marker=marker, linestyle="-",
                    color=color, alpha=0.8,
                    markeredgecolor="#222222", markeredgewidth=0.6,
                    label=f"LA = {int(la)}", markersize=7)

        # Annotate each point with (LA, B)
        for _, row in pred_rows.iterrows():
            la = int(row["lookahead"]) if pd.notna(row.get("lookahead")) else "?"
            b  = int(row["prefetch_budget"]) if pd.notna(row.get("prefetch_budget")) else "?"
            ax.annotate(
                f"LA{la}/B{b}",
                (row["recall"], row["rel_tps"]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=6.5, alpha=0.85, color="#333333",
            )

        # LRU reference line at y=1
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.5,
                   label="LRU baseline (1.0×)", zorder=2)

        # Colorbar for precision
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(precision_label, fontsize=9)

        ax.set_xlabel(recall_label, fontsize=10)
        ax.set_ylabel(f"Relative TPS  (LRU = 1.0×)", fontsize=10)
        ax.set_title(
            f"Predictor Precision–Recall vs Relative Throughput\n"
            f"Expert Cache = {cache_size} experts / layer",
            fontsize=11, fontweight="bold", pad=10,
        )
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.5)
        ax.set_xlim(left=max(0, pred_rows["recall"].min() - 3),
                    right=min(102, pred_rows["recall"].max() + 3))
        ax.set_ylim(bottom=0.85, top=max(1.5, pred_rows["rel_tps"].max() + 0.05))

        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved → {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="Path(s) to sweep.csv file(s).")
    p.add_argument("--cache-size", type=int, required=True,
                   help="Expert cache size to plot (e.g. 24 or 48).")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path. Defaults to precision_recall_tps_C<N>.png in the script directory.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"precision_recall_tps_C{args.cache_size}.png",
    )
    df = load_data(args.csv)
    make_plot(df, args.cache_size, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
