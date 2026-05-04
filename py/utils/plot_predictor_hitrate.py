#!/usr/bin/env python3
"""
plot_predictor_hitrate.py
=========================
Reads training_metrics.json from every layer of every trained predictor and
plots union_recall@K vs lookahead depth — no inference runs required.

Usage:
    python py/utils/plot_predictor_hitrate.py \
        --model-dir trainingData/qwen3_30b/final_multi_input_model \
        --out-dir lookahead_4way_5prompts
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

PLOT_STYLE = {
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d", "axes.labelcolor": "#e0e4f0",
    "xtick.color": "#a0a8c0", "ytick.color": "#a0a8c0",
    "text.color": "#e0e4f0", "grid.color": "#2a2d3d",
    "grid.linestyle": "--", "grid.alpha": 0.6,
    "legend.facecolor": "#1a1d27", "legend.edgecolor": "#3a3d4d",
}
LOOKAHEAD_COLORS = {
    1: "#4fc3f7", 2: "#81d4fa", 3: "#80cbc4", 4: "#a5d6a7",
    5: "#c5e1a5", 6: "#fff176", 8: "#ffcc80", 10: "#ffab40",
    12: "#ce93d8", 14: "#f48fb1", 16: "#ef9a9a",
}


def _color(la: int) -> str:
    return LOOKAHEAD_COLORS.get(la, "#aaaaaa")


def load_metrics(model_dir: str) -> Dict[int, Dict[int, dict]]:
    """Return {lookahead: {layer_idx: training_metrics_dict}}."""
    data: Dict[int, Dict[int, dict]] = {}
    root = Path(model_dir)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        # Expect names like eh1_h32_f4
        parts = subdir.name.split("_")
        f_part = next((p for p in parts if p.startswith("f") and p[1:].isdigit()), None)
        if f_part is None:
            continue
        lookahead = int(f_part[1:])
        data[lookahead] = {}
        for layer_dir in sorted(subdir.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            layer_idx = int(layer_dir.name.split("_")[1])
            metrics_path = layer_dir / "training_metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                data[lookahead][layer_idx] = json.load(f)
    return data


def extract_recall(metrics_by_la: Dict[int, Dict[int, dict]]) -> Tuple[List[int], Dict[int, List[float]]]:
    """Return sorted lookaheads and {lookahead: [recall_per_layer]}."""
    lookaheads = sorted(metrics_by_la.keys())
    recalls: Dict[int, List[float]] = {}
    for la in lookaheads:
        layer_data = metrics_by_la[la]
        vals = []
        for layer_idx in sorted(layer_data.keys()):
            m = layer_data[layer_idx]
            # Try best_val_recalls first, then best
            val = None
            bvr = m.get("best_val_recalls", {})
            for k, v in bvr.items():
                if "recall" in k.lower():
                    val = v
                    break
            if val is None:
                val = m.get("best")
            if val is not None:
                vals.append(float(val) * 100.0)  # convert to %
        recalls[la] = vals
    return lookaheads, recalls


def plot_hitrate_vs_lookahead(
    lookaheads: List[int],
    recalls: Dict[int, List[float]],
    out_dir: str,
    label: str = "",
) -> None:
    means = [float(np.mean(recalls[la])) for la in lookaheads if recalls.get(la)]
    stds  = [float(np.std(recalls[la]))  for la in lookaheads if recalls.get(la)]
    las   = [la for la in lookaheads if recalls.get(la)]

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Predictor Hit Rate (union_recall@K) vs Lookahead Depth",
                     fontsize=13, fontweight="bold")

        # Left: mean ± std across layers
        ax = axes[0]
        ax.errorbar(las, means, yerr=stds, fmt="o-", color="#4fc3f7",
                    linewidth=2, markersize=7, capsize=4, elinewidth=1.5,
                    label="Mean ± std across layers")
        ax.set_xlabel("Lookahead depth (tokens)")
        ax.set_ylabel("union_recall@K (%)")
        ax.set_title("Mean Hit Rate vs Lookahead")
        ax.set_xticks(las)
        ax.grid(True)
        ax.legend(fontsize=8)

        # Right: per-layer heatmap
        ax2 = axes[1]
        max_layer = max(max(recalls[la]) and len(recalls[la]) for la in las if recalls.get(la))
        mat = np.full((len(las), max_layer), np.nan)
        for r, la in enumerate(las):
            vals = recalls[la]
            mat[r, :len(vals)] = vals
        im = ax2.imshow(np.ma.masked_invalid(mat), aspect="auto",
                        cmap="viridis", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax2, label="union_recall@K (%)")
        ax2.set_yticks(range(len(las)))
        ax2.set_yticklabels([str(la) for la in las])
        ax2.set_xlabel("Layer index")
        ax2.set_ylabel("Lookahead depth")
        ax2.set_title("Per-layer Hit Rate Heatmap")

        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"predictor_hitrate_vs_lookahead{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved {out_path}")

    # Print summary table
    print(f"\n{'Lookahead':>10} {'Mean recall%':>14} {'Std':>8} {'Layers':>8}")
    print("-" * 44)
    for la, mean, std in zip(las, means, stds):
        n = len(recalls[la])
        print(f"{la:>10} {mean:>14.2f} {std:>8.2f} {n:>8}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True,
                   help="Directory containing eh*_h*_fN subdirs with per-layer training_metrics.json")
    p.add_argument("--out-dir", default=".", help="Output directory for plots")
    p.add_argument("--label", default="", help="Optional suffix appended to output filename")
    args = p.parse_args()

    if not HAS_PLOT:
        print("ERROR: matplotlib and numpy are required. pip install matplotlib numpy")
        raise SystemExit(1)

    print(f"[plot] Loading metrics from {args.model_dir}")
    metrics = load_metrics(args.model_dir)
    if not metrics:
        print(f"[plot] No training_metrics.json files found under {args.model_dir}")
        raise SystemExit(1)
    print(f"[plot] Found lookaheads: {sorted(metrics.keys())}")

    lookaheads, recalls = extract_recall(metrics)
    plot_hitrate_vs_lookahead(lookaheads, recalls, args.out_dir, label=f"_{args.label}" if args.label else "")


if __name__ == "__main__":
    main()
