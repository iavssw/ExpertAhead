#!/usr/bin/env python3
"""
plot.py — Comprehensive Expert Predictor Results Plotter
=========================================================
Automatically discovers all metrics recorded in training_metrics.json files
and generates a full suite of plots.

Generated figures
-----------------
1. per_layer/layer_XX.png
       One figure per layer with subplots for every metric:
       train vs val curves across epochs.

2. cross_layer/
       One figure per metric showing best-val across all layers.
       Good for seeing which layers are easier / harder to predict.

3. heatmap.png
       2-D grid (Y=metric, X=layer) showing best val value.
       At a glance overview of everything.

4. summary.png
       Key metrics only (full_match, any_correct, top1_exact, mean_overlap)
       overlaid on a single cross-layer figure.

Usage
-----
    python plot.py --sweep_dir ../../trainingData/mixtral_8x7b/predictor_models
    python plot.py --sweep_dir ../../trainingData/mixtral_8x7b/predictor_models \\
                   --out_dir  ../../trainingData/mixtral_8x7b/plots \\
                   --phase    val          # or train or both (default: val)
"""

import argparse
import json
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "acc":         "Full Match Acc",
    "any_correct": "Any Correct (≥1)",
    "top1_exact":  "Top-1 Exact",
    "mean_overlap":"Mean Overlap",
    "mean_prefix": "Mean Prefix Length",
    "loss":        "Loss",
}

SUMMARY_METRICS       = ["acc", "any_correct", "top1_exact"]   # 0–1 range
SUMMARY_COUNT_METRICS = ["mean_overlap", "mean_prefix"]         # range > 1 (0..top_k)
KEY_METRIC      = "acc"   # used for "best" in cross-layer / heatmap by default


def _metric_label(key: str) -> str:
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    # auto-label topN_in_pred, exact_set_k, ...
    if key.startswith("top") and key.endswith("_in_pred"):
        n = key[3:-8]
        return f"Top-{n} in Pred"
    if key.startswith("exact_set_"):
        k = key[10:]
        return f"Exact Set@{k}"
    return key.replace("_", " ").title()


def load_all(sweep_dir: pathlib.Path):
    """
    Walk sweep_dir for training_metrics.json files.

    Returns
    -------
    results : dict
        {layer_idx -> {hidden_dim -> metrics_dict}}
    all_metrics : list[str]
        Sorted list of all metric keys found across all files (excluding 'loss'
        from train since it dominates scale; loss gets its own subplot).
    """
    results: Dict[int, Dict[int, dict]] = {}
    all_val_keys: set = set()

    for p in sorted(sweep_dir.rglob("training_metrics.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"[warn] {p}: {e}")
            continue

        layer  = data.get("layer_idx")
        hidden = data.get("hidden_dim")
        if layer is None or hidden is None:
            # Try to infer from path
            parts = p.parts
            for part in parts:
                if part.startswith("layer_"):
                    try:
                        layer = int(part[6:])
                    except ValueError:
                        pass
                if part.startswith("hidden_"):
                    try:
                        hidden = int(part[7:])
                    except ValueError:
                        pass
        if layer is None or hidden is None:
            print(f"[warn] cannot determine layer/hidden for {p}, skipping")
            continue

        # Collect metric keys from epochs
        for ep in data.get("epochs", []):
            val_dict = ep.get("val", {})
            # also handle flat keys for old format
            if not val_dict:
                val_dict = {k[4:]: v for k, v in ep.items() if k.startswith("val_")
                            and isinstance(v, float)}
            all_val_keys.update(val_dict.keys())

        results.setdefault(layer, {})[hidden] = data

    # Preferred ordering: important metrics first
    priority = ["loss", "acc", "any_correct", "top1_exact", "mean_overlap", "mean_prefix"]
    others   = sorted(k for k in all_val_keys if k not in priority)
    ordered  = [k for k in priority if k in all_val_keys] + others

    return results, ordered


def _get_epoch_series(data: dict, phase: str, metric: str) -> Tuple[List[int], List[float]]:
    """Extract (epoch_numbers, metric_values) for a given phase and metric key."""
    xs, ys = [], []
    for ep in data.get("epochs", []):
        epoch_num = ep.get("epoch", len(xs) + 1)
        # New format: ep['val'] / ep['train'] dicts
        phase_dict = ep.get(phase, {})
        if metric in phase_dict:
            xs.append(epoch_num)
            ys.append(phase_dict[metric])
            continue
        # Flat fallback: val_acc, train_loss, etc.
        flat_key = f"{phase}_{metric}"
        if flat_key in ep:
            xs.append(epoch_num)
            ys.append(ep[flat_key])
    return xs, ys


def _best_val(data: dict, metric: str = "acc") -> Optional[float]:
    """Return the best (max or min) val value for a metric across all epochs."""
    vals = _get_epoch_series(data, "val", metric)[1]
    if not vals:
        return None
    return min(vals) if metric == "loss" else max(vals)


def _colors(n: int):
    return [cm.tab10(i / max(n - 1, 1)) for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Per-layer: all metrics in subplots
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_layer(results, all_metrics, phases, out_dir: pathlib.Path):
    """One PNG per layer, with one subplot per metric."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_colors = {"train": "#4C72B0", "val": "#DD8452"}

    for layer_idx in sorted(results.keys()):
        hidden_runs = results[layer_idx]
        hidden_dims = sorted(hidden_runs.keys())

        ncols = 3
        nrows = math.ceil(len(all_metrics) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 5, nrows * 3.5),
                                 squeeze=False)
        fig.suptitle(f"Layer {layer_idx} — Training Curves", fontsize=14, fontweight="bold")

        h_colors = _colors(len(hidden_dims))

        for ax_idx, metric in enumerate(all_metrics):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            ax.set_title(_metric_label(metric), fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(True, alpha=0.25)
            if metric not in ["loss"] + SUMMARY_COUNT_METRICS:
                ax.set_ylim(0, 1.05)

            for h_color, hidden in zip(h_colors, hidden_dims):
                data = hidden_runs[hidden]
                for phase in phases:
                    xs, ys = _get_epoch_series(data, phase, metric)
                    if not ys:
                        continue
                    p_color = phase_colors.get(phase, h_color)
                    ls = "-" if phase == "val" else "--"
                    label = f"h={hidden}" + (" val" if len(phases) > 1 else "")
                    ax.plot(xs, ys, color=p_color if len(hidden_dims) == 1 else h_color,
                            linestyle=ls, marker="o", markersize=3, label=label,
                            alpha=0.85)

            if ax_idx == 0:
                ax.legend(fontsize=7, ncol=2)

        # Hide unused axes
        for ax_idx in range(len(all_metrics), nrows * ncols):
            axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

        fig.tight_layout()
        path = out_dir / f"layer_{layer_idx:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Per-layer: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Cross-layer: one figure per metric
# ──────────────────────────────────────────────────────────────────────────────

def plot_cross_layer(results, all_metrics, out_dir: pathlib.Path):
    """For each metric, one figure with best-val vs layer (one line per hidden_dim)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_hidden = sorted({h for d in results.values() for h in d})
    all_layers = sorted(results.keys())
    h_colors   = _colors(len(all_hidden))

    for metric in all_metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.35), 4))
        ax.set_title(f"Best Val {_metric_label(metric)} by Layer", fontsize=11)
        ax.set_xlabel("Layer Index"); ax.set_ylabel(_metric_label(metric))
        ax.set_xticks(all_layers)
        ax.grid(True, alpha=0.25)
        if metric not in ["loss"] + SUMMARY_COUNT_METRICS:
            ax.set_ylim(0, 1.05)

        for color, h in zip(h_colors, all_hidden):
            xs, ys = [], []
            for layer in all_layers:
                data = results.get(layer, {}).get(h)
                if data is None:
                    continue
                best = _best_val(data, metric)
                if best is not None:
                    xs.append(layer); ys.append(best)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=5, color=color,
                        label=f"hidden={h}", linewidth=1.8)

        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        safe = metric.replace("/", "_")
        path = out_dir / f"{safe}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Cross-layer: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Heatmap: metric × layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap_all_metrics(results, all_metrics, out_dir: pathlib.Path):
    """2-D heatmap: rows=metric, columns=layer. Uses the best hidden_dim per cell."""
    all_layers = sorted(results.keys())
    # Pick layers that have data for at least one metric
    metrics_to_plot = [m for m in all_metrics if m != "loss"]

    grid = np.full((len(metrics_to_plot), len(all_layers)), np.nan)

    for col, layer in enumerate(all_layers):
        hidden_runs = results.get(layer, {})
        for row, metric in enumerate(metrics_to_plot):
            # Take the best across all hidden dims
            bests = [_best_val(d, metric) for d in hidden_runs.values()]
            bests = [b for b in bests if b is not None]
            if bests:
                grid[row, col] = max(bests)   # higher = better for acc metrics

    n_rows, n_cols = len(metrics_to_plot), len(all_layers)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.55), max(5, n_rows * 0.7)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(grid) if not np.all(np.isnan(grid)) else 0,
                   vmax=np.nanmax(grid) if not np.all(np.isnan(grid)) else 1)

    ax.set_xticks(range(n_cols)); ax.set_xticklabels(all_layers, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_metric_label(m) for m in metrics_to_plot], fontsize=8)
    ax.set_xlabel("Layer Index"); ax.set_title("Best Val — All Metrics × Layer", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Best Val")

    mean_v = np.nanmean(grid)
    for row in range(n_rows):
        for col in range(n_cols):
            v = grid[row, col]
            if not np.isnan(v):
                ax.text(col, row, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v < mean_v else "black")

    fig.tight_layout()
    path = out_dir / "heatmap_all_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Summary: key metrics on one cross-layer figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary(results, all_metrics, out_dir: pathlib.Path):
    """Two summary figures:
    1. summary.png     — 0-1 range metrics (acc, any_correct, top1_exact, topN_in_pred, exact_set_k)
    2. summary_counts.png — count-range metrics (mean_overlap, mean_prefix)
    """
    all_layers = sorted(results.keys())

    # --- Figure 1: 0-1 range metrics ---
    # Include SUMMARY_METRICS + any topN_in_pred / exact_set_k that are present
    rate_metrics = [
        m for m in all_metrics
        if m in SUMMARY_METRICS
        or m.endswith("_in_pred")
        or m.startswith("exact_set_")
    ]
    if rate_metrics:
        colors = _colors(len(rate_metrics))
        fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.35), 5))
        ax.set_title("Summary — Accuracy Metrics by Layer (Best Val)", fontsize=12)
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Metric Value (0–1)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(all_layers); ax.grid(True, alpha=0.25)

        for color, metric in zip(colors, rate_metrics):
            xs, ys = [], []
            for layer in all_layers:
                hidden_runs = results.get(layer, {})
                bests = [_best_val(d, metric) for d in hidden_runs.values()]
                bests = [b for b in bests if b is not None]
                if bests:
                    xs.append(layer); ys.append(max(bests))
            if xs:
                ax.plot(xs, ys, marker="o", markersize=6, color=color,
                        label=_metric_label(metric), linewidth=2)

        ax.legend(fontsize=9, ncol=3)
        fig.tight_layout()
        path = out_dir / "summary.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Summary (accuracy): {path}")

    # --- Figure 2: count-range metrics ---
    count_metrics = [m for m in SUMMARY_COUNT_METRICS if m in all_metrics]
    if count_metrics:
        colors = _colors(len(count_metrics))
        fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.35), 5))
        ax.set_title("Summary — Count Metrics by Layer (Best Val)", fontsize=12)
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Mean Count")
        ax.set_xticks(all_layers); ax.grid(True, alpha=0.25)

        for color, metric in zip(colors, count_metrics):
            xs, ys = [], []
            for layer in all_layers:
                hidden_runs = results.get(layer, {})
                bests = [_best_val(d, metric) for d in hidden_runs.values()]
                bests = [b for b in bests if b is not None]
                if bests:
                    xs.append(layer); ys.append(max(bests))
            if xs:
                ax.plot(xs, ys, marker="o", markersize=6, color=color,
                        label=_metric_label(metric), linewidth=2)

        ax.legend(fontsize=9)
        fig.tight_layout()
        path = out_dir / "summary_counts.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Summary (counts):   {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Comprehensive expert predictor results plotter."
    )
    p.add_argument("--sweep_dir", required=True,
                   help="Root dir containing training_metrics.json files")
    p.add_argument("--out_dir",   default=None,
                   help="Output dir for figures (default: <sweep_dir>/plots)")
    p.add_argument("--phase",     choices=["val", "train", "both"], default="val",
                   help="Which phase to plot in per-layer curves (default: val)")
    p.add_argument("--no_per_layer",   action="store_true",
                   help="Skip per-layer figures (fast mode)")
    p.add_argument("--no_cross_layer", action="store_true",
                   help="Skip per-metric cross-layer figures")
    args = p.parse_args()

    sweep_dir = pathlib.Path(args.sweep_dir)
    out_dir   = pathlib.Path(args.out_dir) if args.out_dir else sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from: {sweep_dir}")
    results, all_metrics = load_all(sweep_dir)

    if not results:
        print("No training_metrics.json files found. Check --sweep_dir.")
        return

    layers_found  = sorted(results.keys())
    hiddens_found = sorted({h for d in results.values() for h in d})
    print(f"  Layers:      {layers_found}")
    print(f"  Hidden dims: {hiddens_found}")
    print(f"  Metrics:     {all_metrics}")
    print(f"  Saving to:   {out_dir}\n")

    phases = ["train", "val"] if args.phase == "both" else [args.phase]

    if not args.no_per_layer:
        print("=== Per-layer figures ===")
        plot_per_layer(results, all_metrics, phases, out_dir / "per_layer")

    if not args.no_cross_layer:
        print("\n=== Cross-layer figures ===")
        plot_cross_layer(results, all_metrics, out_dir / "cross_layer")

    print("\n=== Heatmap ===")
    plot_heatmap_all_metrics(results, all_metrics, out_dir)

    print("\n=== Summary ===")
    plot_summary(results, all_metrics, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
