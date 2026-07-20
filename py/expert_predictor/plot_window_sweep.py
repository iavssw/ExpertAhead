#!/usr/bin/env python3
"""
plot_window_sweep.py — Thesis-quality figures for the Multi-Step Expert Predictor.

Reads training_metrics.json files produced by expert_predictor_cross_layer.py
and generates six publication-ready plots for the lookahead window analysis.

Usage:
  python plot_window_sweep.py --sweep_dir ../../trainingData/qwen3_30b/final_multi_input_model
  python plot_window_sweep.py --sweep_dir ... --out_dir plots/thesis/
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Colour ramp for lookahead windows (1 → 16)
_CMAP = plt.cm.plasma


def _window_color(fs: int, fs_min: int, fs_max: int):
    norm = (fs - fs_min) / max(fs_max - fs_min, 1)
    return _CMAP(0.15 + 0.7 * norm)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(sweep_dir: Path):
    """
    Returns:
      records: list of dicts, one per (future_steps, layer_idx) combination.
               Keys: future_steps, layer_idx, predict_k, prefetch_k,
                     best_recall, best_precision, best_f1, label_density,
                     epoch_recalls (list), epoch_precisions (list), epoch_f1s (list)
    """
    records = []
    for p in sweep_dir.rglob("training_metrics.json"):
        try:
            with open(p) as f:
                d = json.load(f)

            fs     = d.get("future_steps")
            layer  = d.get("layer_idx")
            pk     = d.get("predict_k")
            if fs is None or layer is None:
                continue

            # Best-epoch recall (primary metric)
            best_recall = d.get("best") or 0.0

            # Derive prefetch_k from the metric string, e.g. "union_recall@32" → 32
            metric_str = d.get("metric", "")
            try:
                prefetch_k = int(metric_str.split("@")[-1])
            except (ValueError, IndexError):
                prefetch_k = pk or 1

            # Best val snapshot (may have precision/f1 if trained after update)
            bvr = d.get("best_val_recalls", {})
            best_prec    = bvr.get(f"union_precision@{prefetch_k}", None)
            best_f1      = bvr.get(f"union_f1@{prefetch_k}", None)
            best_density = bvr.get("label_density", None)

            # Per-epoch series
            epoch_recalls    = []
            epoch_precisions = []
            epoch_f1s        = []
            epoch_densities  = []
            for ep in d.get("epochs", []):
                val = ep.get("val", {})
                epoch_recalls.append(val.get(f"union_recall@{prefetch_k}", None))
                epoch_precisions.append(val.get(f"union_precision@{prefetch_k}", None))
                epoch_f1s.append(val.get(f"union_f1@{prefetch_k}", None))
                epoch_densities.append(val.get("label_density", None))

            # Infer density from final epoch if best-snapshot missing
            if best_density is None:
                valid = [v for v in epoch_densities if v is not None]
                best_density = valid[-1] if valid else None

            # Derive precision/F1 from recall + density if not stored
            # precision@K ≈ recall * avg_union_size / prefetch_k
            #             = recall * (density * E) / prefetch_k
            # (This is an approximation for old runs that lack the new fields.)
            if best_prec is None and best_density is not None:
                E = pk * (128 // pk) if pk else 128  # Qwen3-30B has 128 experts
                E = 128
                avg_union = best_density * E
                best_prec = best_recall * avg_union / max(prefetch_k, 1)
                best_prec = min(best_prec, 1.0)

            if best_f1 is None and best_prec is not None:
                r, p2 = best_recall, best_prec
                best_f1 = 2 * r * p2 / max(r + p2, 1e-9)

            records.append({
                "future_steps":       fs,
                "layer_idx":          layer,
                "predict_k":          pk,
                "prefetch_k":         prefetch_k,
                "best_recall":        best_recall,
                "best_precision":     best_prec,
                "best_f1":            best_f1,
                "label_density":      best_density,
                "epoch_recalls":      epoch_recalls,
                "epoch_precisions":   epoch_precisions,
                "epoch_f1s":          epoch_f1s,
                "epoch_densities":    epoch_densities,
            })
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return records


# ---------------------------------------------------------------------------
# Helper: aggregate across layers
# ---------------------------------------------------------------------------

def aggregate_by_fs(records, metric_key: str):
    """Return {future_steps: (mean, std, [per-layer values])} for `metric_key`."""
    by_fs = defaultdict(list)
    for r in records:
        v = r.get(metric_key)
        if v is not None:
            by_fs[r["future_steps"]].append(v)
    out = {}
    for fs, vals in by_fs.items():
        out[fs] = (np.mean(vals), np.std(vals), vals)
    return out


def build_matrix(records, metric_key: str):
    """Return (matrix, window_sizes, layers) for heatmap plotting."""
    all_fs     = sorted({r["future_steps"] for r in records})
    all_layers = sorted({r["layer_idx"]    for r in records})
    li = {l: i for i, l in enumerate(all_layers)}
    fi = {f: i for i, f in enumerate(all_fs)}
    mat = np.full((len(all_fs), len(all_layers)), np.nan)
    for r in records:
        v = r.get(metric_key)
        if v is not None:
            mat[fi[r["future_steps"]], li[r["layer_idx"]]] = v
    return mat, all_fs, all_layers


# ---------------------------------------------------------------------------
# Plot 1 — Recall vs. Lookahead Window (averaged across layers)
# ---------------------------------------------------------------------------

def plot_recall_vs_window(records, out_dir: Path):
    by_fs = aggregate_by_fs(records, "best_recall")
    if not by_fs:
        return
    fs_sorted = sorted(by_fs)
    means = [by_fs[fs][0] for fs in fs_sorted]
    stds  = [by_fs[fs][1] for fs in fs_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fs_sorted, means, "o-", color="#3A86FF", linewidth=2, markersize=6, zorder=3)
    ax.fill_between(fs_sorted,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.18, color="#3A86FF", label="±1 std (across layers)")

    # Annotate first and last
    ax.annotate(f"{means[0]:.3f}", (fs_sorted[0], means[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.annotate(f"{means[-1]:.3f}", (fs_sorted[-1], means[-1]),
                xytext=(-28, 6), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Lookahead Window $N$ (steps)")
    ax.set_ylabel(f"Union Recall@{records[0]['prefetch_k']}")
    ax.set_title("Expert Prediction Recall vs. Lookahead Horizon\n"
                 "(averaged across all decoder layers)")
    ax.set_xticks(fs_sorted)
    ax.set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.06))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    path = out_dir / "fig1_recall_vs_window.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Heatmap: Lookahead Window × Layer
# ---------------------------------------------------------------------------

def plot_heatmap(records, out_dir: Path, metric_key="best_recall", title_suffix="Union Recall"):
    mat, all_fs, all_layers = build_matrix(records, metric_key)
    if mat.size == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat),
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(title_suffix)

    # y-axis: window sizes
    ax.set_yticks(range(len(all_fs)))
    ax.set_yticklabels(all_fs, fontsize=8)
    ax.set_ylabel("Lookahead $N$ (steps)")

    # x-axis: every 4th layer
    step = max(1, len(all_layers) // 12)
    ax.set_xticks(range(0, len(all_layers), step))
    ax.set_xticklabels([all_layers[i] for i in range(0, len(all_layers), step)], fontsize=8)
    ax.set_xlabel("Decoder Layer Index")

    ax.set_title(f"Expert Predictor {title_suffix} — Lookahead Window × Layer")
    fig.tight_layout()
    slug = metric_key.replace("best_", "").replace("_", "")
    path = out_dir / f"fig2_heatmap_{slug}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Recall & Precision together vs. Window
# ---------------------------------------------------------------------------

def plot_recall_precision_vs_window(records, out_dir: Path):
    by_fs_r = aggregate_by_fs(records, "best_recall")
    by_fs_p = aggregate_by_fs(records, "best_precision")
    by_fs_f = aggregate_by_fs(records, "best_f1")
    if not by_fs_r:
        return

    fs_sorted = sorted(by_fs_r)

    def _vals(d):
        means = [d[fs][0] if fs in d else np.nan for fs in fs_sorted]
        stds  = [d[fs][1] if fs in d else 0      for fs in fs_sorted]
        return np.array(means), np.array(stds)

    r_m, r_s = _vals(by_fs_r)
    p_m, p_s = _vals(by_fs_p)
    f_m, f_s = _vals(by_fs_f)

    fig, ax = plt.subplots(figsize=(7, 4))
    prefetch_k = records[0]["prefetch_k"]
    predict_k  = records[0]["predict_k"] or 8

    for ym, ys, color, label in [
        (r_m, r_s, "#3A86FF", f"Recall@{prefetch_k}"),
        (p_m, p_s, "#FF6B6B", f"Precision@{prefetch_k}"),
        (f_m, f_s, "#2EC4B6", f"F1@{prefetch_k}"),
    ]:
        ax.plot(fs_sorted, ym, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax.fill_between(fs_sorted, ym - ys, ym + ys, alpha=0.12, color=color)

    # Random-guess baselines
    E = 128
    rand_recall = prefetch_k / E
    rand_prec   = predict_k  / E
    rand_f1     = 2 * rand_recall * rand_prec / max(rand_recall + rand_prec, 1e-9)
    ax.axhline(rand_recall, color="#3A86FF", linestyle=":", linewidth=1, alpha=0.5,
               label=f"Random recall ({rand_recall:.2f})")
    ax.axhline(rand_prec,   color="#FF6B6B", linestyle=":", linewidth=1, alpha=0.5,
               label=f"Random precision ({rand_prec:.2f})")

    ax.set_xlabel("Lookahead Window $N$ (steps)")
    ax.set_ylabel("Score")
    ax.set_title("Recall, Precision, and F1 vs. Lookahead Horizon\n"
                 "(averaged across all decoder layers)")
    ax.set_xticks(fs_sorted)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    path = out_dir / "fig3_recall_prec_f1_vs_window.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 4 — Recall per Layer for selected window sizes
# ---------------------------------------------------------------------------

def plot_recall_per_layer(records, out_dir: Path):
    mat, all_fs, all_layers = build_matrix(records, "best_recall")
    if mat.size == 0:
        return

    # Show a few representative windows
    highlight = [1, 4, 8, 12, 16]
    fs_to_plot = [fs for fs in highlight if fs in all_fs]
    if not fs_to_plot:
        fs_to_plot = all_fs[::max(1, len(all_fs) // 5)]

    fs_min, fs_max = min(all_fs), max(all_fs)

    fig, ax = plt.subplots(figsize=(10, 4))
    for fs in fs_to_plot:
        i = all_fs.index(fs)
        y = mat[i, :]
        ax.plot(all_layers, y, "-", linewidth=1.6,
                color=_window_color(fs, fs_min, fs_max),
                label=f"$N$={fs}")

    # Colorbar legend
    sm = ScalarMappable(cmap=_CMAP,
                        norm=Normalize(vmin=fs_min * 0.85, vmax=fs_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Lookahead $N$")

    ax.set_xlabel("Decoder Layer Index")
    ax.set_ylabel(f"Union Recall@{records[0]['prefetch_k']}")
    ax.set_title("Expert Prediction Recall Across Decoder Layers\n"
                 "for Selected Lookahead Windows")
    ax.set_xlim(all_layers[0], all_layers[-1])
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)

    path = out_dir / "fig4_recall_per_layer.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 5 — Label Density per Layer
# ---------------------------------------------------------------------------

def plot_label_density(records, out_dir: Path):
    # Only fs=1 (shortest window → smallest union → best density estimate)
    # and fs=max for comparison.
    all_fs     = sorted({r["future_steps"] for r in records})
    all_layers = sorted({r["layer_idx"]    for r in records})
    if not all_layers:
        return

    fs_show = [all_fs[0], all_fs[-1]] if len(all_fs) >= 2 else all_fs
    fs_min, fs_max = min(all_fs), max(all_fs)

    fig, ax = plt.subplots(figsize=(10, 4))
    for fs in fs_show:
        by_layer = {}
        for r in records:
            if r["future_steps"] == fs and r["label_density"] is not None:
                by_layer[r["layer_idx"]] = r["label_density"]
        if not by_layer:
            continue
        xs = sorted(by_layer)
        ys = [by_layer[x] for x in xs]
        ax.plot(xs, ys, "-o", markersize=4, linewidth=1.8,
                color=_window_color(fs, fs_min, fs_max),
                label=f"$N$={fs} (avg={np.mean(ys):.3f})")

    ax.set_xlabel("Decoder Layer Index")
    ax.set_ylabel("Label Density  (avg union size / E)")
    ax.set_title("Union Label Density per Decoder Layer\n"
                 "(fraction of 128 experts selected in target union)")
    ax.set_xlim(all_layers[0], all_layers[-1])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)

    path = out_dir / "fig5_label_density_per_layer.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 6 — Recall Degradation vs. Lookahead (per-step analysis)
#
# We estimate "step-1 recall" vs "union recall" to show degradation.
# For a window of N=k, the union recall represents covering *all* k future
# steps at once; recall at step t+1 alone should be substantially higher.
# We compare across windows using the *same budget K* to show that as N grows
# the predictor must cover an increasingly diverse set, reducing per-step coverage.
# ---------------------------------------------------------------------------

def plot_degradation_curve(records, out_dir: Path):
    """
    For each window N, compute:
      - recall_union@K  = best recorded recall (covers all N steps together)
      - estimated recall_per_step@K ≈ recall_union / N * ln(N+1) (concave decay)
        (This is an informative approximation; exact per-step recall would
         require individual step targets, which aren't stored in existing runs.)

    We also show the 1/N inverse-proportional lower bound for context.
    """
    by_fs = aggregate_by_fs(records, "best_recall")
    if not by_fs:
        return

    fs_sorted = sorted(by_fs)
    means     = np.array([by_fs[fs][0] for fs in fs_sorted])

    # Estimated per-step recall: if we use the same K budget to cover 1 step vs N steps,
    # performance at step t+i degrades approximately proportional to 1/i (locality effect).
    # Here we compute average across steps 1..N:  sum_{i=1}^{N} R_i / N
    # Assuming R_i ≈ R_union * decay_i where decay_i is derived from empirical union:
    #   avg_step_recall = recall_union * N / union_size_avg
    # For Qwen3-30B, active_k=8 experts per step, so union_size grows sub-linearly.
    density_by_fs = aggregate_by_fs(records, "label_density")
    E = 128
    predict_k = records[0].get("predict_k") or 8

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: Union recall vs window ---
    ax = axes[0]
    ax.plot(fs_sorted, means, "o-", color="#3A86FF", linewidth=2, markersize=6, label="Union Recall@K")
    if density_by_fs:
        # avg per-step: if union covers D experts on avg, and active_k active per step,
        # fraction of the K-slot budget that covers one target step = min(K, active_k)/K * R
        per_step = []
        for fs in fs_sorted:
            d_mean = density_by_fs[fs][0] if fs in density_by_fs else None
            r      = by_fs[fs][0]
            if d_mean is not None and d_mean > 0:
                union_sz     = d_mean * E
                # Avg experts from a *single* step that appear in union = predict_k
                # recall for that single step = TP_single / predict_k
                # TP_single ≈ TP_union / (union_sz / predict_k) = TP_union * predict_k / union_sz
                # recall_single = TP_single / predict_k = (TP_union / union_sz) = recall * predict_k / union_sz? 
                # Simpler: avg per-step recall ≈ recall * union_sz / (fs * predict_k)
                per_step.append(min(r * union_sz / (fs * predict_k), 1.0))
            else:
                per_step.append(None)
        valid = [(fs, v) for fs, v in zip(fs_sorted, per_step) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "s--", color="#FF6B6B", linewidth=1.8, markersize=5,
                    label="Est. Avg Per-Step Recall@K")

    ax.set_xlabel("Lookahead Window $N$")
    ax.set_ylabel("Recall")
    ax.set_title("Union vs. Per-Step Recall\nas Lookahead Grows")
    ax.set_xticks(fs_sorted[::2])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # --- Right: Marginal gain in recall per extra step ---
    ax2 = axes[1]
    marginal = np.diff(means)           # Δrecall adding one more step
    xs_m     = fs_sorted[1:]
    ax2.bar(xs_m, marginal, color=np.where(marginal >= 0, "#2EC4B6", "#FF6B6B"),
            alpha=0.8, edgecolor="none", width=0.6)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Lookahead Window $N$")
    ax2.set_ylabel("Δ Union Recall (vs. previous $N$)")
    ax2.set_title("Marginal Recall Gain per Extra\nLookahead Step")
    ax2.set_xticks(xs_m[::2])
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    path = out_dir / "fig6_recall_degradation.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 7 — Training curves for selected layers (val recall across epochs)
# ---------------------------------------------------------------------------

def plot_training_curves(records, out_dir: Path):
    """Show val-recall learning curves for 3 selected window sizes, at a middle layer."""
    all_layers = sorted({r["layer_idx"] for r in records})
    if not all_layers:
        return
    target_layer = all_layers[len(all_layers) // 2]  # middle layer

    fs_show = [1, 4, 8, 16]
    all_fs  = sorted({r["future_steps"] for r in records})
    fs_show = [fs for fs in fs_show if fs in all_fs] or all_fs[:4]
    fs_min, fs_max = min(all_fs), max(all_fs)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in records:
        if r["future_steps"] not in fs_show or r["layer_idx"] != target_layer:
            continue
        vals = [v for v in r["epoch_recalls"] if v is not None]
        if vals:
            ax.plot(range(1, len(vals) + 1), vals,
                    "-", linewidth=1.8, markersize=4, marker="o",
                    color=_window_color(r["future_steps"], fs_min, fs_max),
                    label=f"$N$={r['future_steps']}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Val Union Recall@{records[0]['prefetch_k']}")
    ax.set_title(f"Training Curves — Layer {target_layer}\n"
                 "for Selected Lookahead Windows")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)

    path = out_dir / "fig7_training_curves.pdf"
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Plot 8 — Precision vs recall at multiple cache sizes (set-based metrics)
# ---------------------------------------------------------------------------

def load_multik_from_sweep(sweep_dir: Path):
    """
    Load union_recall@K and union_precision@K for every cache size K logged
    during training.  Returns nested dict:
      data[future_steps][cache_k] = {'recall': [...], 'precision': [...], 'union_size': [...]}
    per-layer lists for averaging.
    """
    data = defaultdict(lambda: defaultdict(lambda: {
        'recall': [], 'precision': [], 'union_size': [],
    }))

    for p in sweep_dir.rglob("training_metrics.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            fs = d.get("future_steps")
            layer = d.get("layer_idx")
            if fs is None or layer is None:
                continue

            bvr = d.get("best_val_recalls", {})
            density = bvr.get("label_density")
            union_size = density * 128 if density is not None else None

            for key, val in bvr.items():
                if not key.startswith("union_recall@"):
                    continue
                k = int(key.split("@")[-1])
                prec_key = f"union_precision@{k}"
                if prec_key not in bvr:
                    continue
                data[fs][k]['recall'].append(val)
                data[fs][k]['precision'].append(bvr[prec_key])
                if union_size is not None:
                    data[fs][k]['union_size'].append(union_size)
        except Exception as e:
            print(f"  Warning: {p}: {e}")

    return data


def plot_cache_precision_recall_tradeoff(sweep_dir: Path, out_dir: Path,
                                         horizons=None, highlight_ks=None):
    """
    Show set-based precision/recall vs cache size K.

    Definitions (per sample, then averaged over layers):
      predicted set = top-K experts from the predictor
      true set      = union of top-predict_k experts over the next N tokens
      precision@K   = |pred ∩ true| / K
      recall@K      = |pred ∩ true| / |true|
    """
    data = load_multik_from_sweep(sweep_dir)
    if not data:
        return

    if horizons is None:
        horizons = sorted(data.keys())
    if highlight_ks is None:
        highlight_ks = [8, 12, 16, 24, 32]

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4),
                             squeeze=False)

    for ax, fs in zip(axes[0], horizons):
        if fs not in data:
            continue
        ks = sorted(data[fs].keys())
        recalls = [np.mean(data[fs][k]['recall']) for k in ks]
        precs   = [np.mean(data[fs][k]['precision']) for k in ks]
        avg_union = np.mean(data[fs][ks[0]]['union_size']) if data[fs][ks[0]]['union_size'] else None

        ax2 = ax.twinx()
        ax.plot(ks, recalls, "o-", color="#3A86FF", linewidth=2, label="Recall")
        ax2.plot(ks, precs, "s--", color="#FF6B6B", linewidth=2, label="Precision")
        for hk in highlight_ks:
            if hk in ks:
                i = ks.index(hk)
                ax.annotate(f"K={hk}", (ks[i], recalls[i]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        if avg_union:
            ax.axhline(1.0, color="#3A86FF", linestyle=":", alpha=0.2)
            ax.axvline(avg_union, color="gray", linestyle=":", alpha=0.5,
                       label=f"avg |true|≈{avg_union:.0f}")
        ax.set_xlabel("Cache size K (predicted experts)")
        ax.set_ylabel("Recall = |pred ∩ true| / |true|", color="#3A86FF")
        ax2.set_ylabel("Precision = |pred ∩ true| / K", color="#FF6B6B")
        ax.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax.set_title(f"N={fs} steps ahead")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.suptitle(
        "Set-Based Expert Cache Metrics\n"
        "(precision = fraction of cache that is needed; "
        "recall = fraction of needed experts in cache)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "fig8_cache_precision_recall_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")

    # Precision–recall curve (one point per cache size)
    fig, ax = plt.subplots(figsize=(6, 5))
    for fs in horizons:
        if fs not in data:
            continue
        ks = sorted(data[fs].keys())
        recalls = [np.mean(data[fs][k]['recall']) for k in ks]
        precs   = [np.mean(data[fs][k]['precision']) for k in ks]
        ax.plot(recalls, precs, "o-", linewidth=2, markersize=6, label=f"N={fs}")
        for k, r, p in zip(ks, recalls, precs):
            ax.annotate(str(k), (r, p), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Recall  (|pred ∩ true| / |true|)")
    ax.set_ylabel("Precision  (|pred ∩ true| / K)")
    ax.set_title("Precision–Recall Tradeoff by Cache Size")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    path = out_dir / "fig8b_precision_recall_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records):
    by_fs = aggregate_by_fs(records, "best_recall")
    by_fsp = aggregate_by_fs(records, "best_precision")
    by_fsf = aggregate_by_fs(records, "best_f1")
    print(f"\n{'='*65}")
    print(f"  WINDOW SWEEP SUMMARY  (prefetch_k={records[0]['prefetch_k']}, "
          f"predict_k={records[0]['predict_k']})")
    print(f"{'='*65}")
    print(f"  {'N':>4}  {'Recall':>8}  {'Prec':>8}  {'F1':>8}  {'Layers':>6}")
    print(f"  {'-'*50}")
    for fs in sorted(by_fs):
        r_m = by_fs[fs][0]
        p_m = by_fsp.get(fs, (None,))[0]
        f_m = by_fsf.get(fs, (None,))[0]
        n   = len(by_fs[fs][2])
        print(f"  {fs:>4}  {r_m:>8.4f}  "
              f"{p_m:>8.4f}  " if p_m else f"  {fs:>4}  {r_m:>8.4f}  {'N/A':>8}  ",
              end="")
        print(f"{f_m:>8.4f}  {n:>6}" if f_m else f"{'N/A':>8}  {n:>6}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality plots for the lookahead window sweep.")
    parser.add_argument("--sweep_dir", required=True,
                        help="Root dir containing eh1_h32_fN/layer_K/training_metrics.json")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: sweep_dir/thesis_plots)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else sweep_dir / "thesis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {sweep_dir}")
    records = load_results(sweep_dir)
    if not records:
        print("No training_metrics.json files found.")
        return

    n_fs     = len({r["future_steps"] for r in records})
    n_layers = len({r["layer_idx"]    for r in records})
    print(f"  Loaded {len(records)} records ({n_fs} window sizes × {n_layers} layers)")
    print(f"  Saving figures to: {out_dir}\n")

    plot_recall_vs_window(records, out_dir)
    plot_heatmap(records, out_dir, "best_recall",    "Union Recall")
    plot_heatmap(records, out_dir, "best_f1",        "F1 Score")
    plot_recall_precision_vs_window(records, out_dir)
    plot_recall_per_layer(records, out_dir)
    plot_label_density(records, out_dir)
    plot_degradation_curve(records, out_dir)
    plot_training_curves(records, out_dir)
    plot_cache_precision_recall_tradeoff(sweep_dir, out_dir, horizons=[1, 4, 8, 16])
    print_summary(records)
    print("Done.")


if __name__ == "__main__":
    main()
