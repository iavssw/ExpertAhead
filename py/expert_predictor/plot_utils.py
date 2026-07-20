#!/usr/bin/env python3
"""
plot_utils.py — Shared constants, loaders, and helpers for expert-predictor plot scripts.

Imported by:
  plot_ablation_comparison.py  — ablation study (any variants)
  plot_emb_markov_paper.py     — single-variant performance figures
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.edgecolor":  "0.8",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}


def apply_style() -> None:
    """Call once at import time or at the top of main()."""
    matplotlib.rcParams.update(RCPARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Variant registry — covers every known ablation variant
# ─────────────────────────────────────────────────────────────────────────────

# Canonical ordering for grouped bar charts / legends
VARIANT_ORDER: list[str] = [
    "emb_only",
    "markov_only",
    "emb_markov",
    "pfill_only",
    "prev_only",
    "prev_layers_only",
    "emb_prev",
    "emb_prev_layers",
    "all_features",
    "tx_emb_only",
    "tx_emb_markov",
    "tx_emb_markov_pfill",
    "mlp_emb_only",
    "mlp_emb_markov",
    "mlp_hist4_emb_only",
    "mlp_hist4_emb_markov",
    "mlp_hist4_emb_markov_pfill",
]

# Per-variant display properties
VARIANT_STYLES: dict[str, dict] = {
    "emb_only":        dict(color="#E63946", marker="s",  ls="-",   label="Embeddings only",           lw=2.2),
    "markov_only":     dict(color="#457B9D", marker="^",  ls="--",  label="Markov only",         lw=2.0),
    "emb_markov":      dict(color="#2EC4B6", marker="o",  ls="-",   label="Embeddings + Markov",   lw=2.5),
    "pfill_only":      dict(color="#55A868", marker="D",  ls=":",   label="Prefill only",        lw=1.8),
    "prev_only":       dict(color="#C44E52", marker="v",  ls="--",  label="Prev only",           lw=1.8),
    "prev_layers_only":dict(color="#C4A252", marker="P",  ls=":",   label="Prev layers only",    lw=1.8),
    "emb_prev":        dict(color="#CCB974", marker="h",  ls="-.",  label="Embeddings + Prev",          lw=1.8),
    "emb_prev_layers": dict(color="#C95B94", marker="*",  ls="-.",  label="Embeddings + Prev layers",   lw=1.8),
    "all_features":    dict(color="#222222", marker="X",  ls="-",   label="All features",        lw=2.5),
    "tx_emb_only":        dict(color="#E63946", marker="s",  ls="-",   label="Tx: Embeddings only",           lw=2.2),
    "tx_emb_markov":      dict(color="#2EC4B6", marker="o",  ls="-",   label="Tx: Embeddings + Markov",       lw=2.5),
    "tx_emb_markov_pfill":dict(color="#55A868", marker="*",  ls="-",   label="Tx: Embeddings + Markov + Pfill", lw=2.5),
    "mlp_emb_only":       dict(color="#E63946", marker="s",  ls="--",  label="MLP (Hist=1): Embeddings only",          lw=2.0),
    "mlp_emb_markov":     dict(color="#2EC4B6", marker="o",  ls="--",  label="MLP (Hist=1): Embeddings + Markov",      lw=2.0),
    "mlp_hist4_emb_only":       dict(color="#E63946", marker="v",  ls=":",  label="MLP (Hist=4): Embeddings only",          lw=2.0),
    "mlp_hist4_emb_markov":     dict(color="#2EC4B6", marker="v",  ls=":",  label="MLP (Hist=4): Embeddings + Markov",      lw=2.0),
    "mlp_hist4_emb_markov_pfill":dict(color="#55A868", marker="v",  ls=":",  label="MLP (Hist=4): Embeddings + Markov + Pfill", lw=2.0),
}

# Bar-chart fill colors (same hues, no line style needed)
VARIANT_COLORS: dict[str, str] = {k: v["color"] for k, v in VARIANT_STYLES.items()}

# Human-readable labels (for axes / legends)
VARIANT_LABELS: dict[str, str] = {k: v["label"] for k, v in VARIANT_STYLES.items()}


def style_for(variant: str) -> dict:
    """Return style dict for a variant, falling back to a gray default."""
    return VARIANT_STYLES.get(variant, dict(color="#888888", marker="o", ls="-", label=variant, lw=1.8))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep(sweep_dir: Path, variant_tag: str | None = None) -> dict[int, dict[int, dict]]:
    """
    Walk *sweep_dir* recursively for training_metrics.json files.

    Args:
        sweep_dir:    Root directory to search.
        variant_tag:  If given, only files whose path contains this string are loaded.
                      Pass None to load all files.

    Returns:
        records[future_steps][layer_idx] = best_val_recalls dict
    """
    records: dict[int, dict[int, dict]] = {}
    for p in sweep_dir.rglob("training_metrics.json"):
        if variant_tag is not None and variant_tag not in str(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            fs    = d.get("future_steps")
            layer = d.get("layer_idx")
            if fs is None or layer is None:
                continue
            bvr = d.get("best_val_recalls", {})
            
            # Read num_experts from best.json in same dir
            cfg_p = p.parent / "best.json"
            if cfg_p.exists():
                try:
                    with open(cfg_p) as cf:
                        bvr["num_experts"] = json.load(cf).get("num_experts")
                except Exception:
                    pass

            records.setdefault(fs, {})[layer] = bvr
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return records


def load_sweep_full(sweep_dir: Path, variant_tag: str | None = None) -> dict[int, dict[int, dict]]:
    """
    Like load_sweep() but records[fs][layer] contains the full metadata dict,
    not just best_val_recalls.  Used by the single-variant performance script.

    records[fs][layer] = {
        "metrics":            best_val_recalls dict,
        "eval_mode":          str,
        "val_mean_union_size": float | None,
        "predict_k":          int,
        "best":               float | None,
        "metric":             str,
        "num_experts":        int | None,
    }
    """
    records: dict[int, dict[int, dict]] = {}
    for p in sweep_dir.rglob("training_metrics.json"):
        if variant_tag is not None and variant_tag not in str(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            fs    = d.get("future_steps")
            layer = d.get("layer_idx")
            if fs is None or layer is None:
                continue
            
            num_experts = None
            cfg_p = p.parent / "best.json"
            if cfg_p.exists():
                try:
                    with open(cfg_p) as cf:
                        num_experts = json.load(cf).get("num_experts")
                except Exception:
                    pass

            records.setdefault(fs, {})[layer] = {
                "metrics":             d.get("best_val_recalls", {}),
                "eval_mode":           d.get("eval_mode", "adaptive_union_k"),
                "val_mean_union_size": d.get("val_mean_union_size"),
                "predict_k":           d.get("predict_k", 8),
                "best":                d.get("best"),
                "metric":              d.get("metric", "adaptive_recall"),
                "num_experts":         num_experts,
            }
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records_fs: dict, key: str) -> tuple[float, float, int]:
    """
    Mean, std, n over layers for metric *key*.

    records_fs can be either:
      - {layer: metrics_dict}                  (from load_sweep)
      - {layer: {"metrics": metrics_dict, ...}} (from load_sweep_full)
    """
    vals = []
    for ent in records_fs.values():
        m = ent.get("metrics", ent) if isinstance(ent, dict) else ent
        v = m.get(key)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def best_param_aggregate(records_fs: dict, key_prefix: str) -> tuple[float, float, int]:
    """
    For each layer, find the *maximum* value over all keys matching
    '{key_prefix}@{param}' (e.g. 'topk_f1' or 'thresh_f1'), then aggregate
    (mean, std, n) over layers.

    This gives the oracle best-K or best-τ performance per horizon.
    """
    layer_bests: list[float] = []
    for ent in records_fs.values():
        m = ent.get("metrics", ent) if isinstance(ent, dict) else ent
        vals = [
            float(v)
            for k, v in m.items()
            if k.startswith(key_prefix + "@") and v is not None
        ]
        if vals:
            layer_bests.append(max(vals))
    if not layer_bests:
        return float("nan"), float("nan"), 0
    return float(np.mean(layer_bests)), float(np.std(layer_bests)), len(layer_bests)


def get_best_k_prf(recs_fs: dict) -> tuple[int, dict[str, tuple[float, float]]]:
    """
    For a given future_steps dictionary (mapping layer_idx to bvr metrics),
    find the K (from topk_f1@K) that maximizes the average F1 over all layers.
    Then return that best K, and a dictionary of the aggregated (mean, std)
    metrics for precision, recall, and f1 at that K.
    """
    ks = set()
    for bvr in recs_fs.values():
        metrics = bvr.get("metrics", bvr) if isinstance(bvr, dict) else bvr
        for k in metrics:
            if k.startswith("topk_f1@"):
                try:
                    ks.add(int(k.split("@")[1]))
                except ValueError:
                    pass

    if not ks:
        return 0, {}

    best_k = -1
    best_mean_f1 = -1.0
    best_metrics = {}

    for k in sorted(ks):
        f1_vals = []
        prec_vals = []
        rec_vals = []
        for bvr in recs_fs.values():
            metrics = bvr.get("metrics", bvr) if isinstance(bvr, dict) else bvr
            f1 = metrics.get(f"topk_f1@{k}")
            prec = metrics.get(f"topk_precision@{k}")
            rec = metrics.get(f"topk_recall@{k}")
            if f1 is not None:
                f1_vals.append(f1)
            if prec is not None:
                prec_vals.append(prec)
            if rec is not None:
                rec_vals.append(rec)

        if f1_vals:
            mean_f1 = float(np.mean(f1_vals))
            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_k = k
                best_metrics = {
                    "f1": (mean_f1, float(np.std(f1_vals))),
                    "precision": (float(np.mean(prec_vals)) if prec_vals else 0.0, float(np.std(prec_vals)) if prec_vals else 0.0),
                    "recall": (float(np.mean(rec_vals)) if rec_vals else 0.0, float(np.std(rec_vals)) if rec_vals else 0.0),
                }

    return best_k, best_metrics


def get_num_experts_from_records(recs_fs: dict) -> int:
    """Find the total number of experts from the loaded records config/metadata."""
    for bvr in recs_fs.values():
        if isinstance(bvr, dict):
            if "num_experts" in bvr and bvr["num_experts"] is not None:
                return int(bvr["num_experts"])
            metrics = bvr.get("metrics", {})
            if isinstance(metrics, dict) and "num_experts" in metrics and metrics["num_experts"] is not None:
                return int(metrics["num_experts"])
    return 128  # Fallback


def discover_metric_keys(records: dict, prefix: str) -> list[float]:
    """
    Return sorted list of parameter values for keys matching '{prefix}@{value}'.
    Works across all layers and horizons.  Values are returned as floats.
    """
    found: set[float] = set()
    for fs_d in records.values():
        for ent in fs_d.values():
            m = ent.get("metrics", ent) if isinstance(ent, dict) else ent
            for k in m:
                match = re.match(rf"{re.escape(prefix)}@([\d.]+)", k)
                if match:
                    found.add(float(match.group(1)))
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, base: Path) -> None:
    """Save figure as both .png and .pdf, then close."""
    fig.savefig(base.with_suffix(".png"))
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved {base.name}")


def set_horizon_xticks(ax: plt.Axes, horizons: list[int]) -> None:
    """Set integer x-ticks for horizon values."""
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])


def ordered_variants(available: list[str]) -> list[str]:
    """Return *available* variants in VARIANT_ORDER, then any extras alphabetically."""
    seen = {v: i for i, v in enumerate(VARIANT_ORDER)}
    in_order  = [v for v in VARIANT_ORDER if v in available]
    extras    = sorted(v for v in available if v not in seen)
    return in_order + extras
