#!/usr/bin/env python3
"""
Plot top-K vs threshold precision/recall for a single-layer ablation sweep.

Expects training_metrics.json from expert_predictor_cross_layer.py with
topk_precision@K, topk_recall@K, thresh_precision@T, thresh_recall@T.
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = ["emb_only", "markov_only", "emb_markov", "all_features"]
VARIANT_LABELS = {
    "emb_only": "Embeddings only",
    "markov_only": "Markov only",
    "emb_markov": "Embeddings + Markov",
    "all_features": "All features",
}
VARIANT_COLORS = {
    "emb_only": "#4C72B0",
    "markov_only": "#8172B2",
    "emb_markov": "#2EC4B6",
    "all_features": "#222222",
}


def parse_variant(name: str) -> str:
    s = name.replace("ablation_", "")
    if "_hist" in s:
        s = s.split("_hist")[0]
    return s


def load_records(sweep_dir: Path, layer: int, extra_dirs: list[Path] | None = None):
    """records[fs][variant] = best_val dict"""
    roots = [sweep_dir] + list(extra_dirs or [])
    out = defaultdict(dict)
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("training_metrics.json"):
            if p in seen:
                continue
            seen.add(p)
            with open(p) as f:
                d = json.load(f)
            if d.get("layer_idx") != layer:
                continue
            fs = d.get("future_steps")
            name = d.get("config_name", p.parent.parent.name)
            variant = parse_variant(name)
            if "_f" in name and fs is not None:
                suffix = f"_f{fs}"
                if variant.endswith(suffix):
                    variant = variant[: -len(suffix)]
            bvr = d.get("best_val_recalls", {})
            if bvr:
                out[fs][variant] = {
                    "metrics": bvr,
                    "metric": d.get("metric", ""),
                    "main_eval_topk": d.get("main_eval_topk"),
                }
    return out


def pick_topk_k(entry: dict, default_k: int = 8) -> int:
    if entry.get("main_eval_topk"):
        return int(entry["main_eval_topk"])
    metric = entry.get("metric", "")
    m = re.search(r"@(\d+)", metric)
    if m:
        return int(m.group(1))
    bvr = entry.get("metrics", entry)
    ks = []
    for key in bvr:
        m = re.match(r"topk_recall@(\d+)", key) or re.match(r"union_recall@(\d+)", key)
        if m:
            ks.append(int(m.group(1)))
    return max(ks) if ks else default_k


def bvr(entry: dict) -> dict:
    return entry.get("metrics", entry)


def discover_thresholds(records) -> list[float]:
    found = set()
    for fs in records:
        for ent in records[fs].values():
            for key in bvr(ent):
                m = re.match(r"thresh_recall@([\d.]+)", key)
                if m:
                    found.add(float(m.group(1)))
    return sorted(found)


def plot_threshold_sweep(records, layer: int, out_dir: Path, thresholds: list[float],
                         topk_k: int = 8, horizon: int | None = None):
    """Bar chart: recall & precision vs threshold (one lookahead, all variants)."""
    horizons = sorted(records.keys())
    fs = horizon if horizon is not None else horizons[0]
    if fs not in records:
        return
    variants = [v for v in VARIANT_ORDER if v in records[fs]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(thresholds))
    w = 0.8 / max(len(variants), 1)

    for vi, variant in enumerate(variants):
        m = bvr(records[fs][variant])
        recs = [m.get(f"thresh_recall@{t:g}", np.nan) for t in thresholds]
        precs = [m.get(f"thresh_precision@{t:g}", np.nan) for t in thresholds]
        off = (vi - len(variants) / 2 + 0.5) * w
        axes[0].bar(x + off, recs, w, label=VARIANT_LABELS[variant],
                    color=VARIANT_COLORS[variant], alpha=0.9)
        axes[1].bar(x + off, precs, w, label=VARIANT_LABELS[variant],
                    color=VARIANT_COLORS[variant], alpha=0.9)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(t) for t in thresholds])
    axes[0].set_xlabel("Sigmoid threshold τ")
    axes[0].set_ylabel("Recall")
    axes[0].set_title(f"Layer {layer} N={fs} — threshold vs recall")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(t) for t in thresholds])
    axes[1].set_xlabel("Sigmoid threshold τ")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Layer {layer} N={fs} — threshold vs precision")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"Threshold sweep (experts with sigmoid(logit) ≥ τ)", fontsize=11, y=1.02)
    fig.tight_layout()
    p = out_dir / f"layer{layer:02d}_N{fs}_threshold_sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def plot_layer_study(records, layer: int, out_dir: Path,
                     main_threshold: float = 0.5, main_eval_topk: int = 8,
                     plot_all_thresholds: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(records.keys())
    variants = [v for v in VARIANT_ORDER if any(v in records[h] for h in horizons)]
    if not variants or not horizons:
        print("No matching records found.")
        return

    # ── Figure 1: Top-K precision & recall vs horizon ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant in variants:
        precs, recs = [], []
        for fs in horizons:
            ent = records[fs].get(variant, {})
            k = main_eval_topk
            m = bvr(ent)
            precs.append(m.get(f"topk_precision@{k}", m.get(f"union_precision@{k}", np.nan)))
            recs.append(m.get(f"topk_recall@{k}", m.get(f"union_recall@{k}", np.nan)))
        axes[0].plot(horizons, recs, "o-", label=VARIANT_LABELS[variant],
                     color=VARIANT_COLORS[variant], linewidth=2)
        axes[1].plot(horizons, precs, "o-", label=VARIANT_LABELS[variant],
                     color=VARIANT_COLORS[variant], linewidth=2)

    axes[0].set_xlabel("Lookahead N (tokens)")
    axes[0].set_ylabel("Recall (top-K cache)")
    axes[0].set_title(f"Layer {layer} — Top-K recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].set_xlabel("Lookahead N (tokens)")
    axes[1].set_ylabel("Precision (top-K cache)")
    axes[1].set_title(f"Layer {layer} — Top-K precision")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle(f"Top-{main_eval_topk} coverage of next-N expert union", fontsize=12, y=1.02)
    fig.tight_layout()
    p = out_dir / f"layer{layer:02d}_topk_prec_recall_vs_horizon.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── Figure 2: Threshold @ main_threshold ────────────────────────────────
    tag = f"{main_threshold:g}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant in variants:
        precs, recs, sizes = [], [], []
        for fs in horizons:
            m = bvr(records[fs].get(variant, {}))
            precs.append(m.get(f"thresh_precision@{tag}", np.nan))
            recs.append(m.get(f"thresh_recall@{tag}", np.nan))
            sizes.append(m.get(f"thresh_avg_pred@{tag}", np.nan))
        axes[0].plot(horizons, recs, "o-", label=VARIANT_LABELS[variant],
                     color=VARIANT_COLORS[variant], linewidth=2)
        axes[1].plot(horizons, precs, "o-", label=VARIANT_LABELS[variant],
                     color=VARIANT_COLORS[variant], linewidth=2)

    axes[0].set_xlabel("Lookahead N")
    axes[0].set_ylabel(f"Recall (sigmoid ≥ {main_threshold})")
    axes[0].set_title(f"Layer {layer} — Threshold recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[1].set_xlabel("Lookahead N")
    axes[1].set_ylabel(f"Precision (sigmoid ≥ {main_threshold})")
    axes[1].set_title(f"Layer {layer} — Threshold precision")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle(f"Threshold selection (high-confidence experts)", fontsize=12, y=1.02)
    fig.tight_layout()
    p = out_dir / f"layer{layer:02d}_thresh{tag}_prec_recall_vs_horizon.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── Figure 3: Grouped bars at each horizon (both methods) ─────────────
    x = np.arange(len(horizons))
    width = 0.8 / max(len(variants), 1)

    for metric, ylab, fname in [
        ("recall", "Recall", "recall"),
        ("precision", "Precision", "precision"),
    ]:
        fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4), squeeze=False)
        for hi, fs in enumerate(horizons):
            ax = axes[0, hi]
            for vi, variant in enumerate(variants):
                ent = records[fs].get(variant, {})
                m = bvr(ent)
                k = main_eval_topk
                if metric == "recall":
                    topk_v = m.get(f"topk_recall@{k}", m.get(f"union_recall@{k}", 0))
                    thr_v = m.get(f"thresh_recall@{tag}", 0)
                else:
                    topk_v = m.get(f"topk_precision@{k}", m.get(f"union_precision@{k}", 0))
                    thr_v = m.get(f"thresh_precision@{tag}", 0)
                off = (vi - len(variants) / 2 + 0.5) * width
                ax.bar(off - width / 4, topk_v, width / 2, color=VARIANT_COLORS[variant], alpha=0.55)
                ax.bar(off + width / 4, thr_v, width / 2, color=VARIANT_COLORS[variant], alpha=1.0,
                       hatch="//", edgecolor="white")
            ax.set_xticks([(i - len(variants) / 2 + 0.5) * width for i in range(len(variants))])
            ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=25, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"N={fs}")
            ax.set_ylabel(ylab)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.legend(["Top-K (light)", f"Thresh≥{main_threshold} (hatched)"],
                   loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(f"Layer {layer} — {ylab}: Top-K vs Threshold", fontsize=12, y=1.12)
        fig.tight_layout()
        p = out_dir / f"layer{layer:02d}_{fname}_topk_vs_thresh.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {p}")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Layer {layer} — best-val metrics")
    print(f"{'='*72}")
    for fs in horizons:
        print(f"\n  N={fs}")
        print(f"    {'Variant':<14}  {'Method':<10}  {'Recall':>8}  {'Precision':>10}  {'K/size':>8}")
        for variant in variants:
            ent = records[fs].get(variant, {})
            m = bvr(ent)
            k = main_eval_topk
            tr = m.get(f"topk_recall@{k}", m.get(f"union_recall@{k}", 0))
            tp = m.get(f"topk_precision@{k}", m.get(f"union_precision@{k}", 0))
            thr_r = m.get(f"thresh_recall@{tag}", 0)
            thr_p = m.get(f"thresh_precision@{tag}", 0)
            avg_p = m.get(f"thresh_avg_pred@{tag}", 0)
            print(f"    {VARIANT_LABELS[variant]:<14}  {'top-K':<10}  {tr:>8.4f}  {tp:>10.4f}  K={k}")
            print(f"    {'':<14}  {f'thr≥{tag}':<10}  {thr_r:>8.4f}  {thr_p:>10.4f}  sz={avg_p:.1f}")
    print(f"{'='*72}\n")

    if plot_all_thresholds:
        avail = discover_thresholds(records)
        if len(avail) > 1:
            plot_threshold_sweep(records, layer, out_dir, avail, topk_k=main_eval_topk)

    _plot_topk_at_k_bar(records, layer, out_dir, main_eval_topk, horizons, variants)


def _plot_topk_at_k_bar(records, layer, out_dir, k, horizons, variants):
    """Single bar chart: recall@K and precision@K for each variant (fixed K)."""
    if not horizons:
        return
    fs = horizons[0] if len(horizons) == 1 else None
    if fs is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(variants))
    recs, precs = [], []
    for v in variants:
        m = bvr(records[fs].get(v, {}))
        recs.append(m.get(f"topk_recall@{k}", m.get(f"union_recall@{k}", 0)))
        precs.append(m.get(f"topk_precision@{k}", m.get(f"union_precision@{k}", 0)))
    w = 0.35
    ax.bar(x - w / 2, recs, w, label=f"Recall@{k}", color="#3A86FF", alpha=0.9)
    ax.bar(x + w / 2, precs, w, label=f"Precision@{k}", color="#FF6B6B", alpha=0.9)
    for i, (r, p) in enumerate(zip(recs, precs)):
        ax.text(i - w / 2, r + 0.02, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Layer {layer} N={fs} — Top-{k} union coverage (best val)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"layer{layer:02d}_N{fs}_top{k}_recall_precision.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot top-K vs threshold precision/recall. "
                    "Replot with a different --threshold without retraining if that τ was logged.")
    ap.add_argument("--sweep_dir", required=True,
                    help="Primary sweep output (e.g. emb_markov_<timestamp>)")
    ap.add_argument("--extra_sweep_dirs", nargs="+", default=None,
                    help="Other variant dirs to merge (emb_only_*, markov_only_*)")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Sigmoid τ for top-K vs threshold comparison plots (default: 0.5 or first in --thresholds)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=None,
                    help="τ values for sweep chart (default: auto-detect from training_metrics.json)")
    ap.add_argument("--main_eval_topk", type=int, default=8,
                    help="Fixed top-K for primary union-recall curves")
    ap.add_argument("--no-threshold-sweep", action="store_true",
                    help="Skip multi-threshold bar chart")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    out = Path(args.out_dir) if args.out_dir else sweep / "plots_prec_recall"
    extra = [Path(d) for d in (args.extra_sweep_dirs or [])]
    records = load_records(sweep, args.layer, extra_dirs=extra)
    print(f"Loaded horizons {sorted(records.keys())} for layer {args.layer}")

    avail = args.thresholds or discover_thresholds(records)
    if avail:
        print(f"  Thresholds in data / requested: {avail}")
    main_thr = args.threshold if args.threshold is not None else (avail[0] if avail else 0.5)
    if args.threshold is not None and args.threshold not in avail and avail:
        print(f"  Warning: τ={args.threshold} not in logged metrics {avail}; plots may be empty for threshold panels.")

    plot_layer_study(records, args.layer, out,
                     main_threshold=main_thr, main_eval_topk=args.main_eval_topk,
                     plot_all_thresholds=not args.no_threshold_sweep)
    if args.thresholds and len(args.thresholds) > 1:
        plot_threshold_sweep(records, args.layer, out, sorted(args.thresholds),
                             topk_k=args.main_eval_topk)


if __name__ == "__main__":
    main()
