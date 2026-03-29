#!/usr/bin/env python3
"""
plot_ablation.py — Plotting tool for expert predictor ablation studies.
Supports both top-1 accuracy (Mixtral) and recall@K (Qwen) metrics.

Usage:
  python plot_ablation.py --sweep_dir trainingData/qwen3_30b/ablation
  python plot_ablation.py --sweep_dir trainingData/qwen3_30b/ablation --out_dir plots/
"""

import argparse
import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

VARIANT_ORDER = [
    "emb_only",
    "pfill_only",
    "prev_only",
    "markov_only",
    "emb_prev",
    "emb_markov",
    "all_features",
]

VARIANT_LABELS = {
    "emb_only":     "Emb Only",
    "pfill_only":   "Prefill Only",
    "prev_only":    "Prev Only",
    "markov_only":  "Markov Only",
    "emb_prev":     "Emb + Prev",
    "emb_markov":   "Emb + Markov",
    "all_features": "All Features",
}

VARIANT_COLORS = {
    "emb_only":     "#4C72B0",
    "pfill_only":   "#55A868",
    "prev_only":    "#C44E52",
    "markov_only":  "#8172B2",
    "emb_prev":     "#CCB974",
    "emb_markov":   "#64B5CD",
    "all_features": "#222222",
}


def load_ablations(sweep_dir: pathlib.Path):
    """
    Returns:
      results[layer_idx][(hist, hidden)] = {variant_name: metric_value}
      meta: dict with keys 'metric_name', 'predict_k'
    """
    results = {}
    meta = {'metric_name': 'Validation Accuracy', 'predict_k': 1}

    for p in sweep_dir.rglob("training_metrics.json"):
        try:
            with open(p) as f:
                data = json.load(f)

            layer  = data.get("layer_idx")
            hist   = data.get("history", 1)
            hidden = data.get("hidden_dim")
            name   = data.get("config_name", p.parent.parent.name)

            # Support both old key (best_acc) and new key (best)
            acc = data.get("best_acc") or data.get("best")

            # Pull metric metadata
            if data.get("metric"):
                meta['metric_name'] = data["metric"]
            if data.get("predict_k"):
                meta['predict_k'] = data["predict_k"]

            # Infer hist/hidden from name if missing
            if "hist" in name and hist == 1:
                try:
                    for part in name.split("_"):
                        if part.startswith("hist"):  hist   = int(part[4:])
                        if part.startswith("h") and part[1:].isdigit(): hidden = int(part[1:])
                except Exception:
                    pass

            if layer is None or acc is None:
                continue

            results.setdefault(layer, {}).setdefault((hist, hidden), {})

            # Normalize variant name: "ablation_emb_only_hist1_h128" → "emb_only"
            short = name.replace("ablation_", "")
            if "_hist" in short:
                short = short.split("_hist")[0]

            results[layer][(hist, hidden)][short] = acc

        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")

    return results, meta


def plot_ablations(results, meta, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_label = meta['metric_name'].replace("_", " ").replace("@", "@")
    predict_k    = meta['predict_k']

    # ── 1. Per-layer grouped bar chart ────────────────────────────────────────
    for layer, configs in sorted(results.items()):
        all_keys     = sorted(configs.keys())  # (hist, hidden)
        all_variants = [v for v in VARIANT_ORDER if any(v in c for c in configs.values())]

        fig, ax = plt.subplots(figsize=(max(12, len(all_variants) * 1.6), 6))
        x     = np.arange(len(all_variants))
        width = 0.8 / max(len(all_keys), 1)

        for i, key in enumerate(all_keys):
            hist, hidden = key
            vals = [configs[key].get(v, 0.0) for v in all_variants]
            offset = (i - len(all_keys) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=f"hist={hist}, h={hidden}", alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_title(f"Ablation — Layer {layer}  (predict_k={predict_k})",
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in all_variants], rotation=15, ha='right')
        ax.set_ylabel(metric_label.title())
        ax.set_ylim(0, 1.08)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.axhline(predict_k / 128 if predict_k > 1 else 0, color='red',
                   linestyle=':', linewidth=1, label='random baseline')
        fig.tight_layout()

        path = out_dir / f"ablation_layer_{layer:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 2. Across-layers line plot (one line per variant) ─────────────────────
    # Use the first (hist, hidden) config for simplicity, or all if only one
    all_layers = sorted(results.keys())

    # Pick the best (hist, hidden) key = most data
    key_counts = {}
    for configs in results.values():
        for k, v in configs.items():
            key_counts[k] = key_counts.get(k, 0) + len(v)
    best_key = max(key_counts, key=key_counts.get)
    hist, hidden = best_key

    available_variants = [v for v in VARIANT_ORDER
                          if any(v in results[l].get(best_key, {}) for l in all_layers)]

    if len(all_layers) > 1 and available_variants:
        fig, ax = plt.subplots(figsize=(max(10, len(all_layers) * 0.7), 5))

        for variant in available_variants:
            ys = [results[l].get(best_key, {}).get(variant, None) for l in all_layers]
            # Only plot where data exists
            xs_v = [x for x, y in zip(all_layers, ys) if y is not None]
            ys_v = [y for y in ys if y is not None]
            if xs_v:
                ax.plot(xs_v, ys_v, marker='o', markersize=4,
                        label=VARIANT_LABELS.get(variant, variant),
                        color=VARIANT_COLORS.get(variant),
                        linewidth=1.8 if variant == "all_features" else 1.2,
                        zorder=3 if variant == "all_features" else 2)

        if predict_k > 1:
            ax.axhline(predict_k / 128, color='red', linestyle=':', linewidth=1, label='random baseline')

        ax.set_title(f"Ablation Across Layers  (hist={hist}, h={hidden}, predict_k={predict_k})",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_label.title())
        ax.set_ylim(0, 1.05)
        ax.set_xticks(all_layers)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        ax.grid(linestyle='--', alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"ablation_across_layers_hist{hist}_h{hidden}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 3. Summary table (stdout) ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY  |  metric={metric_label}  |  layers={sorted(results.keys())}")
    print(f"{'='*70}")
    print(f"  {'Variant':<20}  {'Avg':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*50}")
    for variant in VARIANT_ORDER:
        vals = [
            results[l].get(k, {}).get(variant)
            for l in results for k in results[l]
            if results[l].get(k, {}).get(variant) is not None
        ]
        if vals:
            print(f"  {VARIANT_LABELS.get(variant, variant):<20}  "
                  f"{np.mean(vals):>8.4f}  {np.min(vals):>8.4f}  {np.max(vals):>8.4f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True, help="Dir containing ablation training_metrics.json files")
    parser.add_argument("--out_dir",   default=None,  help="Output dir for plots (default: sweep_dir/plots)")
    args = parser.parse_args()

    sweep_path = pathlib.Path(args.sweep_dir)
    out_path   = pathlib.Path(args.out_dir) if args.out_dir else sweep_path / "plots"

    results, meta = load_ablations(sweep_path)
    if not results:
        print("No ablation results found.")
        return

    print(f"Loaded results for {len(results)} layers, metric={meta['metric_name']}")
    plot_ablations(results, meta, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
