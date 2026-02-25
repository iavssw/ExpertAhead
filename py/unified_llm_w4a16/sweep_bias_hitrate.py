#!/usr/bin/env python3
"""
sweep_bias_hitrate.py
=====================
Sweeps the expert bias `correlation_constant` (alpha) and measures how it
affects the MLP predictor's top-k hit rate at generation time.

The model is loaded ONCE; we change `set_layer_correlation_constants` between
runs, so no subprocess overhead.

For each alpha value we collect per-layer:
  - hit_rate_no_bias  : fraction of tokens where the pure-MLP top-k prediction
                        overlaps with the actual experts (no bias applied)
  - hit_rate_with_bias: same metric with bias applied at the given alpha
  - total             : number of generation tokens evaluated

At alpha=0 both rates will be identical (no bias); non-zero alphas reveal the
marginal benefit/cost of the frequency prior.

Usage
-----
python sweep_bias_hitrate.py \\
    --predictor-path /path/to/embedding_only_predictors \\
    --alpha-values 0 0.25 0.5 0.75 1.0 1.5 2.0 \\
    --num-tokens 100 \\
    --prefetch-count 2 \\
    --output bias_hitrate_results.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

# ── allow running from the script's directory ──────────────────────────────
_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))
from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sweep expert-bias alpha and measure predictor hit rate.")
    p.add_argument("--predictor-path", required=True,
                   help="Path to the per-layer TorchScript predictor models directory")
    p.add_argument("--model-path", default="TheBloke/mixtral-8x7b-v0.1-AWQ",
                   help="HuggingFace model path or local directory")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--config-path", default=None,
                   help="JSON5 config for the C++ backend (optional)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--expert-cache", type=int, default=2,
                   help="Max cached experts per layer")
    p.add_argument("--prefetch-count", type=int, default=2,
                   help="Number of predicted experts checked for hit-rate")
    p.add_argument("--num-layers", type=int, default=32,
                   help="Number of MoE layers in the model")
    p.add_argument("--alpha-values", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                   help="List of correlation_constant (alpha) values to sweep")
    p.add_argument("--num-tokens", type=int, default=100,
                   help="Number of generation tokens to run per alpha value")
    p.add_argument("--prompts", type=str, nargs="+",
                   default=["In a surprising discovery, researchers found that"],
                   help="One or more prompt strings to run during each alpha")
    p.add_argument("--output", default="bias_hitrate_results.csv",
                   help="Output CSV path")
    p.add_argument("--predictor-device", default="gpu", choices=["gpu", "cpu", "auto"])
    p.add_argument("--plot-only", action="store_true",
                   help="Just re-plot from an existing CSV")
    return p.parse_args()


def compute_summary(stats):
    """
    Aggregate per-layer tuple list into summary dict.
    stats: list of (no_bias_hits, with_bias_hits, total) per layer
    """
    layer_rows = []
    for layer_idx, (nb, wb, tot) in enumerate(stats):
        layer_rows.append({
            "layer": layer_idx,
            "hits_no_bias":   nb,
            "hits_with_bias": wb,
            "total":          tot,
            "hit_rate_no_bias":   nb / tot if tot > 0 else 0.0,
            "hit_rate_with_bias": wb / tot if tot > 0 else 0.0,
        })
    return layer_rows


def run_sweep(args):
    print("=" * 70)
    print("Expert Bias Hit-Rate Sweep")
    print(f"  alpha values   : {args.alpha_values}")
    print(f"  prefetch_count : {args.prefetch_count}")
    print(f"  tokens/run     : {args.num_tokens}")
    print(f"  predictor path : {args.predictor_path}")
    print("=" * 70)

    print("\nLoading model (once)...")
    model = Mixtral8x7BW4A16Model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        backend="predict",
        predictor_models_dir=args.predictor_path,
        max_cached_experts_per_layer=args.expert_cache,
        prefetch_experts_count=args.prefetch_count,
        predictor_device=args.predictor_device,
        config_path=args.config_path,
    )
    print("Model loaded.\n")

    all_rows = []  # accumulated CSV rows

    for alpha in args.alpha_values:
        print(f"\n{'─'*60}")
        print(f"  alpha = {alpha:.4f}")
        print(f"{'─'*60}")

        # Set correlation constant for every layer
        model.set_layer_correlation_constants([alpha] * args.num_layers)
        model.reset_predictor_stats()
        model.reset_cache_stats()

        tokens_generated = 0
        for prompt_text in args.prompts:
            if model.tokenizer is None:
                print("  WARNING: no tokenizer available, skipping prompt")
                break
            input_ids = model.tokenize(prompt_text)
            per_prompt = max(1, args.num_tokens // len(args.prompts))
            try:
                model.generate(input_ids, max_new_tokens=per_prompt,
                               temperature=0.0)
                tokens_generated += per_prompt
            except Exception as e:
                print(f"  [Error] generation failed: {e}")

        stats = model.get_predictor_stats()
        if not stats:
            print("  WARNING: no predictor stats returned "
                  "(is predictor available for these layers?)")
            continue

        summary = compute_summary(stats)
        total_nb  = sum(r["hits_no_bias"]   for r in summary)
        total_wb  = sum(r["hits_with_bias"]  for r in summary)
        total_tok = sum(r["total"]           for r in summary)

        overall_nb  = total_nb  / total_tok if total_tok > 0 else 0.0
        overall_wb  = total_wb  / total_tok if total_tok > 0 else 0.0

        print(f"  tokens generated : {tokens_generated}")
        print(f"  total predictions: {total_tok}")
        print(f"  hit_rate (no bias) : {overall_nb:.4f}  ({overall_nb*100:.2f}%)")
        print(f"  hit_rate (w/ bias) : {overall_wb:.4f}  ({overall_wb*100:.2f}%)")
        print(f"  delta              : {overall_wb - overall_nb:+.4f}")

        for row in summary:
            row["alpha"] = alpha
            all_rows.append(row)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if all_rows:
        fieldnames = ["alpha", "layer", "hits_no_bias", "hits_with_bias",
                      "total", "hit_rate_no_bias", "hit_rate_with_bias"]
        out_path = args.output
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved → {out_path}")
        generate_plots(out_path, args)
    else:
        print("\nNo results collected.")


# ──────────────────────────────────────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(csv_path, args=None):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError as e:
        print(f"Plotting skipped (missing library: {e})")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty CSV — nothing to plot.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(csv_path).parent

    alphas = sorted(df["alpha"].unique())
    layers = sorted(df["layer"].unique())

    # ── Plot 1: overall hit rate vs alpha ────────────────────────────────────
    agg = df.groupby("alpha")[["hits_no_bias", "hits_with_bias", "total"]].sum()
    agg["hr_no_bias"]   = agg["hits_no_bias"]   / agg["total"].clip(lower=1)
    agg["hr_with_bias"] = agg["hits_with_bias"]  / agg["total"].clip(lower=1)
    agg = agg.reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(agg["alpha"], agg["hr_no_bias"]   * 100, "o--", label="No bias (MLP only)", color="steelblue")
    ax.plot(agg["alpha"], agg["hr_with_bias"] * 100, "s-",  label="With bias", color="darkorange")
    ax.set_xlabel("Correlation constant α")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Expert predictor hit rate vs. bias alpha (all layers)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fname = str(out_dir / f"bias_hitrate_overall_{ts}.png")
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)

    # ── Plot 2: delta (with_bias - no_bias) vs alpha per layer (heatmap) ────
    pivot_delta = df.copy()
    pivot_delta["delta"] = (pivot_delta["hits_with_bias"] - pivot_delta["hits_no_bias"]) / pivot_delta["total"].clip(lower=1)
    heat = pivot_delta.pivot_table(index="layer", columns="alpha", values="delta", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(6, len(alphas) * 1.2), max(6, len(layers) * 0.35)))
    vmax = max(abs(heat.values.max()), abs(heat.values.min()), 0.01)
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in heat.columns], fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([f"L{l}" for l in heat.index], fontsize=7)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Layer")
    ax.set_title("Δ Hit rate (with_bias − no_bias) per layer × alpha")
    plt.colorbar(im, ax=ax, label="Δ hit rate")
    fig.tight_layout()
    fname = str(out_dir / f"bias_hitrate_delta_heatmap_{ts}.png")
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)

    # ── Plot 3: per-layer hit rate curves at best alpha vs baseline ──────────
    # Find the alpha with highest overall with_bias hit rate
    best_alpha = agg.loc[agg["hr_with_bias"].idxmax(), "alpha"]
    df_best   = df[df["alpha"] == best_alpha].copy()
    df_best["hr_no_bias"]   = df_best["hits_no_bias"]   / df_best["total"].clip(lower=1)
    df_best["hr_with_bias"] = df_best["hits_with_bias"] / df_best["total"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df_best["layer"] - 0.2, df_best["hr_no_bias"]   * 100, 0.4, label="No bias", color="steelblue", alpha=0.8)
    ax.bar(df_best["layer"] + 0.2, df_best["hr_with_bias"] * 100, 0.4, label=f"With bias (α={best_alpha:.2f})", color="darkorange", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Per-layer hit rate at best alpha (α={best_alpha:.2f})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fname = str(out_dir / f"bias_hitrate_per_layer_{ts}.png")
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# entry
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        generate_plots(args.output, args)
    else:
        run_sweep(args)
