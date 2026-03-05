#!/usr/bin/env python3
"""
sweep_bias_hitrate.py
=====================
Sweeps the expert bias `correlation_constant` (alpha) and measures how it
affects the expert cache hit rate at generation time.

The model is loaded ONCE; we change `set_layer_correlation_constants` between
runs so no subprocess overhead.

For each alpha value we run generation and record the cache hit rate.
We also always run alpha=0.0 as the baseline and compare.

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

_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))
from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model


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
                   help="Number of predicted experts to prefetch")
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


def run_sweep(args):
    print("=" * 70)
    print("Expert Bias Hit-Rate Sweep (Separate Runs)")
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

    all_rows = []

    for alpha in args.alpha_values:
        print(f"\n{'─'*60}")
        print(f"  alpha = {alpha:.4f}")
        print(f"{'─'*60}")

        model.set_layer_correlation_constants([alpha] * args.num_layers)
        model.reset_cache_stats()

        tokens_generated = 0
        for prompt_text in args.prompts:
            if model.tokenizer is None:
                print("  WARNING: no tokenizer available, skipping prompt")
                break
            input_ids = model.tokenize(prompt_text)
            per_prompt = max(1, args.num_tokens // len(args.prompts))
            try:
                model.generate(input_ids, max_new_tokens=per_prompt, temperature=0.0)
                tokens_generated += per_prompt
            except Exception as e:
                print(f"  [Error] generation failed: {e}")

        # Collect overall cache stats
        total_hits, total_misses = model.get_cache_stats()
        total = total_hits + total_misses
        hit_rate = total_hits / total if total > 0 else 0.0

        print(f"  tokens generated : {tokens_generated}")
        print(f"  cache hits       : {total_hits} / {total}")
        print(f"  hit_rate         : {hit_rate:.4f}  ({hit_rate*100:.2f}%)")

        all_rows.append({
            "alpha": alpha,
            "hits": total_hits,
            "misses": total_misses,
            "total": total,
            "hit_rate": hit_rate,
        })

    # Save CSV
    if all_rows:
        fieldnames = ["alpha", "hits", "misses", "total", "hit_rate"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved → {args.output}")
        generate_plots(args.output, args)
    else:
        print("\nNo results collected.")


def generate_plots(csv_path, args=None):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Plotting skipped (missing library: {e})")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty CSV — nothing to plot.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(csv_path).parent

    baseline = df[df["alpha"] == 0.0]["hit_rate"].values
    baseline_hr = baseline[0] if len(baseline) > 0 else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["alpha"], df["hit_rate"] * 100, "o-", color="darkorange", label="Hit rate")
    if baseline_hr is not None:
        ax.axhline(y=baseline_hr * 100, color="steelblue", linestyle="--", label=f"Baseline (α=0): {baseline_hr*100:.1f}%")
    ax.set_xlabel("Correlation constant α")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_title("Expert cache hit rate vs. prefill bias alpha")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fname = str(out_dir / f"bias_hitrate_{ts}.png")
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        generate_plots(args.output, args)
    else:
        run_sweep(args)
