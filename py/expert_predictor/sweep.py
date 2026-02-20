#!/usr/bin/env python3
"""
sweep.py — Hidden-Dim Sweep for Expert Predictor
=================================================
Trains one (layer, hidden_dim) combination per run.

Example:
    python sweep.py \\
        --data_dir  ../../trainingData/mixtral_8x7b \\
        --output_dir ../../trainingData/mixtral_8x7b/sweep_results \\
        --model mixtral_8x7b \\
        --hidden_dims 256 512 1024 2048 \\
        --epochs 15
"""

import argparse
import sys

from train import train_embedding_predictor, MODEL_DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description="Sweep hidden_dim for the expert predictor.")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--model",       choices=list(MODEL_DEFAULTS.keys()), default=None)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--layers",      nargs="+", type=int, default=None,
                   help="Layer indices to sweep (default: all layers for the model)")
    p.add_argument("--num_layers",  type=int, default=32)
    p.add_argument("--embedding_dim", type=int, default=4096)
    p.add_argument("--num_experts",   type=int, default=8)
    p.add_argument("--top_k",         type=int, default=2)
    p.add_argument("--history",       type=int, default=1)
    p.add_argument("--epochs",        type=int, default=10)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    emb_dim, n_exp, k, n_layers = (
        MODEL_DEFAULTS[args.model] if args.model else
        (args.embedding_dim, args.num_experts, args.top_k, args.num_layers)
    )
    layers = args.layers if args.layers is not None else list(range(n_layers))

    total = len(layers) * len(args.hidden_dims)
    done  = 0

    print(f"Sweep: {len(layers)} layers × {len(args.hidden_dims)} hidden dims = {total} runs")
    print(f"  Layers:      {layers}")
    print(f"  Hidden dims: {args.hidden_dims}")

    shared = dict(
        embedding_history_size=args.history,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        base_embedding_dim=emb_dim,
        num_experts=n_exp,
        top_k=k,
    )

    results = {}
    for layer_idx in layers:
        for hidden_dim in args.hidden_dims:
            done += 1
            print(
                f"\n{'='*60}\n"
                f"  Run {done}/{total}  —  layer={layer_idx}  hidden_dim={hidden_dim}\n"
                f"{'='*60}"
            )
            try:
                acc = train_embedding_predictor(
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    layer_idx=layer_idx,
                    hidden_dim=hidden_dim,
                    **shared,
                )
                results[(layer_idx, hidden_dim)] = acc
            except Exception as e:
                print(f"[ERROR] layer={layer_idx} hidden={hidden_dim}: {e}", file=sys.stderr)
                continue

    print(f"\nSweep complete. Results in: {args.output_dir}")
    if results:
        print("\nSummary (layer, hidden_dim) → best val acc:")
        for (l, h), acc in sorted(results.items()):
            print(f"  layer={l:3d}  hidden={h:5d}  → {acc:.4f}")


if __name__ == "__main__":
    main()
