#!/usr/bin/env python3
"""
comprehensive_sweep.py — History and Hidden-Dim Sweep for Expert Predictor
========================================================================
Sweeps both embedding_history_size and hidden_dim to find the optimal architecture
while respecting a model size constraint (e.g., < 50 MB).

Example:
    python comprehensive_sweep.py \\
        --data_dir ../../trainingData/mixtral_8x7b \\
        --output_dir ../../trainingData/mixtral_8x7b/sweep_results \\
        --model mixtral_8x7b \\
        --hidden_dims 256 512 1024 \\
        --histories 1 2 3 \\
        --layers 0 7 15 23 31 \\
        --max_size_mb 50
"""

import argparse
import sys
import json
from pathlib import Path
import torch
from train import train_embedding_predictor, MODEL_DEFAULTS

def calculate_model_size_mb(input_dim, hidden_dim, output_dim):
    # Layer 1: input_dim -> hidden_dim
    params = (input_dim * hidden_dim) + hidden_dim
    # Layer 2: hidden_dim -> hidden_dim // 2
    params += (hidden_dim * (hidden_dim // 2)) + (hidden_dim // 2)
    # Layer 3: hidden_dim // 2 -> output_dim
    params += ((hidden_dim // 2) * output_dim) + output_dim
    
    # LayerNorms
    params += (hidden_dim * 2) + ((hidden_dim // 2) * 2)
    
    per_layer_mb = (params * 4) / (1024 * 1024)
    return per_layer_mb

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive sweep for expert predictor.")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--model",       choices=list(MODEL_DEFAULTS.keys()), default=None)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--histories",   nargs="+", type=int, default=[1, 2])
    p.add_argument("--layers",      nargs="+", type=int, default=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
                   help="Layer indices to sweep (representative layers)")
    p.add_argument("--max_size_mb", type=float, default=70.0, help="Max size PER LAYER model in MB")
    
    p.add_argument("--best_by",     type=str, default="top1_exact", help="Metric to define best model")
    p.add_argument("--loss_type",   type=str, default="ce", help="Loss type (ce, bce, focal)")
    p.add_argument("--num_layers",  type=int, default=32)
    p.add_argument("--embedding_dim", type=int, default=4096)
    p.add_argument("--num_experts",   type=int, default=8)
    p.add_argument("--top_k",         type=int, default=2)
    p.add_argument("--model_type",   type=str, default="dual_mlp",
                   choices=["dual_mlp", "expert_predictor"],
                   help="Model architecture to train (default: dual_mlp)")
    p.add_argument("--epochs",        type=int, default=6)
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    emb_dim, n_exp, k, n_layers = (
        MODEL_DEFAULTS[args.model] if args.model else
        (args.embedding_dim, args.num_experts, args.top_k, args.num_layers)
    )
    layers = args.layers if args.layers is not None else list(range(n_layers))

    # Pre-filter architectures based on size
    valid_configs = []
    OVERHEAD = 1.25 # 25% buffer for metadata/config strings in .pt file
    
    for history in args.histories:
        for h_dim in args.hidden_dims:
            input_dim = emb_dim * history
            per_layer_mb = calculate_model_size_mb(input_dim, h_dim, n_exp)
            
            est_per_layer = per_layer_mb * OVERHEAD
            
            if est_per_layer <= args.max_size_mb:
                valid_configs.append((history, h_dim, est_per_layer))
                print(f"Valid: emb_hist={history}, hidden={h_dim} (Est Per-Layer: {est_per_layer:.1f}MB)")
            else:
                print(f"Skipping: emb_hist={history}, hidden={h_dim} (Est Per-Layer {est_per_layer:.1f}MB > {args.max_size_mb}MB)")

    total = len(layers) * len(valid_configs)
    done = 0

    print(f"\nSweep Summary:")
    print(f"  Layers: {layers}")
    print(f"  Valid Archs: {len(valid_configs)}")
    print(f"  Total Runs: {total}")

    results = []
    
    for history, hidden_dim, size_mb in valid_configs:
        for layer_idx in layers:
            done += 1
            print(f"\n{'='*60}")
            print(f" Run {done}/{total}: Layer {layer_idx}, EmbHist {history}, Hidden {hidden_dim} ({size_mb:.1f}MB)")
            print(f"{'='*60}")
            
            # Adjust output dir to include architecture details
            arch_specific_out = output_path / f"eh{history}_h{hidden_dim}"
            
            try:
                metrics = train_embedding_predictor(
                    data_dir=args.data_dir,
                    output_dir=str(arch_specific_out),
                    layer_idx=layer_idx,
                    embedding_history_size=history,
                    hidden_dim=hidden_dim,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    base_embedding_dim=emb_dim,
                    num_experts=n_exp,
                    top_k=k,
                    loss_type=args.loss_type,
                    best_by=args.best_by,
                    model_type=args.model_type,
                )
                
                results.append({
                    "layer": layer_idx,
                    "history": history,
                    "hidden_dim": hidden_dim,
                    "size_mb": size_mb,
                    **{f"val_{k}": v for k, v in metrics.items()}
                })
                
                # Save intermediate results
                with open(output_path / "sweep_summary.json", "w") as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                import traceback
                print(f"[ERROR] Run failed: {e}", file=sys.stderr)
                # traceback.print_exc()
                continue

    print(f"\nSweep complete. Best architectures per layer (by {args.best_by}):")
    # Simple summary logic
    for layer in layers:
        layer_results = [r for r in results if r['layer'] == layer]
        if not layer_results: continue
        best = max(layer_results, key=lambda x: x[f'val_{args.best_by}'])
        print(f"  Layer {layer:2d}: Best is EmbHist={best['history']}, Hidden={best['hidden_dim']} "
              f"(Top-1 Exact: {best['val_top1_exact']:.4f}, Full Match: {best['val_acc']:.4f}, Mean Overlap: {best['val_mean_overlap']:.4f})")

if __name__ == "__main__":
    main()
