#!/usr/bin/env python3
"""
analyze_expert_reuse.py — Token Window Cache Hit Analyzer
=========================================================
Analyzes the rate of expert reuse over sliding token lookahead windows
to mathematically derive optimal physical cache boundaries.
"""

import argparse
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv

MODEL_DEFAULTS = {
    "mixtral_8x7b":  (8,   2, 32),
    "mixtral_8x22b": (8,   2, 56),
    "qwen3_30b":     (128, 8, 48),
    "qwen3_480b":    (128, 8, 94),
}

def analyze_reuse(args):
    if args.model:
        n_exp, active_k, _ = MODEL_DEFAULTS[args.model]
        args.n_exp, args.active_k = n_exp, active_k
    else:
        assert args.n_exp and args.active_k, "Must supply --n_exp and --active_k if --model is not used"

    data_path = Path(args.data_dir)
    if not data_path.exists():
        sys.exit(f"ERROR: data_dir does not exist: {data_path.resolve()}")

    files = sorted(data_path.glob("*.pt"))
    if not files:
        sys.exit(f"ERROR: No .pt files found in: {data_path.resolve()}")

    # Determine which layers to process
    if args.layers:
        target_layers = args.layers
    else:
        target_layers = [args.layer]

    # Pre-extract expert IDs for all target layers to avoid re-reading files
    layer_to_expert_ids = {l: [] for l in target_layers}
    
    for fp in tqdm(files, desc="Pre-loading layer data"):
        payload = torch.load(fp, map_location='cpu', weights_only=False)
        for layer_info in payload.get('layers', []):
            l_idx = layer_info['layer_idx']
            if l_idx in layer_to_expert_ids:
                logits = layer_info['router_logits']
                topk_ids = logits.topk(args.active_k, dim=-1).indices
                layer_to_expert_ids[l_idx].append(topk_ids)

    # Setup logging
    log_file = None
    output_prefix = args.output_file if args.output_file else "expert_reuse"
    output_file_full = f"{output_prefix}_{args.model}.txt"
    
    if args.output_file:
        log_file = Path(output_file_full).open('w')

    def log_print(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    # Data for plotting
    plot_data = {} # {layer_idx: [avg_uniques]}

    for layer_idx in sorted(target_layers):
        all_expert_ids = layer_to_expert_ids[layer_idx]
        if not all_expert_ids:
            log_print(f"\n[SKIP] No data found for Layer {layer_idx}")
            continue

        log_print(f"\nAnalyzing Expert Usage: Layer {layer_idx} (active_k={args.active_k}, model={args.model})")
        log_print(f"{'Window (Tokens)':<18} | {'Avg Unique Experts':<20} | {'Max Unique Experts':<20} | {'Theoretical Max':<20} | {'Reuse %':<10}")
        log_print("-" * 104)

        layer_avgs = []
        for w in range(1, args.max_window + 1):
            total_windows = 0
            total_unique = 0
            max_unique = 0
            
            for seq in all_expert_ids:
                T = seq.size(0)
                if T < w: continue
                
                # Using unfold to create sliding windows: [num_windows, w, active_k]
                windows = seq.unfold(0, w, 1).transpose(1, 2)
                
                # Flatten the experts: [num_windows, w * active_k]
                flat_windows = windows.reshape(windows.size(0), -1)
                
                num_windows = flat_windows.size(0)
                multi_hot = torch.zeros(num_windows, args.n_exp, device='cpu')
                multi_hot.scatter_(1, flat_windows, 1.0)
                unique_counts = multi_hot.sum(dim=1)
                
                total_unique += unique_counts.sum().item()
                total_windows += num_windows
                batch_max = unique_counts.max().item()
                if batch_max > max_unique:
                    max_unique = batch_max

            if total_windows > 0:
                avg_unique = total_unique / total_windows
                layer_avgs.append(avg_unique)
                theoretical = min(args.n_exp, w * args.active_k)
                reuse_pct = (1.0 - (avg_unique / theoretical)) * 100
                log_print(f"{w:<18} | {avg_unique:<20.2f} | {max_unique:<20.1f} | {theoretical:<20} | {reuse_pct:.1f}%")
        
        plot_data[layer_idx] = layer_avgs

    if log_file:
        log_file.close()
        print(f"\nFull report saved to: {output_file_full}")

    # --- CSV Export ---
    csv_filename = f"{output_prefix}_{args.model}.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            sorted_layers = sorted(plot_data.keys())
            # Header: window_size, layer_0, layer_1, ...
            header = ['window_size'] + [f'layer_{l}' for l in sorted_layers]
            writer.writerow(header)
            
            for w_idx in range(args.max_window):
                window_size = w_idx + 1
                row = [window_size]
                for l_idx in sorted_layers:
                    if w_idx < len(plot_data[l_idx]):
                        row.append(f"{plot_data[l_idx][w_idx]:.4f}")
                    else:
                        row.append("")
                writer.writerow(row)
        print(f"Data saved to CSV: {csv_filename}")
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")

    if args.plot:
        # Plot 1: Unique Experts vs Window Size (Normalized to total experts)
        plt.figure(figsize=(10, 6))
        for l_idx, avgs in plot_data.items():
            wins = list(range(1, len(avgs) + 1))
            # Normalize to total number of experts (percentage)
            normalized_avgs = [(a / args.n_exp) * 100 for a in avgs]
            plt.plot(wins, normalized_avgs, label=f"Layer {l_idx}", marker='o', markersize=4)
        
        plt.title(f"Avg Unique Experts % vs Window Size ({args.model})")
        plt.xlabel("Window Size (Tokens)")
        plt.ylabel("Unique Experts % of Total (Lower = Better Locality)")
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        out_name = output_file_full.replace('.txt', '_curves.png')
        plt.savefig(out_name)
        print(f"Plot saved to: {out_name}")

        # Plot 2: Reuse Profile across layers
        if len(plot_data) > 1:
            plt.figure(figsize=(10, 6))
            fix_w = min(8, args.max_window)
            l_indices = sorted(plot_data.keys())
            reuse_vals = []
            for l in l_indices:
                avg = plot_data[l][fix_w-1]
                theoretical = min(args.n_exp, fix_w * args.active_k)
                reuse_vals.append((1.0 - (avg / theoretical)) * 100)
            
            plt.bar(l_indices, reuse_vals, color='skyblue', edgecolor='navy')
            plt.title(f"Expert Reuse Rate % at Window={fix_w} across Layers")
            plt.xlabel("Layer Index")
            plt.ylabel("Reuse % (Higher = Better Locality)")
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            out_name = output_file_full.replace('.txt', '_profile.png')
            plt.savefig(out_name)
            print(f"Profile plot saved to: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True, help="Directory containing .pt dataset chunks")
    parser.add_argument('--model',      choices=MODEL_DEFAULTS.keys(), help="Model config to use")
    parser.add_argument('--layer',      type=int, default=0, help="Target layer index (if --layers not used)")
    parser.add_argument('--layers',     type=int, nargs='+', help="List of layers to analyze", default=range(48))
    parser.add_argument('--max_window', type=int, default=16, help="Maximum lookahead window size (tokens)")
    parser.add_argument('--output_file', default=None, help="Save report to this file")
    parser.add_argument('--plot',       action='store_true', help="Generate plots (PNGs)")
    parser.add_argument('--n_exp',      type=int, help="Total number of experts (if not using --model)")
    parser.add_argument('--active_k',   type=int, help="Number of active experts per token (if not using --model)")
    
    args = parser.parse_args()
    analyze_reuse(args)
