#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

def plot_pareto_layer(df, layer_idx, output_file, predict_k=None):
    layer_df = df[df['layer_idx'] == layer_idx]
    if layer_df.empty:
        print(f"No data found for layer {layer_idx}.")
        return

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # --- Plot 1: Recall vs Prefetch Budget (K) ---
    future_steps_list = sorted(layer_df['future_steps'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(future_steps_list)))
    color_map = {fs: colors[i] for i, fs in enumerate(future_steps_list)}
    
    markers = ['o', 's', 'D', '^', 'v', 'p']
    predict_ks = sorted(layer_df['predict_k'].unique())
    marker_map = {pk: markers[i % len(markers)] for i, pk in enumerate(predict_ks)}
    
    for fs in future_steps_list:
        fs_df = layer_df[layer_df['future_steps'] == fs]
        for pk in sorted(fs_df['predict_k'].unique()):
            group = fs_df[fs_df['predict_k'] == pk].sort_values('prefetch_k')
            ax1.plot(group['prefetch_k'], group['recall'], 
                     label=f"fs={fs}, pk={pk}",
                     color=color_map[fs],
                     marker=marker_map[pk], 
                     markersize=5, alpha=0.7)

    ax1.set_xlabel('Prefetch $K$ (Budget)', fontsize=12)
    ax1.set_ylabel('Recall', fontsize=12)
    ax1.set_title(f'Recall vs Budget (Layer {layer_idx})', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Recall vs Future Steps (Horizon Trade-off) ---
    # Pick a few key budgets to show the trade-off
    all_k = sorted(layer_df['prefetch_k'].unique())
    # Aim for ~4-5 budget levels for clarity
    step = max(1, len(all_k) // 4)
    selected_ks = all_k[::step]
    if all_k[-1] not in selected_ks:
        selected_ks.append(all_k[-1])
        
    for k in selected_ks:
        k_df = layer_df[layer_df['prefetch_k'] == k]
        # For each (k, fs), pick the best pk
        frontier_pts = []
        for fs in future_steps_list:
            fs_df = k_df[k_df['future_steps'] == fs]
            if predict_k is not None:
                # Use specified pk
                pk_df = fs_df[fs_df['predict_k'] == predict_k]
                best_recall = pk_df['recall'].max() if not pk_df.empty else np.nan
            else:
                # Pick best pk for this (k, fs)
                best_recall = fs_df['recall'].max()
            
            if not np.isnan(best_recall):
                frontier_pts.append((fs, best_recall))
        
        if frontier_pts:
            pts = sorted(frontier_pts)
            x, y = zip(*pts)
            ax2.plot(x, y, 'o-', label=f"Budget $K$={k}", linewidth=2)

    ax2.set_xlabel('Future Steps (Horizon)', fontsize=12)
    ax2.set_ylabel('Best Recall', fontsize=12)
    pk_str = f" (pk={predict_k})" if predict_k is not None else " (Best pk)"
    ax2.set_title(f'Recall-Lookahead Trade-off{pk_str} (Layer {layer_idx})', fontsize=14)
    ax2.set_xticks(future_steps_list)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Expert Prediction Pareto Frontiers (Layer {layer_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_pareto(sweep_dir, layer_idx=None, predict_k=None):

    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"Error: {sweep_dir} does not exist.")
        return

    data = []
    
    # Walk through the directory to find all training_metrics.json files
    metrics_files = list(sweep_path.glob("**/training_metrics.json"))
    print(f"Found {len(metrics_files)} metrics files.")

    for mf in metrics_files:
        try:
            with open(mf, 'r') as f:
                res = json.load(f)
            
            curr_layer = res.get('layer_idx')
            # Filter by layer if specified
            if layer_idx is not None and curr_layer != layer_idx:
                continue
                
            fs = res.get('future_steps')
            pk = res.get('predict_k')
            
            recalls = res.get('best_val_recalls', {})
            for key, val in recalls.items():
                # key is "union_recall@K"
                try:
                    prefetch_k = int(key.split('@')[-1])
                    data.append({
                        'layer_idx': curr_layer,
                        'future_steps': fs,
                        'predict_k': pk,
                        'prefetch_k': prefetch_k,
                        'recall': val,
                    })
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"Skipping {mf}: {e}")

    if not data:
        print(f"No data found.")
        return

    df = pd.DataFrame(data)
    
    # Create plots directory inside sweep_dir
    plots_dir = sweep_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if layer_idx is not None:
        layers_to_plot = [layer_idx]
    else:
        layers_to_plot = sorted(df['layer_idx'].unique())
    
    for l in layers_to_plot:
        out_file = plots_dir / f"pareto_frontier_layer{l}.png"
        plot_pareto_layer(df, l, out_file, predict_k=predict_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pareto frontier from sweep results")
    parser.add_argument("sweep_dir", help="Directory containing sweep results")
    parser.add_argument("--layer", type=int, help="Specific layer index to plot (default: all discovered layers)")
    parser.add_argument("--pk", type=int, help="Specific predict_k value for the recall-lookahead trade-off plot")
    
    args = parser.parse_args()
    
    plot_pareto(args.sweep_dir, args.layer, args.pk)
