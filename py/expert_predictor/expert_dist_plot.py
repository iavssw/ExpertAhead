import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_layer(layer_idx, out_filename):
    data_dir = Path(f"../../trainingDataExtended/qwen3_30b_sharded/layer_{layer_idx}")
    files = list(data_dir.glob("*.pt"))
    if not files:
        print(f"No .pt files found in {data_dir}")
        return

    n_exp = 128
    active_k = 8
    overall_counts = np.zeros(n_exp)
    
    prompt_pcts = None
    
    # Load first prompt
    fp = files[0]
    payload = torch.load(fp, map_location='cpu', weights_only=False)
    for layer_info in payload.get('layers', []):
        if layer_info['layer_idx'] == layer_idx:
            logits = layer_info['router_logits']
            topk_ids = logits.topk(active_k, dim=-1).indices
            T = topk_ids.size(0)
            unique, counts = np.unique(topk_ids.numpy(), return_counts=True)
            prompt_counts = np.zeros(n_exp)
            prompt_counts[unique] = counts
            prompt_pcts = (prompt_counts / T) * 100
            break

    # Load 100 prompts for overall distribution
    total_T = 0
    for i in range(min(100, len(files))):
        fp = files[i]
        payload = torch.load(fp, map_location='cpu', weights_only=False)
        for layer_info in payload.get('layers', []):
            if layer_info['layer_idx'] == layer_idx:
                logits = layer_info['router_logits']
                topk_ids = logits.topk(active_k, dim=-1).indices
                T = topk_ids.size(0)
                total_T += T
                unique, counts = np.unique(topk_ids.numpy(), return_counts=True)
                overall_counts[unique] += counts

    overall_pcts = (overall_counts / total_T) * 100

    # Sort pcts for plotting
    sorted_prompt = np.sort(prompt_pcts)[::-1]
    sorted_overall = np.sort(overall_pcts)[::-1]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(range(n_exp), sorted_prompt, color='cyan')
    ax1.set_title(f"Expert Usage Layer {layer_idx} (Single Prompt)")
    ax1.set_xlabel("Expert Rank")
    ax1.set_ylabel("% of Forward Passes")
    ax1.set_ylim(0, 100)
    
    ax2.bar(range(n_exp), sorted_overall, color='magenta')
    ax2.set_title(f"Expert Usage Layer {layer_idx} (Across 100 Prompts)")
    ax2.set_xlabel("Expert Rank")
    ax2.set_ylabel("% of Forward Passes")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.abspath(out_filename)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    analyze_layer(23, "expert_dist.png")
    analyze_layer(0, "expert_dist_layer0.png")
