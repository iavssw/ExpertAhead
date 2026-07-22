import torch
from pathlib import Path
import sys
import numpy as np

def analyze_expert_distribution():
    data_dir = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/layer_23")
    files = list(data_dir.glob("*.pt"))
    if not files:
        print("No .pt files found in", data_dir)
        return

    n_exp = 128
    active_k = 8

    overall_counts = np.zeros(n_exp)
    
    # Load first 10 files (prompts)
    print("--- Single Prompt Distribution ---")
    for i in range(5):
        fp = files[i]
        payload = torch.load(fp, map_location='cpu', weights_only=False)
        for layer_info in payload.get('layers', []):
            if layer_info['layer_idx'] == 23:
                logits = layer_info['router_logits']
                topk_ids = logits.topk(active_k, dim=-1).indices # shape (T, active_k)
                
                # count for this prompt
                unique, counts = np.unique(topk_ids.numpy(), return_counts=True)
                prompt_counts = np.zeros(n_exp)
                prompt_counts[unique] = counts
                
                # Update overall counts
                overall_counts += prompt_counts
                
                # Print stats for this prompt
                non_zero = (prompt_counts > 0).sum()
                print(f"Prompt {i} (Length {topk_ids.size(0)} tokens):")
                print(f"  Used {non_zero}/{n_exp} experts.")
                sorted_counts = np.sort(prompt_counts)[::-1]
                print(f"  Top 5 expert counts: {sorted_counts[:5]}")
                print(f"  Bottom 5 expert counts: {sorted_counts[-5:]}")
                
                # Calculate entropy to show uniformity
                probs = prompt_counts[prompt_counts > 0] / prompt_counts.sum()
                entropy = -np.sum(probs * np.log(probs))
                max_entropy = np.log(n_exp)
                print(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} (lower means more skewed)")

    print("\n--- Overall Distribution (across many prompts) ---")
    # Load more files for overall distribution
    for i in range(5, min(100, len(files))):
        fp = files[i]
        payload = torch.load(fp, map_location='cpu', weights_only=False)
        for layer_info in payload.get('layers', []):
            if layer_info['layer_idx'] == 23:
                logits = layer_info['router_logits']
                topk_ids = logits.topk(active_k, dim=-1).indices
                unique, counts = np.unique(topk_ids.numpy(), return_counts=True)
                overall_counts[unique] += counts

    non_zero = (overall_counts > 0).sum()
    print(f"Across 100 prompts:")
    print(f"  Used {non_zero}/{n_exp} experts overall.")
    sorted_idx = np.argsort(overall_counts)[::-1]
    sorted_counts = overall_counts[sorted_idx]
    
    print(f"  Top 10 most used experts:")
    for j in range(10):
        print(f"    Expert {sorted_idx[j]}: {sorted_counts[j]} times")
        
    print(f"  Top 10 least used experts:")
    for j in range(1, 11):
        print(f"    Expert {sorted_idx[-j]}: {sorted_counts[-j]} times")
        
    probs = overall_counts[overall_counts > 0] / overall_counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    print(f"  Overall Entropy: {entropy:.2f} / {np.log(n_exp):.2f}")

if __name__ == "__main__":
    analyze_expert_distribution()
