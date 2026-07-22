import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Directories
dir_mlp_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/mlp_hist4_ablation_20260616_080608")
dir_tx_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/transformer_ablation_20260614_151137")
dir_emb_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_only_full_20260602_172025")

variants = {
    "MLP Emb (H=1)": {"dir": dir_emb_hist1, "match": "emb_only"},
    "MLP Emb (H=4)": {"dir": dir_mlp_hist4, "match": "emb_only"},
    "Tx Emb (H=4)":  {"dir": dir_tx_hist4, "match": "emb_only"},
}

horizons = [1, 4, 8, 16]
budgets = [8, 16, 32]

results = {v: {h: {b: {"recall": [], "precision": []} for b in budgets} for h in horizons} for v in variants}

for v_key, info in variants.items():
    directory = info["dir"]
    match_str = info["match"]
    if not directory.exists(): continue
    for path in directory.glob("*"):
        if not path.is_dir(): continue
        name = path.name
        
        if match_str == "emb_only" and "markov" in name: continue
        if match_str not in name: continue
        
        ablation_layers = ["layer_0", "layer_23", "layer_47"]
        for layer_name in ablation_layers:
            layer_dir = path / layer_name
            if not (layer_dir / "training_metrics.json").exists(): continue
            with open(layer_dir / "training_metrics.json") as f:
                metrics = json.load(f)
            
            matched_h = next((h for h in horizons if name.endswith(f"_f{h}") or f"_f{h}_" in name), None)
            if matched_h:
                topk_metrics = metrics.get('best_val_recalls', {})
                alt_topk_metrics = metrics.get('val_best_topk', {})
                for b in budgets:
                    r_key = f"topk_recall@{b}"
                    p_key = f"topk_precision@{b}"
                    if r_key in topk_metrics and p_key in topk_metrics:
                        results[v_key][matched_h][b]['recall'].append(topk_metrics[r_key])
                        results[v_key][matched_h][b]['precision'].append(topk_metrics[p_key])
                    else:
                        b_str = str(b)
                        if b_str in alt_topk_metrics:
                            results[v_key][matched_h][b]['recall'].append(alt_topk_metrics[b_str]['recall'])
                            results[v_key][matched_h][b]['precision'].append(alt_topk_metrics[b_str]['precision'])

# Plotting: One plot per budget with Recall and Precision subplots
colors = {"Tx Emb (H=4)": "#d62728", "MLP Emb (H=4)": "#2ca02c", "MLP Emb (H=1)": "#1f77b4"}
markers = {"Tx Emb (H=4)": "o", "MLP Emb (H=4)": "s", "MLP Emb (H=1)": "^"}

for b in budgets:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Architecture Embedding Ablation (Budget={b})', fontsize=16)

    for v in variants.keys():
        # Recall Subplot
        y_recall = [np.mean(results[v][h][b]['recall']) * 100 if results[v][h][b]['recall'] else 0 for h in horizons]
        ax1.plot(horizons, y_recall, label=v, color=colors[v], marker=markers[v], linewidth=2)
        
        # Precision Subplot
        y_prec = [np.mean(results[v][h][b]['precision']) * 100 if results[v][h][b]['precision'] else 0 for h in horizons]
        ax2.plot(horizons, y_prec, label=v, color=colors[v], marker=markers[v], linewidth=2)

    # Format Recall Subplot
    ax1.set_xlabel('Lookaheads (Horizon)', fontsize=12)
    ax1.set_ylabel('Recall (%)', fontsize=12)
    ax1.set_title(f'Recall vs Lookaheads (B={b})', fontsize=14)
    ax1.set_xticks(horizons)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Format Precision Subplot
    ax2.set_xlabel('Lookaheads (Horizon)', fontsize=12)
    ax2.set_ylabel('Precision (%)', fontsize=12)
    ax2.set_title(f'Precision vs Lookaheads (B={b})', fontsize=14)
    ax2.set_xticks(horizons)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'/home/michaelg/.gemini/antigravity-ide/brain/ed35921d-0698-4a9e-a6da-4d7eba8d593f/arch_ablation_B{b}.png')
    plt.close()

print("Plots saved to artifact directory: arch_ablation_B8.png, arch_ablation_B16.png, arch_ablation_B32.png")
