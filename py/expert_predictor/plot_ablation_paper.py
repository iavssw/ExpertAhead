import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import the user's plotting utilities for consistent styling
from plot_utils import apply_style, VARIANT_STYLES

apply_style()

# Directories
dir_mlp_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/mlp_hist4_ablation_20260616_080608")
dir_tx_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/transformer_ablation_20260614_151137")
dir_emb_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_only_full_20260602_172025")

# Map our keys to the plot_utils keys for styling
variants = {
    "mlp_emb_only": {"dir": dir_emb_hist1, "match": "emb_only"},
    "mlp_hist4_emb_only": {"dir": dir_mlp_hist4, "match": "emb_only"},
    "tx_emb_only":  {"dir": dir_tx_hist4, "match": "emb_only"},
    "tx_emb_markov": {"dir": dir_tx_hist4, "match": "emb_markov"},
    "tx_emb_markov_pfill": {"dir": dir_tx_hist4, "match": "emb_markov_pfill"},
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
        
        if match_str == "emb_markov" and "pfill" in name: continue
        if match_str == "emb_only" and "markov" in name: continue
        if match_str not in name: continue
        
        # Apples-to-apples evaluation layers
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

def create_ablation_plot(variant_keys, title_prefix, filename_prefix, color_override=None):
    if color_override is None:
        color_override = {}
        
    for b in budgets:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'hspace': 0.45})
        
        for v in variant_keys:
            s = VARIANT_STYLES[v]
            c = color_override.get(v, s['color'])
            
            label_text = s['label'].replace("Tx:", "Transformer (History=4):").replace("Hist=", "History=")
            
            # Recall
            y_recall = [np.mean(results[v][h][b]['recall']) * 100 if results[v][h][b]['recall'] else 0 for h in horizons]
            ax1.plot(horizons, y_recall, label=label_text, color=c, 
                     marker=s['marker'], ls=s['ls'], lw=s['lw'], markersize=7)
            
            # Precision
            y_prec = [np.mean(results[v][h][b]['precision']) * 100 if results[v][h][b]['precision'] else 0 for h in horizons]
            ax2.plot(horizons, y_prec, label=label_text, color=c, 
                     marker=s['marker'], ls=s['ls'], lw=s['lw'], markersize=7)

        # Style Recall
        ax1.set_xlabel("Stride (S)")
        ax1.set_ylabel(f"Recall @ B={b}")
        ax1.set_title(f"Recall", fontweight="bold")
        ax1.set_xticks(horizons)
        ax1.set_ylim(0, 105)
        ax1.legend(loc='best')

        # Style Precision
        ax2.set_xlabel("Stride (S)")
        ax2.set_ylabel(f"Precision @ B={b}")
        ax2.set_title(f"Precision", fontweight="bold")
        ax2.set_xticks(horizons)
        ax2.set_ylim(0, 105)
        ax2.legend(loc='best')

        fig.suptitle(f"{title_prefix} — Budget B={b}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        plt.savefig(f"paper_plots/{filename_prefix}_B{b}.png", dpi=300, bbox_inches="tight")
        plt.close()

# 1. Architecture Ablation Plot (MLP vs Tx on Embeddings)
create_ablation_plot(
    variant_keys=["mlp_emb_only", "mlp_hist4_emb_only", "tx_emb_only"],
    title_prefix="Architecture Ablation (Embedding Branch)",
    filename_prefix="paper_arch_ablation",
    color_override={
        "mlp_emb_only": "#1f77b4",        # Blue
        "mlp_hist4_emb_only": "#2ca02c",  # Green
        "tx_emb_only": "#E63946"          # Red (keep Transformer as the primary red)
    }
)

# 2. Feature Ablation Plot (Monotonic Improvement with Tx H=4)
create_ablation_plot(
    variant_keys=["tx_emb_only", "tx_emb_markov", "tx_emb_markov_pfill"],
    title_prefix="Feature Component Ablation (Transformer)",
    filename_prefix="paper_feat_ablation"
)

print("Paper plots generated and saved to artifact directory.")
