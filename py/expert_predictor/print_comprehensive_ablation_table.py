import json
import numpy as np
from pathlib import Path
import os

# Directories
dir_mlp_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/mlp_hist4_ablation_20260616_080608")
dir_tx_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/transformer_ablation_20260614_151137")
dir_markov_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/markov_only_20260602_022107")
dir_emb_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_only_full_20260602_172025")
dir_emb_markov_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_markov_20260601_154323")

variants = {
    "markov_hist1": {"dir": dir_markov_hist1, "match": "markov"},
    "mlp_emb_hist1": {"dir": dir_emb_hist1, "match": "emb_only"},
    "mlp_emb_markov_hist1": {"dir": dir_emb_markov_hist1, "match": "emb_markov"},
    "mlp_emb_hist4": {"dir": dir_mlp_hist4, "match": "emb_only"},
    "mlp_emb_markov_hist4": {"dir": dir_mlp_hist4, "match": "emb_markov"},
    "mlp_emb_markov_pfill_hist4": {"dir": dir_mlp_hist4, "match": "emb_markov_pfill"},
    "tx_emb_hist4": {"dir": dir_tx_hist4, "match": "emb_only"},
    "tx_emb_markov_hist4": {"dir": dir_tx_hist4, "match": "emb_markov"},
    "tx_emb_markov_pfill_hist4": {"dir": dir_tx_hist4, "match": "emb_markov_pfill"},
}

horizons = [1, 4, 8, 16]
budgets = [8, 16, 32]

results = {v: {h: {b: {"recall": [], "precision": []} for b in budgets} for h in horizons} for v in variants}

for v_key, info in variants.items():
    directory = info["dir"]
    match_str = info["match"]
    if not directory.exists(): 
        print(f"Warning: Directory not found for {v_key}: {directory}")
        continue
    for path in directory.glob("*"):
        if not path.is_dir(): continue
        name = path.name
        
        # Need to be precise to differentiate emb_markov from emb_markov_pfill
        if match_str == "emb_markov" and "pfill" in name: continue
        if match_str == "emb_only" and "markov" in name: continue
        if match_str not in name: continue
        
        # Only compare apples-to-apples on the specific layers used in the ablation
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
                    # check new format
                    r_key = f"topk_recall@{b}"
                    p_key = f"topk_precision@{b}"
                    if r_key in topk_metrics and p_key in topk_metrics:
                        results[v_key][matched_h][b]['recall'].append(topk_metrics[r_key])
                        results[v_key][matched_h][b]['precision'].append(topk_metrics[p_key])
                    else:
                        # check old format
                        b_str = str(b)
                        if b_str in alt_topk_metrics:
                            results[v_key][matched_h][b]['recall'].append(alt_topk_metrics[b_str]['recall'])
                            results[v_key][matched_h][b]['precision'].append(alt_topk_metrics[b_str]['precision'])

print("\n### Performance Comparison Across Models and Input Features\n")
print("| Horizon | Budget | Metric | Markov | MLP: Emb (H=1) | MLP: Emb+Markov (H=1) | MLP: Emb (H=4) | MLP: Emb+Markov (H=4) | MLP: All (H=4) | Tx: Emb (H=4) | Tx: Emb+Markov (H=4) | Tx: All (H=4) |")
print("|---------|--------|--------|--------|----------------|-----------------------|----------------|-----------------------|----------------|---------------|----------------------|---------------|")

v_order = [
    "markov_hist1", "mlp_emb_hist1", "mlp_emb_markov_hist1",
    "mlp_emb_hist4", "mlp_emb_markov_hist4", "mlp_emb_markov_pfill_hist4",
    "tx_emb_hist4", "tx_emb_markov_hist4", "tx_emb_markov_pfill_hist4"
]

latex_str = """
\\begin{table*}[h]
\\centering
\\caption{Comprehensive performance comparison of predictor architectures and input features (Recall and Precision @ $B$). Highest values across all variants are bolded.}
\\label{tab:comprehensive_ablation}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{cc l ccc ccc ccc}
\\toprule
\\multirow{2}{*}{Horizon} & \\multirow{2}{*}{Budget} & \\multirow{2}{*}{Metric} & \\multicolumn{3}{c}{History = 1 (Baselines)} & \\multicolumn{3}{c}{MLP (History = 4)} & \\multicolumn{3}{c}{Transformer (History = 4)} \\\\
\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-12}
 & & & Markov & MLP: Emb & MLP: Emb+Markov & Emb & Emb+Markov & Emb+Markov+Pfill & Emb & Emb+Markov & Emb+Markov+Pfill \\\\
\\midrule
"""

for h in horizons:
    latex_str += f"\\multirow{{6}}{{*}}{{{h}}} "
    for b in budgets:
        # Recall
        r_vals_raw = []
        for v in v_order:
            arr = results[v][h][b]['recall']
            r_vals_raw.append(np.mean(arr)*100 if arr else 0.0)
            
        best_r = max(r_vals_raw)
        r_strs = []
        for val in r_vals_raw:
            s = f"{val:.1f}%" if val > 0 else "N/A"
            if val == best_r and val > 0: s = f"**{s}**"
            r_strs.append(s)
            
        print(f"| {h} | {b} | Recall | " + " | ".join(r_strs) + " |")
        
        # Latex formatting for recall
        lr_strs = [f"\\textbf{{{v:.1f}\\%}}" if v == best_r and v > 0 else (f"{v:.1f}\\%" if v > 0 else "-") for v in r_vals_raw]
        latex_str += f"& \\multirow{{2}}{{*}}{{{b}}} & Recall & {' & '.join(lr_strs)} \\\\\n"
        
        # Precision
        p_vals_raw = []
        for v in v_order:
            arr = results[v][h][b]['precision']
            p_vals_raw.append(np.mean(arr)*100 if arr else 0.0)
            
        best_p = max(p_vals_raw)
        p_strs = []
        for val in p_vals_raw:
            s = f"{val:.1f}%" if val > 0 else "N/A"
            if val == best_p and val > 0: s = f"**{s}**"
            p_strs.append(s)
            
        print(f"| | | Precision | " + " | ".join(p_strs) + " |")
        
        # Latex formatting for precision
        lp_strs = [f"\\textbf{{{v:.1f}\\%}}" if v == best_p and v > 0 else (f"{v:.1f}\\%" if v > 0 else "-") for v in p_vals_raw]
        latex_str += f"& & Precision & {' & '.join(lp_strs)} \\\\\n"
        
        if b != budgets[-1]:
            latex_str += "\\cmidrule{2-12}\n"
    
    if h != horizons[-1]:
        latex_str += "\\midrule\n"

latex_str += """\\bottomrule
\\end{tabular}
}
\\end{table*}
"""

print("\n\n### LaTeX Source code for the paper:\n")
print("```latex")
print(latex_str.strip())
print("```")
