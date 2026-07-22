import json
import numpy as np
from pathlib import Path

# Paths
tx_dir = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/transformer_ablation_20260614_151137")
markov_dir = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_markov_20260601_154323")

variants = {
    "markov_only": "markov_only",
    "emb_only": "tx_emb_only",
    "emb_markov": "tx_emb_markov",
    "emb_markov_pfill": "tx_emb_markov_pfill"
}

horizons = [1, 4, 8, 16]
budgets = [8, 16, 32]

results = {v: {h: {b: {"recall": [], "precision": []} for b in budgets} for h in horizons} for v in variants}

def load_metrics(directory, v_key, match_str):
    for path in directory.glob("*"):
        if not path.is_dir(): continue
        name = path.name
        if match_str not in name: continue
        
        for layer_dir in path.glob("layer_*"):
            if not (layer_dir / "training_metrics.json").exists():
                continue
                
            with open(layer_dir / "training_metrics.json") as f:
                metrics = json.load(f)
            
        # Identify horizon
        matched_h = None
        for h in horizons:
            # Need to match exact horizon, e.g. _f1 vs _f16
            if name.endswith(f"_f{h}") or f"_f{h}_" in name:
                matched_h = h
                break
        
        if matched_h is None: continue
        
        topk_metrics = metrics.get('best_val_recalls', {})
        for b in budgets:
            r_key = f"topk_recall@{b}"
            p_key = f"topk_precision@{b}"
            if r_key in topk_metrics and p_key in topk_metrics:
                results[v_key][matched_h][b]['recall'].append(topk_metrics[r_key])
                results[v_key][matched_h][b]['precision'].append(topk_metrics[p_key])

# Load Transformer variants
load_metrics(tx_dir, "emb_markov_pfill", "emb_markov_pfill")
load_metrics(tx_dir, "emb_markov", "emb_markov") # This will also match pfill, so we filter it below
# Re-filter emb_markov to remove pfill matches
results["emb_markov"] = {h: {b: {"recall": [], "precision": []} for b in budgets} for h in horizons}
for path in tx_dir.glob("*"):
    if not path.is_dir(): continue
    name = path.name
    if "emb_markov" in name and "pfill" not in name:
        for layer_dir in path.glob("layer_*"):
            if not (layer_dir / "training_metrics.json").exists(): continue
            with open(layer_dir / "training_metrics.json") as f: metrics = json.load(f)
            matched_h = next((h for h in horizons if name.endswith(f"_f{h}") or f"_f{h}_" in name), None)
            if matched_h:
                topk_metrics = metrics.get('best_val_recalls', {})
                for b in budgets:
                    r_key = f"topk_recall@{b}"
                    p_key = f"topk_precision@{b}"
                    if r_key in topk_metrics and p_key in topk_metrics:
                        results["emb_markov"][matched_h][b]['recall'].append(topk_metrics[r_key])
                        results["emb_markov"][matched_h][b]['precision'].append(topk_metrics[p_key])

load_metrics(tx_dir, "emb_only", "emb_only")

# Load Markov Only
load_metrics(markov_dir, "markov_only", "markov_only")


print("\n### Performance Comparison Across Input Features\n")
print("| Horizon $N$ | Budget $B$ | Metric | Markov Only | Tx: Emb Only | Tx: Emb + Markov | Tx: Emb + Markov + Pfill |")
print("|-------------|------------|--------|-------------|--------------|------------------|--------------------------|")

latex_str = """
\\begin{table}[h]
\\centering
\\caption{Performance comparison of predictor input features (Recall and Precision @ $B$). Highest values are bolded.}
\\label{tab:feature_ablation}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{cc l cccc}
\\toprule
\\multirow{2}{*}{Horizon $N$} & \\multirow{2}{*}{Budget $B$} & \\multirow{2}{*}{Metric} & \\multicolumn{4}{c}{Predictor Architecture} \\\\
\\cmidrule(lr){4-7}
 & & & Markov Only & Tx: Emb & Tx: Emb+Markov & Tx: Emb+Markov+Pfill \\\\
\\midrule
"""

for h in horizons:
    latex_str += f"\\multirow{{6}}{{*}}{{{h}}} "
    for b in budgets:
        # Recall
        r_vals_raw = []
        for v in ["markov_only", "emb_only", "emb_markov", "emb_markov_pfill"]:
            arr = results[v][h][b]['recall']
            r_vals_raw.append(np.mean(arr)*100 if arr else 0.0)
            
        best_r = max(r_vals_raw)
        r_strs = []
        for val in r_vals_raw:
            s = f"{val:.1f}\\%" if val > 0 else "N/A"
            if val == best_r and val > 0: s = f"**{s}**"
            r_strs.append(s)
            
        print(f"| {h} | {b} | Recall | {r_strs[0]} | {r_strs[1]} | {r_strs[2]} | {r_strs[3]} |")
        
        # Latex formatting for recall
        lr_strs = [f"\\textbf{{{v:.1f}\\%}}" if v == best_r and v > 0 else (f"{v:.1f}\\%" if v > 0 else "-") for v in r_vals_raw]
        latex_str += f"& \\multirow{{2}}{{*}}{{{b}}} & Recall & {lr_strs[0]} & {lr_strs[1]} & {lr_strs[2]} & {lr_strs[3]} \\\\\n"
        
        # Precision
        p_vals_raw = []
        for v in ["markov_only", "emb_only", "emb_markov", "emb_markov_pfill"]:
            arr = results[v][h][b]['precision']
            p_vals_raw.append(np.mean(arr)*100 if arr else 0.0)
            
        best_p = max(p_vals_raw)
        p_strs = []
        for val in p_vals_raw:
            s = f"{val:.1f}\\%" if val > 0 else "N/A"
            if val == best_p and val > 0: s = f"**{s}**"
            p_strs.append(s)
            
        print(f"| | | Precision | {p_strs[0]} | {p_strs[1]} | {p_strs[2]} | {p_strs[3]} |")
        
        # Latex formatting for precision
        lp_strs = [f"\\textbf{{{v:.1f}\\%}}" if v == best_p and v > 0 else (f"{v:.1f}\\%" if v > 0 else "-") for v in p_vals_raw]
        latex_str += f"& & Precision & {lp_strs[0]} & {lp_strs[1]} & {lp_strs[2]} & {lp_strs[3]} \\\\\n"
        
        if b != budgets[-1]:
            latex_str += "\\cmidrule{2-7}\n"
    
    if h != horizons[-1]:
        latex_str += "\\midrule\n"

latex_str += """\\bottomrule
\\end{tabular}
}
\\end{table}
"""

print("\n\n### LaTeX Source code for the paper:\n")
print("```latex")
print(latex_str.strip())
print("```")
