import json
import numpy as np
from pathlib import Path

dir_mlp_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/mlp_hist4_ablation_20260616_080608")
dir_tx_hist4 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b_sharded/transformer_ablation_20260614_151137")
dir_emb_hist1 = Path("/mnt/storage/Michael/michaelg/heteroPredict/trainingDataExtended/qwen3_30b/emb_only_full_20260602_172025")

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

print(json.dumps(
    {v: {h: {b: {
        "recall": float(np.mean(results[v][h][b]['recall'])) * 100 if results[v][h][b]['recall'] else 0.0,
        "precision": float(np.mean(results[v][h][b]['precision'])) * 100 if results[v][h][b]['precision'] else 0.0
    } for b in budgets} for h in horizons} for v in variants}, indent=2
))
