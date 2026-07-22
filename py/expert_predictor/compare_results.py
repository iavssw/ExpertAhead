import json
import glob
from pathlib import Path

# Paths to search
search_paths = [
    "../../trainingDataExtended/qwen3_30b_sharded/transformer_study/**/*.json",
    "../../trainingDataExtended/qwen3_30b_sharded/mlp_study/**/*.json",
    "../../trainingDataExtended/qwen3_30b_sharded/transformer_full_20260611_165858/**/*.json",
    "../../trainingDataExtended/qwen3_30b/emb_only_full_20260602_172025/**/*.json"
]

results = []
for pattern in search_paths:
    for fpath in glob.glob(pattern, recursive=True):
        if "training_metrics.json" not in fpath:
            continue
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
            
        config = data.get("config_name", "unknown")
        layer = data.get("layer_idx", "unknown")
        
        # Future steps might be explicitly stated or inferred from config
        n = data.get("future_steps")
        if n is None and "_f" in config:
            n = int(config.split("_f")[-1])
        elif n is None:
            n = "unknown"
            
        best_recall = data.get("best", 0.0)
        
        # Determine source label
        if "transformer_full_20260611_165858" in fpath:
            arch = "Old Trans. Capped (H=4)"
        elif "emb_only_full_20260602_172025" in fpath:
            arch = "Old MLP Baseline"
        elif "transformer_study" in fpath:
            if "transformer_eh4" in config:
                arch = "Transformer (H=4)"
            elif "transformer_eh6" in config:
                arch = "Transformer (H=6)"
            else:
                arch = f"Trans: {config}"
        elif "mlp_study" in fpath:
            arch = "MLP Baseline (H=1)"
        else:
            arch = config
            
        # Only add to results if we have valid Layer and N
        if layer != "unknown" and n != "unknown":
            # Don't add duplicate entries if there are old checkpoints etc
            if not any(r['arch'] == arch and r['layer'] == layer and r['n'] == n for r in results):
                results.append({
                    "arch": arch,
                    "layer": layer,
                    "n": n,
                    "recall": best_recall * 100
                })

print(f"{'Architecture':<25} | {'Layer':<5} | {'Lookahead (N)':<13} | {'Best Adaptive Recall':<20}")
print("-" * 75)

# Sort by Layer -> N -> Arch
results.sort(key=lambda x: (int(x["layer"]) if isinstance(x["layer"], int) else 999, 
                           int(x["n"]) if isinstance(x["n"], int) else 999, 
                           x["arch"]))

for r in results:
    print(f"{r['arch']:<25} | {r['layer']:<5} | {r['n']:<13} | {r['recall']:.2f}%")
