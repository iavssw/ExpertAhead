import json, glob, sys

search_path = sys.argv[1] + "/**/*.json"
results = []
for fpath in glob.glob(search_path, recursive=True):
    if "training_metrics.json" not in fpath: continue
    try:
        with open(fpath, "r") as f: data = json.load(f)
    except: continue
    
    config = data.get("config_name", "unknown")
    if "ablation_" not in config: continue
    variant = config.split("ablation_")[1].split("_hist")[0]
    
    layer = data.get("layer_idx", "unknown")
    n = data.get("future_steps", "unknown")
    best = data.get("best", 0.0) * 100
    results.append({"variant": variant, "layer": layer, "n": n, "best": best})

print(f"{'Variant':<25} | {'Layer':<5} | {'N':<5} | {'Best Recall'}")
print("-" * 55)
results.sort(key=lambda x: (x["layer"], x["n"], x["variant"]))
for r in results:
    print(f"{r['variant']:<25} | {r['layer']:<5} | {r['n']:<5} | {r['best']:.2f}%")
