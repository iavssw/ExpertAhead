import os
import csv
import subprocess
import sys
import tempfile

sys.path.insert(0, "/home/michael/heteroPredict/py/utils")
from sweep_cache_policy import generate_plots

def main():
    base_dir = "/home/michael/heteroPredict/py/utils/final_results_runs/sec1_cache_policy/20260616_203559"
    main_csv = os.path.join(base_dir, "sweep.csv")
    tmp_prefix = os.path.join(base_dir, "missing_sweep")
    tmp_csv = f"{tmp_prefix}.csv"

    cmd = [
        "/home/michael/heteroPredict/utils/rocmPytorch/bin/python3",
        "py/utils/sweep_cache_policy.py",
        "--model", "qwen",
        "--backend", "cached",
        "--dataset", "wikitext",
        "--policies", "LRU", "MRU", "LFU", "MFU", "RANDOM", "LFRU", "PREFILL",
        "--cache-sizes", "24", "40", "56",
        "--lambdas", "0.0",
        "--mode", "generation",
        "--num-prompts", "10",
        "--max-new-tokens", "256",
        "--output-prefix", tmp_prefix
    ]

    print("Running sweep for missing cache sizes (24, 40, 56)...")
    subprocess.run(cmd, check=True, cwd="/home/michael/heteroPredict")

    print("Reading new results...")
    new_rows = []
    with open(tmp_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_rows.append(row)
            
    print("Reading existing results...")
    all_rows = []
    fieldnames = []
    with open(main_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)
            
    # Append
    all_rows.extend(new_rows)
    
    # Sort logically so they aren't just tacked on the end when plotting
    def sort_key(r):
        return (r['policy'], float(r['lambda']), int(r['cache_size']))
    
    all_rows.sort(key=sort_key)
    
    print("Writing updated sweep.csv...")
    with open(main_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print("Re-generating plots with all cache sizes...")
    # Convert types back for generate_plots
    for r in all_rows:
        r['cache_size'] = int(r['cache_size'])
        r['lambda'] = float(r['lambda'])
        if r.get('cache_hit_rate'): r['cache_hit_rate'] = float(r['cache_hit_rate'])
        if r.get('tps'): r['tps'] = float(r['tps'])
        
    # generate_plots will overwrite the original plots since we pass base_dir/sweep as prefix
    generate_plots(all_rows, os.path.join(base_dir, "sweep"))
    
    print("Cleaning up temporary files...")
    os.remove(tmp_csv)
    # also remove the temporary plots it created
    for lam in ["0.0"]:
        for ptype in ["hitrate_bar", "tps_bar", "ppl_bar"]:
            fpath = f"{tmp_prefix}_{ptype}_lambda_{lam}.png"
            if os.path.exists(fpath):
                os.remove(fpath)
                
    print("Done! Added 24, 40, and 56 to sec1.")

if __name__ == "__main__":
    main()
