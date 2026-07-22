import os
import csv
import subprocess
import sys

sys.path.insert(0, "/home/michael/heteroPredict/py/utils")
from sweep_cache_policy import generate_plots

def main():
    base_dir = "/home/michael/heteroPredict/py/utils/final_results_runs/sec1_cache_policy/20260616_203559"
    main_csv = os.path.join(base_dir, "sweep.csv")
    tmp_prefix = os.path.join(base_dir, "missing_random")
    tmp_csv = f"{tmp_prefix}.csv"

    cmd = [
        "/home/michael/heteroPredict/utils/rocmPytorch/bin/python3",
        "py/utils/sweep_cache_policy.py",
        "--model", "qwen",
        "--backend", "cached",
        "--dataset", "wikitext",
        "--policies", "RANDOM",
        "--cache-sizes", "8", "16", "24", "32", "40", "48", "56", "64",
        "--lambdas", "0.0",
        "--mode", "generation",
        "--num-prompts", "10",
        "--max-new-tokens", "256",
        "--output-prefix", tmp_prefix
    ]

    print("Running sweep for RANDOM baseline across all cache sizes...")
    subprocess.run(cmd, check=True, cwd="/home/michael/heteroPredict")

    print("Reading new RANDOM results...")
    new_rows = []
    with open(tmp_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_rows.append(row)
            
    print("Reading existing results from sweep.csv...")
    all_rows = []
    fieldnames = []
    with open(main_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Skip old RANDOM rows
            if row.get("policy") != "RANDOM":
                all_rows.append(row)
            
    # Append the fresh RANDOM rows
    all_rows.extend(new_rows)
    
    # Sort logically
    def sort_key(r):
        return (r['policy'], float(r['lambda']), int(r['cache_size']))
    
    all_rows.sort(key=sort_key)
    
    print("Writing updated sweep.csv...")
    with open(main_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print("Re-generating plots...")
    # Convert types back for generate_plots
    for r in all_rows:
        r['cache_size'] = int(r['cache_size'])
        r['lambda'] = float(r['lambda'])
        if r.get('cache_hit_rate'): r['cache_hit_rate'] = float(r['cache_hit_rate'])
        if r.get('tps'): r['tps'] = float(r['tps'])
        
    generate_plots(all_rows, os.path.join(base_dir, "sweep"))
    
    print("Cleaning up temporary files...")
    os.remove(tmp_csv)
    for lam in ["0.0"]:
        for ptype in ["hitrate_bar", "tps_bar", "ppl_bar"]:
            fpath = f"{tmp_prefix}_{ptype}_lambda_{lam}.png"
            if os.path.exists(fpath):
                os.remove(fpath)
                
    print("Done! RANDOM baseline has been updated in sweep.csv.")

if __name__ == "__main__":
    main()
