import subprocess
import csv
import os
import re

QWEN_SCRIPT = "../unified_llm_w4a16/qwen3_30B-A3B_w4a16_model.py"
WEIGHTS_DIR = "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"
CONFIG_PATH = "../unified_llm_w4a16/configs/configs_strixH_qwen3_30B_A3B.json5"
PREDICTOR_DIR = "/home/michael/heteroPredict/trainingData/qwen3_30b/transformer_final_pfill_markov_emb/transformer_eh4_h64_f6"

def parse_tps(output):
    match = re.search(r"Average Time per Token: ([\d.]+) seconds", output)
    if match:
        return 1.0 / float(match.group(1))
    return None

def parse_avg_load(output):
    # Bandwidth: MissLoads=149, AvgLoadTime=0.93ms
    matches = re.findall(r"AvgLoadTime=([\d.]+)ms", output)
    if matches:
        return sum(float(x) for x in matches) / len(matches)
    return None

def run_cmd(backend, cache_size, policy="LRU", budget_frac=None, lookahead=None):
    cmd = [
        "python3", QWEN_SCRIPT,
        "--backend", backend,
        "--expert-weights-dir", WEIGHTS_DIR,
        "--config-path", CONFIG_PATH,
        "--max-cached-experts", str(cache_size),
        "--max-new-tokens", "128",
        "--sweep-prompts-file", "/tmp/_tmp_wikitext_prompts.json",
        "--cache-policy", policy
    ]
    
    if backend == "predict":
        budget = max(1, int(cache_size * budget_frac))
        cmd.extend([
            "--predictor-model", PREDICTOR_DIR,
            "--predictor-lookahead", str(lookahead),
            "--prefetch-experts-count", str(budget)
        ])
    
    env = os.environ.copy()
    env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "0"
    
    print(f"Running: {' '.join(cmd)}")
    import sys
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
        
    process.wait()
    full_output = "".join(output_lines)
    
    tps = parse_tps(full_output)
    avg_load = parse_avg_load(full_output)
    
    if process.returncode != 0:
        print("Error! Command failed.")
        
    return tps, avg_load

def main():
    import json
    
    with open("/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/wikitext-103-raw-v1_test.txt", "r", encoding="utf-8") as f:
        blob = f.read()
        
    paragraphs = [p.strip() for p in blob.split("\n\n") if p.strip()]
    
    prompts = []
    current = ""
    max_chars = 4096
    
    for para in paragraphs:
        if para.startswith("=") and para.endswith("="):
            continue
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) >= max_chars:
            if current:
                prompts.append(current[:max_chars])
                if len(prompts) >= 3:
                    break
            current = para
        else:
            current = candidate
            
    with open("/tmp/_tmp_wikitext_prompts.json", "w") as f:
        json.dump(prompts, f)
        
    cache_sizes = [8, 24, 40]
    budgets = [0.25, 0.50, 0.75, 1.0]
    lookahead = 6
    
    results = []
    
    for c in cache_sizes:
        print(f"\n--- Cache Size {c} ---")
        
        # LRU Baseline
        tps, load = run_cmd("cached", c, policy="LRU")
        results.append(["LRU", c, "N/A", f"{tps:.2f}" if tps else "err", f"{load:.2f}" if load else "err"])
        
        # RANDOM Baseline
        tps, load = run_cmd("cached", c, policy="RANDOM")
        results.append(["RANDOM", c, "N/A", f"{tps:.2f}" if tps else "err", f"{load:.2f}" if load else "err"])
        
        # Predictor
        for b in budgets:
            tps, load = run_cmd("predict", c, policy="LRU", budget_frac=b, lookahead=lookahead)
            budget_val = max(1, int(c * b))
            results.append(["Predictor (f6)", c, budget_val, f"{tps:.2f}" if tps else "err", f"{load:.2f}" if load else "err"])
            
    with open("mini_sweep.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Cache_Size", "Prefetch_Budget", "TPS", "AvgLoadTime_ms"])
        writer.writerows(results)
        
    print("\nDone! Results saved to mini_sweep.csv")

if __name__ == "__main__":
    main()
