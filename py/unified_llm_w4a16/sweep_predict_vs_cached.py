#!/usr/bin/env python3
import subprocess
import re
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def run_subprocess(cmd):
    """Run a command and return its output string, printing in real-time."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="")
                output_lines.append(line)
        
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            return None
        
        return "".join(output_lines)
    except Exception as e:
        print(f"Exception running command: {e}")
        return None

def parse_cache_stats(output):
    """Parse cache hits, misses, and hit rate from output."""
    match = re.search(r"Cache Stats: Hits=(\d+), Misses=(\d+), HitRate=([\d\.]+)%", output)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None, None, None

def run_sweep():
    parser = argparse.ArgumentParser(description="Sweep 'cached' vs 'predict' backends with different lambda values.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="List of lambda values to test")
    parser.add_argument("--cache-size", type=int, default=2, help="Cache size (experts per layer) to use for both backends")
    parser.add_argument("--predictor-path", type=str, required=True, help="Path to predictor models base directory")
    parser.add_argument("--dataset", type=str, choices=["default", "fineweb"], default="default", help="Dataset to use")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens for generation mode")
    parser.add_argument("--text", type=str, default=None, help="Text for single prompt mode")
    
    args = parser.parse_args()

    # Load prompts
    prompts = []
    if args.dataset == "fineweb":
        try:
            from datasets import load_dataset
            print("Loading FineWeb dataset...")
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            count = 0
            for item in dataset:
                text = item['text']
                if len(text) > 100: 
                     prompts.append(text[:300]) 
                     count += 1
                     if count >= args.num_prompts:
                         break
        except Exception as e:
            print(f"Error loading dataset: {e}. Install datasets with pip install datasets")
            sys.exit(1)
    else:
        default_text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        prompts = [args.text if args.text else default_text] * args.num_prompts

    print(f"Sweeping {len(prompts)} prompt(s).")
    print(f"Backends: ['cached', 'predict']")
    print(f"Cache Size: {args.cache_size}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Predictor Path: {args.predictor_path}")
    
    results = [] 
    script_path = os.path.join(os.path.dirname(__file__), "mixtral_8x7B_w4a16_model.py")
    
    # Define configurations
    configs = [
        {"name": "Cached (LRU)", "backend": "cached", "predictor": ""},
        {"name": "Predict (Spec)", "backend": "predict", "predictor": args.predictor_path}
    ]

    for lambda_val in args.lambdas:
        for config in configs:
            backend_name = config["backend"]
            predictor = config["predictor"]
            display_name = config["name"]
            
            print(f"\n{'='*40} Run: Lambda={lambda_val}, Backend={display_name} {'='*40}")
            
            # Aggregate stats across prompts
            total_ppl = 0.0
            valid_ppl_count = 0
            total_hits = 0
            total_misses = 0
            
            for i, prompt in enumerate(prompts):
                print(f">> Prompt {i+1}/{len(prompts)}...")
                
                cmd = [
                    sys.executable, script_path, 
                    "--backend", backend_name, 
                    "--device", "cuda",
                    "--expert-cache", str(args.cache_size), 
                    "--lambda-val", str(lambda_val), 
                    "--text", prompt,
                    "--generation-perplexity"
                ]
                
                if predictor:
                    cmd.extend(["--predictor-model", predictor])
                
                output = run_subprocess(cmd)
                
                if output:
                    # Parse PPL
                    ppl_match = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
                    if ppl_match:
                        total_ppl += float(ppl_match.group(1))
                        valid_ppl_count += 1
                    
                    # Parse Cache Stats
                    hits, misses, _ = parse_cache_stats(output)
                    if hits is not None:
                        total_hits += hits
                        total_misses += misses
            
            # Average results
            avg_ppl = total_ppl / valid_ppl_count if valid_ppl_count > 0 else None
            total_reqs = total_hits + total_misses
            hit_rate = (total_hits / total_reqs * 100.0) if total_reqs > 0 else None
            
            result = {
                "lambda": lambda_val,
                "backend": display_name,
                "gen_perplexity": avg_ppl,
                "hit_rate": hit_rate,
                "cache_size": args.cache_size
            }
            results.append(result)
            
            print(f"Result: PPL={avg_ppl}, HitRate={hit_rate}%")

    # Save Results
    df = pd.DataFrame(results)
    csv_file = "sweep_predict_vs_cached_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
    
    # Plotting
    try:
        generate_plots(df)
    except Exception as e:
        print(f"Error generating plots: {e}")

def generate_plots(df):
    if df.empty:
        print("No data to plot")
        return
        
    backends = df['backend'].unique()
    
    # 1. Lambda vs Perplexity
    plt.figure(figsize=(10, 6))
    for backend in backends:
        subset = df[df['backend'] == backend].sort_values('lambda')
        plt.plot(subset['lambda'], subset['gen_perplexity'], marker='o', label=backend)
        
    plt.xlabel("Lambda")
    plt.ylabel("Generation Perplexity")
    plt.title("Cached vs Predict: Perplexity")
    plt.legend()
    plt.grid(True)
    plt.savefig("predict_vs_cached_ppl.png")
    print("Saved predict_vs_cached_ppl.png")
    
    # 2. Lambda vs Hit Rate
    plt.figure(figsize=(10, 6))
    for backend in backends:
        subset = df[df['backend'] == backend].sort_values('lambda')
        plt.plot(subset['lambda'], subset['hit_rate'], marker='s', label=backend)
        
    plt.xlabel("Lambda")
    plt.ylabel("Cache Hit Rate (%)")
    plt.title("Cached vs Predict: Hit Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("predict_vs_cached_hitrate.png")
    print("Saved predict_vs_cached_hitrate.png")

if __name__ == "__main__":
    run_sweep()
