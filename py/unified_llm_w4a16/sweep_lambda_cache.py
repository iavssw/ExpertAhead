#!/usr/bin/env python3
import subprocess
import re
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Try importing datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

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
    parser = argparse.ArgumentParser(description="Sweep lambda and cache size for Generation Perplexity/TPS.")
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=[2,3,4,5,6,7,8], help="List of cache sizes to test")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="List of lambda values to test")
    parser.add_argument("--mode", type=str, choices=["perplexity", "generation", "both"], default="both",
                        help="Mode: 'perplexity' for quality eval, 'generation' for speed eval, 'both' for combined")
    parser.add_argument("--text", type=str, default=None, help="Text for single prompt mode")
    parser.add_argument("--dataset", type=str, choices=["default", "fineweb"], default="default", help="Dataset to use")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to evaluate from dataset")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens for generation mode")
    parser.add_argument("--backend", type=str, default="cached", choices=["cached", "predict", "base"],
                        help="Backend to use (default: cached)")
    parser.add_argument("--predictor-model", type=str, default="", help="Path to predictor model for 'predict' backend")

    args = parser.parse_args()

    # Load prompts
    prompts = []
    if args.dataset == "fineweb":
        if not HAS_DATASETS:
            print("Error: 'datasets' library needed for fineweb. Install with `pip install datasets`.")
            sys.exit(1)
        print("Loading FineWeb dataset...")
        try:
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            count = 0
            for item in dataset:
                text = item['text']
                if len(text) > 100: 
                     prompts.append(text[:300]) # 300 chars ~ 60-80 tokens
                     count += 1
                     if count >= args.num_prompts:
                         break
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
    else:
        default_text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        prompts = [args.text if args.text else default_text]

    print(f"Sweeping {len(prompts)} prompt(s).")
    print(f"Mode: {args.mode}")
    print(f"Cache Sizes: {args.cache_sizes}")
    print(f"Lambdas: {args.lambdas}")
    
    results = [] # List of dicts
    script_path = os.path.join(os.path.dirname(__file__), "mixtral_8x7B_w4a16_model.py")
    
    # Header
    print("-" * 120)
    if args.mode == "perplexity":
        print(f"{'Cache':<6} | {'Lambda':<6} | {'Prompt':<6} | {'Gen PPL':<10} | {'Hit Rate':<10}")
    elif args.mode == "generation":
        print(f"{'Cache':<6} | {'Lambda':<6} | {'Prompt':<6} | {'TPS':<10} | {'Hit Rate':<10}")
    else:
        print(f"{'Cache':<6} | {'Lambda':<6} | {'Prompt':<6} | {'Gen PPL':<10} | {'PPL Hit%':<10} | {'TPS':<10} | {'Gen Hit%':<10}")
    print("-" * 120)

    for prompt_idx, prompt in enumerate(prompts):
        for cache_size in args.cache_sizes:
            lambdas_to_run = [args.lambdas[0]] if cache_size == 8 else args.lambdas
            
            for lambda_val in lambdas_to_run:
                print(f"\n{'='*40} Run: Cache={cache_size}, Lambda={lambda_val}, Prompt={prompt_idx} {'='*40}")
                
                result = {
                    "cache_size": cache_size, "lambda": lambda_val, "prompt_idx": prompt_idx, "mode": args.mode,
                    "gen_perplexity": None, "tps": None,
                    "ppl_hit_rate": None, "gen_hit_rate": None
                }
                
                # --- PHASE 1: PERPLEXITY ---
                if args.mode in ["perplexity", "both"]:
                    print(">> Running Perplexity Phase...")
                    cmd = [
                        sys.executable, script_path, "--backend", args.backend, "--device", "cuda",
                        "--expert-cache", str(cache_size), "--lambda-val", str(lambda_val), "--text", prompt,
                        "--generation-perplexity"
                    ]
                    if args.backend == "predict" and args.predictor_model:
                        cmd.extend(["--predictor-model", args.predictor_model])
                    output = run_subprocess(cmd)
                    
                    if output:
                        # Parse PPL
                        ppl_match = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
                        if ppl_match:
                            result["gen_perplexity"] = float(ppl_match.group(1))
                        
                        # Parse Cache Stats
                        hits, misses, rate = parse_cache_stats(output)
                        result["ppl_hit_rate"] = rate
                        result["ppl_hits"] = hits
                        result["ppl_misses"] = misses

                # --- PHASE 2: GENERATION ---
                if args.mode in ["generation", "both"]:
                    print(">> Running Generation Phase...")
                    cmd = [
                        sys.executable, script_path, "--backend", args.backend, "--device", "cuda",
                        "--expert-cache", str(cache_size), "--lambda-val", str(lambda_val), "--text", prompt,
                        "--generate", "--max-new-tokens", str(args.max_new_tokens)
                    ]
                    if args.backend == "predict" and args.predictor_model:
                        cmd.extend(["--predictor-model", args.predictor_model])
                    output = run_subprocess(cmd)

                    if output:
                        # Parse TPS
                        tps_match = re.search(r"Average Time per Token:\s+([\d\.]+)\s+seconds", output)
                        if tps_match:
                            time_per_token = float(tps_match.group(1))
                            result["tps"] = 1.0 / time_per_token if time_per_token > 0 else 0.0

                        # Parse Cache Stats
                        hits, misses, rate = parse_cache_stats(output)
                        result["gen_hit_rate"] = rate
                        result["gen_hits"] = hits
                        result["gen_misses"] = misses

                # --- Print Result Row ---
                cache_str = str(cache_size)
                lambda_str = str(lambda_val)
                prompt_str = str(prompt_idx)
                
                ppl_str = f"{result['gen_perplexity']:.4f}" if result['gen_perplexity'] is not None else "N/A"
                tps_str = f"{result['tps']:.4f}" if result['tps'] is not None else "N/A"
                
                ppl_rate_str = f"{result['ppl_hit_rate']:.2f}%" if result['ppl_hit_rate'] is not None else "N/A"
                gen_rate_str = f"{result['gen_hit_rate']:.2f}%" if result['gen_hit_rate'] is not None else "N/A"

                print("-" * 120)
                if args.mode == "perplexity":
                    print(f"{cache_str:<6} | {lambda_str:<6} | {prompt_str:<6} | {ppl_str:<10} | {ppl_rate_str:<10}")
                elif args.mode == "generation":
                    print(f"{cache_str:<6} | {lambda_str:<6} | {prompt_str:<6} | {tps_str:<10} | {gen_rate_str:<10}")
                else:
                    print(f"{cache_str:<6} | {lambda_str:<6} | {prompt_str:<6} | {ppl_str:<10} | {ppl_rate_str:<10} | {tps_str:<10} | {gen_rate_str:<10}")
                
                results.append(result)
                sys.stdout.flush()

    print("-" * 120)
    
    if not results:
        print("No results collected.")
        return

    # Save CSV
    df = pd.DataFrame(results)
    csv_file = f"sweep_results_{args.mode}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
    
    # Generate Visualizations
    try:
        generate_plots(df, args.mode)
    except Exception as e:
        print(f"Error generating plots: {e}")

def generate_plots(df, mode):
    # Determine numeric columns based on mode
    numeric_cols = []
    if "gen_perplexity" in df.columns: numeric_cols.append("gen_perplexity")
    if "tps" in df.columns: numeric_cols.append("tps")
    if "ppl_hit_rate" in df.columns: numeric_cols.append("ppl_hit_rate")
    if "gen_hit_rate" in df.columns: numeric_cols.append("gen_hit_rate")
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_cols, how='all')
    if df.empty:
        print("No valid data for plotting.")
        return

    grouped = df.groupby(['cache_size', 'lambda']).mean(numeric_only=True).reset_index()
    cache_sizes = grouped['cache_size'].unique()
    
    # 1. Lambda vs TPS
    if "tps" in grouped.columns and not grouped["tps"].isnull().all():
        plt.figure(figsize=(10, 6))
        for size in cache_sizes:
            subset = grouped[grouped['cache_size'] == size]
            if not subset.empty:
                plt.plot(subset['lambda'], subset['tps'], marker='o', label=f"Cache Size {size}")
        plt.xlabel("Lambda")
        plt.ylabel("Tokens Per Second (TPS)")
        plt.title("Impact of Lambda on Generation Speed")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"lambda_vs_tps_{mode}.png")
        print(f"Saved lambda_vs_tps_{mode}.png")

    # 2. Lambda vs Perplexity
    if "gen_perplexity" in grouped.columns and not grouped["gen_perplexity"].isnull().all():
        plt.figure(figsize=(10, 6))
        for size in cache_sizes:
            subset = grouped[grouped['cache_size'] == size]
            if not subset.empty:
                plt.plot(subset['lambda'], subset['gen_perplexity'], marker='s', label=f"Cache Size {size}")
        plt.xlabel("Lambda")
        plt.ylabel("Generation Perplexity")
        plt.title("Impact of Lambda on Generation Perplexity")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"lambda_vs_gen_perplexity_{mode}.png")
        print(f"Saved lambda_vs_gen_perplexity_{mode}.png")

    # 3. Lambda vs Hit Rate
    # Plot both PPL and Gen hit rates if available
    hit_rate_cols = []
    if "ppl_hit_rate" in grouped.columns and not grouped["ppl_hit_rate"].isnull().all():
        hit_rate_cols.append(("ppl_hit_rate", "PPL Hit Rate", "s"))
    if "gen_hit_rate" in grouped.columns and not grouped["gen_hit_rate"].isnull().all():
        hit_rate_cols.append(("gen_hit_rate", "Gen Hit Rate", "^"))

    if hit_rate_cols:
        plt.figure(figsize=(10, 6))
        for col, label_suffix, marker in hit_rate_cols:
            for size in cache_sizes:
                subset = grouped[grouped['cache_size'] == size]
                if not subset.empty:
                    label = f"Cache {size} ({label_suffix})"
                    plt.plot(subset['lambda'], subset[col], marker=marker, label=label)
        
        plt.xlabel("Lambda")
        plt.ylabel("Cache Hit Rate (%)")
        plt.title("Impact of Lambda on Cache Hit Rate")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        plt.savefig(f"lambda_vs_hit_rate_{mode}.png")
        print(f"Saved lambda_vs_hit_rate_{mode}.png")

if __name__ == "__main__":
    run_sweep()
