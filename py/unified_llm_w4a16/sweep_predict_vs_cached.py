#!/usr/bin/env python3
import subprocess
import re
import sys
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json

import threading
import psutil

def run_subprocess(cmd, timeout=300):
    """Run a command and return its output string, printing in real-time. Includes timeout."""
    try:
        env = os.environ.copy()
        # Disabling SDMA can fix random ROCm driver hangs on back-to-back runs
        env["HSA_ENABLE_SDMA"] = "0"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid  # Put process in its own group to kill reliably
        )
        
        output_lines = []
        timeout_reached = False
        
        def kill_process():
            nonlocal timeout_reached
            timeout_reached = True
            try:
                import signal
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception as e:
                pass
                
        timer = threading.Timer(timeout, kill_process)
        timer.start()
        
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end="")
                    sys.stdout.flush()
                    output_lines.append(line)
        finally:
            timer.cancel()
            
        if timeout_reached:
            print(f"Command timed out after {timeout} seconds")
            return None
            
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            return None
            
        # Add a brief sleep to let ROCm gracefully clean up IPC buffers
        time.sleep(2.0)
        
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
    parser.add_argument("--cache-size", type=int, nargs="+", default=[2], help="Cache sizes (key per layer) to use for both backends")
    parser.add_argument("--predictor-path", type=str, default="/home/michael/mixtral_project/expert_prediction_full/embedding_only_predictors", help="Path to predictor models base directory")
    parser.add_argument("--dataset", type=str, choices=["default", "fineweb", "orca", "wikitext", "txt"], default="default", help="Dataset to use")
    parser.add_argument("--predictor-device", type=str, default="gpu", choices=["gpu", "cpu", "auto"],
                        help="Device for predictor inference: 'gpu' (default), 'cpu' (NPU path on Strix), 'auto'")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens for generation mode")
    parser.add_argument("--text", type=str, default=None, help="Text for single prompt mode")
    parser.add_argument("--mode", type=str, choices=["perplexity", "generation", "both"], default="generation",
                        help="Mode: 'perplexity' for quality eval, 'generation' for speed eval, 'both' for combined")
    parser.add_argument("--prefetch-count", type=int, nargs="+", default=[1], help="Number of predicted experts to prefetched.")
    parser.add_argument("--expert-correlation-csv", type=str, default=None, help="Path to CSV containing layer correlation multipliers.")
    parser.add_argument("--plot-only", action="store_true", help="Just plot the results from sweep_predict_vs_cached_results.csv")

    parser.add_argument(
        "--predict-layers",
        type=int,
        nargs="+",
        default=None,
        help="List of layer indices to enable the predictor (e.g. 0 1 2 3). If omitted, predicts all layers."
    )
    
    args = parser.parse_args()

    if args.plot_only:
        csv_file = "sweep_predict_vs_cached_results.csv"
        if not os.path.exists(csv_file):
            print(f"Error: {csv_file} does not exist.")
            return
        df = pd.read_csv(csv_file)
        generate_plots(df, args.mode)
        return

    # Load prompts
    prompts = []
    if args.dataset in ["fineweb", "orca", "wikitext"]:
        try:
            from datasets import load_dataset
            print(f"Loading {args.dataset} dataset...")
            if args.dataset == "fineweb":
                dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            elif args.dataset == "orca":
                dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            elif args.dataset == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
                
            count = 0
            for item in dataset:
                if args.dataset == "fineweb":
                    text = item.get('text', "")
                elif args.dataset == "orca":
                    system = item.get("system_prompt", "")
                    question = item.get("question", "")
                    response = item.get("response", "")
                    text = f"{system}\n{question}\n{response}".strip()
                elif args.dataset == "wikitext":
                    text = item.get('text', "")
                    
                if len(text) > 100: 
                     prompts.append(text[:300]) 
                     count += 1
                     if count >= args.num_prompts:
                         break
        except Exception as e:
            print(f"Error loading dataset: {e}. Install datasets with pip install datasets")
            sys.exit(1)
    elif args.dataset == "txt":
        with open("prompts.txt", "r") as f:
            content = f.read()
            prompts = [p.strip() for p in content.split("\n\n") if p.strip()]
        args.num_prompts = len(prompts)
    else:
        default_text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        prompts = [args.text if args.text else default_text] * args.num_prompts

    print(f"Sweeping {len(prompts)} prompt(s).")
    print(f"Mode: {args.mode}")
    print(f"Backends: ['cached', 'predict']")
    print(f"Cache Sizes: {args.cache_size}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Prefetch Counts: {args.prefetch_count}")
    print(f"Predictor Path: {args.predictor_path}")

    results = []
    script_path = os.path.join(os.path.dirname(__file__), "mixtral_8x7B_w4a16_model.py")

    # Define configurations
    configs = [
        {"name": "Cached (LRU)", "backend": "cached", "predictor": ""},
        {"name": "Predict (Spec)", "backend": "predict", "predictor": args.predictor_path}
    ]

    import tempfile
    temp_prompts_fd, temp_prompts_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(temp_prompts_fd, 'w') as f:
        json.dump(prompts, f)

    # Header
    print("-" * 120)
    if args.mode == "perplexity":
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'Gen PPL':<10} | {'Hit Rate':<10}")
    elif args.mode == "generation":
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'TPS':<10} | {'Hit Rate':<10}")
    else:
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'Gen PPL':<10} | {'TPS':<10} | {'Hit Rate':<10}")
    print("-" * 120)

    for cache_size in args.cache_size:
        for prefetch_count in args.prefetch_count:
            for lambda_val in args.lambdas:
                for config in configs:
                    backend_name = config["backend"]
                    if backend_name == "cached" and prefetch_count != args.prefetch_count[0]:
                        continue
                    if backend_name == "predict" and prefetch_count > cache_size:
                        continue
                    
                    predictor = config["predictor"]
                    display_name = config["name"]
        
                    print(f"\n{'='*30} Run: Cache={cache_size}, Prefetch={prefetch_count}, Lambda={lambda_val}, Backend={display_name} {'='*30}")
        
                    # Aggregate stats across prompts
                    total_ppl = 0.0
                    valid_ppl_count = 0
                    total_tps = 0.0
                    valid_tps_count = 0
                    total_hits = 0
                    total_misses = 0
        
                    # --- PHASE 1: PERPLEXITY ---
                    if args.mode in ["perplexity", "both"]:
                        print(">> Running Perplexity Phase...")
                        cmd = [
                            sys.executable, script_path,
                            "--backend", backend_name,
                            "--device", "cuda",
                            "--expert-cache", str(cache_size),
                            "--lambda-val", str(lambda_val),
                            "--sweep-prompts-file", temp_prompts_path,
                            "--generation-perplexity",
                            "--no-generate"
                        ]
                        if predictor:
                            cmd.extend(["--predictor-model", predictor])
                        if backend_name == "predict":
                            cmd.extend([
                                "--predictor-device", args.predictor_device,
                                "--prefetch-experts-count", str(prefetch_count)
                            ])
                            if args.predict_layers is not None:
                                cmd.append("--predict-layers")
                                cmd.extend(map(str, args.predict_layers))
                        
                        if args.expert_correlation_csv:
                            cmd.extend(["--expert-correlation-csv", args.expert_correlation_csv])
                        
                        output = run_subprocess(cmd)
    
                        if output:
                            ppl_match = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
                            if ppl_match:
                                total_ppl = float(ppl_match.group(1))
                                valid_ppl_count = 1
                            hits, misses, _ = parse_cache_stats(output)
                            if hits is not None:
                                total_hits += hits
                                total_misses += misses

                    # --- PHASE 2: GENERATION (TPS) ---
                    if args.mode in ["generation", "both"]:
                        print(">> Running Generation Phase...")
                        cmd = [
                            sys.executable, script_path,
                            "--backend", backend_name,
                            "--device", "cuda",
                            "--expert-cache", str(cache_size),
                            "--lambda-val", str(lambda_val),
                            "--sweep-prompts-file", temp_prompts_path,
                            "--generate",
                            "--max-new-tokens", str(args.max_new_tokens)
                        ]
                        if predictor:
                            cmd.extend(["--predictor-model", predictor])
                        if backend_name == "predict":
                            cmd.extend([
                                "--predictor-device", args.predictor_device,
                                "--prefetch-experts-count", str(prefetch_count)
                            ])
                            if args.predict_layers is not None:
                                cmd.append("--predict-layers")
                                cmd.extend(map(str, args.predict_layers))
                        
                        if args.expert_correlation_csv:
                            cmd.extend(["--expert-correlation-csv", args.expert_correlation_csv])

                        output = run_subprocess(cmd)
    
                        if output:
                            tps_match = re.search(r"End-to-End TPS:\s+([\d\.]+)", output)
                            if tps_match:
                                total_tps = float(tps_match.group(1))
                                valid_tps_count = 1
                            hits, misses, _ = parse_cache_stats(output)
                            if hits is not None:
                                total_hits += hits
                                total_misses += misses

                    # Average results
                    avg_ppl = total_ppl / valid_ppl_count if valid_ppl_count > 0 else None
                    avg_tps = total_tps / valid_tps_count if valid_tps_count > 0 else None
                    total_reqs = total_hits + total_misses
                    hit_rate = (total_hits / total_reqs * 100.0) if total_reqs > 0 else None

                    result = {
                        "cache_size": cache_size,
                        "prefetch_count": prefetch_count if backend_name == "predict" else None,
                        "lambda": lambda_val,
                        "backend": display_name,
                        "gen_perplexity": avg_ppl,
                        "hit_rate": hit_rate,
                        "tokens_per_second": avg_tps
                    }
                    results.append(result)

                    # Print result row
                    ppl_str = f"{avg_ppl:.4f}" if avg_ppl is not None else "N/A"
                    tps_str = f"{avg_tps:.4f}" if avg_tps is not None else "N/A"
                    rate_str = f"{hit_rate:.2f}%" if hit_rate is not None else "N/A"
                    print("-" * 120)
                    if args.mode == "perplexity":
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {ppl_str:<10} | {rate_str:<10}")
                    elif args.mode == "generation":
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {tps_str:<10} | {rate_str:<10}")
                    else:
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {ppl_str:<10} | {tps_str:<10} | {rate_str:<10}")

                    sys.stdout.flush()

    print("-" * 120)

    try:
        os.remove(temp_prompts_path)
    except:
        pass

    if not results:
        print("No results collected.")
        return

    # Save CSV
    df = pd.DataFrame(results)
    csv_file = "sweep_predict_vs_cached_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

    # Plotting
    try:
        generate_plots(df, args.mode)
    except Exception as e:
        print(f"Error generating plots: {e}")

def generate_plots(df, mode):
    # Single timestamp for all plots from this run
    ts = time.strftime("%Y%m%d_%H%M%S")

    import textwrap
    cmd_args_str = " ".join(sys.argv)
    wrapped_args = textwrap.fill(f"Cmd: {cmd_args_str}", width=110)
    num_lines = wrapped_args.count('\n') + 1
    bottom_margin = 0.03 + 0.025 * num_lines

    if df.empty:
        print("No data to plot")
        return

    def make_label(row):
        lbl = str(row['backend'])
        if "Predict" in lbl:
            p_val = "N/A" if pd.isna(row.get('prefetch_count')) else int(row['prefetch_count'])
            lbl += f" (C={int(row['cache_size'])}, P={p_val})"
        else:
            lbl += f" (C={int(row['cache_size'])})"
        return lbl

    df['plot_label'] = df.apply(make_label, axis=1)
    plot_labels = sorted(df['plot_label'].unique())

    # 1. Lambda vs Tokens per Second
    if mode in ["generation", "both"] and "tokens_per_second" in df.columns:
        if not df["tokens_per_second"].isnull().all():
            plt.figure(figsize=(10, 6))
            for lbl in plot_labels:
                subset = df[df['plot_label'] == lbl].sort_values('lambda')
                m = 's' if "Predict" in lbl else 'o'
                plt.plot(subset['lambda'], subset['tokens_per_second'], marker=m, label=lbl)
            plt.xlabel("Lambda")
            plt.ylabel("Tokens per Second")
            plt.title("Cached vs Predict: Tokens per Second")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.figtext(0.5, 0.02, wrapped_args, ha="center", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
            plt.tight_layout(rect=[0, bottom_margin, 1, 1])
            fname = f"predict_vs_cached_tps_strix_{ts}.png"
            plt.savefig(fname)
            print(f"Saved {fname}")

    # 2. Lambda vs Generation Perplexity
    if mode in ["perplexity", "both"] and "gen_perplexity" in df.columns:
        if not df["gen_perplexity"].isnull().all():
            plt.figure(figsize=(10, 6))
            for lbl in plot_labels:
                subset = df[df['plot_label'] == lbl].sort_values('lambda')
                m = 's' if "Predict" in lbl else 'o'
                plt.plot(subset['lambda'], subset['gen_perplexity'], marker=m, label=lbl)
            plt.xlabel("Lambda")
            plt.ylabel("Generation Perplexity")
            plt.title("Cached vs Predict: Generation Perplexity")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.figtext(0.5, 0.02, wrapped_args, ha="center", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
            plt.tight_layout(rect=[0, bottom_margin, 1, 1])
            fname = f"predict_vs_cached_ppl_strix_{ts}.png"
            plt.savefig(fname)
            print(f"Saved {fname}")

    # 3. Lambda vs Hit Rate
    if "hit_rate" in df.columns and not df["hit_rate"].isnull().all():
        plt.figure(figsize=(10, 6))
        for lbl in plot_labels:
            subset = df[df['plot_label'] == lbl].sort_values('lambda')
            m = 's' if "Predict" in lbl else 'o'
            plt.plot(subset['lambda'], subset['hit_rate'], marker=m, label=lbl)
        plt.xlabel("Lambda")
        plt.ylabel("Cache Hit Rate (%)")
        plt.title("Cached vs Predict: Cache Hit Rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 100)
        plt.figtext(0.5, 0.02, wrapped_args, ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])
        fname = f"predict_vs_cached_hitrate_strix_{ts}.png"
        plt.savefig(fname)
        print(f"Saved {fname}")

    # 4. Bar Chart side-by-side for Tokens per Second
    if mode in ["generation", "both"] and "tokens_per_second" in df.columns:
        if not df["tokens_per_second"].isnull().all():
            try:
                pivot_df = df.pivot_table(index='lambda', columns='plot_label', values='tokens_per_second', aggfunc='mean')
                ax = pivot_df.plot(kind='bar', figsize=(12, 6))
                plt.xlabel("Lambda")
                plt.ylabel("Tokens per Second")
                plt.title("Cached vs Predict: Tokens per Second (Bar Chart)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=0)
                plt.figtext(0.5, 0.02, wrapped_args, ha="center", va="bottom", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
                plt.tight_layout(rect=[0, bottom_margin, 1, 1])
                fname = f"predict_vs_cached_tps_bar_strix_{ts}.png"
                plt.savefig(fname)
                print(f"Saved {fname}")
            except Exception as e:
                print(f"Error generating bar chart: {e}")

if __name__ == "__main__":
    run_sweep()
