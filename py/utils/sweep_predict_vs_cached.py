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
        last_output_time = [time.time()]  # list for mutability in closures
        start_time = time.time()

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
                    last_output_time[0] = time.time()
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
    """Parse cache hits, misses, and hit rate from output. Also looks for Predictor Stats and Bandwidth."""
    cache_hits, cache_misses, cache_rate = None, None, None
    pred_hits, pred_total, pred_rate = None, None, None
    stall_loads, prefetch_loads, avg_load_ms = None, None, None
    
    match = re.search(r"Cache Stats: Hits=(\d+), Misses=(\d+), HitRate=([\d\.]+)%", output)
    if match:
        cache_hits, cache_misses, cache_rate = int(match.group(1)), int(match.group(2)), float(match.group(3))
        
    pred_match = re.search(r"Predictor Stats: Hits=(\d+), Total=(\d+), HitRate=([\d\.]+)%", output)
    if pred_match:
        pred_hits, pred_total, pred_rate = int(pred_match.group(1)), int(pred_match.group(2)), float(pred_match.group(3))

    bw_match = re.search(r"Bandwidth: StallLoads=(\d+),\s*PrefetchLoads=(\d+),\s*AvgLoadTime=([\d\.]+)ms", output)
    if bw_match:
        stall_loads, prefetch_loads, avg_load_ms = int(bw_match.group(1)), int(bw_match.group(2)), float(bw_match.group(3))
        
    return cache_hits, cache_misses, cache_rate, pred_hits, pred_total, pred_rate, stall_loads, prefetch_loads, avg_load_ms

def run_sweep():
    parser = argparse.ArgumentParser(description="Sweep 'cached' vs 'predict' backends with different lambda values.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="List of lambda values to test")
    parser.add_argument("--cache-size", type=int, nargs="+", default=[2], help="Cache sizes (key per layer) to use for both backends")
    parser.add_argument("--predictor-path", type=str, default="/home/michael/heteroPredict/trainingData/mixtral_8x7b/sweep_3_5", help="Path to predictor models base directory")
    parser.add_argument("--dataset", type=str, choices=["default", "fineweb", "orca", "wikitext", "txt"], default="default", help="Dataset to use")
    parser.add_argument("--predictor-device", type=str, default="gpu", choices=["gpu", "cpu", "auto"],
                        help="Device for predictor inference: 'gpu' (default), 'cpu' (NPU path on Strix), 'auto'")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens for generation mode")
    parser.add_argument("--text", type=str, default=None, help="Text for single prompt mode")
    parser.add_argument("--mode", type=str, choices=["perplexity", "generation", "both"], default="generation",
                        help="Mode: 'perplexity' for quality eval, 'generation' for speed eval, 'both' for combined")
    parser.add_argument("--prefetch-count", type=int, nargs="+", default=[1], help="Number of predicted experts to prefetched.")
    parser.add_argument("--expert-correlation-csv", type=str, default=None, help="Path to CSV containing layer correlation multipliers.")
    parser.add_argument("--plot-only", action="store_true", help="Just plot the results from sweep_predict_vs_cached_results.csv")
    parser.add_argument(
        "--backends", type=str, nargs="+", default=["cached", "predict"], choices=["cached", "predict"],
        help="Which backends to sweep (default: both)."
    )
    parser.add_argument(
        "--model", type=str, default="mixtral", choices=["mixtral", "qwen"],
        help="Model to sweep: 'mixtral' (default) or 'qwen'."
    )
    parser.add_argument(
        "--forced-top-n", type=int, default=0,
        help="(cached backend) Force the top-N unbiased experts into the cache mask every step. 0 = disabled."
    )

    parser.add_argument(
        "--predict-layer-subsets",
        type=str,
        nargs="+",
        default=["all"],
        help="List of layer subsets to sweep over. Use 'all' for all layers, or comma commands like '0,1,2' '0,2,4,6'. Default is ['all']."
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each subprocess call. Defaults to max(300, max_new_tokens * 20 + 180)."
    )
    
    args = parser.parse_args()

    # Compute timeout: default scales with max_new_tokens to avoid spurious timeouts
    if args.subprocess_timeout is None:
        args.subprocess_timeout = max(300, args.max_new_tokens * 20 + 180)

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
    print(f"Backends: {args.backends}")
    print(f"Cache Sizes: {args.cache_size}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Prefetch Counts: {args.prefetch_count}")
    print(f"Predictor Path: {args.predictor_path}")
    print(f"Subprocess Timeout: {args.subprocess_timeout}s")
    print(f"Layer Subsets: {args.predict_layer_subsets}")

    results = []
    script_path = os.path.join(
        os.path.dirname(__file__),
        "../unified_llm_w4a16/"
        + ("qwen3_30B-A3B_w4a16_model.py" if args.model == "qwen" else "mixtral_8x7B_w4a16_model.py")
    )

    # Define configurations
    configs = []
    if "cached" in args.backends:
        configs.append({"name": "Cached (LRU)", "backend": "cached", "predictor": ""})
    if "predict" in args.backends:
        configs.append({"name": "Predict (Spec)", "backend": "predict", "predictor": args.predictor_path})

    import tempfile
    temp_prompts_fd, temp_prompts_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(temp_prompts_fd, 'w') as f:
        json.dump(prompts, f)

    # Header
    print("-" * 165)
    if args.mode == "perplexity":
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'Layers':<15} | {'Gen PPL':<10} | {'Hit Rate':<10} | {'Pred Rate':<10} | {'Stalls':<8} | {'AvgLoad':<10}")
    elif args.mode == "generation":
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'Layers':<15} | {'TPS':<10} | {'Hit Rate':<10} | {'Pred Rate':<10} | {'Stalls':<8} | {'AvgLoad':<10}")
    else:
        print(f"{'Cache':<6} | {'Prefetch':<8} | {'Lambda':<8} | {'Backend':<16} | {'Layers':<15} | {'Gen PPL':<10} | {'TPS':<10} | {'Hit Rate':<10} | {'Pred Rate':<10} | {'Stalls':<8} | {'AvgLoad':<10}")
    print("-" * 165)

    for cache_size in args.cache_size:
        for prefetch_count in args.prefetch_count:
            for lambda_val in args.lambdas:
                for layer_subset_str in args.predict_layer_subsets:
                    for config in configs:
                        backend_name = config["backend"]
                        if backend_name == "cached" and prefetch_count != args.prefetch_count[0]:
                            continue
                        if backend_name == "cached" and layer_subset_str != args.predict_layer_subsets[0]:
                            # Cached backend doesn't care about predict layers, so don't re-run it
                            continue
                        if backend_name == "predict" and prefetch_count > cache_size:
                            continue
                        
                        predictor = config["predictor"]
                        display_name = config["name"]
                        
                        layer_subset_name = layer_subset_str
                        parsed_predict_layers = None
                        if layer_subset_str.lower() != "all":
                            parsed_predict_layers = [int(x.strip()) for x in layer_subset_str.split(",") if x.strip()]
                            layer_subset_name = layer_subset_str
            
                        print(f"\n{'='*25} Run: Cache={cache_size}, Prefetch={prefetch_count}, Lambda={lambda_val}, Layers={layer_subset_name}, Backend={display_name} {'='*25}")
        
                    # Aggregate stats across prompts
                    total_ppl = 0.0
                    valid_ppl_count = 0
                    total_tps = 0.0
                    valid_tps_count = 0
                    total_hits = 0
                    total_misses = 0
                    total_pred_hits = 0
                    total_pred_total = 0
                    total_stall_loads = 0
                    total_prefetch_loads = 0
                    sum_avg_load_ms = 0.0
                    valid_load_stats_count = 0
        
                    # --- PHASE 1: PERPLEXITY ---
                    if args.mode in ["perplexity", "both"]:
                        print(">> Running Perplexity Phase...")
                        cmd = [
                            sys.executable, script_path,
                            "--backend", backend_name,
                            "--device", "cuda",
                            "--lambda-val", str(lambda_val),
                            "--sweep-prompts-file", temp_prompts_path,
                            "--generation-perplexity",
                            "--generate",
                            "--max-new-tokens",
                            str(args.max_new_tokens),
                            "--temperature",
                            str(getattr(args, "temperature", 0.0)),
                            "--top-p",
                            str(getattr(args, "top_p", 0.9)),
                            "--top-k",
                            str(getattr(args, "top_k", 50)),
                        ]
                        
                        cmd.extend(["--max-cached-experts" if args.model == "qwen" else "--expert-cache", str(cache_size)])
                        cmd.extend(["--prefetch-experts-count", str(prefetch_count)])
                        if predictor:
                            cmd.extend(["--predictor-model", predictor])
                            if parsed_predict_layers is not None:
                                cmd.append("--predict-layers")
                                cmd.extend(map(str, parsed_predict_layers))
                        if args.forced_top_n > 0:
                            cmd.extend(["--forced-top-n", str(args.forced_top_n)])
                        
                        if args.expert_correlation_csv:
                            cmd.extend(["--expert-correlation-csv", args.expert_correlation_csv])
                        
                        output = run_subprocess(cmd, timeout=args.subprocess_timeout)
    
                        if output:
                            ppl_match = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
                            if ppl_match:
                                total_ppl = float(ppl_match.group(1))
                                valid_ppl_count = 1
                            hits, misses, _, phits, ptotal, _, stall, prefetch, avg_ms = parse_cache_stats(output)
                            if hits is not None:
                                total_hits += hits
                                total_misses += misses
                            if phits is not None:
                                total_pred_hits += phits
                                total_pred_total += ptotal
                            if stall is not None:
                                total_stall_loads += stall
                                total_prefetch_loads += prefetch
                                sum_avg_load_ms += avg_ms
                                valid_load_stats_count += 1

                    # --- PHASE 2: GENERATION (TPS) ---
                    if args.mode in ["generation", "both"]:
                        print(">> Running Generation Phase...")
                        cmd = [
                            sys.executable, script_path,
                            "--backend", backend_name,
                            "--device", "cuda",
                            "--lambda-val", str(lambda_val),
                            "--sweep-prompts-file", temp_prompts_path,
                            "--generate",
                            "--max-new-tokens", str(args.max_new_tokens)
                        ]
                        
                        cmd.extend(["--max-cached-experts" if args.model == "qwen" else "--expert-cache", str(cache_size)])
                        cmd.extend(["--prefetch-experts-count", str(prefetch_count)])
                        if predictor:
                            cmd.extend(["--predictor-model", predictor])
                            if parsed_predict_layers is not None:
                                cmd.append("--predict-layers")
                                cmd.extend(map(str, parsed_predict_layers))
                        if args.forced_top_n > 0:
                            cmd.extend(["--forced-top-n", str(args.forced_top_n)])
                        
                        if args.expert_correlation_csv:
                            cmd.extend(["--expert-correlation-csv", args.expert_correlation_csv])

                        output = run_subprocess(cmd, timeout=args.subprocess_timeout)
    
                        if output:
                            tps_match = re.search(r"End-to-End TPS:\s+([\d\.]+)", output)
                            if tps_match:
                                total_tps = float(tps_match.group(1))
                                valid_tps_count = 1
                            hits, misses, _, phits, ptotal, _, stall, prefetch, avg_ms = parse_cache_stats(output)
                            if hits is not None:
                                total_hits += hits
                                total_misses += misses
                            if phits is not None:
                                total_pred_hits += phits
                                total_pred_total += ptotal
                            if stall is not None:
                                total_stall_loads += stall
                                total_prefetch_loads += prefetch
                                sum_avg_load_ms += avg_ms
                                valid_load_stats_count += 1

                    # Average results
                    avg_ppl = total_ppl / valid_ppl_count if valid_ppl_count > 0 else None
                    avg_tps = total_tps / valid_tps_count if valid_tps_count > 0 else None
                    total_reqs = total_hits + total_misses
                    hit_rate = (total_hits / total_reqs * 100.0) if total_reqs > 0 else None
                    pred_rate = (total_pred_hits / total_pred_total * 100.0) if total_pred_total > 0 else None

                    avg_load_ms = sum_avg_load_ms / valid_load_stats_count if valid_load_stats_count > 0 else None

                    result = {
                        "cache_size": cache_size,
                        "prefetch_count": prefetch_count if backend_name == "predict" else None,
                        "layers": layer_subset_name if backend_name == "predict" else "all",
                        "lambda": lambda_val,
                        "backend": display_name,
                        "gen_perplexity": avg_ppl,
                        "hit_rate": hit_rate,
                        "tokens_per_second": avg_tps,
                        "pred_rate": pred_rate,
                        "stall_loads": total_stall_loads,
                        "prefetch_loads": total_prefetch_loads,
                        "avg_load_ms": avg_load_ms
                    }
                    results.append(result)

                    # Print result row
                    ppl_str = f"{avg_ppl:.4f}" if avg_ppl is not None else "N/A"
                    tps_str = f"{avg_tps:.4f}" if avg_tps is not None else "N/A"
                    rate_str = f"{hit_rate:.2f}%" if hit_rate is not None else "N/A"
                    prate_str = f"{pred_rate:.2f}%" if pred_rate is not None else "N/A"
                    stall_str = str(total_stall_loads) if total_stall_loads is not None else "N/A"
                    avg_ld_str = f"{avg_load_ms:.2f}ms" if avg_load_ms is not None else "0.00ms"
                    
                    print("-" * 165)
                    if args.mode == "perplexity":
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {layer_subset_name:<15} | {ppl_str:<10} | {rate_str:<10} | {prate_str:<10} | {stall_str:<8} | {avg_ld_str:<10}")
                    elif args.mode == "generation":
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {layer_subset_name:<15} | {tps_str:<10} | {rate_str:<10} | {prate_str:<10} | {stall_str:<8} | {avg_ld_str:<10}")
                    else:
                        print(f"{cache_size:<6} | {prefetch_count:<8} | {lambda_val:<8} | {display_name:<16} | {layer_subset_name:<15} | {ppl_str:<10} | {tps_str:<10} | {rate_str:<10} | {prate_str:<10} | {stall_str:<8} | {avg_ld_str:<10}")

                    sys.stdout.flush()

    print("-" * 165)

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
            l_val = row.get('layers', 'all')
            lbl += f" (C={int(row['cache_size'])}, P={p_val}, L={l_val})"
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
