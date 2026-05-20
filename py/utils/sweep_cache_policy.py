#!/usr/bin/env python3
"""
sweep_cache_policy.py — benchmark different cache eviction policies

Replaces LRU with multiple cache eviction methods (LRU, MRU, LFU, MFU, CLOCK, RANDOM, LFRU)
across multiple cache-sizes and lambda values using the unified_llm_w4a16 model.

Typical usage examples:
python sweep_cache_policy.py --policies LRU LFU CLOCK --cache-sizes 2 4 --lambdas 0.0 1.0
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def run_subprocess(cmd, timeout=600):
    try:
        env = os.environ.copy()
        env["HSA_ENABLE_SDMA"] = "0"
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, preexec_fn=os.setsid,
        )
        output_lines = []
        timeout_reached = False

        def _kill():
            nonlocal timeout_reached
            timeout_reached = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass

        timer = threading.Timer(timeout, _kill)
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
            print(f"[sweep] Command timed out after {timeout}s")
            return None

        if process.returncode != 0:
            print(f"[sweep] Command failed (return code {process.returncode})")
            return None

        time.sleep(1.0)
        return "".join(output_lines)
    except Exception as exc:
        print(f"[sweep] Exception running command: {exc}")
        return None

def parse_output(output):
    result = {}
    m = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
    if m: result["gen_ppl"] = float(m.group(1))

    m = re.search(r"Cache Stats: Hits=(\d+), Misses=(\d+), HitRate=([\d\.]+)%", output)
    if m:
        result["cache_hits"] = int(m.group(1))
        result["cache_misses"] = int(m.group(2))
        result["cache_hit_rate"] = float(m.group(3))

    m = re.search(r"End-to-End TPS:\s+([\d\.]+)", output)
    if m:
        result["tps"] = float(m.group(1))
    else:
        m2 = re.search(r"Average Time per Token:\s+([\d\.]+)\s+seconds", output)
        if m2:
            tpt = float(m2.group(1))
            result["tps"] = 1.0 / tpt if tpt > 0 else 0.0

    return result

def load_prompts(args):
    dataset_name = args.dataset
    num = args.num_prompts

    if dataset_name in ("fineweb", "orca", "wikitext"):
        from datasets import load_dataset
        print(f"[sweep] Loading {dataset_name} dataset …")
        if dataset_name == "fineweb":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        elif dataset_name == "orca":
            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        elif dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

        prompts = []
        for item in ds:
            text = ""
            if dataset_name == "fineweb": text = item.get("text", "")
            elif dataset_name == "orca": text = item.get("question", "")
            elif dataset_name == "wikitext": text = item.get("text", "")
            if len(text.strip()) > 100:
                prompts.append(text.strip()[:600])
                if len(prompts) >= num: break
        return prompts

    return ["In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."] * num

def build_cmd(args, model_script, policy, cache_size, lambda_val, temp_prompts_path, phase):
    cmd = [
        sys.executable, model_script,
        "--backend", args.backend,
        "--device", "cuda",
        "--cache-policy", policy,
        "--expert-cache", str(cache_size),
        "--lambda-val", str(lambda_val),
        "--sweep-prompts-file", temp_prompts_path,
    ]
    if getattr(args, "prefill_top_n", 0) > 0:
        cmd.extend(["--prefill-top-n", str(args.prefill_top_n)])

    if phase == "perplexity":
        cmd += [
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
    else:
        cmd += ["--generate", "--max-new-tokens", str(args.max_new_tokens)]
    return cmd

def run_sweep(args, model_script, prompts):
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(prompts, f)

    timeout = args.subprocess_timeout or max(300, args.max_new_tokens * 20 + 180)
    results = []

    try:
        for policy in args.policies:
            for cache_size in sorted(args.cache_sizes):
                for lambda_val in sorted(args.lambdas):
                    print(f"\n{'='*25} policy={policy} cache={cache_size} λ={lambda_val} {'='*25}")
                    row = {
                        "policy": policy, "cache_size": cache_size, "lambda": lambda_val,
                        "gen_ppl": None, "tps": None, "cache_hit_rate": None
                    }

                    if args.mode in ("perplexity", "both"):
                        cmd = build_cmd(args, model_script, policy, cache_size, lambda_val, temp_path, "perplexity")
                        out = run_subprocess(cmd, timeout=timeout)
                        if out:
                            parsed = parse_output(out)
                            row["gen_ppl"] = parsed.get("gen_ppl")
                            row["cache_hit_rate"] = parsed.get("cache_hit_rate")

                    if args.mode in ("generation", "both"):
                        cmd = build_cmd(args, model_script, policy, cache_size, lambda_val, temp_path, "generation")
                        out = run_subprocess(cmd, timeout=timeout)
                        if out:
                            parsed = parse_output(out)
                            row["tps"] = parsed.get("tps")
                            row["cache_hit_rate"] = parsed.get("cache_hit_rate")

                    results.append(row)
    finally:
        try: os.remove(temp_path)
        except: pass

    return results

def save_csv(results, csv_path):
    if not results: return
    fieldnames = sorted(set().union(*(r.keys() for r in results)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[sweep] Results saved to: {csv_path}")

def generate_plots(results, output_prefix):
    if not HAS_MATPLOTLIB or not results: return
    
    policies = sorted(set(r["policy"] for r in results))
    cache_sizes = sorted(set(r["cache_size"] for r in results))
    lambdas = sorted(set(r["lambda"] for r in results))

    def save(fig, name):
        fname = f"{output_prefix}_{name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot 1: Policy vs Hit Rate (Bar chart grouped by Cache Size) for a fixed lambda
    for lam in lambdas:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(policies))
        width = 0.8 / len(cache_sizes)
        
        for i, cs in enumerate(cache_sizes):
            y = [next((r["cache_hit_rate"] for r in results if r["policy"] == p and r["cache_size"] == cs and r["lambda"] == lam and r["cache_hit_rate"] is not None), 0) for p in policies]
            ax.bar([pos + i * width for pos in x], y, width, label=f"Cache = {cs}")
            
        ax.set_xticks([pos + width * (len(cache_sizes) - 1) / 2 for pos in x])
        ax.set_xticklabels(policies)
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title(f"Hit Rate by Policy (λ={lam})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save(fig, f"hitrate_bar_lambda_{lam}")

    # Plot 2: PPL vs policy
    for lam in lambdas:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cs in enumerate(cache_sizes):
            y = [next((r["gen_ppl"] for r in results if r["policy"] == p and r["cache_size"] == cs and r["lambda"] == lam and r["gen_ppl"] is not None), 0) for p in policies]
            ax.bar([pos + i * width for pos in x], y, width, label=f"Cache = {cs}")
            
        ax.set_xticks([pos + width * (len(cache_sizes) - 1) / 2 for pos in x])
        ax.set_xticklabels(policies)
        ax.set_ylabel("Generation PPL")
        ax.set_title(f"PPL by Policy (λ={lam})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save(fig, f"ppl_bar_lambda_{lam}")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cached", "predict", "base"], default="cached", help="Model backend to evaluate")
    p.add_argument("--policies", nargs="+", default=["LRU", "MRU", "LFU", "MFU", "CLOCK", "RANDOM", "LFRU", "PREFILL"], help="Policies to test")
    p.add_argument("--cache-sizes", type=int, nargs="+", default=[2])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0])
    p.add_argument("--prefill-top-n", type=int, default=0, help="For PREFILL policy: how many top experts to lock")
    p.add_argument("--dataset", choices=["default", "wikitext", "fineweb", "orca"], default="default")
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--mode", choices=["perplexity", "generation", "both"], default="both")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--output-prefix", type=str, default=None)
    p.add_argument("--subprocess-timeout", type=int, default=300)
    return p

def main():
    args = build_parser().parse_args()
    args.output_prefix = args.output_prefix or f"sweep_policy_{time.strftime('%Y%m%d_%H%M%S')}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_script = os.path.abspath(os.path.join(script_dir, "..", "unified_llm_w4a16", "mixtral_8x7B_w4a16_model.py"))

    prompts = load_prompts(args)
    results = run_sweep(args, model_script, prompts)
    save_csv(results, f"{args.output_prefix}.csv")
    generate_plots(results, args.output_prefix)
    return 0

if __name__ == "__main__":
    exit(main())
