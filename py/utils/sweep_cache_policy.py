#!/usr/bin/env python3
"""
sweep_cache_policy.py — thesis section 1 (expert cache policy).

Benchmarks expert cache eviction policies (LRU, MRU, LFU, MFU, CLOCK, RANDOM, LFRU, PREFILL)
across several expert cache sizes and lambda values using the unified_llm_w4a16 model.
Default mode is ``generation`` (end-to-end TPS + cache hit rate). Use ``--mode perplexity``
or ``both`` only if you explicitly want generation perplexity.

Typical usage examples:
python sweep_cache_policy.py --policies LRU LFU CLOCK --cache-sizes 2 4 --lambdas 0.0 1.0

For PREFILL, locked expert count N defaults to round(prefill_cache_fraction * cache_size)
(use --prefill-top-n to override with a fixed N).
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
from typing import Optional

from prompt_datasets import HF_EVAL_CHOICES

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_prompts(args):
    dataset_name = args.dataset
    num = args.num_prompts
    max_chars = getattr(args, "prompt_max_chars", 4096)

    if dataset_name in HF_EVAL_CHOICES:
        from prompt_datasets import load_eval_prompts
        prompts = load_eval_prompts(
            dataset_name,
            num,
            max_chars=max_chars,
            stream_skip=getattr(args, "dataset_offset", None),
        )
        if prompts:
            return prompts

    return ["In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."] * num

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

# def parse_output(output):
#     result = {}
#     m = re.search(r"Generation Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
#     if m:
#         result["gen_ppl"] = float(m.group(1))

#     # Qwen sweep summary format.
#     m = re.search(r"Cache Stats:\s*Hits=(\d+),\s*Misses=(\d+),\s*HitRate=([-+]?\d*\.?\d+)%", output)
#     if m:
#         result["cache_hits"] = int(m.group(1))
#         result["cache_misses"] = int(m.group(2))
#         result["cache_hit_rate"] = float(m.group(3))
#     else:
#         # Fallback: aggregate per-layer cached-backend lines:
#         #   Layer N: Hits=H, Misses=M, HitRate=...
#         layer_stats = re.findall(r"Layer\s+\d+:\s+Hits=(\d+),\s*Misses=(\d+),\s*HitRate=", output)
#         if layer_stats:
#             hits = sum(int(h) for h, _ in layer_stats)
#             misses = sum(int(miss) for _, miss in layer_stats)
#             total = hits + misses
#             result["cache_hits"] = hits
#             result["cache_misses"] = misses
#             result["cache_hit_rate"] = (100.0 * hits / total) if total > 0 else 0.0

#     m = re.search(r"End-to-End TPS:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
#     if m:
#         result["tps"] = float(m.group(1))
#     else:
#         # Accept both "... 0.123 seconds" and "... 0.123" variants.
#         m2 = re.search(r"Average Time per Token:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s+seconds)?", output)
#         if m2:
#             tpt = float(m2.group(1))
#             result["tps"] = 1.0 / tpt if tpt > 0 else 0.0

#     m_ppl_std = re.search(r"Generation Perplexity StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
#     if m_ppl_std:
#         result["gen_ppl_std"] = float(m_ppl_std.group(1))

#     m_tps_std = re.search(r"TPS StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
#     if m_tps_std:
#         result["tps_std"] = float(m_tps_std.group(1))

#     return result

def parse_output(output):
    result = {}
    m = re.search(r"Generation Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if m:
        result["gen_ppl"] = float(m.group(1))

    m = re.search(r"Cache Stats:\s*Hits=(\d+),\s*Misses=(\d+),\s*HitRate=([-+]?\d*\.?\d+)%", output)
    if m:
        result["cache_hits"] = int(m.group(1))
        result["cache_misses"] = int(m.group(2))
        result["cache_hit_rate"] = float(m.group(3))
    else:
        layer_stats = re.findall(r"Layer\s+\d+:\s+Hits=(\d+),\s*Misses=(\d+),\s*HitRate=", output)
        if layer_stats:
            hits = sum(int(h) for h, _ in layer_stats)
            misses = sum(int(miss) for _, miss in layer_stats)
            total = hits + misses
            result["cache_hits"] = hits
            result["cache_misses"] = misses
            result["cache_hit_rate"] = (100.0 * hits / total) if total > 0 else 0.0

    # Prefer decode-only TPS from per-prompt C++ backend lines (excludes prefill),
    # matching sweep_predict_cached_cache_metrics.py's parse_tps. Falls back to the
    # Python wrapper's own end-to-end aggregate (includes prefill) if those aren't present.
    totals = [float(x) for x in re.findall(r"Total Generation Time:\s*([\d.eE+-]+)\s+seconds", output)]
    per_token = [float(x) for x in re.findall(r"Average Time per Token:\s*([\d.eE+-]+)\s+seconds", output)]
    if totals and per_token:
        total_generated_est = 0.0
        total_decode_time = 0.0
        for total_s, tpt_s in zip(totals, per_token):
            if total_s > 0 and tpt_s > 0:
                total_generated_est += total_s / tpt_s
                total_decode_time += total_s
        if total_decode_time > 0 and total_generated_est > 0:
            result["tps"] = total_generated_est / total_decode_time

    if "tps" not in result:
        m = re.search(r"End-to-End TPS:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
        if m:
            result["tps"] = float(m.group(1))
        else:
            m2 = re.search(r"Average Time per Token:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s+seconds)?", output)
            if m2:
                tpt = float(m2.group(1))
                if tpt > 0:
                    result["tps"] = 1.0 / tpt

    m_ppl_std = re.search(r"Generation Perplexity StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if m_ppl_std:
        result["gen_ppl_std"] = float(m_ppl_std.group(1))

    m_tps_std = re.search(r"TPS StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if m_tps_std:
        result["tps_std"] = float(m_tps_std.group(1))

    return result

def resolve_prefill_top_n(policy: str, cache_size: int, args) -> Optional[int]:
    """Experts to pin for PREFILL: explicit --prefill-top-n, else fraction of cache size."""
    if policy.upper() != "PREFILL":
        return None
    if getattr(args, "prefill_top_n", 0) > 0:
        return args.prefill_top_n
    frac = getattr(args, "prefill_cache_fraction", 0.25)
    return max(1, round(cache_size * frac))


def model_script_for(args: argparse.Namespace) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    name = "qwen3_30B-A3B_w4a16_model.py" if args.model == "qwen" else "mixtral_8x7B_w4a16_model.py"
    return os.path.abspath(os.path.join(script_dir, "..", "unified_llm_w4a16", name))


# def build_cmd(args, model_script, policy, cache_size, lambda_val, temp_prompts_path, phase):
#     cmd = [
#         sys.executable, model_script,
#         "--backend", args.backend,
#         "--device", "cuda",
#         "--cache-policy", policy,
#         "--lambda-val", str(lambda_val),
#         "--sweep-prompts-file", temp_prompts_path,
#     ]
#     if args.model == "qwen":
#         cmd.extend(["--max-cached-experts", str(cache_size)])
#     else:
#         cmd.extend(["--expert-cache", str(cache_size)])
#     prefill_n = resolve_prefill_top_n(policy, cache_size, args)
#     if prefill_n is not None:
#         cmd.extend(["--prefill-top-n", str(prefill_n)])
#     # Mixtral only: Qwen always prewarms on cached/predict (no --no-prewarm flag).
#     if args.model == "mixtral" and not getattr(args, "prewarm", False):
#         cmd.append("--no-prewarm")

#     if phase == "perplexity":
#         cmd += [
#             "--generation-perplexity",
#             "--generate",
#             "--max-new-tokens",
#             str(args.max_new_tokens),
#             "--temperature",
#             str(getattr(args, "temperature", 0.0)),
#             "--top-p",
#             str(getattr(args, "top_p", 0.9)),
#             "--top-k",
#             str(getattr(args, "top_k", 50)),
#         ]
#     else:
#         cmd += ["--generate", "--max-new-tokens", str(args.max_new_tokens)]
#     return cmd

def build_cmd(args, model_script, policy, cache_size, lambda_val, temp_prompts_path, phase):
    prefill_n = resolve_prefill_top_n(policy, cache_size, args)

    run_config = {
        "backend": args.backend,
        "device": "cuda",
        "cache_policy": policy,
        "lambda_val": lambda_val,
        "sweep_prompts_file": temp_prompts_path,
        "generate": True,
        "max_new_tokens": args.max_new_tokens,
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", 0.9),
        "top_k": getattr(args, "top_k", 50),
    }

    if args.model == "qwen":
        run_config["max_cached_experts"] = cache_size
    else:
        # NOT VERIFIED: mixtral_8x7B_w4a16_model.py schema unconfirmed.
        run_config["expert_cache"] = cache_size

    if prefill_n is not None:
        run_config["prefill_top_n"] = prefill_n

    if args.model == "mixtral" and not getattr(args, "prewarm", False):
        # NOT VERIFIED: qwen script has no run_config key to suppress prewarm at all.
        run_config["no_prewarm"] = True

    if phase == "perplexity":
        run_config["generation_perplexity"] = True

    fd, temp_json_path = tempfile.mkstemp(suffix=".json", prefix="sweep_cache_policy_run_config_")
    with os.fdopen(fd, "w") as f:
        json.dump(run_config, f)

    return [sys.executable, model_script, "--run-config", temp_json_path]

def resolve_subprocess_timeout(args) -> int:
    """Per-model subprocess budget (one full sweep over all prompts)."""
    if args.subprocess_timeout is not None:
        return args.subprocess_timeout
    # Same scaling as sweep_predict_cached_cache_metrics (100×256 needs ~14h).
    return max(900, args.num_prompts * args.max_new_tokens * 2 + 600)


def csv_fieldnames(mode: str) -> list[str]:
    """Column order for section-1 results (TPS + hit rate by default)."""
    fields = ["policy", "cache_size", "lambda", "prefill_top_n"]
    if mode in ("perplexity", "both"):
        fields.extend(["gen_ppl", "gen_ppl_std"])
    fields.extend(["tps", "tps_std", "cache_hit_rate"])
    return fields


def run_sweep(args, model_script, prompts, csv_path: str):
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(prompts, f)

    timeout = resolve_subprocess_timeout(args)
    results = []
    print(f"[sweep] CSV output: {csv_path}", flush=True)
    print(f"[sweep] Subprocess timeout: {timeout}s per run", flush=True)

    try:
        for policy in args.policies:
            for cache_size in sorted(args.cache_sizes):
                for lambda_val in sorted(args.lambdas):
                    print(f"\n{'='*25} policy={policy} cache={cache_size} λ={lambda_val} {'='*25}")
                    row = {
                        "policy": policy,
                        "cache_size": cache_size,
                        "lambda": lambda_val,
                        "prefill_top_n": resolve_prefill_top_n(policy, cache_size, args),
                        "tps": None,
                        "tps_std": None,
                        "cache_hit_rate": None,
                    }
                    if args.mode in ("perplexity", "both"):
                        row["gen_ppl"] = None
                        row["gen_ppl_std"] = None

                    if args.mode in ("perplexity", "both"):
                        cmd = build_cmd(args, model_script, policy, cache_size, lambda_val, temp_path, "perplexity")
                        out = run_subprocess(cmd, timeout=timeout)
                        if out:
                            parsed = parse_output(out)
                            row["gen_ppl"] = parsed.get("gen_ppl")
                            row["gen_ppl_std"] = parsed.get("gen_ppl_std")
                            row["cache_hit_rate"] = parsed.get("cache_hit_rate")

                    if args.mode in ("generation", "both"):
                        cmd = build_cmd(args, model_script, policy, cache_size, lambda_val, temp_path, "generation")
                        out = run_subprocess(cmd, timeout=timeout)
                        if out:
                            parsed = parse_output(out)
                            row["tps"] = parsed.get("tps")
                            row["tps_std"] = parsed.get("tps_std")
                            row["cache_hit_rate"] = parsed.get("cache_hit_rate")

                    results.append(row)
                    save_csv(results, csv_path, fieldnames=csv_fieldnames(args.mode))
    finally:
        try: os.remove(temp_path)
        except: pass

    return results

def save_csv(results, csv_path, *, fieldnames: Optional[list[str]] = None) -> None:
    if not results:
        return
    fieldnames = fieldnames or sorted(set().union(*(r.keys() for r in results)))
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        f.flush()
        os.fsync(f.fileno())

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

    # Plot 2: TPS vs policy
    for lam in lambdas:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cs in enumerate(cache_sizes):
            y = [next((r["tps"] for r in results if r["policy"] == p and r["cache_size"] == cs and r["lambda"] == lam and r["tps"] is not None), 0) for p in policies]
            ax.bar([pos + i * width for pos in x], y, width, label=f"Cache = {cs}")

        ax.set_xticks([pos + width * (len(cache_sizes) - 1) / 2 for pos in x])
        ax.set_xticklabels(policies)
        ax.set_ylabel("End-to-End TPS")
        ax.set_title(f"TPS by Policy (λ={lam})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save(fig, f"tps_bar_lambda_{lam}")

    # Plot 3: PPL vs policy (only when perplexity mode was used)
    if any(r.get("gen_ppl") is not None for r in results):
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
    p.add_argument("--model", choices=["qwen", "mixtral"], default="qwen", help="Which model script to run.")
    p.add_argument("--backend", choices=["cached", "predict", "base"], default="cached", help="Model backend to evaluate")
    p.add_argument("--policies", nargs="+", default=["LRU", "MRU", "LFU", "MFU", "CLOCK", "RANDOM", "LFRU", "PREFILL"], help="Policies to test")
    p.add_argument("--cache-sizes", type=int, nargs="+", default=[2])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0])
    p.add_argument(
        "--prefill-top-n", type=int, default=0,
        help="PREFILL only: fixed N locked experts (overrides --prefill-cache-fraction)",
    )
    p.add_argument(
        "--prefill-cache-fraction", type=float, default=0.25,
        help="PREFILL only: N = round(fraction * cache_size), at least 1 (default 0.25)",
    )
    p.add_argument(
        "--dataset",
        choices=["default", *HF_EVAL_CHOICES],
        default="default",
        help="Prompt source. Eval uses official test splits (or a hash holdout for FineWeb/Orca).",
    )
    p.add_argument(
        "--dataset-offset",
        type=int,
        default=None,
        help="Skip this many eval-partition prompts (never train examples).",
    )
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--mode", choices=["perplexity", "generation", "both"], default="generation")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--output-prefix", type=str, default=None)
    p.add_argument(
        "--subprocess-timeout",
        type=int,
        default=None,
        help="Kill model subprocess after N seconds (default: scales with --num-prompts and --max-new-tokens)",
    )
    p.add_argument(
        "--prewarm", action="store_true",
        help="Preload experts at init (default off in this sweep; large caches can OOM)",
    )
    p.add_argument(
        "--prompt-max-chars", type=int, default=4096,
        help="Characters per wikitext prompt chunk (default 4096 ≈ 1024 tokens at ~4 chars/tok).",
    )
    return p

def main():
    args = build_parser().parse_args()
    args.output_prefix = args.output_prefix or f"sweep_policy_{time.strftime('%Y%m%d_%H%M%S')}"

    model_script = model_script_for(args)
    prompts = load_prompts(args)
    csv_path = f"{args.output_prefix}.csv"
    results = run_sweep(args, model_script, prompts, csv_path)
    print(f"[sweep] Done — {len(results)} rows in {csv_path}", flush=True)
    generate_plots(results, args.output_prefix)
    return 0

if __name__ == "__main__":
    exit(main())
