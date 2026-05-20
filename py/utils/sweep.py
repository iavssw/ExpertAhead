#!/usr/bin/env python3
"""
sweep.py — unified evaluation sweep for heteroPredict / Mixtral 8x7B

Replaces:
  sweep_extreme_lambda.py      — extreme lambda range, tradeoff plots
  sweep_predict_vs_cached.py   — cached vs predict comparison, layer subsets
  sweep_wikitext_metrics.py    — wikitext dataset, TopK match stats
  sweep_lambda_cache.py        — basic lambda × cache sweep (retired)

Typical usage examples
──────────────────────
# Compare cached vs predict over default lambda range
python sweep.py --backends cached predict --cache-sizes 2 --lambdas 0.0 0.2 0.4 0.6 0.8 1.0

# Wikitext perplexity + TPS sweep (replaces sweep_wikitext_metrics.py)
python sweep.py --dataset wikitext --num-prompts 10 --backends predict \
    --cache-sizes 1 2 3 4 --lambdas 0.0 0.5 1.0 1.5 2.0

# Extreme lambda sweep for cache-biasing study (replaces sweep_extreme_lambda.py)
python sweep.py --backends cached \
    --lambdas 0.0 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 \
    --extreme-lambda-plots

# Just regenerate plots from a previous run
python sweep.py --plot-only --csv-file my_results.csv
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

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helper
# ──────────────────────────────────────────────────────────────────────────────

def run_subprocess(cmd, timeout=600):
    """Run *cmd* and return its combined stdout/stderr as a string.

    Prints output in real-time.  Returns None on timeout or non-zero exit.
    Sets HSA_ENABLE_SDMA=0 to avoid ROCm driver hangs on back-to-back runs.
    """
    try:
        env = os.environ.copy()
        env["HSA_ENABLE_SDMA"] = "0"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,  # own process group → reliable kill
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

        # Brief pause to let ROCm clean up IPC buffers
        time.sleep(1.0)
        return "".join(output_lines)

    except Exception as exc:
        print(f"[sweep] Exception running command: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_output(output):
    """Parse all known metrics from a model script's stdout."""
    result = {}

    # Generation Perplexity
    m = re.search(r"Generation Perplexity:\s+([\d\.]+)", output)
    if m:
        result["gen_ppl"] = float(m.group(1))

    # Average NLL
    m = re.search(r"Average NLL:\s+([\d\.]+)", output)
    if m:
        result["avg_nll"] = float(m.group(1))

    # Cache Stats
    m = re.search(r"Cache Stats: Hits=(\d+), Misses=(\d+), HitRate=([\d\.]+)%", output)
    if m:
        result["cache_hits"] = int(m.group(1))
        result["cache_misses"] = int(m.group(2))
        result["cache_hit_rate"] = float(m.group(3))

    # Predictor Stats
    m = re.search(r"Predictor Stats: Hits=(\d+), Total=(\d+), HitRate=([\d\.]+)%", output)
    if m:
        result["pred_hits"] = int(m.group(1))
        result["pred_total"] = int(m.group(2))
        result["pred_hit_rate"] = float(m.group(3))

    # Sequential Top-1 fallback (used when predictor stats absent)
    m2 = re.search(r"Sequential Top1 Stats: Hits=(\d+), Total=(\d+), HitRate=([\d\.]+)%", output)
    if m2 and "pred_hit_rate" not in result:
        result["pred_hits"] = int(m2.group(1))
        result["pred_total"] = int(m2.group(2))
        result["pred_hit_rate"] = float(m2.group(3))

    # Bandwidth / load stats
    m = re.search(r"Bandwidth: StallLoads=(\d+),\s*PrefetchLoads=(\d+),\s*AvgLoadTime=([\d\.]+)ms", output)
    if m:
        result["stall_loads"] = int(m.group(1))
        result["prefetch_loads"] = int(m.group(2))
        result["avg_load_ms"] = float(m.group(3))

    # Predictor TopK matches: 0=N 1=N >=2=N
    m = re.search(r"Predictor TopK matches: 0=(\d+)\s+1=(\d+)\s+>=2=(\d+)", output)
    if m:
        result["pred_match_0"] = int(m.group(1))
        result["pred_match_1"] = int(m.group(2))
        result["pred_match_2"] = int(m.group(3))

    # TPS — prefer End-to-End TPS, fall back to Average Time per Token
    m = re.search(r"End-to-End TPS:\s+([\d\.]+)", output)
    if m:
        result["tps"] = float(m.group(1))
    else:
        m2 = re.search(r"Average Time per Token:\s+([\d\.]+)\s+seconds", output)
        if m2:
            tpt = float(m2.group(1))
            result["tps"] = 1.0 / tpt if tpt > 0 else 0.0

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_prompts(args):
    """Return a list of text prompts according to *args.dataset*."""
    dataset_name = args.dataset
    num = args.num_prompts

    if dataset_name in ("fineweb", "orca", "wikitext"):
        try:
            from datasets import load_dataset
        except ImportError:
            print("Error: 'datasets' package required. Run: pip install datasets")
            sys.exit(1)

        print(f"[sweep] Loading {dataset_name} dataset …")
        if dataset_name == "fineweb":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        elif dataset_name == "orca":
            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        elif dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

        prompts = []
        for item in ds:
            if dataset_name == "fineweb":
                text = item.get("text", "")
            elif dataset_name == "orca":
                text = "\n".join([
                    item.get("system_prompt", ""),
                    item.get("question", ""),
                    item.get("response", ""),
                ]).strip()
            elif dataset_name == "wikitext":
                text = item.get("text", "")

            if len(text.strip()) > 100:
                prompts.append(text.strip()[:600])
                if len(prompts) >= num:
                    break

        if not prompts:
            print("[sweep] Warning: no prompts loaded; using default text.")

        return prompts

    if dataset_name == "txt":
        prompt_file = args.text or "prompts.txt"
        if not os.path.exists(prompt_file):
            print(f"Error: prompt file '{prompt_file}' not found.")
            sys.exit(1)
        with open(prompt_file) as f:
            prompts = [p.strip() for p in f.read().split("\n\n") if p.strip()]
        return prompts

    # default
    text = args.text or (
        "In a shocking finding, scientist discovered a herd of unicorns living "
        "in a remote, previously unexplored valley, in the Andes Mountains. "
        "Even more surprising to the researchers was the fact that the unicorns "
        "spoke perfect English."
    )
    return [text] * num


# ──────────────────────────────────────────────────────────────────────────────
# Core sweep
# ──────────────────────────────────────────────────────────────────────────────

def build_cmd(args, model_script, backend, cache_size, lambda_val,
              prefetch_count, predict_layers, temp_prompts_path, phase):
    """Build the subprocess command for one evaluation run.

    *phase* is either ``"perplexity"`` or ``"generation"``.
    """
    cmd = [
        sys.executable, model_script,
        "--backend", backend,
        "--device", "cuda",
    ]
    if args.model == "qwen":
        cmd.extend(["--max-cached-experts", str(cache_size)])
    else:
        cmd.extend(["--expert-cache", str(cache_size)])

    cmd.extend([
        "--lambda-val", str(lambda_val),
        "--sweep-prompts-file", temp_prompts_path,
    ])

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

    if getattr(args, "forced_top_n", 0) > 0:
        cmd += ["--forced-top-n", str(args.forced_top_n)]

    if backend == "predict":
        cmd += ["--predictor-model", args.predictor_path]
        cmd += ["--predictor-device", args.predictor_device]
        cmd += ["--prefetch-experts-count", str(prefetch_count)]
        if predict_layers is not None:
            cmd.append("--predict-layers")
            cmd.extend(map(str, predict_layers))

    if getattr(args, "expert_correlation_csv", None):
        cmd += ["--expert-correlation-csv", args.expert_correlation_csv]

    if getattr(args, "config_path", None):
        cmd += ["--config-path", args.config_path]

    return cmd


def run_sweep(args, model_script, prompts):
    """Execute the full sweep and return a list of result dicts."""

    # Write prompts to a temp file once
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(prompts, f)

    # Compute per-subprocess timeout
    if args.subprocess_timeout is not None:
        timeout = args.subprocess_timeout
    else:
        timeout = max(300, args.max_new_tokens * 20 + 180)

    # Normalise lambda list: always include 0.0 so baseline is available
    lambdas = sorted(set(args.lambdas))

    # Resolve layer subsets
    layer_subsets = []  # list of (name, list_or_None)
    for s in args.predict_layer_subsets:
        if s.lower() == "all":
            layer_subsets.append(("all", None))
        else:
            indices = [int(x.strip()) for x in s.split(",") if x.strip()]
            layer_subsets.append((s, indices))

    results = []

    _print_header(args)

    try:
        for cache_size in sorted(args.cache_sizes):
            for prefetch_count in sorted(args.prefetch_counts):
                for lambda_val in lambdas:
                    for layers_name, layers in layer_subsets:
                        for backend in args.backends:
                            # Skip nonsensical combinations
                            if backend == "cached" and prefetch_count != args.prefetch_counts[0]:
                                continue
                            if backend == "cached" and layers_name != layer_subsets[0][0]:
                                # cached backend is independent of predict-layers
                                continue
                            if backend == "predict" and prefetch_count > cache_size:
                                continue

                            row = _run_one(
                                args, model_script, backend, cache_size,
                                lambda_val, prefetch_count, layers_name, layers,
                                temp_path, timeout,
                            )
                            results.append(row)
                            _print_row(args, row)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return results


def _run_one(args, model_script, backend, cache_size, lambda_val,
             prefetch_count, layers_name, layers, temp_path, timeout):
    """Run perplexity and/or generation phase and return a merged result dict."""

    print(f"\n{'='*25} backend={backend} cache={cache_size} λ={lambda_val} "
          f"prefetch={prefetch_count} layers={layers_name} {'='*25}")

    row = {
        "backend": backend,
        "cache_size": cache_size,
        "lambda": lambda_val,
        "prefetch_count": prefetch_count if backend == "predict" else None,
        "layers": layers_name if backend == "predict" else "N/A",
        "gen_ppl": None,
        "tps": None,
        "cache_hit_rate": None,
        "pred_hit_rate": None,
        "stall_loads": None,
        "prefetch_loads": None,
        "avg_load_ms": None,
        "pred_match_0": None,
        "pred_match_1": None,
        "pred_match_2": None,
    }

    mode = args.mode

    # ── perplexity phase ──────────────────────────────────────────────────
    if mode in ("perplexity", "both"):
        print(">> Perplexity phase …")
        cmd = build_cmd(args, model_script, backend, cache_size, lambda_val,
                        prefetch_count, layers, temp_path, "perplexity")
        out = run_subprocess(cmd, timeout=timeout)
        if out:
            parsed = parse_output(out)
            row["gen_ppl"] = parsed.get("gen_ppl")
            # Take cache/pred stats from perplexity pass (may be overwritten below)
            for k in ("cache_hit_rate", "pred_hit_rate", "stall_loads",
                      "prefetch_loads", "avg_load_ms",
                      "pred_match_0", "pred_match_1", "pred_match_2"):
                if k in parsed:
                    row[k] = parsed[k]

    # ── generation phase ──────────────────────────────────────────────────
    if mode in ("generation", "both"):
        print(">> Generation phase …")
        cmd = build_cmd(args, model_script, backend, cache_size, lambda_val,
                        prefetch_count, layers, temp_path, "generation")
        out = run_subprocess(cmd, timeout=timeout)
        if out:
            parsed = parse_output(out)
            row["tps"] = parsed.get("tps")
            # Overwrite cache/pred stats with generation-phase values
            for k in ("cache_hit_rate", "pred_hit_rate", "stall_loads",
                      "prefetch_loads", "avg_load_ms",
                      "pred_match_0", "pred_match_1", "pred_match_2"):
                if k in parsed:
                    row[k] = parsed[k]

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Console output helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_header(args):
    mode = args.mode
    sep = "-" * 160
    print("=" * 160)
    print("SWEEP")
    print(f"  backends:  {args.backends}")
    print(f"  cache:     {sorted(args.cache_sizes)}")
    print(f"  lambdas:   {sorted(args.lambdas)}")
    print(f"  prefetch:  {sorted(args.prefetch_counts)}")
    print(f"  layers:    {args.predict_layer_subsets}")
    print(f"  dataset:   {args.dataset} ({args.num_prompts} prompts)")
    print(f"  mode:      {mode}")
    print("=" * 160)
    cols = (f"{'Backend':<12} {'Cache':>5} {'λ':>6} {'Prefetch':>8} "
            f"{'Layers':<12}")
    if mode in ("perplexity", "both"):
        cols += f" {'PPL':>10}"
    if mode in ("generation", "both"):
        cols += f" {'TPS':>8}"
    cols += f" {'HitRate':>9} {'PredRate':>9} {'Stalls':>7} {'AvgLoad':>9}"
    print(sep)
    print(cols)
    print(sep)


def _print_row(args, row):
    mode = args.mode
    sep = "-" * 160
    ppl   = row.get("gen_ppl");       ppl_s  = f"{ppl:.4f}"   if ppl  is not None else "N/A"
    tps   = row.get("tps");           tps_s  = f"{tps:.3f}"   if tps  is not None else "N/A"
    hr    = row.get("cache_hit_rate"); hr_s   = f"{hr:.2f}%"   if hr   is not None else "N/A"
    pr    = row.get("pred_hit_rate");  pr_s   = f"{pr:.2f}%"   if pr   is not None else "N/A"
    st    = row.get("stall_loads");    st_s   = str(st)         if st   is not None else "N/A"
    al    = row.get("avg_load_ms");    al_s   = f"{al:.2f}ms"  if al   is not None else "N/A"

    line = (f"{row['backend']:<12} {row['cache_size']:>5} {row['lambda']:>6.2f} "
            f"{str(row['prefetch_count'] or '–'):>8} {str(row['layers']):<12}")
    if mode in ("perplexity", "both"):
        line += f" {ppl_s:>10}"
    if mode in ("generation", "both"):
        line += f" {tps_s:>8}"
    line += f" {hr_s:>9} {pr_s:>9} {st_s:>7} {al_s:>9}"
    print(sep)
    print(line)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _figtext(cmd_str):
    """Return (wrapped_str, bottom_margin) for a command annotation."""
    import textwrap
    wrapped = textwrap.fill(f"Cmd: {cmd_str}", width=110)
    nlines = wrapped.count("\n") + 1
    return wrapped, 0.03 + 0.025 * nlines


def generate_plots(results, output_prefix, args):
    if not HAS_MATPLOTLIB:
        print("[sweep] matplotlib not available, skipping plots.")
        return
    if not results:
        print("[sweep] No results to plot.")
        return

    cmd_str = " ".join(sys.argv)
    wrapped_args, bot = _figtext(cmd_str)

    # Convert to a simple list-of-dicts for easy querying
    # (avoid hard pandas dependency in plot code)
    rows = results  # list of dicts

    def vals(key):
        return [r.get(key) for r in rows]

    lambdas   = sorted(set(r["lambda"]     for r in rows))
    caches    = sorted(set(r["cache_size"] for r in rows))
    backends  = sorted(set(r["backend"]    for r in rows))

    def make_label(r):
        lbl = r["backend"]
        if r["backend"] == "predict":
            lbl += f" (C={r['cache_size']}, P={r['prefetch_count']}, L={r['layers']})"
        else:
            lbl += f" (C={r['cache_size']})"
        return lbl

    label_set = sorted(set(make_label(r) for r in rows))

    def save(fig, name):
        fname = f"{output_prefix}_{name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[sweep] Saved: {fname}")

    extreme = getattr(args, "extreme_lambda_plots", False)
    xscale_kw = dict(xscale="symlog", linthresh=1.0) if extreme else {}

    # 1. λ vs PPL (one line per backend/config label)
    ppl_data = [(make_label(r), r["lambda"], r["gen_ppl"]) for r in rows if r.get("gen_ppl") is not None]
    if ppl_data:
        fig, ax = plt.subplots(figsize=(11, 6))
        for lbl in label_set:
            pts = sorted([(lam, p) for l, lam, p in ppl_data if l == lbl], key=lambda x: x[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=lbl)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Generation Perplexity")
        ax.set_title("Perplexity vs Lambda")
        if extreme: ax.set_xscale("symlog", linthresh=1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.01, wrapped_args, ha="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
        plt.tight_layout(rect=[0, bot, 1, 1])
        save(fig, "ppl_vs_lambda")

    # 2. λ vs TPS
    tps_data = [(make_label(r), r["lambda"], r["tps"]) for r in rows if r.get("tps") is not None]
    if tps_data:
        fig, ax = plt.subplots(figsize=(11, 6))
        for lbl in label_set:
            pts = sorted([(lam, t) for l, lam, t in tps_data if l == lbl], key=lambda x: x[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="s", label=lbl)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Tokens per Second")
        ax.set_title("TPS vs Lambda")
        if extreme: ax.set_xscale("symlog", linthresh=1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.01, wrapped_args, ha="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
        plt.tight_layout(rect=[0, bot, 1, 1])
        save(fig, "tps_vs_lambda")

    # 3. λ vs Cache Hit Rate
    hr_data = [(make_label(r), r["lambda"], r["cache_hit_rate"]) for r in rows if r.get("cache_hit_rate") is not None]
    if hr_data:
        fig, ax = plt.subplots(figsize=(11, 6))
        for lbl in label_set:
            pts = sorted([(lam, h) for l, lam, h in hr_data if l == lbl], key=lambda x: x[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="^", label=lbl)
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title("Cache Hit Rate vs Lambda")
        ax.set_ylim(0, 100)
        if extreme: ax.set_xscale("symlog", linthresh=1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.01, wrapped_args, ha="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
        plt.tight_layout(rect=[0, bot, 1, 1])
        save(fig, "hitrate_vs_lambda")

    # 4. Cache Size vs PPL (one line per lambda) — useful for memory sweep
    if len(caches) > 1:
        ppl_cs = [(r["cache_size"], r["lambda"], r.get("gen_ppl")) for r in rows if r.get("gen_ppl") is not None]
        if ppl_cs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for lam in lambdas:
                pts = sorted([(cs, p) for cs, l, p in ppl_cs if l == lam and p is not None])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=f"λ={lam}")
            ax.set_xlabel("Cache Size (# experts)")
            ax.set_ylabel("Generation Perplexity")
            ax.set_title("Perplexity vs Cache Size")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, "ppl_vs_cache")

    # 5. Cache Size vs Cache Hit Rate
    if len(caches) > 1:
        hr_cs = [(r["cache_size"], r["lambda"], r.get("cache_hit_rate")) for r in rows if r.get("cache_hit_rate") is not None]
        if hr_cs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for lam in lambdas:
                pts = sorted([(cs, h) for cs, l, h in hr_cs if l == lam and h is not None])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=f"λ={lam}")
            ax.set_xlabel("Cache Size (# experts)")
            ax.set_ylabel("Cache Hit Rate (%)")
            ax.set_title("Cache Hit Rate vs Cache Size")
            ax.set_ylim(0, 100)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, "hitrate_vs_cache")

    # 6. Extreme-lambda: PPL degradation vs hit-rate improvement scatter
    if extreme:
        baseline_rows = {r["cache_size"]: r for r in rows if r["lambda"] == 0.0}
        scatter_pts = []
        for r in rows:
            cs = r["cache_size"]
            ppl = r.get("gen_ppl")
            hr  = r.get("cache_hit_rate")
            base = baseline_rows.get(cs, {})
            base_ppl = base.get("gen_ppl")
            base_hr  = base.get("cache_hit_rate")
            if ppl and hr and base_ppl and base_hr and base_ppl > 0:
                scatter_pts.append((
                    hr - base_hr,
                    (ppl - base_ppl) / base_ppl * 100,
                    r["lambda"],
                ))
        if scatter_pts:
            fig, ax = plt.subplots(figsize=(10, 6))
            for dx, dy, lam in scatter_pts:
                ax.scatter(dx, dy, s=100, zorder=5)
                ax.annotate(f"λ={lam}", (dx, dy), xytext=(8, 5),
                            textcoords="offset points", fontsize=9)
            ax.set_xlabel("Cache Hit Rate Improvement (pp)")
            ax.set_ylabel("Perplexity Increase (%)")
            ax.set_title("Perplexity Cost of Cache Hit Rate Gains")
            ax.axhline(0, color="gray", alpha=0.3)
            ax.axvline(0, color="gray", alpha=0.3)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, "tradeoff_scatter")


# ──────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results, csv_path):
    if not results:
        return
    fieldnames = sorted(set().union(*(r.keys() for r in results)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[sweep] Results saved to: {csv_path}")


def load_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        row[k] = None if v in ("", "None") else v
            rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── what to sweep ──────────────────────────────────────────────────────
    sweep = p.add_argument_group("sweep axes")
    sweep.add_argument("--backends", nargs="+", default=["cached", "predict"],
                       choices=["cached", "predict"],
                       help="Backends to run (default: cached predict)")
    sweep.add_argument("--lambdas", type=float, nargs="+",
                       default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       help="Lambda values to sweep")
    sweep.add_argument("--cache-sizes", type=int, nargs="+", default=[2],
                       help="Expert cache sizes per layer")
    sweep.add_argument("--prefetch-counts", type=int, nargs="+", default=[1],
                       help="Number of experts to prefetch (predict backend)")
    sweep.add_argument("--predict-layer-subsets", nargs="+", default=["all"],
                       help="Layer subsets for predict backend. 'all' or comma-"
                            "separated indices e.g. '0,2,4'")

    # ── evaluation ─────────────────────────────────────────────────────────
    ev = p.add_argument_group("evaluation")
    ev.add_argument("--mode", choices=["perplexity", "generation", "both"],
                    default="both",
                    help="Evaluation mode (default: both)")
    ev.add_argument("--max-new-tokens", type=int, default=30,
                    help="Tokens to generate per prompt (generation mode)")
    ev.add_argument("--forced-top-n", type=int, default=0,
                    help="Force top-N unbiased experts into cache mask (cached/base models).")

    # ── dataset / prompts ──────────────────────────────────────────────────
    ds = p.add_argument_group("dataset")
    ds.add_argument("--dataset",
                    choices=["default", "wikitext", "fineweb", "orca", "txt"],
                    default="default", help="Dataset (default: default)")
    ds.add_argument("--num-prompts", type=int, default=5,
                    help="Number of prompts to evaluate")
    ds.add_argument("--text", type=str, default=None,
                    help="Custom prompt text (default dataset) or path to prompts file (txt dataset)")

    # ── model / predictor ──────────────────────────────────────────────────
    mdl = p.add_argument_group("model")
    mdl.add_argument("--model", choices=["mixtral", "qwen"], default="mixtral",
                     help="Model architecture to evaluate (default: mixtral)")
    mdl.add_argument("--predictor-path", type=str,
                     default="/home/michael/heteroPredict/trainingData/mixtral_8x7b/sweep_3_5",
                     help="Path to predictor model directory")
    mdl.add_argument("--predictor-device", choices=["gpu", "cpu", "auto"],
                     default="gpu", help="Device for predictor inference")
    mdl.add_argument("--config-path", type=str, default=None,
                     help="Path to JSON5 config file passed to model script")
    mdl.add_argument("--expert-correlation-csv", type=str, default=None,
                     help="CSV of per-layer correlation multipliers")

    # ── output ─────────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument("--output-prefix", type=str, default=None,
                     help="Prefix for CSV and plot files (default: sweep_<timestamp>)")
    out.add_argument("--extreme-lambda-plots", action="store_true",
                     help="Use symlog x-axis and add tradeoff scatter plot "
                          "(useful when lambdas span 0–50+)")
    out.add_argument("--plot-only", action="store_true",
                     help="Skip sweep and regenerate plots from --csv-file")
    out.add_argument("--csv-file", type=str, default=None,
                     help="CSV file to use with --plot-only")

    # ── misc ───────────────────────────────────────────────────────────────
    misc = p.add_argument_group("misc")
    misc.add_argument("--subprocess-timeout", type=int, default=None,
                      help="Per-subprocess timeout in seconds "
                           "(default: max(300, max_new_tokens*20+180))")

    return p


def main():

    parser = build_parser()
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output_prefix is None:
        args.output_prefix = f"sweep_{ts}"

    # ── plot-only mode ─────────────────────────────────────────────────────
    if args.plot_only:
        csv_path = args.csv_file
        if csv_path is None:
            # Try to find the most recent sweep CSV
            candidates = sorted(
                [f for f in os.listdir(".") if f.startswith("sweep_") and f.endswith(".csv")],
                reverse=True,
            )
            if candidates:
                csv_path = candidates[0]
                print(f"[sweep] --plot-only: using {csv_path}")
            else:
                print("Error: no CSV found. Specify --csv-file.")
                sys.exit(1)
        results = load_csv(csv_path)
        prefix = args.output_prefix or os.path.splitext(csv_path)[0]
        generate_plots(results, prefix, args)
        return

    # ── locate model script ────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "mixtral_8x7B_w4a16_model.py" if args.model == "mixtral" else "qwen3_30B-A3B_w4a16_model.py"
    model_script = os.path.join(
        script_dir, "..", "unified_llm_w4a16", model_filename
    )
    if not os.path.exists(model_script):
        print(f"Error: model script not found at {model_script}")
        sys.exit(1)

    # ── load prompts ───────────────────────────────────────────────────────
    prompts = load_prompts(args)
    if not prompts:
        print("Error: no prompts loaded.")
        sys.exit(1)
    print(f"[sweep] Loaded {len(prompts)} prompt(s).")

    # ── run sweep ──────────────────────────────────────────────────────────
    results = run_sweep(args, model_script, prompts)

    print("\n" + "=" * 160)
    print("SWEEP COMPLETE")
    print("=" * 160)

    # ── save CSV ───────────────────────────────────────────────────────────
    csv_path = f"{args.output_prefix}.csv"
    save_csv(results, csv_path)

    # ── generate plots ─────────────────────────────────────────────────────
    try:
        generate_plots(results, args.output_prefix, args)
    except Exception as exc:
        print(f"[sweep] Error generating plots: {exc}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
