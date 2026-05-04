#!/usr/bin/env python3
"""
Qwen/Mixtral cached-vs-predict sweep utility with two modes:

1. Simple mode (default):
   Compare cached vs predict with the same cache sizes and predictor path.

2. Comprehensive mode (`--sweep-question ...`):
   Run larger Qwen lookahead-window experiments across baseline, predictor,
   cache-conditioning, and forced-top-N questions, writing CSV output and
   generating plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
QWEN_MODEL_SCRIPT = os.path.join(ROOT_DIR, "unified_llm_w4a16", "qwen3_30B-A3B_w4a16_model.py")
MIXTRAL_MODEL_SCRIPT = os.path.join(ROOT_DIR, "unified_llm_w4a16", "mixtral_8x7B_w4a16_model.py")
TS = time.strftime("%Y%m%d_%H%M%S")

PREFETCH_BUDGETS = {
    1: 8,
    2: 13,
    3: 16,
    4: 20,
    5: 22,
    6: 25,
    7: 27,
    8: 30,
    9: 32,
    10: 34,
    11: 35,
    12: 37,
    13: 38,
    14: 40,
    15: 41,
    16: 42,
}
DEFAULT_QWEN_REUSE_CSV = os.path.join(ROOT_DIR, "expert_predictor", "expert_reuse_qwen3_30b.csv")


def run_subprocess(cmd: List[str], timeout: int, log_file: Optional[str] = None) -> Optional[str]:
    """Run command; optional tee to log_file. Returns merged stdout+stderr."""
    try:
        env = os.environ.copy()
        env.setdefault("HSA_ENABLE_SDMA", "0")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,
        )

        log_fp = open(log_file, "a", encoding="utf-8") if log_file else None
        output_lines: List[str] = []
        timeout_reached = False

        def kill_process() -> None:
            nonlocal timeout_reached
            timeout_reached = True
            try:
                import signal

                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass

        timer = threading.Timer(float(timeout), kill_process)
        timer.start()
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end="")
                    sys.stdout.flush()
                    output_lines.append(line)
                    if log_fp:
                        log_fp.write(line)
                        log_fp.flush()
        finally:
            timer.cancel()
            if log_fp:
                log_fp.close()

        if timeout_reached:
            print(f"[sweep_predict_cached_cache_metrics] Command timed out after {timeout}s", flush=True)
            return None
        if process.returncode != 0:
            print(f"[sweep_predict_cached_cache_metrics] exit code {process.returncode}", flush=True)
            return None
        time.sleep(1.0)
        return "".join(output_lines)
    except Exception as e:
        print(f"[sweep_predict_cached_cache_metrics] Exception: {e}", flush=True)
        return None


def parse_tps(text: str) -> Optional[float]:
    """
    Prefer decode-only TPS emitted by the C++ backends.

    The Python sweep wrapper also prints an "Average Time per Token" summary, but
    that value is derived from end-to-end elapsed time and therefore includes
    prefill. The C++ backends print:
      - Total Generation Time: <sec> seconds
      - Average Time per Token: <sec> seconds
    once per prompt, after prefill has completed. We aggregate those prompt-level
    measurements into a single decode-only TPS.
    """
    totals = [float(x) for x in re.findall(r"Total Generation Time:\s*([\d.eE+-]+)\s+seconds", text)]
    per_token = [float(x) for x in re.findall(r"Average Time per Token:\s*([\d.eE+-]+)\s+seconds", text)]
    if totals and per_token:
        total_generated_est = 0.0
        total_decode_time = 0.0
        for total_s, tpt_s in zip(totals, per_token):
            if total_s > 0 and tpt_s > 0:
                total_generated_est += total_s / tpt_s
                total_decode_time += total_s
        if total_decode_time > 0 and total_generated_est > 0:
            return total_generated_est / total_decode_time

    m_cpp = re.search(r"Average Time per Token:\s*([\d.eE+-]+)\s+seconds", text)
    if m_cpp:
        tpt = float(m_cpp.group(1))
        if tpt > 0:
            return 1.0 / tpt

    m = re.search(r"End-to-End TPS:\s*([\d.]+)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"Average Time per Token:\s*([\d.eE+-]+)", text)
    if m2:
        tpt = float(m2.group(1))
        if tpt > 0:
            return 1.0 / tpt
    return None


def parse_aggregate_cache_line(text: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    # Match only true aggregate lines (not "Layer X Cache Stats: ...").
    m = re.search(r"^Cache Stats:\s*Hits=(\d+),\s*Misses=(\d+),\s*HitRate=([\d.]+)%\s*$", text, re.MULTILINE)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_predictor_aggregate(text: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    # Match only true aggregate lines.
    m = re.search(r"^Predictor Stats:\s*Hits=(\d+),\s*Total=(\d+),\s*HitRate=([\d.]+)%\s*$", text, re.MULTILINE)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_layer_cache_lines(text: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    rows = re.findall(r"Layer\s+\d+\s+Cache Stats:\s*Hits=(\d+),\s*Misses=(\d+),\s*HitRate=([\d.]+)%", text)
    if not rows:
        return None, None, None
    hits = sum(int(r[0]) for r in rows)
    misses = sum(int(r[1]) for r in rows)
    total = hits + misses
    hit_rate = (100.0 * hits / total) if total > 0 else 0.0
    return hits, misses, hit_rate


def parse_predict_bandwidth_lines(text: str) -> Tuple[int, int, Optional[float]]:
    rows = re.findall(
        r"Bandwidth:\s*StallLoads=(\d+),\s*PrefetchLoads=(\d+),\s*AvgLoadTime=([\d.]+)ms",
        text,
    )
    if not rows:
        return 0, 0, None
    stall = sum(int(r[0]) for r in rows)
    pref = sum(int(r[1]) for r in rows)
    num = 0.0
    den = 0
    for s, p, a in rows:
        loads = int(s) + int(p)
        den += loads
        num += float(a) * loads
    avg_ms = (num / den) if den > 0 else None
    return stall, pref, avg_ms


def parse_output(output: str) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "gen_perplexity": None,
        "tokens_per_second": None,
        "cache_hits": None,
        "cache_misses": None,
        "hit_rate_pct": None,
        "pred_hits": None,
        "pred_total": None,
        "pred_hit_rate_pct": None,
        "pred_hits_routed_forced_n": None,
        "pred_total_routed_forced_n": None,
        "pred_hit_rate_routed_forced_n_pct": None,
        "pred_hits_routed_topk": None,
        "pred_total_routed_topk": None,
        "pred_hit_rate_routed_topk_pct": None,
        "pred_requested_hits_forced_n": None,
        "pred_requested_total_forced_n": None,
        "pred_requested_rate_forced_n_pct": None,
        "pred_requested_hits_topk": None,
        "pred_requested_total_topk": None,
        "pred_requested_rate_topk_pct": None,
        "stall_loads": None,
        "prefetch_loads": None,
        "avg_ms_per_expert_load": None,
    }

    ppl_matches = re.findall(r"Generation Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if ppl_matches:
        result["gen_perplexity"] = float(ppl_matches[-1])

    result["tokens_per_second"] = parse_tps(output)

    ch, cm, hr = parse_aggregate_cache_line(output)
    if ch is None:
        # Backward-compatible fallback for outputs that only print per-layer stats.
        ch, cm, hr = parse_layer_cache_lines(output)
    result["cache_hits"] = ch
    result["cache_misses"] = cm
    result["hit_rate_pct"] = hr

    ph, pt, pr = parse_predictor_aggregate(output)
    if ph is not None:
        result["pred_hits"] = ph
        result["pred_total"] = pt
        result["pred_hit_rate_pct"] = pr
    else:
        cpp_matches = re.findall(r"Predictor Routed-Expert HitRate=([\d.]+)% \((\d+)/(\d+)", output)
        if cpp_matches:
            total_hits = sum(int(m[1]) for m in cpp_matches)
            total_eval = sum(int(m[2]) for m in cpp_matches)
            if total_eval > 0:
                result["pred_hits"] = total_hits
                result["pred_total"] = total_eval
                result["pred_hit_rate_pct"] = (100.0 * total_hits) / total_eval

    forced_matches = re.findall(r"Predictor RoutedForcedN HitRate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if forced_matches:
        total_hits = sum(int(m[1]) for m in forced_matches)
        total_eval = sum(int(m[2]) for m in forced_matches)
        if total_eval > 0:
            result["pred_hits_routed_forced_n"] = total_hits
            result["pred_total_routed_forced_n"] = total_eval
            result["pred_hit_rate_routed_forced_n_pct"] = (100.0 * total_hits) / total_eval

    topk_matches = re.findall(r"Predictor RoutedTopK HitRate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if topk_matches:
        total_hits = sum(int(m[1]) for m in topk_matches)
        total_eval = sum(int(m[2]) for m in topk_matches)
        if total_eval > 0:
            result["pred_hits_routed_topk"] = total_hits
            result["pred_total_routed_topk"] = total_eval
            result["pred_hit_rate_routed_topk_pct"] = (100.0 * total_hits) / total_eval

    req_forced_matches = re.findall(r"Predicted RequestedByRouterForcedN Rate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if req_forced_matches:
        total_hits = sum(int(m[1]) for m in req_forced_matches)
        total_eval = sum(int(m[2]) for m in req_forced_matches)
        if total_eval > 0:
            result["pred_requested_hits_forced_n"] = total_hits
            result["pred_requested_total_forced_n"] = total_eval
            result["pred_requested_rate_forced_n_pct"] = (100.0 * total_hits) / total_eval

    req_topk_matches = re.findall(r"Predicted RequestedByRouterTopK Rate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if req_topk_matches:
        total_hits = sum(int(m[1]) for m in req_topk_matches)
        total_eval = sum(int(m[2]) for m in req_topk_matches)
        if total_eval > 0:
            result["pred_requested_hits_topk"] = total_hits
            result["pred_requested_total_topk"] = total_eval
            result["pred_requested_rate_topk_pct"] = (100.0 * total_hits) / total_eval

    stall, pref, avg_load = parse_predict_bandwidth_lines(output)
    if stall or pref or avg_load is not None:
        result["stall_loads"] = stall
        result["prefetch_loads"] = pref
        result["avg_ms_per_expert_load"] = avg_load

    return result


def calculate_fair_metrics(csv_path: str, predictor_path: str) -> Tuple[int, int]:
    window_size = 1
    match = re.search(r"f(\d+)", predictor_path)
    if match:
        window_size = int(match.group(1))
        print(f"[Calibration] Detected window_size {window_size} from predictor path")
    else:
        print(f"[Calibration] Warning: Could not detect window_size from path '{predictor_path}', defaulting to f1")

    counts = None
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["window_size"]) == window_size:
                    counts = [float(v) for k, v in row.items() if k.startswith("layer_") and v != ""]
                    break
    except Exception as e:
        print(f"[Calibration] Error reading CSV {csv_path}: {e}")

    if not counts:
        print(f"[Calibration] Warning: window_size {window_size} not found in CSV, using fallback 8")
        return 8, 8

    fair_cache = int(math.ceil(1.2 * max(counts)))
    fair_prefetch = int(round(sum(counts) / len(counts)))
    return fair_cache, fair_prefetch


def load_min_cache_per_lookahead(csv_path: str) -> Dict[int, int]:
    result: Dict[int, int] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                window = int(row["window_size"])
                layer_vals = [float(v) for k, v in row.items() if k != "window_size" and v != ""]
                if layer_vals:
                    result[window] = int(math.ceil(max(layer_vals)))
    except FileNotFoundError:
        print(
            f"[sweep] WARNING: Expert reuse CSV not found at {csv_path}; cache/lookahead constraint disabled.",
            flush=True,
        )
    return result


def drop_page_cache() -> None:
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
            f.write("3\n")
        print("[sweep] Dropped OS page cache.", flush=True)
    except PermissionError:
        print("[sweep] Could not drop page cache (permission denied).", flush=True)
    except OSError as e:
        print(f"[sweep] drop_caches not available: {e}", flush=True)


@dataclass
class SimpleRunRow:
    ts: str
    model: str
    backend: str
    cache_size: int
    prefetch_count: Optional[int]
    lambda_val: float
    tps: Optional[float]
    cache_hits: Optional[int]
    cache_misses: Optional[int]
    hit_rate_pct: Optional[float]
    pred_hits: Optional[int]
    pred_total: Optional[int]
    pred_hit_rate_pct: Optional[float]
    pred_hits_routed_forced_n: Optional[int]
    pred_total_routed_forced_n: Optional[int]
    pred_hit_rate_routed_forced_n_pct: Optional[float]
    pred_hits_routed_topk: Optional[int]
    pred_total_routed_topk: Optional[int]
    pred_hit_rate_routed_topk_pct: Optional[float]
    pred_requested_hits_forced_n: Optional[int]
    pred_requested_total_forced_n: Optional[int]
    pred_requested_rate_forced_n_pct: Optional[float]
    pred_requested_hits_topk: Optional[int]
    pred_requested_total_topk: Optional[int]
    pred_requested_rate_topk_pct: Optional[float]
    stall_loads: Optional[int]
    prefetch_loads: Optional[int]
    avg_ms_per_expert_load: Optional[float]
    ok: bool


def build_model_cmd(
    *,
    model: str,
    backend: str,
    cache_size: int,
    lambda_val: float,
    prompts_json: str,
    predictor_path: str = "",
    prefetch_count: Optional[int] = None,
    predict_layers: Optional[List[int]] = None,
    forced_top_n: int = 0,
    prob_mass_threshold: float = -1.0,
    forced_top_p: float = -1.0,
    config_path: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 80,
    mode_generate: bool = True,
    mode_generation_perplexity: bool = False,
    expert_reuse_csv: Optional[str] = None,
    predictor_device: Optional[str] = None,
    expert_weights_dir: Optional[str] = None,
) -> List[str]:
    script = QWEN_MODEL_SCRIPT if model == "qwen" else MIXTRAL_MODEL_SCRIPT
    cmd: List[str] = [
        sys.executable,
        script,
        "--backend",
        backend,
        "--device",
        "cuda",
        "--lambda-val",
        str(lambda_val),
        "--sweep-prompts-file",
        prompts_json,
    ]
    if predictor_device:
        cmd.extend(["--predictor-device", predictor_device])

    if mode_generation_perplexity:
        cmd.extend(["--generation-perplexity", "--no-generate"])
    elif mode_generate:
        cmd.extend(
            [
                "--generate",
                "--max-new-tokens",
                str(max_new_tokens),
                "--temperature",
                str(temperature),
                "--top-p",
                str(top_p),
                "--top-k",
                str(top_k),
            ]
        )

    if model == "qwen":
        cmd.extend(["--max-cached-experts", str(cache_size)])
    else:
        cmd.extend(["--expert-cache", str(cache_size)])

    if backend == "predict":
        if prefetch_count is not None:
            cmd.extend(["--prefetch-experts-count", str(prefetch_count)])
        if predictor_path:
            cmd.extend(["--predictor-model", predictor_path])
            # Auto-detect lookahead depth from the fN suffix in the predictor path
            # and pass it so the C++ backend fires the predictor every N tokens.
            m_la = re.search(r"f(\d+)", os.path.basename(os.path.normpath(predictor_path)))
            if m_la:
                cmd.extend(["--predictor-lookahead", m_la.group(1)])
        if predict_layers is not None:
            cmd.append("--predict-layers")
            cmd.extend(str(x) for x in predict_layers)

    if forced_top_n > 0:
        cmd.extend(["--forced-top-n", str(forced_top_n)])
    # Preferred interface: probability-mass routing threshold.
    # Backward compatibility: if legacy forced_top_p is set, treat it as the same threshold.
    effective_prob_mass = prob_mass_threshold if prob_mass_threshold >= 0.0 else forced_top_p
    if effective_prob_mass >= 0.0:
        cmd.extend(["--mass-threshold-substitution-p", str(effective_prob_mass)])
    if config_path:
        cmd.extend(["--config-path", config_path])
    if expert_reuse_csv:
        cmd.extend(["--expert-reuse-csv", expert_reuse_csv])
    if expert_weights_dir:
        cmd.extend(["--expert-weights-dir", expert_weights_dir])
    return cmd


def _is_wikitext_header(text: str) -> bool:
    """Return True for wikitext section/article title lines (e.g. ' = Title = \\n')."""
    stripped = text.strip()
    return stripped.startswith("=") and stripped.endswith("=")


def load_prompts(args: argparse.Namespace) -> List[str]:
    dataset = args.dataset
    n = args.num_prompts
    max_chars: int = getattr(args, "prompt_max_chars", 500)

    if dataset == "wikitext":
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)

            # Concatenate the full corpus into one blob (drop headers and blank lines),
            # split by "\n\n" to recover paragraph boundaries, then greedily pack
            # paragraphs into fixed-size chunks of `max_chars` characters.
            # At ~4 chars/token this gives ~1024-token context windows when
            # --prompt-max-chars is left at its default of 4096.
            blob_parts: List[str] = []
            for item in ds:
                text = item.get("text", "").strip()
                if not text or _is_wikitext_header(text):
                    continue
                blob_parts.append(text)

            blob = "\n\n".join(blob_parts)
            paragraphs = [p.strip() for p in blob.split("\n\n") if p.strip()]

            prompts: List[str] = []
            current: str = ""
            for para in paragraphs:
                candidate = (current + "\n\n" + para) if current else para
                if len(candidate) >= max_chars:
                    if current:
                        prompts.append(current[:max_chars])
                        if len(prompts) >= n:
                            break
                    # paragraph itself may exceed max_chars — chunk it directly
                    while len(para) >= max_chars:
                        prompts.append(para[:max_chars])
                        para = para[max_chars:]
                        if len(prompts) >= n:
                            break
                    current = para
                else:
                    current = candidate
            if current and len(prompts) < n:
                prompts.append(current[:max_chars])

            if prompts:
                print(
                    f"[sweep] Loaded {len(prompts)} wikitext chunk(s), "
                    f"chunk_chars={max_chars} (~{max_chars // 4} tokens)",
                    flush=True,
                )
                return prompts
        except Exception as e:
            print(f"[sweep] Could not load wikitext ({e}), using default.", flush=True)

    if dataset == "fineweb":
        try:
            from datasets import load_dataset

            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            prompts = []
            for item in ds:
                text = item.get("text", "")
                if len(text) > 100:
                    prompts.append(text[:max_chars])
                if len(prompts) >= n:
                    break
            if prompts:
                return prompts
        except Exception as e:
            print(f"[sweep] fineweb load failed: {e}", flush=True)

    if dataset == "orca":
        try:
            from datasets import load_dataset

            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            prompts = []
            for item in ds:
                text = f"{item.get('system_prompt', '')}\n{item.get('question', '')}".strip()
                if len(text) > 100:
                    prompts.append(text[:max_chars])
                if len(prompts) >= n:
                    break
            if prompts:
                return prompts
        except Exception as e:
            print(f"[sweep] orca load failed: {e}", flush=True)

    if dataset == "txt":
        path = args.prompts_txt or os.path.join(ROOT_DIR, "unified_llm_w4a16", "prompts.txt")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        prompts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        return prompts[:n]

    default = (
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    return [default] * n


def predictor_path_for_lookahead(base_dir: str, lookahead: int) -> str:
    return os.path.join(os.path.abspath(base_dir), f"eh1_h32_f{lookahead}")


def cache_large_enough(cache_size: int, lookahead: int, min_cache_map: Dict[int, int]) -> bool:
    required = min_cache_map.get(lookahead)
    if required is None:
        return True
    if cache_size >= required:
        return True
    print(
        f"[sweep] Skipping cache_size={cache_size} / lookahead={lookahead}: requires cache >= {required}",
        flush=True,
    )
    return False


def cfg(
    question: str,
    label: str,
    backend: str,
    cache_size: int,
    lambda_val: float,
    forced_top_n: int,
    *,
    prob_mass_threshold: float = -1.0,
    forced_top_p: float = -1.0,
    lookahead: Optional[int] = None,
    prefetch_budget: Optional[int] = None,
    mode_perplexity: bool = True,
    mode_generate: bool = True,
) -> Dict[str, Any]:
    return {
        "question": question,
        "label": label,
        "backend": backend,
        "cache_size": cache_size,
        "lambda_val": lambda_val,
        "forced_top_n": forced_top_n,
        # Canonical field for alternate probability-mass routing mode.
        "prob_mass_threshold": prob_mass_threshold if prob_mass_threshold >= 0.0 else forced_top_p,
        # Legacy compatibility column (kept for existing plots/readers).
        "forced_top_p": prob_mass_threshold if prob_mass_threshold >= 0.0 else forced_top_p,
        "lookahead": lookahead,
        "prefetch_budget": prefetch_budget,
        "mode_perplexity": mode_perplexity,
        "mode_generate": mode_generate,
    }


def baseline_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [cfg("BASELINE", "LRU-Baseline", "cached", cs, 0.0, 8) for cs in args.cache_sizes]


def predict_configs(args: argparse.Namespace, min_cache_map: Dict[int, int]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    lookaheads = args.lookaheads or list(range(1, 17))
    for cache_size in args.cache_sizes:
        for lookahead in lookaheads:
            predictor_path = predictor_path_for_lookahead(args.predictor_base_dir, lookahead)
            if not os.path.isdir(predictor_path):
                print(f"[sweep] Warning: predictor path missing for LA={lookahead}: {predictor_path}", flush=True)
                continue
            if not cache_large_enough(cache_size, lookahead, min_cache_map):
                continue
            if args.budget_equals_cache_size:
                budgets = [cache_size]
            else:
                budgets = [b for b in args.prefetch_budgets if b <= cache_size]
            for budget in budgets:
                for lambda_val in args.lambdas:
                    configs.append(
                        cfg(
                            "PREDICT",
                            f"Predict LA={lookahead} B={budget}",
                            "predict",
                            cache_size,
                            lambda_val,
                            args.routing_bias_top_n,
                            lookahead=lookahead,
                            prefetch_budget=budget,
                        )
                    )
    return configs


def cache_cond_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for cache_size in args.cache_sizes:
        for lambda_val in args.cache_cond_lambdas:
            for forced_top_n in args.cache_cond_forced_top_ns:
                if forced_top_n == 8 and lambda_val == 0.0:
                    continue
                configs.append(
                    cfg(
                        "CACHE_COND",
                        f"CacheCond FN={forced_top_n}",
                        "cached",
                        cache_size,
                        lambda_val,
                        forced_top_n,
                        mode_perplexity=True,
                        mode_generate=False,
                    )
                )
    return configs


def forced_n_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for cache_size in args.cache_sizes:
        for lambda_val in args.cache_cond_lambdas:
            for forced_top_n in args.forced_top_ns:
                if forced_top_n == 8 and lambda_val == 0.0:
                    continue
                configs.append(
                    cfg(
                        "FORCED_N",
                        f"Lambda={lambda_val}",
                        "cached",
                        cache_size,
                        lambda_val,
                        forced_top_n,
                        mode_perplexity=True,
                        mode_generate=False,
                    )
                )
    return configs


def custom_1_16_configs(args: argparse.Namespace, predictor_base_dir: Optional[str] = None, predictor_tag: str = "") -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    selected_lookaheads = args.lookaheads if args.lookaheads else [1, 2, 4, 6, 8, 16]
    do_ppl = getattr(args, "custom_include_ppl", False)
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""
    for cache_size in args.cache_sizes:
        for lookahead in selected_lookaheads:
            # LRU baseline: one per (cache_size, lookahead), independent of budget fraction
            # Only emit if no predictor_tag (avoid duplicated baselines across predictors)
            if not predictor_tag:
                configs.append(cfg(
                    "CUSTOM_1_16", "Neither (LRU)", "cached",
                    cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=None,
                    mode_perplexity=do_ppl, mode_generate=True,
                ))
            for fraction in args.budget_fractions:
                budget = max(1, int(cache_size * fraction))
                c = cfg(
                    "CUSTOM_1_16", f"Prefetch Only B={budget}{tag_suffix}",
                    "predict", cache_size, 0.0,
                    0,
                    lookahead=lookahead, prefetch_budget=budget,
                    mode_perplexity=do_ppl, mode_generate=True,
                )
                if predictor_base_dir:
                    c["predictor_base_dir"] = predictor_base_dir
                    c["predictor_tag"] = predictor_tag
                configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    cc = cfg(
                        "CUSTOM_1_16", f"Cache-Cond Only λ={lambda_val} B={budget}{tag_suffix}",
                        "cached", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=do_ppl, mode_generate=True,
                    )
                    if predictor_base_dir:
                        cc["predictor_base_dir"] = predictor_base_dir
                        cc["predictor_tag"] = predictor_tag
                    configs.append(cc)
                    bc = cfg(
                        "CUSTOM_1_16", f"Both λ={lambda_val} B={budget}{tag_suffix}",
                        "predict", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=do_ppl, mode_generate=True,
                    )
                    if predictor_base_dir:
                        bc["predictor_base_dir"] = predictor_base_dir
                        bc["predictor_tag"] = predictor_tag
                    configs.append(bc)
    return configs


def custom_1_16_no_ppl_configs(args: argparse.Namespace, predictor_base_dir: Optional[str] = None, predictor_tag: str = "") -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    selected_lookaheads = [1, 2, 4, 6, 8, 12, 16]
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""
    for cache_size in args.cache_sizes:
        for lookahead in selected_lookaheads:
            if not predictor_tag:
                configs.append(cfg(
                    "CUSTOM_1_16_NO_PPL", "Neither (LRU)", "cached",
                    cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=None,
                    mode_perplexity=False, mode_generate=True,
                ))
            for fraction in args.budget_fractions:
                budget = max(1, int(cache_size * fraction))
                c = cfg(
                    "CUSTOM_1_16_NO_PPL", f"Prefetch Only B={budget}{tag_suffix}",
                    "predict", cache_size, 0.0,
                    0,
                    lookahead=lookahead, prefetch_budget=budget,
                    mode_perplexity=False, mode_generate=True,
                )
                if predictor_base_dir:
                    c["predictor_base_dir"] = predictor_base_dir
                    c["predictor_tag"] = predictor_tag
                configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    cc = cfg(
                        "CUSTOM_1_16_NO_PPL", f"Cache-Cond Only λ={lambda_val} B={budget}{tag_suffix}",
                        "cached", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=False, mode_generate=True,
                    )
                    if predictor_base_dir:
                        cc["predictor_base_dir"] = predictor_base_dir
                        cc["predictor_tag"] = predictor_tag
                    configs.append(cc)
                    bc = cfg(
                        "CUSTOM_1_16_NO_PPL", f"Both λ={lambda_val} B={budget}{tag_suffix}",
                        "predict", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=False, mode_generate=True,
                    )
                    if predictor_base_dir:
                        bc["predictor_base_dir"] = predictor_base_dir
                        bc["predictor_tag"] = predictor_tag
                    configs.append(bc)
    return configs


def custom_1_16_ppl_only_configs(args: argparse.Namespace, predictor_base_dir: Optional[str] = None, predictor_tag: str = "") -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    selected_lookaheads = [1, 2, 4, 6, 8, 12, 16]
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""
    for cache_size in args.cache_sizes:
        for lookahead in selected_lookaheads:
            if not predictor_tag:
                configs.append(cfg(
                    "CUSTOM_1_16_PPL_ONLY", "Neither (LRU)", "cached",
                    cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=None,
                    mode_perplexity=True, mode_generate=False,
                ))
            for fraction in args.budget_fractions:
                budget = max(1, int(cache_size * fraction))
                c = cfg(
                    "CUSTOM_1_16_PPL_ONLY", f"Prefetch Only B={budget}{tag_suffix}",
                    "predict", cache_size, 0.0,
                    0,
                    lookahead=lookahead, prefetch_budget=budget,
                    mode_perplexity=True, mode_generate=False,
                )
                if predictor_base_dir:
                    c["predictor_base_dir"] = predictor_base_dir
                    c["predictor_tag"] = predictor_tag
                configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    cc = cfg(
                        "CUSTOM_1_16_PPL_ONLY", f"Cache-Cond Only λ={lambda_val} B={budget}{tag_suffix}",
                        "cached", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=True, mode_generate=False,
                    )
                    if predictor_base_dir:
                        cc["predictor_base_dir"] = predictor_base_dir
                        cc["predictor_tag"] = predictor_tag
                    configs.append(cc)
                    bc = cfg(
                        "CUSTOM_1_16_PPL_ONLY", f"Both λ={lambda_val} B={budget}{tag_suffix}",
                        "predict", cache_size, lambda_val,
                        args.routing_bias_top_n,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=True, mode_generate=False,
                    )
                    if predictor_base_dir:
                        bc["predictor_base_dir"] = predictor_base_dir
                        bc["predictor_tag"] = predictor_tag
                    configs.append(bc)
    return configs


def lambda_fn_sweep_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Sweep λ × forced_top_n × probability_mass_threshold at representative lookahead depths.

    Generates Cache-Cond Only (cached backend) configs to isolate the routing policy
    effect independently of prefetching. Also includes the LRU baseline per (cache_size, lookahead).
    Designed to produce a 2-D heatmap of TPS and PPL as a function of (λ, forcing policy).
    """
    configs: List[Dict[str, Any]] = []
    lookaheads = args.lookaheads or [4, 8]
    for cache_size in args.cache_sizes:
        for lookahead in lookaheads:
            # LRU baseline: one per (cache_size, lookahead), independent of budget fraction
            configs.append(cfg(
                "LAMBDA_FN_SWEEP", "LRU Baseline", "cached",
                cache_size, 0.0, 0,
                lookahead=lookahead, prefetch_budget=None,
                # Run generation perplexity pass + generation (TPS); merge fills gen_perplexity + tokens_per_second.
                mode_perplexity=True, mode_generate=True,
            ))

            for fraction in args.budget_fractions:
                budget = max(1, int(cache_size * fraction))

                # λ × forced_top_n grid (count-based forcing)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    for fn in args.cache_cond_forced_top_ns:
                        configs.append(cfg(
                            "LAMBDA_FN_SWEEP", f"CacheCond λ={lambda_val} FN={fn} B={budget}",
                            "cached", cache_size, lambda_val, fn,
                            lookahead=lookahead, prefetch_budget=budget,
                            mode_perplexity=True, mode_generate=True,
                        ))

                # λ × probability-mass threshold grid
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    for fp in args.probability_mass_thresholds:
                        configs.append(cfg(
                            "LAMBDA_FN_SWEEP", f"CacheCond λ={lambda_val} PM={fp} B={budget}",
                            "cached", cache_size, lambda_val, 0,
                            prob_mass_threshold=fp,
                            lookahead=lookahead, prefetch_budget=budget,
                            mode_perplexity=True, mode_generate=True,
                        ))

    return configs


def probability_mass_calibration_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Sweep probability-mass threshold finely in PPL mode to find the minimum p that keeps
    perplexity degradation under a target threshold (e.g. 5%).

    Generates:
      - LRU baseline (no forcing, no lambda) — reference PPL
      - Cache-Cond Only at each (lambda, probability_mass_threshold, budget_fraction) triple in PPL mode

    Use --probability-mass-thresholds with a fine grid, e.g.:
      --probability-mass-thresholds 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99
    """
    configs: List[Dict[str, Any]] = []
    lookaheads = args.lookaheads or [4, 8]
    for cache_size in args.cache_sizes:
        for lookahead in lookaheads:
            # LRU baseline — reference PPL, no forcing, no lambda
            configs.append(cfg(
                "PROBABILITY_MASS_CALIBRATION", "LRU Baseline", "cached",
                cache_size, 0.0, 0,
                lookahead=lookahead, prefetch_budget=None,
                mode_perplexity=True, mode_generate=False,
            ))

            for fraction in args.budget_fractions:
                budget = max(1, int(cache_size * fraction))
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    for fp in args.probability_mass_thresholds:
                        configs.append(cfg(
                            "PROBABILITY_MASS_CALIBRATION", f"λ={lambda_val} PM={fp} B={budget}",
                            "cached", cache_size, lambda_val, 0,
                            prob_mass_threshold=fp,
                            lookahead=lookahead, prefetch_budget=budget,
                            mode_perplexity=True, mode_generate=False,
                        ))

    return configs


def cfg_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    prob_mass_threshold = config.get("prob_mass_threshold", config.get("forced_top_p", -1.0))
    return (
        config["question"],
        config["backend"],
        config["cache_size"],
        config["lambda_val"],
        config["forced_top_n"],
        prob_mass_threshold,
        config.get("lookahead"),
        config.get("prefetch_budget"),
        config.get("mode_perplexity"),
        config.get("mode_generate"),
    )


def row_has_data(row: Dict[str, Any]) -> bool:
    for col in (
        "gen_perplexity",
        "tokens_per_second",
        "hit_rate_pct",
        "pred_hit_rate_pct",
        "pred_hit_rate_routed_forced_n_pct",
        "pred_hit_rate_routed_topk_pct",
        "pred_requested_rate_forced_n_pct",
        "pred_requested_rate_topk_pct",
    ):
        value = row.get(col)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return True
    return False


def aggregate_cold_prompt_metrics(outputs: Iterable[str]) -> Dict[str, Optional[float]]:
    metrics_list = [parse_output(text) for text in outputs]
    good = [m for m in metrics_list if any(v is not None for v in m.values())]
    if not good:
        return parse_output("")

    result: Dict[str, Optional[float]] = {k: None for k in good[0].keys()}
    for key in (
        "gen_perplexity",
        "tokens_per_second",
        "hit_rate_pct",
        "pred_hit_rate_pct",
        "pred_hit_rate_routed_forced_n_pct",
        "pred_hit_rate_routed_topk_pct",
        "pred_requested_rate_forced_n_pct",
        "pred_requested_rate_topk_pct",
        "avg_ms_per_expert_load",
    ):
        values = [float(m[key]) for m in good if m.get(key) is not None]
        if values:
            result[key] = sum(values) / len(values)

    for key in (
        "cache_hits",
        "cache_misses",
        "pred_hits",
        "pred_total",
        "pred_hits_routed_forced_n",
        "pred_total_routed_forced_n",
        "pred_hits_routed_topk",
        "pred_total_routed_topk",
        "pred_requested_hits_forced_n",
        "pred_requested_total_forced_n",
        "pred_requested_hits_topk",
        "pred_requested_total_topk",
        "stall_loads",
        "prefetch_loads",
    ):
        values = [int(m[key]) for m in good if m.get(key) is not None]
        if values:
            result[key] = sum(values)

    return result


def execute_comprehensive_run(
    config: Dict[str, Any],
    prompts_file: str,
    prompts: List[str],
    args: argparse.Namespace,
    run_idx: int,
    total_runs: int,
) -> Dict[str, Any]:
    cache_size = config["cache_size"]
    lambda_val = config["lambda_val"]
    forced_top_n = config["forced_top_n"]
    prob_mass_threshold = config.get("prob_mass_threshold", config.get("forced_top_p", -1.0))
    lookahead = config.get("lookahead")
    budget = config.get("prefetch_budget")
    predictor_tag = config.get("predictor_tag", "")
    pm_str = f" PM={prob_mass_threshold}" if prob_mass_threshold >= 0.0 else ""
    tag_str = f" predictor={predictor_tag}" if predictor_tag else ""
    print(
        f"\n{'=' * 18} {config['question']} | {config['label']} | "
        f"C={cache_size} λ={lambda_val} FN={forced_top_n}{pm_str}"
        f"{f' LA={lookahead} B={budget}' if lookahead is not None else ''}{tag_str} "
        f"[{run_idx + 1}/{total_runs}] {'=' * 18}",
        flush=True,
    )

    # Use per-config predictor_base_dir override if present (multi-predictor comparison)
    effective_predictor_base_dir = config.get("predictor_base_dir", args.predictor_base_dir)
    predictor_path = ""
    if config["backend"] == "predict":
        assert lookahead is not None
        predictor_path = predictor_path_for_lookahead(effective_predictor_base_dir, lookahead)

    def run_pass(*, mode_perplexity: bool, mode_generate: bool, one_prompt_path: str) -> Optional[Dict[str, Optional[float]]]:
        cmd = build_model_cmd(
            model="qwen",
            backend=config["backend"],
            cache_size=cache_size,
            lambda_val=lambda_val,
            prompts_json=one_prompt_path,
            predictor_path=predictor_path,
            prefetch_count=budget,
            predict_layers=args.predict_layers,
            forced_top_n=forced_top_n,
            prob_mass_threshold=prob_mass_threshold,
            config_path=args.config_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            mode_generate=mode_generate,
            mode_generation_perplexity=mode_perplexity,
            predictor_device=args.predictor_device,
            expert_weights_dir=getattr(args, "expert_weights_dir", None),
        )
        out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
        if out is None:
            return None
        return parse_output(out)

    merged: Dict[str, Optional[float]] = {
        "gen_perplexity": None,
        "tokens_per_second": None,
        "cache_hits": None,
        "cache_misses": None,
        "hit_rate_pct": None,
        "pred_hits": None,
        "pred_total": None,
        "pred_hit_rate_pct": None,
        "pred_hits_routed_forced_n": None,
        "pred_total_routed_forced_n": None,
        "pred_hit_rate_routed_forced_n_pct": None,
        "pred_hits_routed_topk": None,
        "pred_total_routed_topk": None,
        "pred_hit_rate_routed_topk_pct": None,
        "pred_requested_hits_forced_n": None,
        "pred_requested_total_forced_n": None,
        "pred_requested_rate_forced_n_pct": None,
        "pred_requested_hits_topk": None,
        "pred_requested_total_topk": None,
        "pred_requested_rate_topk_pct": None,
        "stall_loads": None,
        "prefetch_loads": None,
        "avg_ms_per_expert_load": None,
    }

    def merge_metrics(metrics: Optional[Dict[str, Optional[float]]]) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if value is not None:
                merged[key] = value

    if not args.cold_per_prompt:
        if config.get("mode_perplexity", False):
            merge_metrics(run_pass(mode_perplexity=True, mode_generate=False, one_prompt_path=prompts_file))
        if config.get("mode_generate", False):
            merge_metrics(run_pass(mode_perplexity=False, mode_generate=True, one_prompt_path=prompts_file))
    else:
        if config.get("mode_perplexity", False):
            outputs: List[str] = []
            for prompt in prompts:
                if args.drop_page_cache_between_prompts:
                    drop_page_cache()
                fd, one_prompt_path = tempfile.mkstemp(suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump([prompt], f)
                    cmd = build_model_cmd(
                        model="qwen",
                        backend=config["backend"],
                        cache_size=cache_size,
                        lambda_val=lambda_val,
                        prompts_json=one_prompt_path,
                        predictor_path=predictor_path,
                        prefetch_count=budget,
                        predict_layers=args.predict_layers,
                        forced_top_n=forced_top_n,
                        prob_mass_threshold=prob_mass_threshold,
                        config_path=args.config_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        mode_generate=False,
                        mode_generation_perplexity=True,
                        predictor_device=args.predictor_device,
                        expert_weights_dir=getattr(args, "expert_weights_dir", None),
                    )
                    out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
                    if out:
                        outputs.append(out)
                finally:
                    try:
                        os.remove(one_prompt_path)
                    except OSError:
                        pass
            merge_metrics(aggregate_cold_prompt_metrics(outputs))

        if config.get("mode_generate", False):
            outputs = []
            for prompt in prompts:
                if args.drop_page_cache_between_prompts:
                    drop_page_cache()
                fd, one_prompt_path = tempfile.mkstemp(suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump([prompt], f)
                    cmd = build_model_cmd(
                        model="qwen",
                        backend=config["backend"],
                        cache_size=cache_size,
                        lambda_val=lambda_val,
                        prompts_json=one_prompt_path,
                        predictor_path=predictor_path,
                        prefetch_count=budget,
                        predict_layers=args.predict_layers,
                        forced_top_n=forced_top_n,
                        prob_mass_threshold=prob_mass_threshold,
                        config_path=args.config_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        mode_generate=True,
                        mode_generation_perplexity=False,
                        predictor_device=args.predictor_device,
                        expert_weights_dir=getattr(args, "expert_weights_dir", None),
                    )
                    out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
                    if out:
                        outputs.append(out)
                finally:
                    try:
                        os.remove(one_prompt_path)
                    except OSError:
                        pass
            merge_metrics(aggregate_cold_prompt_metrics(outputs))

    return {
        "question": config["question"],
        "label": config["label"],
        "backend": config["backend"],
        "cache_size": cache_size,
        "lambda_val": lambda_val,
        "forced_top_n": forced_top_n,
        "prob_mass_threshold": prob_mass_threshold,
        # Keep legacy column for downstream scripts.
        "forced_top_p": prob_mass_threshold,
        "lookahead": lookahead,
        "prefetch_budget": budget,
        "predictor_tag": predictor_tag,
        "mode_perplexity": config.get("mode_perplexity", False),
        "mode_generate": config.get("mode_generate", False),
        **merged,
    }


PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d",
    "axes.labelcolor": "#e0e4f0",
    "xtick.color": "#a0a8c0",
    "ytick.color": "#a0a8c0",
    "text.color": "#e0e4f0",
    "grid.color": "#2a2d3d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4d",
}
ACCENT = ["#4fc3f7", "#81d4fa", "#ef9a9a", "#ffcc80", "#a5d6a7", "#ce93d8", "#80cbc4"]
LOOKAHEAD_COLORS = {1: "#4fc3f7", 3: "#81d4fa", 5: "#a5d6a7", 8: "#ffcc80", 12: "#ce93d8", 16: "#ef9a9a"}
BUDGET_MARKERS = {8: "o", 16: "s", 32: "^", 48: "D"}
LRU_COLOR = "#ffffff"
LRU_STYLE = dict(color=LRU_COLOR, linestyle="--", linewidth=2.5, zorder=10)


def lookahead_color(lookahead: int) -> str:
    return LOOKAHEAD_COLORS.get(int(lookahead), "#80cbc4")


def budget_marker(budget: int) -> str:
    return BUDGET_MARKERS.get(int(budget), "x")


def style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, pad=8, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True)
    ax.legend(fontsize=7, loc="best")


def base_lookup(df_base: Optional[pd.DataFrame], metric: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if df_base is None:
        return result
    for _, row in df_base[df_base[metric].notna()].iterrows():
        result[int(row["cache_size"])] = float(row[metric])
    return result


def save_plot(fig: plt.Figure, out_dir: str, filename: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}", flush=True)


def have_plot_deps() -> bool:
    return plt is not None and np is not None


def plot_baseline(df_base: pd.DataFrame, out_dir: str, ts: str) -> None:
    df_tps = df_base[df_base["tokens_per_second"].notna()].sort_values("cache_size")
    df_ppl = df_base[df_base["gen_perplexity"].notna()].sort_values("cache_size")
    if df_tps.empty and df_ppl.empty:
        return

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("LRU Baseline (λ=0, FN=8) — TPS and PPL vs Cache Size", fontsize=13, fontweight="bold", y=1.02)
        if not df_tps.empty:
            axes[0].plot(df_tps["cache_size"], df_tps["tokens_per_second"], marker="D", **LRU_STYLE, label="LRU-Baseline")
        style_ax(axes[0], "Tokens per Second", "Cache size (experts/layer)", "TPS")
        if not df_ppl.empty:
            axes[1].plot(df_ppl["cache_size"], df_ppl["gen_perplexity"], marker="D", **LRU_STYLE, label="LRU-Baseline")
        style_ax(axes[1], "Generation Perplexity", "Cache size (experts/layer)", "PPL")
        plt.tight_layout()
        save_plot(fig, out_dir, f"baseline_tps_ppl_{ts}.png")


def plot_tps_vs_cache(df_predict: pd.DataFrame, df_base: Optional[pd.DataFrame], out_dir: str, ts: str) -> None:
    df = df_predict[df_predict["tokens_per_second"].notna()].copy()
    if df.empty:
        return
    base_tps = base_lookup(df_base, "tokens_per_second")
    lookaheads = sorted(df["lookahead"].dropna().unique(), key=int)
    lambdas = sorted(df["lambda_val"].unique())
    budgets = sorted(df["prefetch_budget"].dropna().unique(), key=int)
    ncols = min(len(lookaheads), 3)
    nrows = math.ceil(len(lookaheads) / ncols)

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("TPS vs Cache Size: Predictor vs LRU Baseline", fontsize=13, fontweight="bold", y=1.01)
        for idx, lookahead in enumerate(lookaheads):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["lookahead"] == lookahead]
            if base_tps:
                xs = sorted(base_tps.keys())
                ys = [base_tps[x] for x in xs]
                ax.plot(xs, ys, label="LRU-Baseline", **LRU_STYLE)
            for lambda_val in lambdas:
                for budget in budgets:
                    s = sub[(sub["lambda_val"] == lambda_val) & (sub["prefetch_budget"] == budget)].sort_values("cache_size")
                    if s.empty:
                        continue
                    alpha = 0.5 + 0.5 * (lambda_val / max(lambdas)) if max(lambdas) > 0 else 1.0
                    ax.plot(
                        s["cache_size"],
                        s["tokens_per_second"],
                        marker=budget_marker(int(budget)),
                        color=lookahead_color(int(lookahead)),
                        alpha=alpha,
                        linewidth=1.8,
                        markersize=6,
                        label=f"λ={lambda_val} B={int(budget)}",
                    )
            style_ax(ax, f"Lookahead = {int(lookahead)} tokens", "Cache size (experts/layer)", "Tokens / second")
        for idx in range(len(lookaheads), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"tps_vs_cache_{ts}.png")


def plot_ppl_vs_cache(df_predict: pd.DataFrame, df_base: Optional[pd.DataFrame], out_dir: str, ts: str) -> None:
    df = df_predict[df_predict["gen_perplexity"].notna()].copy()
    if df.empty:
        return
    base_ppl = base_lookup(df_base, "gen_perplexity")
    if not base_ppl:
        return
    lookaheads = sorted(df["lookahead"].dropna().unique(), key=int)
    lambdas = sorted(df["lambda_val"].unique())
    budgets = sorted(df["prefetch_budget"].dropna().unique(), key=int)
    ncols = min(len(lookaheads), 3)
    nrows = math.ceil(len(lookaheads) / ncols)

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("PPL Degradation vs Cache Size (relative to LRU Baseline)", fontsize=13, fontweight="bold", y=1.01)
        for idx, lookahead in enumerate(lookaheads):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["lookahead"] == lookahead]
            for pct, style, color in ((1, ":", "#a5d6a7"), (2, "--", "#ffcc80"), (5, "-.", "#ef9a9a")):
                ax.axhline(pct, linestyle=style, color=color, linewidth=1, alpha=0.7, label=f"+{pct}% threshold")
            ax.axhline(0, color=LRU_COLOR, linewidth=1.2, alpha=0.4, label="Baseline (0%)")
            for lambda_val in lambdas:
                for budget in budgets:
                    s = sub[(sub["lambda_val"] == lambda_val) & (sub["prefetch_budget"] == budget)].sort_values("cache_size")
                    xs: List[int] = []
                    ys: List[float] = []
                    for _, row in s.iterrows():
                        cache_size = int(row["cache_size"])
                        if cache_size not in base_ppl:
                            continue
                        xs.append(cache_size)
                        ys.append((float(row["gen_perplexity"]) - base_ppl[cache_size]) / base_ppl[cache_size] * 100.0)
                    if xs:
                        alpha = 0.5 + 0.5 * (lambda_val / max(lambdas)) if max(lambdas) > 0 else 1.0
                        ax.plot(
                            xs,
                            ys,
                            marker=budget_marker(int(budget)),
                            color=lookahead_color(int(lookahead)),
                            alpha=alpha,
                            linewidth=1.8,
                            markersize=6,
                            label=f"λ={lambda_val} B={int(budget)}",
                        )
            style_ax(ax, f"Lookahead = {int(lookahead)} tokens", "Cache size (experts/layer)", "PPL degradation vs LRU (%)")
        for idx in range(len(lookaheads), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"ppl_vs_cache_{ts}.png")


def plot_tps_ppl_scatter(df_predict: pd.DataFrame, df_base: Optional[pd.DataFrame], out_dir: str, ts: str) -> None:
    df = df_predict[df_predict["tokens_per_second"].notna() & df_predict["gen_perplexity"].notna()].copy()
    if df.empty:
        return
    base_tps = base_lookup(df_base, "tokens_per_second")
    base_ppl = base_lookup(df_base, "gen_perplexity")
    lookaheads = sorted(df["lookahead"].dropna().unique(), key=int)
    cache_sizes = sorted(df["cache_size"].unique(), key=int)
    ncols = min(len(cache_sizes), 3)
    nrows = math.ceil(len(cache_sizes) / ncols)
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("TPS vs PPL: Predictor vs LRU (upper-left = wins on both axes)", fontsize=13, fontweight="bold", y=1.01)
        for idx, cache_size in enumerate(cache_sizes):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["cache_size"] == cache_size]
            if cache_size in base_tps and cache_size in base_ppl:
                ax.scatter(
                    [base_ppl[cache_size]],
                    [base_tps[cache_size]],
                    marker="D",
                    s=140,
                    color=LRU_COLOR,
                    zorder=10,
                    label="LRU-Baseline",
                    edgecolors="#555",
                    linewidths=1,
                )
                ax.axvline(base_ppl[cache_size], color=LRU_COLOR, linewidth=0.8, linestyle=":", alpha=0.3)
                ax.axhline(base_tps[cache_size], color=LRU_COLOR, linewidth=0.8, linestyle=":", alpha=0.3)
            for lookahead in lookaheads:
                pts = sub[sub["lookahead"] == lookahead]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["gen_perplexity"],
                    pts["tokens_per_second"],
                    color=lookahead_color(int(lookahead)),
                    s=55,
                    alpha=0.85,
                    label=f"LA={int(lookahead)}",
                    zorder=5,
                )
            style_ax(ax, f"Cache size = {cache_size}", "Generation Perplexity (↓ better)", "TPS (↑ better)")
        for idx in range(len(cache_sizes), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"tps_ppl_scatter_{ts}.png")


def plot_hitrate_heatmap(df_predict: pd.DataFrame, out_dir: str, ts: str) -> None:
    df = df_predict[df_predict["pred_hit_rate_pct"].notna()].copy()
    if df.empty:
        return
    cache_sizes = sorted(df["cache_size"].unique(), key=int)
    lookaheads = sorted(df["lookahead"].dropna().unique(), key=int)
    budgets = sorted(df["prefetch_budget"].dropna().unique(), key=int)
    for cache_size in cache_sizes:
        sub = df[df["cache_size"] == cache_size]
        if sub.empty:
            continue
        mat = np.full((len(lookaheads), len(budgets)), np.nan)
        for r_idx, lookahead in enumerate(lookaheads):
            for c_idx, budget in enumerate(budgets):
                cell = sub[(sub["lookahead"] == lookahead) & (sub["prefetch_budget"] == budget)]
                if not cell.empty:
                    mat[r_idx, c_idx] = cell["pred_hit_rate_pct"].mean()
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="viridis", vmin=0, vmax=100)
            plt.colorbar(im, ax=ax, label="Predictor Hit Rate (%)")
            ax.set_xticks(range(len(budgets)))
            ax.set_xticklabels([str(int(x)) for x in budgets])
            ax.set_yticks(range(len(lookaheads)))
            ax.set_yticklabels([str(int(x)) for x in lookaheads])
            ax.set_xlabel("Prefetch Budget (top-K)")
            ax.set_ylabel("Lookahead Depth (tokens)")
            ax.set_title(f"Predictor Hit Rate — Cache size = {cache_size}", fontweight="bold")
            for r_idx in range(len(lookaheads)):
                for c_idx in range(len(budgets)):
                    value = mat[r_idx, c_idx]
                    if not np.isnan(value):
                        ax.text(c_idx, r_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value < 60 else "black")
            plt.tight_layout()
            save_plot(fig, out_dir, f"hitrate_heatmap_C{cache_size}_{ts}.png")


def plot_cache_cond(df_cc: pd.DataFrame, df_base: Optional[pd.DataFrame], out_dir: str, ts: str) -> None:
    df = df_cc[df_cc["gen_perplexity"].notna()].copy()
    if df.empty:
        return
    base_ppl = base_lookup(df_base, "gen_perplexity")
    cache_sizes = sorted(df["cache_size"].unique(), key=int)
    fn_vals = sorted(df["forced_top_n"].unique(), key=int)
    fn_colors = {fn: ACCENT[idx % len(ACCENT)] for idx, fn in enumerate(fn_vals)}
    ncols = min(len(cache_sizes), 3)
    nrows = math.ceil(len(cache_sizes) / ncols)
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("Cache-Conditional PPL Degradation vs Lambda (no predictor)", fontsize=13, fontweight="bold", y=1.01)
        for idx, cache_size in enumerate(cache_sizes):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["cache_size"] == cache_size]
            base = base_ppl.get(cache_size)
            if base is None:
                continue
            for pct, style, color in ((1, ":", "#a5d6a7"), (2, "--", "#ffcc80"), (5, "-.", "#ef9a9a")):
                ax.axhline(pct, linestyle=style, color=color, linewidth=1, alpha=0.6, label=f"+{pct}%")
            ax.axhline(0, color=LRU_COLOR, linewidth=1.2, alpha=0.3)
            for forced_top_n in fn_vals:
                s = sub[sub["forced_top_n"] == forced_top_n].sort_values("lambda_val")
                if s.empty:
                    continue
                pct = (s["gen_perplexity"] - base) / base * 100.0
                ax.plot(s["lambda_val"], pct.values, marker="o", color=fn_colors[forced_top_n], linewidth=1.8, markersize=6, label=f"FN={forced_top_n}")
            style_ax(ax, f"Cache size = {cache_size}", "Lambda (λ)", "PPL degradation vs LRU (%)")
        for idx in range(len(cache_sizes), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"cache_cond_ppl_{ts}.png")


def plot_forced_n(df_fn: pd.DataFrame, df_base: Optional[pd.DataFrame], out_dir: str, ts: str) -> None:
    df = df_fn[df_fn["gen_perplexity"].notna()].copy()
    if df.empty:
        return
    base_ppl = base_lookup(df_base, "gen_perplexity")
    cache_sizes = sorted(df["cache_size"].unique(), key=int)
    lambdas = sorted(df["lambda_val"].unique())
    lambda_colors = {lv: ACCENT[idx % len(ACCENT)] for idx, lv in enumerate(lambdas)}
    ncols = min(len(cache_sizes), 3)
    nrows = math.ceil(len(cache_sizes) / ncols)
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("PPL Degradation vs Forced-Router Experts (forced_top_n)", fontsize=13, fontweight="bold", y=1.01)
        for idx, cache_size in enumerate(cache_sizes):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["cache_size"] == cache_size]
            base = base_ppl.get(cache_size)
            if base is None:
                continue
            for pct, style, color in ((1, ":", "#a5d6a7"), (2, "--", "#ffcc80"), (5, "-.", "#ef9a9a")):
                ax.axhline(pct, linestyle=style, color=color, linewidth=1, alpha=0.6, label=f"+{pct}%")
            ax.axhline(0, color=LRU_COLOR, linewidth=1.2, alpha=0.3, label="Baseline")
            for lambda_val in lambdas:
                s = sub[sub["lambda_val"] == lambda_val].sort_values("forced_top_n")
                if s.empty:
                    continue
                pct = (s["gen_perplexity"] - base) / base * 100.0
                ax.plot(s["forced_top_n"], pct.values, marker="o", color=lambda_colors[lambda_val], linewidth=1.8, markersize=6, label=f"λ={lambda_val}")
            ax.invert_xaxis()
            style_ax(ax, f"Cache size = {cache_size}", "forced_top_n  (← less forced | more forced →)", "PPL degradation vs LRU (%)")
        for idx in range(len(cache_sizes), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"forced_n_ppl_{ts}.png")


def plot_custom_1_16(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    df = df_custom.copy()
    if df.empty:
        return
    df = df.sort_values("lookahead")

    cache_sizes = sorted(df["cache_size"].dropna().unique(), key=int)
    lambdas = sorted(df[df["lambda_val"] > 0]["lambda_val"].unique())

    has_tps = df["tokens_per_second"].notna().any()
    has_ppl = df["gen_perplexity"].notna().any()
    has_hitrate = df["hit_rate_pct"].notna().any()
    metrics: List[Tuple[str, str]] = []
    if has_tps:
        metrics.append(("tokens_per_second", "TPS (↑ better)"))
    if has_ppl:
        metrics.append(("gen_perplexity", "Perplexity (↓ better)"))
    if has_hitrate:
        metrics.append(("hit_rate_pct", "Cache Hit Rate % (↑ better)"))
    if not metrics:
        return

    nrows = len(metrics)
    ncols = len(cache_sizes)

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle("Budget-to-Cache Ratio Sweep", fontsize=14, fontweight="bold", y=1.02)

        for col, cache_size in enumerate(cache_sizes):
            sub = df[df["cache_size"] == cache_size]
            lookaheads_sub = sorted(sub["lookahead"].dropna().unique())

            pref_mask = sub["label"].str.startswith("Prefetch Only", na=False)
            budgets = sorted(sub.loc[pref_mask, "prefetch_budget"].dropna().unique(), key=float)
            n_budgets = max(len(budgets), 1)
            budget_cmap = plt.cm.Blues

            for row, (metric_col, ylabel) in enumerate(metrics):
                ax = axes[row][col]
                is_tps = metric_col == "tokens_per_second"
                is_hitrate = metric_col == "hit_rate_pct"
                lru_vals: Dict[float, float] = {}

                # LRU baseline
                s_lru = (
                    sub[sub["label"] == "Neither (LRU)"]
                    .dropna(subset=[metric_col])
                    .sort_values("lookahead")
                )
                if not s_lru.empty:
                    ax.plot(
                        s_lru["lookahead"], s_lru[metric_col],
                        marker="D", **LRU_STYLE, label="LRU Baseline",
                    )
                    for _, r in s_lru.iterrows():
                        lru_vals[float(r["lookahead"])] = float(r[metric_col])

                def _annotate(s: pd.DataFrame, color: str, above: bool) -> None:
                    if s.empty or not lru_vals:
                        return
                    dy, va = (8, "bottom") if above else (-8, "top")
                    for _, r in s.iterrows():
                        la = float(r["lookahead"])
                        val = float(r[metric_col])
                        lru = lru_vals.get(la)
                        if lru is None or lru == 0:
                            continue
                        if is_tps:
                            txt = f"×{val/lru:.2f}"
                        elif is_hitrate:
                            txt = f"{val-lru:+.1f}pp"
                        else:
                            txt = f"{(val-lru)/abs(lru)*100:+.1f}%"
                        ax.annotate(
                            txt, xy=(la, val), xytext=(0, dy),
                            textcoords="offset points", ha="center", va=va,
                            fontsize=6, color=color, fontweight="bold", alpha=0.9,
                        )

                # Prefetch Only lines — one per budget, light→dark blue gradient
                for b_idx, budget in enumerate(budgets):
                    frac = budget / cache_size if cache_size else 0
                    color = budget_cmap(0.3 + 0.6 * b_idx / (n_budgets - 1) if n_budgets > 1 else 0.7)
                    s_b = (
                        sub[pref_mask & (sub["prefetch_budget"] == budget)]
                        .dropna(subset=[metric_col])
                        .sort_values("lookahead")
                    )
                    if s_b.empty:
                        continue
                    ax.plot(
                        s_b["lookahead"], s_b[metric_col],
                        marker="^", color=color, linewidth=2,
                        label=f"Prefetch B={int(budget)} ({frac:.0%})",
                    )
                    _annotate(s_b, color, above=is_tps)

                # Cache-Cond Only and Both lines (when lambdas are present)
                for l_idx, lambda_val in enumerate(lambdas):
                    cc_color   = ["#ffcc80", "#ff9800", "#e65100"][l_idx % 3]
                    both_color = ["#a5d6a7", "#4caf50", "#1b5e20"][l_idx % 3]
                    for pfx, color, marker, above in [
                        (f"Cache-Cond Only λ={lambda_val}", cc_color,   "o", False),
                        (f"Both λ={lambda_val}",            both_color, "s", True),
                    ]:
                        mask = sub["label"].str.startswith(pfx, na=False)
                        for budget in sorted(sub.loc[mask, "prefetch_budget"].dropna().unique(), key=float):
                            frac = budget / cache_size if cache_size else 0
                            s_line = (
                                sub[mask & (sub["prefetch_budget"] == budget)]
                                .dropna(subset=[metric_col])
                                .sort_values("lookahead")
                            )
                            if s_line.empty:
                                continue
                            short = pfx.split(" λ=")[0]
                            ax.plot(
                                s_line["lookahead"], s_line[metric_col],
                                marker=marker, color=color, linewidth=1.5, linestyle=":",
                                label=f"{short} λ={lambda_val} B={int(budget)} ({frac:.0%})",
                            )
                            _annotate(s_line, color, above=above)

                ax.set_xticks(lookaheads_sub)
                if is_hitrate:
                    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(100, ax.get_ylim()[1] + 5))
                style_ax(ax, f"Cache = {cache_size} Experts/Layer", "Lookahead Depth (Tokens)", ylabel)

        plt.tight_layout()
        save_plot(fig, out_dir, f"custom_1_16_summary_{ts}.png")


def plot_probability_mass_calibration(df_cal: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Plot PPL degradation % vs probability-mass threshold for each (lookahead, lambda) pair.

    Shows horizontal threshold lines at 1%, 2%, 5% so the minimum safe FP value
    can be read directly off the chart.
    """
    df = df_cal[df_cal["gen_perplexity"].notna()].copy()
    if df.empty:
        return

    lookaheads = sorted(df["lookahead"].dropna().unique(), key=int)
    lambdas = sorted(df[df["lambda_val"] > 0]["lambda_val"].unique())
    lambda_colors = {lv: ACCENT[i % len(ACCENT)] for i, lv in enumerate(lambdas)}

    ncols = min(len(lookaheads), 3)
    nrows = math.ceil(len(lookaheads) / ncols)

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("PPL Degradation vs Probability-Mass Threshold", fontsize=13, fontweight="bold", y=1.01)

        for idx, lookahead in enumerate(lookaheads):
            ax = axes[idx // ncols][idx % ncols]
            sub = df[df["lookahead"] == lookahead]

            # LRU baseline PPL for this lookahead
            base_rows = sub[sub["label"] == "LRU Baseline"]
            if base_rows.empty or base_rows["gen_perplexity"].isna().all():
                ax.set_title(f"LA={int(lookahead)} (no baseline)")
                continue
            base_ppl = float(base_rows["gen_perplexity"].mean())

            # Threshold lines
            for pct, style, color in ((1, ":", "#a5d6a7"), (2, "--", "#ffcc80"), (5, "-.", "#ef9a9a")):
                ax.axhline(pct, linestyle=style, color=color, linewidth=1.2, alpha=0.8, label=f"{pct}% threshold")
            ax.axhline(0, color=LRU_COLOR, linewidth=1, alpha=0.3, label="LRU baseline (0%)")

            for lambda_val in lambdas:
                pm_col = "prob_mass_threshold" if "prob_mass_threshold" in sub.columns else "forced_top_p"
                pts = sub[(sub["lambda_val"] == lambda_val) & (sub[pm_col] >= 0)].copy()
                if pts.empty:
                    continue
                pts = pts.sort_values(pm_col)
                xs = pts[pm_col].tolist()
                ys = [(float(ppl) - base_ppl) / base_ppl * 100.0 for ppl in pts["gen_perplexity"]]
                ax.plot(xs, ys, marker="o", color=lambda_colors[lambda_val], linewidth=2,
                        markersize=6, label=f"λ={lambda_val}")

                # Annotate the first FP value that stays ≤ 5%
                for x, y in zip(xs, ys):
                    if y <= 5.0:
                        ax.annotate(f"p={x:.2f}", (x, y), textcoords="offset points",
                                    xytext=(4, 6), fontsize=7, color=lambda_colors[lambda_val])
                        break

            style_ax(ax, f"Lookahead = {int(lookahead)}", "Probability-mass threshold", "PPL degradation vs LRU (%)")
            ax.set_xlim(left=0.0)

        for idx in range(len(lookaheads), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        save_plot(fig, out_dir, f"probability_mass_calibration_{ts}.png")


def plot_hit_rates(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Plot cache/predictor rates vs lookahead depth.

    For each (cache_size, budget), shows:
      - Cache Hit Rate % (hit_rate_pct): how often the expert is already resident
        in VRAM — this should be *higher* with the predictor than with LRU.
      - Predictor RoutedTopK Hit Rate % (pred_hit_rate_routed_topk_pct):
        recall-like metric: among routed experts, fraction found in predictions.
      - Predicted RequestedByRouterTopK % (pred_requested_rate_topk_pct):
        precision-like metric: among predicted experts, fraction requested by router.
      - A dashed reference line for the LRU cache hit rate so the prefetch
        improvement (Δpp) is immediately visible.

    One figure is saved per cache_size.
    """
    df = df_custom.copy()
    if df.empty:
        return
    df = df.sort_values("lookahead")

    has_cache_hr = df["hit_rate_pct"].notna().any()
    has_pred_hr = df["pred_hit_rate_routed_topk_pct"].notna().any() or df["pred_hit_rate_pct"].notna().any()
    has_pred_precision = df["pred_requested_rate_topk_pct"].notna().any()
    if not has_cache_hr and not has_pred_hr and not has_pred_precision:
        return

    cache_sizes = sorted(df["cache_size"].dropna().unique(), key=int)

    for cache_size in cache_sizes:
        sub = df[df["cache_size"] == cache_size]
        lookaheads_sub = sorted(sub["lookahead"].dropna().unique())

        pref_mask = sub["label"].str.startswith("Prefetch Only", na=False)
        budgets = sorted(sub.loc[pref_mask, "prefetch_budget"].dropna().unique(), key=float)
        n_budgets = max(len(budgets), 1)

        # Number of sub-panels: cache hit + predictor recall + predictor precision (if available)
        hr_metrics: List[Tuple[str, str]] = []
        if has_cache_hr:
            hr_metrics.append(("hit_rate_pct", "Cache Hit Rate (%)"))
        if has_pred_hr:
            metric_name = "pred_hit_rate_routed_topk_pct" if sub["pred_hit_rate_routed_topk_pct"].notna().any() else "pred_hit_rate_pct"
            hr_metrics.append((metric_name, "Predictor Recall on Routed Top-K (%)"))
        if has_pred_precision:
            hr_metrics.append(("pred_requested_rate_topk_pct", "Predicted Experts Requested by Router (%)"))

        ncols = len(hr_metrics)
        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
            fig.suptitle(
                f"Hit Rates vs Lookahead Depth — Cache = {cache_size} Experts/Layer",
                fontsize=13, fontweight="bold", y=1.03,
            )

            for col, (metric_col, ylabel) in enumerate(hr_metrics):
                ax = axes[0][col]
                budget_cmap = plt.cm.Blues

                # LRU baseline cache hit rate (from cached backend, label "Neither (LRU)")
                lru_sub = (
                    sub[sub["label"] == "Neither (LRU)"]
                    .dropna(subset=[metric_col])
                    .sort_values("lookahead")
                )
                lru_hr_by_la: Dict[float, float] = {}
                if not lru_sub.empty:
                    ax.plot(
                        lru_sub["lookahead"], lru_sub[metric_col],
                        marker="D", **LRU_STYLE, label="LRU Baseline",
                    )
                    for _, r in lru_sub.iterrows():
                        lru_hr_by_la[float(r["lookahead"])] = float(r[metric_col])

                # Prefetch Only lines — light→dark blue gradient per budget
                for b_idx, budget in enumerate(budgets):
                    frac = budget / cache_size if cache_size else 0
                    color = budget_cmap(0.3 + 0.6 * b_idx / (n_budgets - 1) if n_budgets > 1 else 0.7)
                    s_b = (
                        sub[pref_mask & (sub["prefetch_budget"] == budget)]
                        .dropna(subset=[metric_col])
                        .sort_values("lookahead")
                    )
                    if s_b.empty:
                        continue
                    ax.plot(
                        s_b["lookahead"], s_b[metric_col],
                        marker="^", color=color, linewidth=2,
                        label=f"Prefetch B={int(budget)} ({frac:.0%})",
                    )
                    # Annotate Δpp vs LRU for cache hit rate
                    if metric_col == "hit_rate_pct" and lru_hr_by_la:
                        for _, r in s_b.iterrows():
                            la = float(r["lookahead"])
                            val = float(r[metric_col])
                            lru = lru_hr_by_la.get(la)
                            if lru is not None:
                                ax.annotate(
                                    f"{val - lru:+.1f}pp",
                                    xy=(la, val), xytext=(0, 8),
                                    textcoords="offset points",
                                    ha="center", va="bottom",
                                    fontsize=6, color=color, fontweight="bold", alpha=0.9,
                                )

                ax.set_xticks(lookaheads_sub)
                ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
                top_cap = 100.0 if metric_col == "hit_rate_pct" else ax.get_ylim()[1]
                ax.set_ylim(top=min(top_cap, ax.get_ylim()[1] + 5))
                style_ax(ax, ylabel, "Lookahead Depth (Tokens)", ylabel)

            plt.tight_layout()
            save_plot(fig, out_dir, f"hit_rates_C{cache_size}_{ts}.png")


def plot_predictor_comparison(df: pd.DataFrame, out_dir: str, ts: str) -> None:
    """A/B comparison plot for multiple predictors.

    For each (cache_size, lookahead), plots predictor hit rate and TPS
    as a function of prefetch budget, with one line per predictor_tag.
    This directly answers: does a different training objective produce
    better top-K predictions at the same budget?
    """
    if "predictor_tag" not in df.columns:
        return
    df = df.copy()
    df["predictor_tag"] = df["predictor_tag"].fillna("")
    # Only look at rows with a predictor tag (the predict backend rows)
    df_pred = df[df["predictor_tag"].str.len() > 0].copy()
    if df_pred.empty:
        return

    tags = sorted(df_pred["predictor_tag"].unique())
    if len(tags) < 2:
        return  # nothing to compare

    cache_sizes = sorted(df_pred["cache_size"].dropna().unique(), key=int)
    lookaheads = sorted(df_pred["lookahead"].dropna().unique(), key=int)

    # Get LRU baselines from untagged rows
    df_lru = df[(df["label"] == "Neither (LRU)") | (df["label"] == "LRU Baseline")]

    TAG_COLORS = ["#4fc3f7", "#ef9a9a", "#a5d6a7", "#ffcc80", "#ce93d8", "#80cbc4"]
    TAG_MARKERS = ["o", "s", "^", "D", "v", "P"]

    metrics_to_plot: List[Tuple[str, str, bool]] = []  # (col, ylabel, higher_is_better)
    if df_pred["pred_hit_rate_pct"].notna().any():
        metrics_to_plot.append(("pred_hit_rate_pct", "Predictor Hit Rate (%)", True))
    if df_pred["hit_rate_pct"].notna().any():
        metrics_to_plot.append(("hit_rate_pct", "Cache Hit Rate (%)", True))
    if df_pred["tokens_per_second"].notna().any():
        metrics_to_plot.append(("tokens_per_second", "Tokens per Second", True))

    if not metrics_to_plot:
        return

    for cache_size in cache_sizes:
        nrows = len(metrics_to_plot)
        ncols = min(len(lookaheads), 4)
        nrow_panels = math.ceil(len(lookaheads) / ncols)

        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(
                nrows * nrow_panels, ncols,
                figsize=(5.5 * ncols, 4 * nrows * nrow_panels),
                squeeze=False,
            )
            fig.suptitle(
                f"Predictor A/B Comparison — Cache = {cache_size}",
                fontsize=14, fontweight="bold", y=1.01,
            )

            for m_idx, (metric_col, ylabel, higher_better) in enumerate(metrics_to_plot):
                for la_idx, lookahead in enumerate(lookaheads):
                    ax_row = m_idx * nrow_panels + la_idx // ncols
                    ax_col = la_idx % ncols
                    ax = axes[ax_row][ax_col]

                    # LRU baseline horizontal line
                    lru_sub = df_lru[
                        (df_lru["cache_size"] == cache_size)
                        & (df_lru["lookahead"] == lookahead)
                    ]
                    if not lru_sub.empty and lru_sub[metric_col].notna().any():
                        lru_val = lru_sub[metric_col].mean()
                        ax.axhline(lru_val, **LRU_STYLE, label="LRU Baseline")

                    for t_idx, tag in enumerate(tags):
                        color = TAG_COLORS[t_idx % len(TAG_COLORS)]
                        marker = TAG_MARKERS[t_idx % len(TAG_MARKERS)]
                        sub = df_pred[
                            (df_pred["predictor_tag"] == tag)
                            & (df_pred["cache_size"] == cache_size)
                            & (df_pred["lookahead"] == lookahead)
                            & (df_pred["label"].str.startswith("Prefetch Only", na=False))
                        ].dropna(subset=[metric_col]).sort_values("prefetch_budget")
                        if sub.empty:
                            continue
                        ax.plot(
                            sub["prefetch_budget"], sub[metric_col],
                            marker=marker, color=color, linewidth=2,
                            markersize=7, label=tag,
                        )

                    style_ax(
                        ax,
                        f"LA={int(lookahead)}" if m_idx == 0 else f"LA={int(lookahead)}",
                        "Prefetch Budget",
                        ylabel,
                    )

            # Remove unused axes
            total_axes = nrows * nrow_panels * ncols
            used_axes = len(metrics_to_plot) * len(lookaheads)
            for idx in range(used_axes, total_axes):
                r, c = divmod(idx, ncols)
                if r < axes.shape[0] and c < axes.shape[1]:
                    axes[r][c].set_visible(False)

            plt.tight_layout()
            save_plot(fig, out_dir, f"predictor_comparison_C{cache_size}_{ts}.png")


def plot_lambda_fn_sweep(df_in: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Bar charts for routing comparison runs (FN vs probability-mass vs LRU baseline).

    Expects rows with ``question == \"LAMBDA_FN_SWEEP\"`` and populated
    ``gen_perplexity`` / ``tokens_per_second`` when both PPL and generation passes ran.
    """
    df = df_in[df_in["question"] == "LAMBDA_FN_SWEEP"].copy()
    if df.empty or np is None:
        return

    def short_label(r: pd.Series) -> str:
        if float(r.get("lambda_val", 0)) == 0.0 and int(r.get("forced_top_n", 0)) == 0:
            pm = r.get("prob_mass_threshold", r.get("forced_top_p", -1.0))
            try:
                pmf = float(pm) if pm == pm else -1.0
            except (TypeError, ValueError):
                pmf = -1.0
            if pmf < 0.0:
                return "LRU"
        fn = int(r.get("forced_top_n", 0))
        if fn > 0:
            return f"FN={fn}"
        pm = r.get("prob_mass_threshold", r.get("forced_top_p", -1.0))
        try:
            pmf = float(pm)
        except (TypeError, ValueError):
            return "?"
        return f"PM={pmf:g}"

    def sort_key(r: pd.Series) -> Tuple[int, float, int]:
        lbl = short_label(r)
        if lbl == "LRU":
            return (0, 0.0, 0)
        if lbl.startswith("FN="):
            return (1, float(lbl.split("=")[1]), 0)
        if lbl.startswith("PM="):
            return (2, float(lbl.split("=")[1]), 0)
        return (3, 0.0, 0)

    df["_lbl"] = df.apply(short_label, axis=1)
    df["_sk"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sk")

    labels = df["_lbl"].tolist()
    x = np.arange(len(labels))
    ppl = df["gen_perplexity"].astype(float)
    tps = df["tokens_per_second"].astype(float)

    colors = []
    for lbl in labels:
        if lbl == "LRU":
            colors.append("#b0bec5")
        elif lbl.startswith("FN"):
            colors.append("#4fc3f7")
        else:
            colors.append("#a5d6a7")

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.55 * len(labels)), 7), squeeze=False)
        ax0, ax1 = axes[0][0], axes[1][0]

        mask_ppl = ppl.notna()
        if mask_ppl.any():
            ax0.bar(x[mask_ppl], ppl[mask_ppl], color=np.array(colors)[mask_ppl.values], edgecolor="#3a3d4d")
            ax0.set_xticks(x)
            ax0.set_xticklabels(labels, rotation=35, ha="right")
            ax0.set_ylabel("Generation perplexity (↓ better)")
            ax0.set_title("LAMBDA_FN_SWEEP — perplexity by routing config", fontsize=12, fontweight="bold")
            ax0.grid(True, axis="y")
        else:
            ax0.text(0.5, 0.5, "No gen_perplexity (run with mode_perplexity=True)", ha="center", va="center", transform=ax0.transAxes)

        mask_tps = tps.notna()
        if mask_tps.any():
            ax1.bar(x[mask_tps], tps[mask_tps], color=np.array(colors)[mask_tps.values], edgecolor="#3a3d4d")
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=35, ha="right")
            ax1.set_ylabel("Tokens / second (↑ better)")
            ax1.set_title("LAMBDA_FN_SWEEP — throughput by routing config", fontsize=12, fontweight="bold")
            ax1.grid(True, axis="y")
        else:
            ax1.text(0.5, 0.5, "No tokens_per_second", ha="center", va="center", transform=ax1.transAxes)

        plt.tight_layout()
        save_plot(fig, out_dir, f"lambda_fn_sweep_routing_compare_{ts}.png")

    # Tradeoff scatter: PPL vs TPS (same configs)
    both = df["gen_perplexity"].notna() & df["tokens_per_second"].notna()
    if both.any():
        sub = df.loc[both]
        with plt.style.context(PLOT_STYLE):
            fig2, ax = plt.subplots(figsize=(7, 5))
            for lbl in sub["_lbl"].unique():
                m = sub["_lbl"] == lbl
                c = "#b0bec5" if lbl == "LRU" else ("#4fc3f7" if lbl.startswith("FN") else "#a5d6a7")
                ax.scatter(
                    sub.loc[m, "gen_perplexity"],
                    sub.loc[m, "tokens_per_second"],
                    s=120,
                    c=c,
                    edgecolors="#e0e4f0",
                    linewidths=0.6,
                    label=lbl,
                    zorder=3,
                )
            ax.set_xlabel("Generation perplexity (↓ better)")
            ax.set_ylabel("Tokens / second (↑ better)")
            ax.set_title("PPL vs TPS tradeoff (LAMBDA_FN_SWEEP)", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.5)
            ax.legend(fontsize=8, loc="best")
            plt.tight_layout()
            save_plot(fig2, out_dir, f"lambda_fn_sweep_ppl_vs_tps_{ts}.png")


def generate_all_plots(df: pd.DataFrame, out_dir: str, ts: str) -> None:
    if not have_plot_deps():
        raise RuntimeError("Plot dependencies are unavailable. Install matplotlib and numpy to generate plots.")
    os.makedirs(out_dir, exist_ok=True)
    df_lambda_fn = df[df["question"] == "LAMBDA_FN_SWEEP"].copy()
    if not df_lambda_fn.empty:
        plot_lambda_fn_sweep(df, out_dir, ts)
    df_base = df[df["question"] == "BASELINE"].copy()
    df_predict = df[df["question"] == "PREDICT"].copy()
    df_cache_cond = df[df["question"] == "CACHE_COND"].copy()
    df_forced_n = df[df["question"] == "FORCED_N"].copy()
    df_custom = df[df["question"].isin(["CUSTOM_1_16", "CUSTOM_1_16_NO_PPL", "CUSTOM_1_16_PPL_ONLY"])].copy()
    df_top_p_cal = df[df["question"].isin(["TOP_P_CALIBRATION", "PROBABILITY_MASS_CALIBRATION"])].copy()
    base_arg = df_base if not df_base.empty else None
    if not df_base.empty:
        plot_baseline(df_base, out_dir, ts)
    if not df_predict.empty:
        plot_tps_vs_cache(df_predict, base_arg, out_dir, ts)
        plot_ppl_vs_cache(df_predict, base_arg, out_dir, ts)
        plot_tps_ppl_scatter(df_predict, base_arg, out_dir, ts)
        plot_hitrate_heatmap(df_predict, out_dir, ts)
    if not df_cache_cond.empty:
        plot_cache_cond(df_cache_cond, base_arg, out_dir, ts)
    if not df_forced_n.empty:
        plot_forced_n(df_forced_n, base_arg, out_dir, ts)
    if not df_custom.empty:
        plot_custom_1_16(df_custom, out_dir, ts)
        plot_hit_rates(df_custom, out_dir, ts)
        plot_predictor_comparison(df_custom, out_dir, ts)
    if not df_top_p_cal.empty:
        plot_probability_mass_calibration(df_top_p_cal, out_dir, ts)


def run_simple_mode(args: argparse.Namespace) -> int:
    if args.subprocess_timeout is None:
        num_p = max(1, getattr(args, "num_prompts", 1))
        args.subprocess_timeout = max(900, num_p * args.max_new_tokens * 2 + 600)

    if args.expert_reuse_csv and args.predictor_path:
        fair_cache, fair_prefetch = calculate_fair_metrics(args.expert_reuse_csv, args.predictor_path)
        print(f"[Calibration] Fair Metrics for {args.predictor_path}: Cache={fair_cache}, Prefetch={fair_prefetch}")
        if args.cache_sizes == [8]:
            args.cache_sizes = [fair_cache]
        if args.prefetch_counts == [1]:
            args.prefetch_counts = [fair_prefetch]

    prompts = load_prompts(args)
    fd, tmp_json = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(prompts, f)

    rows: List[SimpleRunRow] = []
    ts0 = time.strftime("%Y-%m-%dT%H:%M:%S")
    fieldnames = [f.name for f in fields(SimpleRunRow)]

    if "predict" in args.backends and not (args.predictor_path or "").strip():
        print("predict backend needs --predictor-path", flush=True)
        return 2

    def run_backend(backend: str, cache_size: int, prefetch: int) -> None:
        cmd = build_model_cmd(
            model=args.model,
            backend=backend,
            cache_size=cache_size,
            lambda_val=args.lambda_val,
            prompts_json=tmp_json,
            predictor_path=args.predictor_path,
            prefetch_count=prefetch,
            predict_layers=args.predict_layers,
            forced_top_n=args.forced_top_n,
            prob_mass_threshold=args.probability_mass_threshold,
            config_path=args.config_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            mode_generate=True,
            mode_generation_perplexity=False,
            expert_reuse_csv=args.expert_reuse_csv,
            predictor_device=args.predictor_device if args.model == "qwen" else None,
            expert_weights_dir=getattr(args, "expert_weights_dir", None),
        )
        print("\n" + "=" * 80, flush=True)
        print("RUN", cmd, flush=True)
        out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
        ok = out is not None
        metrics = parse_output(out or "")
        rows.append(
            SimpleRunRow(
                ts=ts0,
                model=args.model,
                backend=backend,
                cache_size=cache_size,
                prefetch_count=prefetch if backend == "predict" else None,
                lambda_val=args.lambda_val,
                tps=metrics["tokens_per_second"],
                cache_hits=metrics["cache_hits"],
                cache_misses=metrics["cache_misses"],
                hit_rate_pct=metrics["hit_rate_pct"],
                pred_hits=metrics["pred_hits"],
                pred_total=metrics["pred_total"],
                pred_hit_rate_pct=metrics["pred_hit_rate_pct"],
                pred_hits_routed_forced_n=metrics["pred_hits_routed_forced_n"],
                pred_total_routed_forced_n=metrics["pred_total_routed_forced_n"],
                pred_hit_rate_routed_forced_n_pct=metrics["pred_hit_rate_routed_forced_n_pct"],
                pred_hits_routed_topk=metrics["pred_hits_routed_topk"],
                pred_total_routed_topk=metrics["pred_total_routed_topk"],
                pred_hit_rate_routed_topk_pct=metrics["pred_hit_rate_routed_topk_pct"],
                pred_requested_hits_forced_n=metrics["pred_requested_hits_forced_n"],
                pred_requested_total_forced_n=metrics["pred_requested_total_forced_n"],
                pred_requested_rate_forced_n_pct=metrics["pred_requested_rate_forced_n_pct"],
                pred_requested_hits_topk=metrics["pred_requested_hits_topk"],
                pred_requested_total_topk=metrics["pred_requested_total_topk"],
                pred_requested_rate_topk_pct=metrics["pred_requested_rate_topk_pct"],
                stall_loads=metrics["stall_loads"] if backend == "predict" else None,
                prefetch_loads=metrics["prefetch_loads"] if backend == "predict" else None,
                avg_ms_per_expert_load=metrics["avg_ms_per_expert_load"] if backend == "predict" else None,
                ok=ok,
            )
        )

    try:
        for cache_size in args.cache_sizes:
            if "cached" in args.backends:
                run_backend("cached", cache_size, prefetch=args.prefetch_counts[0])
            for prefetch in args.prefetch_counts:
                if prefetch > cache_size:
                    print(f"Skip prefetch={prefetch} > cache_size={cache_size}", flush=True)
                    continue
                if "predict" in args.backends:
                    run_backend("predict", cache_size, prefetch=prefetch)
    finally:
        try:
            os.remove(tmp_json)
        except OSError:
            pass

    out_path = os.path.abspath(args.output_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)
    print("\n--- summary (cached vs predict, same cache_size) ---")
    for cache_size in args.cache_sizes:
        cached_row = next((r for r in rows if r.backend == "cached" and r.cache_size == cache_size and r.ok), None)
        pred_rows = [r for r in rows if r.backend == "predict" and r.cache_size == cache_size and r.ok]
        for pred_row in pred_rows:
            if cached_row is None:
                print(
                    f"cache={cache_size} prefetch={pred_row.prefetch_count} "
                    f"TPS={pred_row.tps} hit%={pred_row.hit_rate_pct} "
                    f"avg_ms/load={pred_row.avg_ms_per_expert_load} "
                    f"stall/pref={pred_row.stall_loads}/{pred_row.prefetch_loads}",
                    flush=True,
                )
            else:
                print(
                    f"cache={cache_size} prefetch={pred_row.prefetch_count} "
                    f"TPS predict={pred_row.tps} cached={cached_row.tps} "
                    f"hit% predict={pred_row.hit_rate_pct} cached={cached_row.hit_rate_pct} "
                    f"avg_ms/load(predict)={pred_row.avg_ms_per_expert_load} "
                    f"stall/pref predict={pred_row.stall_loads}/{pred_row.prefetch_loads}",
                    flush=True,
                )
    return 0


def run_comprehensive_mode(args: argparse.Namespace) -> int:
    if pd is None:
        print("[sweep] Comprehensive mode requires pandas in the active Python environment.", flush=True)
        return 2
    if args.model != "qwen":
        print("[sweep] Comprehensive mode currently targets Qwen only. Use --model qwen.", flush=True)
        return 2

    if args.subprocess_timeout is None:
        # Scale with both max_new_tokens and num_prompts.
        # Assume worst-case ~3 TPS under heavy prefetch, plus 600s init overhead.
        num_p = max(1, getattr(args, "num_prompts", 1))
        args.subprocess_timeout = max(900, num_p * args.max_new_tokens * 2 + 600)

    csv_file = args.csv_file or os.path.join(args.out_dir, f"sweep_{TS}.csv")
    plot_dir = args.out_dir

    if args.plot_only:
        if not have_plot_deps():
            print("[sweep] Plot-only mode requires matplotlib and numpy in the active Python environment.", flush=True)
            return 2
        if not os.path.exists(csv_file):
            print(f"Error: CSV not found: {csv_file}", flush=True)
            return 1
        df = pd.read_csv(csv_file)
        generate_all_plots(df, plot_dir, TS)
        return 0

    os.makedirs(plot_dir, exist_ok=True)
    csv_parent = os.path.dirname(os.path.abspath(csv_file))
    if csv_parent:
        os.makedirs(csv_parent, exist_ok=True)

    min_cache_map = load_min_cache_per_lookahead(args.constraint_expert_reuse_csv)

    # Multi-predictor support: if --predictor-base-dirs is provided with >1 dir,
    # generate configs for each predictor with a tag so the CSV tracks which
    # predictor produced each row.  The comparison plot overlays them.
    predictor_dirs = getattr(args, "predictor_base_dirs", None) or []
    multi_predictor = len(predictor_dirs) > 1

    def _make_custom_configs(fn, question_name):
        if not multi_predictor:
            return fn(args)
        cfgs = []
        for pd_dir in predictor_dirs:
            tag = os.path.basename(os.path.normpath(pd_dir))
            cfgs.extend(fn(args, predictor_base_dir=pd_dir, predictor_tag=tag))
        # Add LRU baselines once (not tagged)
        cfgs.extend(fn(args))
        # Deduplicate LRU baselines: keep only first occurrence per cfg_key
        seen: set[Tuple[Any, ...]] = set()
        deduped: List[Dict[str, Any]] = []
        for c in cfgs:
            k = cfg_key(c)
            if k not in seen:
                seen.add(k)
                deduped.append(c)
        return deduped

    question_builders = {
        "baseline": lambda: baseline_configs(args),
        "predict": lambda: predict_configs(args, min_cache_map),
        "cache_cond": lambda: cache_cond_configs(args),
        "forced_n": lambda: forced_n_configs(args),
        "custom_1_16": lambda: _make_custom_configs(custom_1_16_configs, "CUSTOM_1_16"),
        "custom_1_16_no_ppl": lambda: _make_custom_configs(custom_1_16_no_ppl_configs, "CUSTOM_1_16_NO_PPL"),
        "custom_1_16_ppl_only": lambda: _make_custom_configs(custom_1_16_ppl_only_configs, "CUSTOM_1_16_PPL_ONLY"),
        "lambda_fn_sweep": lambda: lambda_fn_sweep_configs(args),
        "probability_mass_calibration": lambda: probability_mass_calibration_configs(args),
        # Backward-compatible alias.
        "top_p_calibration": lambda: probability_mass_calibration_configs(args),
    }
    questions = ["baseline", "predict", "cache_cond"] if args.sweep_question == "all" else [args.sweep_question]
    configs: List[Dict[str, Any]] = []
    for question in questions:
        configs.extend(question_builders[question]())

    existing_rows: List[Dict[str, Any]] = []
    done_keys: set[Tuple[Any, ...]] = set()
    if args.retry_failed and args.csv_file and os.path.exists(args.csv_file):
        df_existing = pd.read_csv(args.csv_file)
        for _, row in df_existing.iterrows():
            row_dict = row.to_dict()
            existing_rows.append(row_dict)
            if row_has_data(row_dict):
                done_keys.add(
                    (
                        row.get("question"),
                        row.get("backend"),
                        int(row["cache_size"]) if not pd.isna(row["cache_size"]) else None,
                        float(row["lambda_val"]),
                        int(row["forced_top_n"]),
                        row.get("lookahead"),
                        row.get("prefetch_budget"),
                        row.get("mode_perplexity"),
                        row.get("mode_generate"),
                    )
                )
        before = len(configs)
        configs = [c for c in configs if cfg_key(c) not in done_keys]
        print(f"[sweep] --retry-failed: {before - len(configs)} configs already done, {len(configs)} to re-run.", flush=True)

    expanded_runs: List[Dict[str, Any]] = []
    for config_id, config in enumerate(configs):
        for run_iter in range(max(1, args.repeat_each_config)):
            cfg_copy = dict(config)
            cfg_copy["_config_id"] = config_id
            cfg_copy["_run_iter"] = run_iter
            expanded_runs.append(cfg_copy)
    if args.shuffle_config_order:
        random.shuffle(expanded_runs)

    print(f"\n[sweep] Total runs : {len(expanded_runs)}", flush=True)
    print(f"[sweep] Questions  : {questions}", flush=True)
    print(f"[sweep] CSV output : {csv_file}", flush=True)
    print(f"[sweep] Timeout    : {args.subprocess_timeout}s per run", flush=True)
    print("-" * 80, flush=True)

    prompts = load_prompts(args)
    print(f"[sweep] Loaded {len(prompts)} prompt(s), dataset={args.dataset}", flush=True)
    fd, prompts_file = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(prompts, f)

    results = list(existing_rows) if args.retry_failed else []
    if args.drop_page_cache_before_first_run:
        drop_page_cache()

    prompt_hash = hashlib.sha1(json.dumps(prompts, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        for idx, config in enumerate(expanded_runs):
            if idx > 0 and args.drop_page_cache_between_runs:
                drop_page_cache()
            row = execute_comprehensive_run(config, prompts_file, prompts, args, idx, len(expanded_runs))
            row["run_id"] = run_id
            row["row_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            row["prompt_hash"] = prompt_hash
            row["config_id"] = config.get("_config_id", -1)
            row["run_iter"] = config.get("_run_iter", 0)
            results.append(row)
            pd.DataFrame(results).to_csv(csv_file, index=False)
    finally:
        try:
            os.remove(prompts_file)
        except OSError:
            pass

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"\n[sweep] Results saved to {csv_file}", flush=True)
    ok_count = sum(1 for row in results if row_has_data(row))
    print(f"[sweep] Runs with data: {ok_count} / {len(results)}", flush=True)
    print("\n[sweep] Generating plots...", flush=True)
    try:
        generate_all_plots(df, plot_dir, TS)
    except Exception as e:
        if not have_plot_deps():
            print("[sweep] Skipping plots because matplotlib/numpy are unavailable in this Python.", flush=True)
        else:
            import traceback

            print(f"[sweep] Error during plotting: {e}", flush=True)
            traceback.print_exc()
    print("[sweep] Done!", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        "--sweep-question",
        choices=[
            "simple",
            "baseline",
            "predict",
            "cache_cond",
            "forced_n",
            "custom_1_16",
            "custom_1_16_no_ppl",
            "custom_1_16_ppl_only",
            "lambda_fn_sweep",
            "probability_mass_calibration",
            "top_p_calibration",
            "all",
        ],
        default="simple",
        help="`simple` keeps the original cached-vs-predict behavior. Any other value enables the comprehensive Qwen sweep.",
    )

    p.add_argument("--model", choices=["qwen", "mixtral"], default="qwen")
    p.add_argument("--backends", nargs="+", default=["predict", "cached"], choices=["predict", "cached"])
    p.add_argument("--cache-sizes", type=int, nargs="+", default=[8], help="Cache sizes to sweep.")
    p.add_argument("--prefetch-counts", type=int, nargs="+", default=[8], help="Predict backend only (simple mode).")
    p.add_argument("--prefetch-budgets", type=int, nargs="+", default=[16, 32], help="Prefetch budgets for comprehensive predict sweeps.")
    p.add_argument("--budget-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="Budget as a fraction of cache size for custom_1_16/lambda_fn/top_p sweeps. "
                        "Sweeps the sweet spot between cache thrashing (high fraction) and under-utilisation (low fraction).")
    p.add_argument("--budget-equals-cache-size", action="store_true", help="Use budget=cache_size in comprehensive predict sweeps.")
    p.add_argument("--lambda-val", type=float, default=0.0, help="Single lambda value for simple mode.")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="Lambda sweep for comprehensive mode.")
    p.add_argument("--cache-cond-lambdas", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--lookaheads", type=int, nargs="+", default=None, help="Lookahead depths for comprehensive mode.")
    p.add_argument("--predictor-path", type=str, default="", help="Required for simple predict backend.")
    p.add_argument(
        "--predictor-base-dir",
        type=str,
        default="/home/michael/heteroPredict/trainingData/qwen3_30b/final_multi_input_model",
        help="Base directory containing eh1_h32_fN predictor dirs for comprehensive mode.",
    )
    p.add_argument(
        "--predictor-base-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple predictor base directories for A/B comparison. Each directory's "
             "basename is used as the predictor tag in the CSV & plots. When provided, "
             "the sweep runs each config once per predictor and generates comparison plots.",
    )
    p.add_argument("--predict-layers", type=int, nargs="*", default=None)
    p.add_argument("--predictor-device", type=str, default="cpu")
    p.add_argument("--forced-top-n", type=int, default=0, help="Forced top-N routing for simple mode.")
    p.add_argument(
        "--probability-mass-threshold",
        type=float,
        default=-1.0,
        help="Probability-mass routing threshold for simple mode (-1 disables).",
    )
    p.add_argument("--routing-bias-top-n", type=int, default=6, dest="routing_bias_top_n",
                   help="Forced top-N for Cache-Cond and Both configs (biases router toward cached experts). "
                        "Prefetch Only always uses 0 (unbiased router).")
    p.add_argument("--cache-cond-forced-top-ns", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--forced-top-ns", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument(
        "--probability-mass-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
        help="Cumulative probability-mass thresholds for alternate mass-based routing sweeps.",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--forced-top-ps",
        type=float,
        nargs="+",
        dest="probability_mass_thresholds",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--config-path", type=str, default=None)
    p.add_argument("--dataset", choices=["default", "txt", "wikitext", "fineweb", "orca"], default="default")
    p.add_argument("--prompts-txt", type=str, default=None)
    p.add_argument("--num-prompts", type=int, default=1)
    p.add_argument(
        "--prompt-max-chars", type=int, default=4096,
        help="Characters per prompt chunk. For wikitext the corpus is concatenated then "
             "chunked into windows of this size (default 4096 ≈ 1024 tokens at ~4 chars/tok). "
             "Shorter values (~500) give faster TPS-only sweeps; longer values improve PPL accuracy.",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--expert-reuse-csv", type=str, default=None, help="Optional per-layer calibration CSV for simple mode.")
    p.add_argument(
        "--constraint-expert-reuse-csv",
        type=str,
        default=DEFAULT_QWEN_REUSE_CSV,
        help="Expert reuse CSV used to enforce minimum cache size per lookahead in comprehensive mode.",
    )
    p.add_argument("--output-csv", type=str, default="predict_cached_cache_metrics.csv", help="CSV output for simple mode.")
    p.add_argument("--csv-file", type=str, default=None, help="CSV path for comprehensive mode / retry / plot-only.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--log-file", type=str, default=None, help="Append full subprocess logs here.")
    p.add_argument("--subprocess-timeout", type=int, default=None)
    p.add_argument("--custom-include-ppl", action="store_true",
                   help="When using custom_1_16, also run a perplexity pass (separate from the TPS generation pass).")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--repeat-each-config", type=int, default=1)
    p.add_argument("--shuffle-config-order", action="store_true")
    p.add_argument("--drop-page-cache-between-runs", action="store_true")
    p.add_argument("--drop-page-cache-before-first-run", action="store_true")
    p.add_argument("--cold-per-prompt", action="store_true")
    p.add_argument("--drop-page-cache-between-prompts", action="store_true")
    p.add_argument(
        "--expert-weights-dir", type=str, default=None,
        help="Override expert weight directory for MoE layers. Supports packed (EXPK, 1 "
             "file/expert) and unpacked (9 files/expert) layouts — auto-detected at runtime.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.sweep_question == "simple":
        return run_simple_mode(args)
    return run_comprehensive_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
