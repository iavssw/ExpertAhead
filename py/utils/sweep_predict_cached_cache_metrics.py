#!/usr/bin/env python3
"""
Cached-vs-predict expert sweep engine behind the thesis results chapter.

Driven by ``finals_experiment_runner.py``; runs a grid of decode configs by
spawning the C++ backends (cached / predict) and parsing their TPS, cache, and
predictor stats into a CSV, then draws summary plots. The three ``--sweep-question``
modes map to the thesis sections:

- ``custom_1_16_no_ppl`` — LRU vs prefetch (and cache-cond / hybrid) over
  lookahead x prefetch budget, lossless (no perplexity pass). Sections 2 and 3.
  Cache-Cond runs once per cache size (cached backend; no lookahead). The cached backend ignores budget.
- ``custom_1_16`` — same grid plus a perplexity pass. Section 5 (four-way).
- ``lambda_fn_sweep`` — cache-conditional routing, forced-top-J vs probability-mass
  threshold, with a perplexity pass. Section 4.

Prefetch budgets are taken from ``--budget-fractions`` x cache size (or explicit
``--prefetch-budgets`` with ``--custom-explicit-prefetch-budgets``).
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
import textwrap
import threading
import time
import uuid
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

from prompt_datasets import HF_EVAL_CHOICES

DEFAULT_QWEN_REUSE_CSV = os.path.join(ROOT_DIR, "expert_predictor", "expert_reuse_qwen3_30b.csv")
DEFAULT_QWEN_EXPERT_WEIGHTS_DIR = os.path.join(
    ROOT_DIR, "unified_llm_w4a16", "model_weights", "Qwen3-30B-A3B-AWQ_packed"
)


def run_subprocess(cmd: List[str], timeout: int, log_file: Optional[str] = None) -> Optional[str]:
    """Run command; optional tee to log_file. Returns merged stdout+stderr.

    Enforces *timeout* using a **reader thread + wall-clock deadline** on the parent.
    The old ``threading.Timer`` + ``readline()`` pattern could hang past *timeout* if
    ``killpg`` failed silently while the child kept running (parent blocked forever on
    the pipe).
    """
    if timeout <= 0:
        print("[sweep] WARNING: non-positive subprocess timeout; using 60s", flush=True)
        timeout = 60

    def kill_child(p: subprocess.Popen, reason: str) -> None:
        try:
            import signal

            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception as e:
            print(f"[sweep] killpg({reason}) failed: {e!r}; trying process.kill()", flush=True)
            try:
                p.kill()
            except Exception as e2:
                print(f"[sweep] process.kill({reason}) failed: {e2!r}", flush=True)

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

        q: Queue[Optional[str]] = Queue()

        def reader() -> None:
            assert process.stdout is not None
            try:
                for line in iter(process.stdout.readline, ""):
                    q.put(line)
            finally:
                q.put(None)

        tr = threading.Thread(target=reader, name="sweep-subproc-reader", daemon=True)
        tr.start()

        deadline = time.monotonic() + float(timeout)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timeout_reached = True
                    print(
                        f"[sweep] Subprocess wall timeout ({timeout}s); SIGKILL pid={process.pid}",
                        flush=True,
                    )
                    kill_child(process, "wall-timeout")
                    break
                try:
                    line = q.get(timeout=min(1.0, max(0.05, remaining)))
                except Empty:
                    if process.poll() is not None:
                        while True:
                            try:
                                line = q.get(timeout=15.0)
                            except Empty:
                                break
                            if line is None:
                                break
                            if line:
                                print(line, end="")
                                sys.stdout.flush()
                                output_lines.append(line)
                                if log_fp:
                                    log_fp.write(line)
                                    log_fp.flush()
                        break
                    continue

                if line is None:
                    break
                if line:
                    print(line, end="")
                    sys.stdout.flush()
                    output_lines.append(line)
                    if log_fp:
                        log_fp.write(line)
                        log_fp.flush()
        finally:
            if log_fp:
                log_fp.close()

        if timeout_reached:
            while True:
                try:
                    extra = q.get(timeout=0.3)
                except Empty:
                    break
                if extra is None:
                    break
                if extra:
                    output_lines.append(extra)

        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            print("[sweep] WARNING: child not reaped within 120s after stdout closed", flush=True)

        if timeout_reached:
            print(f"[sweep_predict_cached_cache_metrics] Command timed out after {timeout}s", flush=True)
            return None
        if process.returncode != 0:
            print(f"[sweep_predict_cached_cache_metrics] exit code {process.returncode}", flush=True)
            return None
        time.sleep(0.2)
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
    # Try predictor format first: Bandwidth: StallLoads=..., PrefetchLoads=..., AvgLoadTime=...
    rows_predict = re.findall(
        r"Bandwidth:\s*StallLoads=(\d+),\s*PrefetchLoads=(\d+),\s*AvgLoadTime=([\d.]+)ms",
        text,
    )
    if rows_predict:
        stall = sum(int(r[0]) for r in rows_predict)
        pref = sum(int(r[1]) for r in rows_predict)
        num = 0.0
        den = 0
        for s, p, a in rows_predict:
            loads = int(s) + int(p)
            den += loads
            num += float(a) * loads
        avg_ms = (num / den) if den > 0 else None
        return stall, pref, avg_ms

    # Try cached format: Bandwidth: MissLoads=..., AvgLoadTime=...
    rows_cached = re.findall(
        r"Bandwidth:\s*MissLoads=(\d+),\s*AvgLoadTime=([\d.]+)ms",
        text,
    )
    if rows_cached:
        stall = sum(int(r[0]) for r in rows_cached)
        num = 0.0
        den = 0
        for m, a in rows_cached:
            loads = int(m)
            den += loads
            num += float(a) * loads
        avg_ms = (num / den) if den > 0 else None
        return stall, 0, avg_ms

    return 0, 0, None


def parse_prefetch_overlap_lines(text: str) -> Tuple[int, int, int]:
    rows = re.findall(
        r"PrefetchOverlap:\s*HiddenReady=(\d+),\s*HiddenWait=(\d+),\s*TicksSkipped=(\d+)",
        text,
    )
    if not rows:
        return 0, 0, 0
    ready = sum(int(r[0]) for r in rows)
    wait = sum(int(r[1]) for r in rows)
    skipped = sum(int(r[2]) for r in rows)
    return ready, wait, skipped


# Qwen3-30B-A3B decode constants (paper §Modeling Steady-State Decode)
MOE_LAYERS_QWEN = 48
EXPERTS_PER_TOKEN_QWEN = 8
E_INVOCATIONS_PER_TOKEN = MOE_LAYERS_QWEN * EXPERTS_PER_TOKEN_QWEN  # 384
T_CEIL_MS_PAPER = 49.0
DELTA_T_MS_PAPER = 0.771


def print_decode_regime_report(row: Dict[str, Any], *, t_ceil_ms: float = T_CEIL_MS_PAPER) -> None:
    """
    Map sweep CSV metrics to §Modeling Steady-State Decode and print which constraint likely binds.

    - LRU: E_miss = E*(1-h), T = T_ceil + E_miss*ΔT
    - Predict: prefetch_loads = issued SSD reads; prefetch_hits_ready ≈ M_prefetch hides;
      prefetch_hits_wait / prefetch_ticks_skipped → overlap (M_cap); π, ρ from predictor columns.
    """
    backend = str(row.get("backend", ""))
    label = row.get("label", "")
    hits = int(row.get("cache_hits") or 0)
    misses = int(row.get("cache_misses") or 0)
    accesses = hits + misses
    if accesses <= 0:
        print(f"[regime] {label}: no cache events", flush=True)
        return

    h = hits / accesses
    n_tok_est = accesses / E_INVOCATIONS_PER_TOKEN
    e_miss = misses / n_tok_est if n_tok_est > 0 else 0.0
    tps = row.get("tokens_per_second")
    t_meas_ms = (1000.0 / float(tps)) if tps and float(tps) > 0 else None

    print(f"\n[regime] {label} ({backend}) C={row.get('cache_size')} LA={row.get('lookahead')} "
          f"B={row.get('prefetch_budget')}", flush=True)
    print(f"  h={100*h:.1f}%  E_miss/token≈{e_miss:.2f}  (est. {n_tok_est:.0f} decode tokens)", flush=True)
    if t_meas_ms is not None:
        print(f"  T_meas≈{t_meas_ms:.1f} ms/token ({tps:.2f} TPS)", flush=True)

    if backend == "cached":
        t_lru = t_ceil_ms + e_miss * DELTA_T_MS_PAPER
        print(f"  LRU model (ΔT={DELTA_T_MS_PAPER} ms): T≈{t_lru:.1f} ms → {1000/t_lru:.2f} TPS", flush=True)
        print("  Binding: cache hit rate h (capacity + workload); all misses block (stall).", flush=True)
        return

    stall = int(row.get("stall_loads") or 0)
    issued = int(row.get("prefetch_loads") or 0)
    hidden = int(row.get("prefetch_hits_ready") or 0)
    wait = int(row.get("prefetch_hits_wait") or 0)
    skipped = int(row.get("prefetch_ticks_skipped") or 0)
    t_ssd = float(row.get("avg_ms_per_expert_load") or DELTA_T_MS_PAPER)
    n_stride = int(row.get("predictor_stride") or row.get("lookahead") or 1)
    b = int(row.get("prefetch_budget") or 0)
    rho = float(row.get("pred_hit_rate_routed_topk_pct") or 0) / 100.0
    # π: prefer top-k requested rate; fall back to forced-N if top-k duplicates recall (known quirk).
    pi_topk = float(row.get("pred_requested_rate_topk_pct") or 0) / 100.0
    pi_fn = float(row.get("pred_requested_rate_forced_n_pct") or 0) / 100.0
    routed_topk_hits = int(row.get("pred_hits_routed_topk") or 0)
    requested_topk_hits = int(row.get("pred_requested_hits_topk") or 0)
    if (
        routed_topk_hits > 0
        and requested_topk_hits == routed_topk_hits
        and abs(pi_topk - rho) < 0.05
    ):
        pi = pi_fn if pi_fn > 0 else pi_topk
        pi_note = " (using requested@FN; top-k column matches recall)"
    else:
        pi = pi_topk
        pi_note = ""

    e_block = stall / n_tok_est if n_tok_est > 0 else 0.0
    e_hidden = hidden / n_tok_est if n_tok_est > 0 else 0.0
    m_cap = (n_stride * t_ceil_ms) / t_ssd if t_ssd > 0 else 0.0
    m_pi = pi * MOE_LAYERS_QWEN * b if b > 0 else 0.0

    misses_saved_vs_lru = max(0, int(row.get("_lru_misses_ref") or 0) - misses)
    hide_frac_issued = (hidden / issued) if issued > 0 else 0.0
    t_pred = t_ceil_ms + e_block * DELTA_T_MS_PAPER

    print(f"  Predictor: ρ≈{rho:.2f}  π≈{pi:.2f}{pi_note}  |  stride N={n_stride}", flush=True)
    print(f"  Loads: issued(prefetch)={issued}  hidden_ready={hidden}  hidden_wait={wait}  "
          f"stall(block)={stall}  ticks_skipped={skipped}", flush=True)
    print(
        f"  Cache: misses={misses}  (vs LRU baseline misses≈{misses_saved_vs_lru + misses} "
        f"→ ~{misses_saved_vs_lru} fewer than LRU if paired run)",
        flush=True,
    )
    print(
        f"  Per token: E_block≈{e_block:.2f}  router_hits_on_prefetch_slots≈{e_hidden:.2f}/tok "
        f"(not same as M_prefetch; can exceed issued)",
        flush=True,
    )
    print(f"  Overlap capacity M_cap≈{m_cap:.1f} loads/interval  "
          f"(N·T_ceil/T_ssd = {n_stride}·{t_ceil_ms:.0f}/{t_ssd:.2f})", flush=True)
    print(f"  Precision cap π·L·B≈{m_pi:.1f} loads/tick  |  Prefetch model T≈{t_pred:.1f} ms "
          f"({1000/t_pred:.2f} TPS)", flush=True)

    # Heuristic binding (compare dominant signals)
    binds: List[str] = []
    if rho < 0.5:
        binds.append("recall (ρ·U)")
    if pi < 0.25:
        binds.append("precision (π·L·B waste)")
    if wait > 0 or skipped > 0:
        binds.append("overlap time (M_cap / ΔT)")
    elif issued > 0 and wait == 0 and skipped == 0:
        binds.append("overlap not skip-limited (wait=0, ticks_skipped=0)")
    if e_block > e_miss * 0.8 if e_miss else e_block > 2:
        binds.append("remaining blocking stalls (E_block)")
    if not binds:
        binds.append("mixed / near ceiling")

    print(f"  Likely binding: {', '.join(binds)}", flush=True)
    print(
        "  Note: prefetch_loads counts SSD issues on speculative path; "
        "prefetch_hits_ready is M_prefetch (non-blocking) in the model.",
        flush=True,
    )


def parse_output(output: str) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "gen_perplexity": None,
        "gen_perplexity_std": None,
        "prompt_gen_perplexity": None,
        "prompt_gen_perplexity_std": None,
        "tokens_per_second": None,
        "tokens_per_second_std": None,
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
        "gating_prefetch_hits": None,
        "gating_prefetch_total": None,
        "gating_prefetch_recall_pct": None,
        "gating_pred_recall_hits": None,
        "gating_pred_recall_total": None,
        "gating_pred_recall_pct": None,
        "gating_pred_precision_hits": None,
        "gating_pred_precision_total": None,
        "gating_pred_precision_pct": None,
        "gating_prefetch_delivery_pct": None,
        "prefetch_already_cached": None,
        "prefetch_dropped_no_victim": None,
        "gating_prefetch_triggers": None,
        "stall_loads": None,
        "prefetch_loads": None,
        "prefetch_hits_ready": None,
        "prefetch_hits_wait": None,
        "prefetch_ticks_skipped": None,
        "avg_ms_per_expert_load": None,
        "ref_perplexity": None,
        "ref_nll_sum": None,
        "ref_tokens": None,
    }

    m_ref_nll = re.search(r"^Reference NLL Sum:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output, re.MULTILINE)
    m_ref_tok = re.search(r"^Reference Tokens:\s+(\d+)", output, re.MULTILINE)
    m_ref_ppl = re.search(r"^Reference Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output, re.MULTILINE)
    if m_ref_nll and m_ref_tok:
        result["ref_nll_sum"] = float(m_ref_nll.group(1))
        result["ref_tokens"] = int(m_ref_tok.group(1))
    if m_ref_ppl:
        result["ref_perplexity"] = float(m_ref_ppl.group(1))

    ppl_matches = re.findall(
        r"^Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output,
        re.MULTILINE,
    )
    if ppl_matches:
        result["gen_perplexity"] = float(ppl_matches[-1])

    m_prompt_ppl = re.search(
        r"Generation Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output,
    )
    if m_prompt_ppl:
        result["prompt_gen_perplexity"] = float(m_prompt_ppl.group(1))

    m_prompt_ppl_std = re.search(
        r"Generation Perplexity StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output,
    )
    if m_prompt_ppl_std:
        result["prompt_gen_perplexity_std"] = float(m_prompt_ppl_std.group(1))

    result["tokens_per_second"] = parse_tps(output)

    m_tps_std = re.search(r"TPS StdDev:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if m_tps_std:
        result["tokens_per_second_std"] = float(m_tps_std.group(1))

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

    wr_matches = re.findall(r"Predictor WindowRecall Rate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if wr_matches:
        total_hits = sum(int(m[1]) for m in wr_matches)
        total_eval = sum(int(m[2]) for m in wr_matches)
        if total_eval > 0:
            result["pred_hits_window_recall"] = total_hits
            result["pred_total_window_recall"] = total_eval
            result["pred_window_recall_pct"] = (100.0 * total_hits) / total_eval

    wp_matches = re.findall(r"Predictor WindowPrecision Rate=([\d.]+)% \((\d+)/(\d+)\)", output)
    if wp_matches:
        total_hits = sum(int(m[1]) for m in wp_matches)
        total_eval = sum(int(m[2]) for m in wp_matches)
        if total_eval > 0:
            result["pred_hits_window_precision"] = total_hits
            result["pred_total_window_precision"] = total_eval
            result["pred_window_precision_pct"] = (100.0 * total_hits) / total_eval

    stall, pref, avg_load = parse_predict_bandwidth_lines(output)
    if stall or pref or avg_load is not None:
        result["stall_loads"] = stall
        result["prefetch_loads"] = pref
        result["avg_ms_per_expert_load"] = avg_load
    ready, wait, skipped = parse_prefetch_overlap_lines(output)
    if ready or wait or skipped:
        result["prefetch_hits_ready"] = ready
        result["prefetch_hits_wait"] = wait
        result["prefetch_ticks_skipped"] = skipped

    gating_matches = re.findall(r"GatingPrefetch Delivery=([\d.]+)% \((\d+)/(\d+)\)", output)
    if not gating_matches:
        gating_matches = re.findall(r"GatingPrefetch Recall=([\d.]+)% \((\d+)/(\d+)\)", output)
    if gating_matches:
        total_hits = sum(int(m[1]) for m in gating_matches)
        total_eval = sum(int(m[2]) for m in gating_matches)
        if total_eval > 0:
            result["gating_prefetch_hits"] = total_hits
            result["gating_prefetch_total"] = total_eval
            result["gating_prefetch_delivery_pct"] = (100.0 * total_hits) / total_eval
            result["gating_prefetch_recall_pct"] = result["gating_prefetch_delivery_pct"]

    gating_pred_recall_matches = re.findall(r"GatingPredict Recall=([\d.]+)% \((\d+)/(\d+)\)", output)
    if gating_pred_recall_matches:
        total_hits = sum(int(m[1]) for m in gating_pred_recall_matches)
        total_eval = sum(int(m[2]) for m in gating_pred_recall_matches)
        if total_eval > 0:
            result["gating_pred_recall_hits"] = total_hits
            result["gating_pred_recall_total"] = total_eval
            result["gating_pred_recall_pct"] = (100.0 * total_hits) / total_eval

    gating_pred_precision_matches = re.findall(r"GatingPredict Precision=([\d.]+)% \((\d+)/(\d+)\)", output)
    if gating_pred_precision_matches:
        total_hits = sum(int(m[1]) for m in gating_pred_precision_matches)
        total_eval = sum(int(m[2]) for m in gating_pred_precision_matches)
        if total_eval > 0:
            result["gating_pred_precision_hits"] = total_hits
            result["gating_pred_precision_total"] = total_eval
            result["gating_pred_precision_pct"] = (100.0 * total_hits) / total_eval

    prefetch_eff_matches = re.findall(
        r"PrefetchEff: Dropped=(\d+), AlreadyCached=(\d+)", output
    )
    if prefetch_eff_matches:
        result["prefetch_dropped_no_victim"] = sum(int(m[0]) for m in prefetch_eff_matches)
        result["prefetch_already_cached"] = sum(int(m[1]) for m in prefetch_eff_matches)

    trigger_matches = re.findall(r"GatingPrefetch Triggers=(\d+)", output)
    if trigger_matches:
        result["gating_prefetch_triggers"] = sum(int(m) for m in trigger_matches)

    return result


def print_decode_regime_reports_from_df(df: "pd.DataFrame") -> None:
    """Print regime alignment for LRU + each predict row in a sweep CSV."""
    if df is None or df.empty:
        return
    lru_misses_ref: Optional[int] = None
    lru_rows = df[df["label"].astype(str).str.contains("LRU", na=False)]
    if not lru_rows.empty and pd.notna(lru_rows.iloc[0].get("cache_misses")):
        lru_misses_ref = int(lru_rows.iloc[0]["cache_misses"])
    for _, row in df.iterrows():
        if row.get("backend") not in ("cached", "predict"):
            continue
        if not row_has_data(row.to_dict()):
            continue
        d = row.to_dict()
        if lru_misses_ref is not None:
            d["_lru_misses_ref"] = lru_misses_ref
        print_decode_regime_report(d)




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




def iter_cache_lookahead_coupled_pairs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
):
    """Yield (cache_size, lookahead) with LA = max(1, cache_size // divisor)."""
    divisor = int(getattr(args, "cache_lookahead_divisor", 8) or 8)
    for cache_size in args.cache_sizes:
        lookahead = max(1, cache_size // divisor)
        yield cache_size, lookahead


def resolve_oracle_trace_path(args: argparse.Namespace, batch_idx: int) -> str:
    default_dir = "/home/michael/heteroPredict/trainingData/wikitext_test_traces"
    if args.model == "mixtral":
        default_dir = "/home/michael/heteroPredict/trainingData/mixtral_traces"
    trace_dir = getattr(args, "oracle_trace_dir", None) or default_dir
    prefix = f"oracle_trace_{args.model}3_30b" if args.model == "qwen" else "oracle_trace_mixtral_8x7b"
    primary = os.path.join(trace_dir, f"{prefix}_{batch_idx:05d}.txt")
    if os.path.exists(primary):
        return primary
    alt = os.path.join(trace_dir, f"{prefix}_{batch_idx:05d}_exact.txt")
    if os.path.exists(alt):
        return alt
    generic = os.path.join(trace_dir, f"trace_{batch_idx:05d}.txt")
    if os.path.exists(generic):
        return generic
    return primary





def iter_cache_lookahead_pairs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
):
    """Yield (cache_size, lookahead) over the --cache-sizes x --lookaheads grid.

    The optional expert-reuse CSV (``--constraint-expert-reuse-csv``) is used only to
    skip pairs whose cache is too small to be lossless at that lookahead.
    """
    lookaheads = args.lookaheads if args.lookaheads else [1, 2, 4, 6, 8, 16]
    if min_cache_map is None:
        min_cache_map = load_min_cache_per_lookahead(
            getattr(args, "constraint_expert_reuse_csv", None) or ""
        )
    slack = int(getattr(args, "cache_lookahead_slack", 0) or 0)
    for cache_size in args.cache_sizes:
        for lookahead in lookaheads:
            # print('disabled cache size checking')
            yield cache_size, lookahead




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




def build_model_cmd(
    *,
    model: str,
    backend: str,
    cache_size: int,
    lambda_val: float,
    prompts_json: str,
    predictor_path: str = "",
    prefetch_count: Optional[int] = None,
    prefetch_threshold: float = 0.0,
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
    mode_wikitext_perplexity: bool = False,
    generation_perplexity: bool = False,
    expert_reuse_csv: Optional[str] = None,
    predictor_device: Optional[str] = None,
    expert_weights_dir: Optional[str] = None,
    predictor_stride: Optional[int] = None,
    disable_measurement: bool = False,
    cache_policy: Optional[str] = None,
    oracle_trace_path: Optional[str] = None,
    oracle_lookahead: Optional[int] = None,
    oracle_full_union: bool = False,
    oracle_routing_agreement: Optional[float] = None,
    oracle_noise_seed: Optional[int] = None,
    predictor_type: Optional[str] = None,
    gating_lookahead: Optional[int] = None,
    prefetch_non_evicting: bool = False,
    gating_score_percentile: Optional[float] = None,
    speculative_cache_fraction: Optional[float] = None,
) -> List[str]:
    script = QWEN_MODEL_SCRIPT if model == "qwen" else MIXTRAL_MODEL_SCRIPT
    
    # Preferred interface: probability-mass routing threshold.
    # Backward compatibility: if legacy forced_top_p is set, treat it as the same threshold.
    effective_prob_mass = prob_mass_threshold if prob_mass_threshold >= 0.0 else forced_top_p

    run_config = {
        "backend": backend,
        "device": "cuda",
        "lambda_val": lambda_val,
        "sweep_prompts_file": prompts_json,
        "predictor_device": predictor_device,
        "generate": mode_generate,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_cached_experts": cache_size if model == "qwen" else 0,
        "expert_cache": cache_size if model != "qwen" else 0,
        "forced_top_n": forced_top_n,
        "mass_threshold_substitution_p": effective_prob_mass if effective_prob_mass >= 0.0 else -1.0,
        "config_path": config_path,
        "expert_reuse_csv": expert_reuse_csv,
        "expert_weights_dir": expert_weights_dir or (DEFAULT_QWEN_EXPERT_WEIGHTS_DIR if model == "qwen" else None),
        "suppress_predictor_stats": disable_measurement,
        "cache_policy": cache_policy,
        "wikitext103_perplexity": mode_wikitext_perplexity,
        "wikitext103_split": "test",
        "wikitext103_max_length": 4096,
        "wikitext103_stride": 2048,
        "wikitext103_max_windows": 8,
        "generation_perplexity": generation_perplexity,
    }

    if backend == "predict":
        if prefetch_count is not None:
            run_config["prefetch_experts_count"] = prefetch_count
        if prefetch_threshold > 0.0:
            run_config["prefetch_threshold"] = prefetch_threshold
        if predictor_path:
            run_config["predictor_model"] = predictor_path
            
            if predictor_stride is not None:
                run_config["predictor_lookahead"] = predictor_stride
            else:
                m_la = re.search(r"f(\d+)", os.path.basename(os.path.normpath(predictor_path)))
                if m_la:
                    run_config["predictor_lookahead"] = int(m_la.group(1))
                else:
                    fs = read_predictor_future_steps(predictor_path)
                    if fs is not None:
                        run_config["predictor_lookahead"] = fs
                        
        if predict_layers is not None:
            run_config["predict_layers"] = predict_layers

    if oracle_trace_path:
        run_config["oracle_trace_path"] = oracle_trace_path
    if oracle_lookahead is not None:
        run_config["oracle_lookahead"] = oracle_lookahead
    if oracle_full_union:
        run_config["oracle_full_union"] = True
    if oracle_routing_agreement is not None and oracle_routing_agreement < 1.0 - 1e-6:
        run_config["oracle_routing_agreement"] = oracle_routing_agreement
    if oracle_noise_seed is not None:
        run_config["oracle_noise_seed"] = oracle_noise_seed
    if predictor_type:
        run_config["predictor_type"] = predictor_type
    if gating_lookahead is not None:
        run_config["gating_lookahead"] = gating_lookahead
    if gating_score_percentile is not None and gating_score_percentile > 0.0:
        run_config["gating_score_percentile"] = gating_score_percentile
    if speculative_cache_fraction is not None and speculative_cache_fraction > 0.0:
        run_config["speculative_cache_fraction"] = speculative_cache_fraction
    if prefetch_non_evicting:
        run_config["prefetch_non_evicting"] = True

    # Create temporary JSON file for the run config
    fd, temp_json_path = tempfile.mkstemp(suffix=".json", prefix="sweep_run_config_")
    with os.fdopen(fd, "w") as f:
        json.dump(run_config, f)

    cmd: List[str] = [
        sys.executable,
        script,
        "--run-config",
        temp_json_path
    ]
    return cmd


def load_prompts(args: argparse.Namespace) -> List[Any]:
    dataset = args.dataset
    n = args.num_prompts
    max_chars: int = getattr(args, "prompt_max_chars", 500)

    examples_json = getattr(args, "examples_json", None)
    if examples_json:
        with open(examples_json, "r", encoding="utf-8") as f:
            examples = json.load(f)
        prompts: List[Any] = []
        for ex in examples:
            if isinstance(ex, dict):
                p = {
                    "text": (ex.get("text") or "")[:max_chars],
                    "domain": ex.get("domain"),
                    "idx": ex.get("idx"),
                    "gold": ex.get("gold"),
                    "metric": ex.get("metric"),
                }
                if ex.get("ref_text"):
                    p["ref_text"] = ex["ref_text"]
                prompts.append(p)
            else:
                prompts.append({"text": str(ex)[:max_chars]})
        if n > 0:
            prompts = prompts[:n]
        print(f"[sweep] Loaded {len(prompts)} example(s) from {examples_json}", flush=True)
        return prompts

    if dataset == "oracle":
        prompts = []
        utils_dir = os.path.join(ROOT_DIR, "utils")
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        from oracle_trace import load_oracle_trace

        for i in range(n):
            trace_file = resolve_oracle_trace_path(args, i)
            if not os.path.exists(trace_file):
                print(f"[sweep] Oracle trace file not found: {trace_file}", flush=True)
                break
            bundle = load_oracle_trace(trace_file)
            if bundle.prompt_token_ids:
                prompts.append({"text": bundle.prompt_text or "", "token_ids": bundle.prompt_token_ids})
                print(
                    f"[sweep] Oracle trace {trace_file}: using exact PROMPT TOKEN IDS for sweep.",
                    flush=True,
                )
            elif bundle.prompt_text:
                prompts.append({"text": bundle.prompt_text.strip()})
            else:
                print(f"[sweep] Oracle trace file empty: {trace_file}", flush=True)
                break
        
        if prompts:
            print(f"[sweep] Loaded {len(prompts)} prompts from oracle traces.", flush=True)
            return prompts

    if dataset == "txt":
        path = args.prompts_txt or os.path.join(ROOT_DIR, "unified_llm_w4a16", "prompts.txt")
        # Structured examples JSON from cache_eval_prompts.py
        if str(path).endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                raw_obj = json.load(f)
            if isinstance(raw_obj, list) and raw_obj and isinstance(raw_obj[0], dict) and "text" in raw_obj[0]:
                args.examples_json = path
                return load_prompts(args)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        prompts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if n <= 0:
            print(f"[sweep] Loaded all {len(prompts)} prompt(s) from {path}", flush=True)
            return prompts
        return prompts[:n]

    if dataset in HF_EVAL_CHOICES:
        try:
            from prompt_datasets import load_eval_prompts

            prompts = load_eval_prompts(
                dataset,
                n,
                max_chars=max_chars,
                stream_skip=getattr(args, "dataset_offset", None),
            )
            if prompts:
                return prompts
            print(f"[sweep] {dataset} returned no prompts, using default.", flush=True)
        except Exception as e:
            print(f"[sweep] {dataset} load failed: {e}, using default.", flush=True)

    default = (
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    return [default] * n


def _is_flat_predictor_dir(base_dir: str) -> bool:
    """True when *base_dir* contains ``layer_0/`` directly (no ``*_fN`` wrapper)."""
    return os.path.isdir(os.path.join(base_dir, "layer_0"))


def read_predictor_future_steps(base_dir: str) -> Optional[int]:
    """Read ``future_steps`` from ``layer_0/best.json`` (flat or nested layout)."""
    for meta_path in (
        os.path.join(base_dir, "layer_0", "best.json"),
        os.path.join(base_dir, "layer_0", "training_metrics.json"),
    ):
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            fs = data.get("future_steps")
            if fs is not None:
                return int(fs)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def predictor_path_for_lookahead(base_dir: str, lookahead: int) -> str:
    """Return the predictor directory to pass as ``--predictor-model``.

    Supports two layouts:
    - **Flat**: ``base_dir/layer_X/best_jit.pt`` (e.g. ``transformer/``).
    - **Nested**: ``base_dir/*_f{lookahead}/layer_X/`` (legacy ablation runs).

    For flat dirs, ``future_steps`` in ``layer_0/best.json`` should match *lookahead*.
    """
    abs_base = os.path.abspath(base_dir)
    if _is_flat_predictor_dir(abs_base):
        fs = read_predictor_future_steps(abs_base)
        if fs is not None and fs != lookahead:
            print(
                f"[sweep] WARNING: flat predictor {abs_base} has future_steps={fs} "
                f"but sweep lookahead={lookahead}",
                flush=True,
            )
        return abs_base
    suffix = f"_f{lookahead}"
    try:
        for entry in os.scandir(abs_base):
            if entry.is_dir() and entry.name.endswith(suffix):
                return entry.path
    except OSError:
        pass
    # Legacy fallback (original naming convention).
    return os.path.join(abs_base, f"eh1_h32_f{lookahead}")



def cache_large_enough(
    cache_size: int,
    lookahead: int,
    min_cache_map: Dict[int, int],
    slack: int = 0,
) -> bool:
    required = min_cache_map.get(lookahead)
    if required is None:
        return True
    effective = max(1, required - max(0, slack))
    if cache_size >= effective:
        return True
    slack_note = f" (effective {effective} with slack {slack})" if slack else ""
    print(
        f"[sweep] Skipping cache_size={cache_size} / lookahead={lookahead}: "
        f"requires cache >= {required}{slack_note}",
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
    prefetch_threshold: float = 0.0,
    mode_perplexity: bool = True,
    mode_generate: bool = True,
    mode_generation_perplexity: bool = False,
    temperature: Optional[float] = None,
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
        "prefetch_threshold": prefetch_threshold,
        "mode_perplexity": mode_perplexity,
        "mode_generate": mode_generate,
        "mode_generation_perplexity": mode_generation_perplexity,
        "temperature": temperature,
    }












def _prefetch_forced_top_ns_list(args: argparse.Namespace) -> List[int]:
    """Extra predict-backend prefetch lines: forced top-J into cache mask (0 = default single line)."""
    raw = getattr(args, "prefetch_forced_top_ns", None)
    if not raw:
        return [0]
    out = sorted({int(x) for x in raw})
    return out


def _routing_bias_top_ns_list(args: argparse.Namespace) -> List[int]:
    """Forced top-J values for Cache-Cond and Hybrid (Both) rows."""
    raw = getattr(args, "routing_bias_top_ns", None)
    if raw:
        return sorted({int(x) for x in raw})
    return [int(args.routing_bias_top_n)]


def iter_custom_prefetch_budgets(args: argparse.Namespace, cache_size: int) -> List[int]:
    """Prefetch budgets for CUSTOM_1_16 family: explicit list, fractions, or their union."""
    budgets: set[int] = set()
    fractions = list(getattr(args, "budget_fractions", []) or [])
    if getattr(args, "no_budget_fractions", False):
        fractions = []
    if getattr(args, "custom_explicit_prefetch_budgets", False):
        budgets.update(int(b) for b in args.prefetch_budgets if 1 <= int(b) <= int(cache_size))
        for f in fractions:
            budgets.add(max(1, int(cache_size * float(f))))
    else:
        for f in fractions:
            budgets.add(max(1, int(cache_size * float(f))))
    return sorted(budgets)


def apply_row_selection_flags(
    configs: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Filter configs for --cache-cond-only / --both-only / --prefetch-only / --skip-baselines."""
    cache_cond_only = bool(getattr(args, "cache_cond_only", False))
    both_only = bool(getattr(args, "both_only", False))
    prefetch_only = bool(getattr(args, "prefetch_only", False))
    skip_baselines = bool(getattr(args, "skip_baselines", False))

    mode_count = sum(bool(x) for x in (cache_cond_only, both_only, prefetch_only))
    if mode_count > 1:
        raise ValueError(
            "Use only one of --cache-cond-only, --both-only, --prefetch-only"
        )

    def label(c: Dict[str, Any]) -> str:
        return str(c.get("label", ""))

    def is_baseline(c: Dict[str, Any]) -> bool:
        return label(c).startswith("Neither")

    if cache_cond_only:
        configs = [c for c in configs if label(c).startswith("Cache-Cond")]
    elif both_only:
        configs = [c for c in configs if label(c).startswith("Both")]
    elif prefetch_only:
        configs = [
            c for c in configs
            if label(c).startswith("Prefetch Only") or is_baseline(c)
        ]

    if skip_baselines:
        configs = [c for c in configs if not is_baseline(c)]

    return configs


def gating_routed_top_k(model: str) -> int:
    """Max experts the next MoE layer routes to; caps cross-layer gating prefetch."""
    return 8 if model == "qwen" else 2


def effective_gating_prefetch_budget(budget: int, model: str) -> int:
    """Gating only prefetches next-layer router top-k — never more than routed width."""
    return min(int(budget), gating_routed_top_k(model))


def gating_budget_label(budget: int, model: str) -> str:
    gating_b = effective_gating_prefetch_budget(budget, model)
    if gating_b < int(budget):
        return f"B={gating_b} (sweep B={budget})"
    return f"B={gating_b}"


def apply_gating_sweep_fields(
    config: Dict[str, Any],
    *,
    budget: int,
    cache_size: int,
    args: argparse.Namespace,
) -> int:
    """Set gating predictor fields; returns effective prefetch budget (≤ routed top-k)."""
    gating_b = effective_gating_prefetch_budget(budget, args.model)
    config["prefetch_budget"] = gating_b
    config["predictor_type"] = "gating"
    config["gating_lookahead"] = 1
    config["speculative_cache_fraction"] = float(gating_b) / float(cache_size)
    config["prefetch_non_evicting"] = bool(getattr(args, "prefetch_non_evicting", False))
    config["cache_policy"] = "LRU"
    return gating_b


def _valid_lookaheads_for_cache(
    cache_size: int,
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]],
) -> List[int]:
    lookaheads = args.lookaheads if args.lookaheads else [1, 2, 4, 6, 8, 16]
    slack = int(getattr(args, "cache_lookahead_slack", 0) or 0)
    return [
        la for la in lookaheads
        if cache_large_enough(cache_size, la, min_cache_map, slack=slack)
    ]


def _append_lru_baselines_once_per_cache(
    configs: List[Dict[str, Any]],
    *,
    question: str,
    cache_sizes: Sequence[int],
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]],
    mode_perplexity: bool,
    predictor_tag: str,
) -> None:
    """One LRU row per cache size (cached backend ignores lookahead)."""
    if predictor_tag:
        return
    for cache_size in cache_sizes:
        valid = _valid_lookaheads_for_cache(cache_size, args, min_cache_map)
        if not valid:
            continue
        configs.append(cfg(
            question, "Neither (LRU)", "cached",
            cache_size, 0.0, 0,
            lookahead=min(valid), prefetch_budget=None,
            mode_perplexity=mode_perplexity, mode_generate=True,
        ))


def _routing_policy_perplexity(lambda_val: float, args: argparse.Namespace) -> bool:
    """Whether to run the wikitext103 perplexity pass for this config."""
    if getattr(args, "ppl_on_lambda_policies", False):
        return lambda_val > 0.0
    return bool(getattr(args, "custom_include_ppl", False))


def _tag_predictor_config(c: Dict[str, Any], predictor_base_dir: Optional[str], predictor_tag: str) -> None:
    if predictor_base_dir:
        c["predictor_base_dir"] = predictor_base_dir
        c["predictor_tag"] = predictor_tag


def _append_cache_cond_once_per_cache(
    configs: List[Dict[str, Any]],
    *,
    question: str,
    cache_size: int,
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    tag_suffix: str = "",
    bookend: Optional[str] = None,
) -> None:
    """Cache-Cond uses the cached backend — one row per cache size (no lookahead)."""
    for lambda_val in args.lambdas:
        if lambda_val == 0.0:
            continue
        j_values = _routing_bias_top_ns_list(args)
        for j in j_values:
            j_label = f" J={j}" if len(j_values) > 1 else ""
            cc = cfg(
                question, f"Cache-Cond Only λ={lambda_val}{j_label}{tag_suffix}",
                "cached", cache_size, lambda_val,
                j,
                lookahead=None, prefetch_budget=None,
                mode_perplexity=_routing_policy_perplexity(lambda_val, args),
                mode_generate=True,
            )
            if bookend:
                cc["bookend_pass"] = bookend
            _tag_predictor_config(cc, predictor_base_dir, predictor_tag)
            configs.append(cc)


def _custom_1_16_configs_for_cache_grouped(
    args: argparse.Namespace,
    *,
    question: str,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
    mode_perplexity_lru: bool = False,
) -> List[Dict[str, Any]]:
    """Per cache size: LRU + Cache-Cond (start), all Prefetch/Hybrid, LRU + Cache-Cond (end verify)."""
    configs: List[Dict[str, Any]] = []
    if predictor_tag:
        return configs
    prefetch_j_list = _prefetch_forced_top_ns_list(args)
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""

    for cache_size in args.cache_sizes:
        valid_las = _valid_lookaheads_for_cache(cache_size, args, min_cache_map)
        if not valid_las:
            continue

        def _lru_cfg(bookend: str) -> Dict[str, Any]:
            c = cfg(
                question, "Neither (LRU)", "cached",
                cache_size, 0.0, 0,
                lookahead=min(valid_las), prefetch_budget=None, prefetch_threshold=0.0,
                mode_perplexity=mode_perplexity_lru, mode_generate=True,
            )
            c["bookend_pass"] = bookend
            return c

        def _cc_cfg(bookend: str) -> None:
            _append_cache_cond_once_per_cache(
                configs,
                question=question,
                cache_size=cache_size,
                args=args,
                predictor_base_dir=predictor_base_dir,
                predictor_tag=predictor_tag,
                tag_suffix=tag_suffix,
                bookend=bookend,
            )

        configs.append(_lru_cfg("start"))
        _cc_cfg("start")

        for lookahead in valid_las:
            for budget in iter_custom_prefetch_budgets(args, cache_size):
                for pj in prefetch_j_list:
                    pj_label = f" J={pj}" if pj else ""
                    c = cfg(
                        question, f"Prefetch Only B={budget}{pj_label}{tag_suffix}",
                        "predict", cache_size, 0.0,
                        pj,
                        lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                        mode_perplexity=_routing_policy_perplexity(0.0, args),
                        mode_generate=True,
                    )
                    c["bookend_pass"] = "main"
                    _tag_predictor_config(c, predictor_base_dir, predictor_tag)
                    configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    j_values = _routing_bias_top_ns_list(args)
                    for j in j_values:
                        j_label = f" J={j}" if len(j_values) > 1 else ""
                        bc = cfg(
                            question, f"Both λ={lambda_val}{j_label} B={budget}{tag_suffix}",
                            "predict", cache_size, lambda_val,
                            j,
                            lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                            mode_perplexity=_routing_policy_perplexity(lambda_val, args),
                            mode_generate=True,
                        )
                        bc["bookend_pass"] = "main"
                        _tag_predictor_config(bc, predictor_base_dir, predictor_tag)
                        configs.append(bc)

        configs.append(_lru_cfg("end"))
        _cc_cfg("end")

    return configs


def custom_1_16_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    if getattr(args, "cache_grouped_bookends", False):
        do_ppl_all = getattr(args, "custom_include_ppl", False)
        do_ppl_lru = do_ppl_all and not getattr(args, "ppl_on_lambda_policies", False)
        return _custom_1_16_configs_for_cache_grouped(
            args,
            question="CUSTOM_1_16",
            predictor_base_dir=predictor_base_dir,
            predictor_tag=predictor_tag,
            min_cache_map=min_cache_map,
            mode_perplexity_lru=do_ppl_lru,
        )
    configs: List[Dict[str, Any]] = []
    do_ppl_all = getattr(args, "custom_include_ppl", False)
    do_ppl_lru = do_ppl_all and not getattr(args, "ppl_on_lambda_policies", False)
    prefetch_j_list = _prefetch_forced_top_ns_list(args)
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""
    _append_lru_baselines_once_per_cache(
        configs,
        question="CUSTOM_1_16",
        cache_sizes=args.cache_sizes,
        args=args,
        min_cache_map=min_cache_map,
        mode_perplexity=do_ppl_lru,
        predictor_tag=predictor_tag,
    )
    for cache_size in args.cache_sizes:
        _append_cache_cond_once_per_cache(
            configs,
            question="CUSTOM_1_16",
            cache_size=cache_size,
            args=args,
            predictor_base_dir=predictor_base_dir,
            predictor_tag=predictor_tag,
            tag_suffix=tag_suffix,
        )
    for cache_size, lookahead in iter_cache_lookahead_pairs(args, min_cache_map):
            for budget in iter_custom_prefetch_budgets(args, cache_size):
                for pj in prefetch_j_list:
                    pj_label = f" J={pj}" if pj else ""
                    c = cfg(
                        "CUSTOM_1_16", f"Prefetch Only B={budget}{pj_label}{tag_suffix}",
                        "predict", cache_size, 0.0,
                        pj,
                        lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                        mode_perplexity=_routing_policy_perplexity(0.0, args),
                        mode_generate=True,
                    )
                    if predictor_base_dir:
                        c["predictor_base_dir"] = predictor_base_dir
                        c["predictor_tag"] = predictor_tag
                    configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    for j in _routing_bias_top_ns_list(args):
                        j_label = f" J={j}" if len(_routing_bias_top_ns_list(args)) > 1 else ""
                        bc = cfg(
                            "CUSTOM_1_16", f"Both λ={lambda_val}{j_label} B={budget}{tag_suffix}",
                            "predict", cache_size, lambda_val,
                            j,
                            lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                            mode_perplexity=_routing_policy_perplexity(lambda_val, args),
                            mode_generate=True,
                        )
                        if predictor_base_dir:
                            bc["predictor_base_dir"] = predictor_base_dir
                            bc["predictor_tag"] = predictor_tag
                        configs.append(bc)
    return configs


# April 2026 ``lookahead_4way_20prompts`` coupled (cache, lookahead, prefetch budget) per row.




def custom_1_16_no_ppl_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    if getattr(args, "cache_grouped_bookends", False):
        return _custom_1_16_configs_for_cache_grouped(
            args,
            question="CUSTOM_1_16_NO_PPL",
            predictor_base_dir=predictor_base_dir,
            predictor_tag=predictor_tag,
            min_cache_map=min_cache_map,
            mode_perplexity_lru=False,
        )
    configs: List[Dict[str, Any]] = []
    prefetch_j_list = _prefetch_forced_top_ns_list(args)
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""
    prefetch_only = getattr(args, "prefetch_only", False)
    if not prefetch_only:
        configs.extend(random_baseline_configs(args, min_cache_map))
        _append_lru_baselines_once_per_cache(
            configs,
            question="CUSTOM_1_16_NO_PPL",
            cache_sizes=args.cache_sizes,
            args=args,
            min_cache_map=min_cache_map,
            mode_perplexity=False,
            predictor_tag=predictor_tag,
        )
    for cache_size in args.cache_sizes:
        if not prefetch_only:
            _append_cache_cond_once_per_cache(
                configs,
                question="CUSTOM_1_16_NO_PPL",
                cache_size=cache_size,
                args=args,
                predictor_base_dir=predictor_base_dir,
                predictor_tag=predictor_tag,
                tag_suffix=tag_suffix,
            )
    for cache_size, lookahead in iter_cache_lookahead_pairs(args, min_cache_map):
            for budget in iter_custom_prefetch_budgets(args, cache_size):
                for pj in prefetch_j_list:
                    pj_label = f" J={pj}" if pj else ""
                    c = cfg(
                        "CUSTOM_1_16_NO_PPL", f"Prefetch Only B={budget}{pj_label}{tag_suffix}",
                        "predict", cache_size, 0.0,
                        pj,
                        lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                        mode_perplexity=_routing_policy_perplexity(0.0, args),
                        mode_generate=True,
                    )
                    if predictor_base_dir:
                        c["predictor_base_dir"] = predictor_base_dir
                        c["predictor_tag"] = predictor_tag
                    configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    if prefetch_only:
                        continue  # skip hybrid (Both) rows
                    j_values = _routing_bias_top_ns_list(args)
                    for j in j_values:
                        j_label = f" J={j}" if len(j_values) > 1 else ""
                        bc = cfg(
                            "CUSTOM_1_16_NO_PPL", f"Both λ={lambda_val}{j_label} B={budget}{tag_suffix}",
                            "predict", cache_size, lambda_val,
                            j,
                            lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                            mode_perplexity=_routing_policy_perplexity(lambda_val, args),
                            mode_generate=True,
                        )
                        if predictor_base_dir:
                            bc["predictor_base_dir"] = predictor_base_dir
                            bc["predictor_tag"] = predictor_tag
                        configs.append(bc)
                
                if getattr(args, "include_oracle_baselines", False):
                    oracle_cfg = cfg(
                        "CUSTOM_1_16_NO_PPL", f"Oracle Prefetch B={budget}{tag_suffix}",
                        "predict", cache_size, 0.0,
                        0,
                        lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                        mode_perplexity=_routing_policy_perplexity(0.0, args),
                        mode_generate=True,
                    )
                    oracle_cfg["is_oracle"] = True
                    configs.append(oracle_cfg)
                    
            for threshold in getattr(args, "prefetch_thresholds", []):
                if threshold <= 0.0:
                    continue
                for pj in prefetch_j_list:
                    pj_label = f" J={pj}" if pj else ""
                    c = cfg(
                        "CUSTOM_1_16_NO_PPL", f"Prefetch Only T={threshold}{pj_label}{tag_suffix}",
                        "predict", cache_size, 0.0,
                        pj,
                        lookahead=lookahead, prefetch_budget=cache_size, prefetch_threshold=threshold,
                        mode_perplexity=_routing_policy_perplexity(0.0, args),
                        mode_generate=True,
                    )
                    if predictor_base_dir:
                        c["predictor_base_dir"] = predictor_base_dir
                        c["predictor_tag"] = predictor_tag
                    configs.append(c)
                for lambda_val in args.lambdas:
                    if lambda_val == 0.0:
                        continue
                    if prefetch_only:
                        continue
                    j_values = _routing_bias_top_ns_list(args)
                    for j in j_values:
                        j_label = f" J={j}" if len(j_values) > 1 else ""
                        bc = cfg(
                            "CUSTOM_1_16_NO_PPL", f"Both λ={lambda_val}{j_label} T={threshold}{tag_suffix}",
                            "predict", cache_size, lambda_val,
                            j,
                            lookahead=lookahead, prefetch_budget=cache_size, prefetch_threshold=threshold,
                            mode_perplexity=_routing_policy_perplexity(lambda_val, args),
                            mode_generate=True,
                        )
                        if predictor_base_dir:
                            bc["predictor_base_dir"] = predictor_base_dir
                            bc["predictor_tag"] = predictor_tag
                        configs.append(bc)
    return configs


def apply_non_baseline_cache_policy(configs: List[Dict[str, Any]], policy: Optional[str]) -> None:
    """Set cache eviction policy on each config. LRU/RANDOM baselines keep their policy; others get ``policy``."""
    if not policy:
        return
    p = policy.upper()
    for c in configs:
        label = str(c.get("label", ""))
        if label.startswith("Neither (LRU)"):
            c["cache_policy"] = "LRU"
        elif label.startswith("Neither (RANDOM)"):
            c["cache_policy"] = "RANDOM"
        elif label.startswith("Cache-Cond Only"):
            c["cache_policy"] = "LRU"
        else:
            c["cache_policy"] = p


def random_baseline_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """One cached λ=0 row per cache size using RANDOM eviction (no predictor)."""
    configs: List[Dict[str, Any]] = []
    question = "CUSTOM_1_16_NO_PPL"
    for cache_size in args.cache_sizes:
        valid = _valid_lookaheads_for_cache(cache_size, args, min_cache_map)
        if not valid:
            continue
        import os
        force_miss = os.environ.get("FORCE_EXPERT_MISS") == "1"
        label = "Forced Miss Anchor (C=0)" if force_miss else "Neither (RANDOM)"
        c = cfg(
            question, label, "cached",
            cache_size, 0.0, 0,
            lookahead=min(valid), prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c["cache_policy"] = "RANDOM"
        configs.append(c)
    return configs


def paper_baseline_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Paper Table 2 style ablations: LRU-only, gating prefetch, ML predictor, oracle ceiling."""
    configs: List[Dict[str, Any]] = []
    question = "PAPER_BASELINE"
    topk = 8 if args.model == "qwen" else 2

    for cache_size in args.cache_sizes:
        lru = cfg(
            question, "LRU only (cached)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None,
            mode_perplexity=False, mode_generate=True,
        )
        lru["cache_policy"] = "LRU"
        configs.append(lru)

        naive = cfg(
            question, f"Naive k={topk} (cached)", "cached",
            topk, 0.0, 0,
            lookahead=1, prefetch_budget=None,
            mode_perplexity=False, mode_generate=True,
        )
        naive["cache_policy"] = "LRU"
        configs.append(naive)

        for budget in iter_custom_prefetch_budgets(args, cache_size):
            gating = cfg(
                question, f"LRU+Gating {gating_budget_label(budget, args.model)}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=budget,
                mode_perplexity=False, mode_generate=True,
            )
            apply_gating_sweep_fields(gating, budget=budget, cache_size=cache_size, args=args)
            configs.append(gating)

            if predictor_base_dir or args.predictor_base_dir:
                ml = cfg(
                    question, f"LRU+ML B={budget}", "predict",
                    cache_size, 0.0, 0,
                    lookahead=1, prefetch_budget=budget,
                    mode_perplexity=False, mode_generate=True,
                )
                ml["cache_policy"] = "LRU"
                if predictor_base_dir:
                    ml["predictor_base_dir"] = predictor_base_dir
                if predictor_tag:
                    ml["predictor_tag"] = predictor_tag
                configs.append(ml)

            oracle = cfg(
                question, f"Oracle B={budget}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=budget,
                mode_perplexity=False, mode_generate=True,
            )
            oracle["is_oracle"] = True
            oracle["oracle_full_union"] = False
            oracle["cache_policy"] = "LRU"
            configs.append(oracle)

    return configs


def gating_vs_ml_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Fair gating vs ML comparison: same cache, budget fraction, partitioned speculative pool."""
    configs: List[Dict[str, Any]] = []
    question = "GATING_VS_ML"
    tag_suffix = f" [{predictor_tag}]" if predictor_tag else ""

    for cache_size in args.cache_sizes:
        lru = cfg(
            question, f"LRU only (cached){tag_suffix}", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None,
            mode_perplexity=False, mode_generate=True,
        )
        lru["cache_policy"] = "LRU"
        configs.append(lru)

        for budget in iter_custom_prefetch_budgets(args, cache_size):
            frac = float(budget) / float(cache_size)

            gating = cfg(
                question, f"Gating {gating_budget_label(budget, args.model)}{tag_suffix}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=budget,
                mode_perplexity=False, mode_generate=True,
            )
            apply_gating_sweep_fields(gating, budget=budget, cache_size=cache_size, args=args)
            configs.append(gating)

            if predictor_base_dir or args.predictor_base_dir:
                for cache_size_la, lookahead in iter_cache_lookahead_pairs(args, min_cache_map):
                    if cache_size_la != cache_size:
                        continue
                    ml = cfg(
                        question, f"ML B={budget} LA={lookahead}{tag_suffix}", "predict",
                        cache_size, 0.0, 0,
                        lookahead=lookahead, prefetch_budget=budget,
                        mode_perplexity=False, mode_generate=True,
                    )
                    ml["speculative_cache_fraction"] = frac
                    ml["cache_policy"] = "LRU"
                    _tag_predictor_config(ml, predictor_base_dir, predictor_tag)
                    configs.append(ml)

    return configs


def gating_budget_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Gating-heuristic prefetch only, over explicit prefetch budgets B."""
    configs: List[Dict[str, Any]] = []
    question = "GATING_BUDGET_SWEEP"
    for cache_size in args.cache_sizes:
        for budget in iter_custom_prefetch_budgets(args, cache_size):
            gating = cfg(
                question, f"Gating {gating_budget_label(budget, args.model)}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=budget,
                mode_perplexity=False, mode_generate=True,
            )
            apply_gating_sweep_fields(gating, budget=budget, cache_size=cache_size, args=args)
            configs.append(gating)
    return configs


def gating_cc_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Cross-layer gating prefetch + cache-conditional routing (λ=1, forced top-J)."""
    configs: List[Dict[str, Any]] = []
    question = "GATING_CC_SWEEP"
    for cache_size in args.cache_sizes:
        for budget in iter_custom_prefetch_budgets(args, cache_size):
            for j in _routing_bias_top_ns_list(args):
                gating_cc = cfg(
                    question,
                    f"Gating+CC λ=1.0 J={j} {gating_budget_label(budget, args.model)}",
                    "predict",
                    cache_size,
                    1.0,
                    j,
                    lookahead=1,
                    prefetch_budget=budget,
                    mode_perplexity=False,
                    mode_generate=True,
                )
                apply_gating_sweep_fields(gating_cc, budget=budget, cache_size=cache_size, args=args)
                gating_cc["lambda_val"] = 1.0
                gating_cc["forced_top_n"] = j
                configs.append(gating_cc)
    return configs


def lru_vs_gating_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Straight LRU (cached) vs LRU + cross-layer gating prefetch over budget B."""
    configs: List[Dict[str, Any]] = []
    question = "LRU_VS_GATING"
    for cache_size in args.cache_sizes:
        lru = cfg(
            question, "LRU only (cached)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None,
            mode_perplexity=False, mode_generate=True,
        )
        lru["cache_policy"] = "LRU"
        configs.append(lru)

        for budget in iter_custom_prefetch_budgets(args, cache_size):
            gating = cfg(
                question, f"LRU+Gating {gating_budget_label(budget, args.model)}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=budget,
                mode_perplexity=False, mode_generate=True,
            )
            apply_gating_sweep_fields(gating, budget=budget, cache_size=cache_size, args=args)
            configs.append(gating)

        for pct in getattr(args, "gating_score_percentiles", []) or []:
            pct_f = float(pct)
            if pct_f <= 0.0:
                continue
            pct_label = f"p{int(round(pct_f * 100))}"
            gating_pct = cfg(
                question, f"LRU+Gating score>={pct_label}", "predict",
                cache_size, 0.0, 0,
                lookahead=1, prefetch_budget=None,
                mode_perplexity=False, mode_generate=True,
            )
            gating_pct["predictor_type"] = "gating"
            gating_pct["gating_lookahead"] = 1
            gating_pct["gating_score_percentile"] = pct_f
            budgets = list(iter_custom_prefetch_budgets(args, cache_size))
            if budgets:
                spec_b = effective_gating_prefetch_budget(max(budgets), args.model)
                gating_pct["speculative_cache_fraction"] = float(spec_b) / float(cache_size)
                gating_pct["prefetch_budget"] = spec_b
            gating_pct["prefetch_non_evicting"] = bool(getattr(args, "prefetch_non_evicting", False))
            gating_pct["cache_policy"] = "LRU"
            configs.append(gating_pct)
    return configs


def oracle_baseline_sweep_configs(
    args: argparse.Namespace,
    predictor_base_dir: Optional[str] = None,
    predictor_tag: str = "",
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Oracle vs LRU vs RANDOM over cache size, coupled lookahead, and prefetch budget B."""
    configs: List[Dict[str, Any]] = []
    question = "ORACLE_BASELINE_SWEEP"
    seen_lru_cache_sizes = set()
    full_union_only = getattr(args, "oracle_full_union_only", False)
    pair_iter = (
        iter_cache_lookahead_coupled_pairs(args, min_cache_map)
        if getattr(args, "couple_lookahead_to_cache", False)
        else iter_cache_lookahead_pairs(args, min_cache_map)
    )
    lru_only = getattr(args, "lru_only", False)
    for cache_size, lookahead in pair_iter:
        if cache_size not in seen_lru_cache_sizes:
            seen_lru_cache_sizes.add(cache_size)
            lru = cfg(
                question, "Neither (LRU)", "cached",
                cache_size, 0.0, 0,
                lookahead=lookahead, prefetch_budget=None, prefetch_threshold=0.0,
                mode_perplexity=False, mode_generate=True,
            )
            lru["cache_policy"] = "LRU"
            configs.append(lru)

            if not lru_only:
                rnd = cfg(
                    question, "Neither (RANDOM)", "cached",
                    cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=None, prefetch_threshold=0.0,
                    mode_perplexity=False, mode_generate=True,
                )
                rnd["cache_policy"] = "RANDOM"
                configs.append(rnd)

        if not full_union_only and not lru_only:
            for budget in iter_custom_prefetch_budgets(args, cache_size):
                top_b = cfg(
                    question, f"Oracle Top-B B={budget}",
                    "predict", cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                    mode_perplexity=False, mode_generate=True,
                )
                top_b["is_oracle"] = True
                top_b["oracle_full_union"] = False
                configs.append(top_b)
                
                if not getattr(args, "no_actual_predictor", False):
                    pf = cfg(
                        question, f"Actual Predictor B={budget}",
                        "predict", cache_size, 0.0, 0,
                        lookahead=lookahead, prefetch_budget=budget, prefetch_threshold=0.0,
                        mode_perplexity=False, mode_generate=True,
                    )
                    # Pass through predictor properties if added by _make_custom_configs
                    if predictor_base_dir:
                        pf["predictor_base_dir"] = predictor_base_dir
                    if predictor_tag:
                        pf["predictor_tag"] = predictor_tag
                    configs.append(pf)

        if getattr(args, "include_oracle_full_union", False) or full_union_only:
            agreements = getattr(args, "oracle_routing_agreements", None) or [1.0]
            for agreement in agreements:
                if agreement >= 1.0 - 1e-6:
                    label = f"Oracle Full Union LA={lookahead}"
                else:
                    label = f"Oracle Noisy A={agreement:g} LA={lookahead}"
                full = cfg(
                    question, label,
                    "predict", cache_size, 0.0, 0,
                    lookahead=lookahead, prefetch_budget=cache_size, prefetch_threshold=0.0,
                    mode_perplexity=False, mode_generate=True,
                )
                full["is_oracle"] = True
                full["oracle_full_union"] = True
                if agreement < 1.0 - 1e-6:
                    full["oracle_routing_agreement"] = agreement
                configs.append(full)
    return configs


def lambda_fn_sweep_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Sweep λ × forced_top_n × probability_mass_threshold at representative lookahead depths.

    Generates Cache-Cond Only (cached backend) configs to isolate the routing policy
    effect independently of prefetching. Also includes the LRU baseline per (cache_size, lookahead).
    Designed to produce a 2-D heatmap of TPS and PPL as a function of (λ, forcing policy).

    The cached backend never prefetches, so prefetch budget is irrelevant here — every
    Cache-Cond row uses ``prefetch_budget=None`` and there is no budget loop.
    """
    configs: List[Dict[str, Any]] = []
    for cache_size, lookahead in iter_cache_lookahead_pairs(args, min_cache_map):
            configs.append(cfg(
                "LAMBDA_FN_SWEEP", "LRU Baseline", "cached",
                cache_size, 0.0, 0,
                lookahead=lookahead, prefetch_budget=None, prefetch_threshold=0.0,
                # Run generation perplexity pass + generation (TPS); merge fills gen_perplexity + tokens_per_second.
                mode_perplexity=True, mode_generate=True,
            ))

            # λ × forced_top_n grid (count-based forcing)
            for lambda_val in args.lambdas:
                if lambda_val == 0.0:
                    continue
                for fn in args.cache_cond_forced_top_ns:
                    configs.append(cfg(
                        "LAMBDA_FN_SWEEP", f"CacheCond λ={lambda_val} FN={fn}",
                        "cached", cache_size, lambda_val, fn,
                        lookahead=lookahead, prefetch_budget=None, prefetch_threshold=0.0,
                        mode_perplexity=True, mode_generate=True,
                    ))

            # λ × probability-mass threshold grid
            for lambda_val in args.lambdas:
                if lambda_val == 0.0:
                    continue
                for fp in args.probability_mass_thresholds:
                    configs.append(cfg(
                        "LAMBDA_FN_SWEEP", f"CacheCond λ={lambda_val} PM={fp}",
                        "cached", cache_size, lambda_val, 0,
                        prob_mass_threshold=fp,
                        lookahead=lookahead, prefetch_budget=None, prefetch_threshold=0.0,
                        mode_perplexity=True, mode_generate=True,
                    ))

    return configs

def unified_sweep_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    question = "UNIFIED_SWEEP"
    for cache_size in args.cache_sizes:
        rnd = cfg(
            question, "Neither (RANDOM)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        rnd["cache_policy"] = "RANDOM"
        configs.append(rnd)

        oracle_cc = cfg(
            question, "Oracle Full Union + CC (J=6)", "predict",
            cache_size, 1.0, 6,
            lookahead=1, prefetch_budget=cache_size, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        oracle_cc["is_oracle"] = True
        oracle_cc["oracle_full_union"] = True
        configs.append(oracle_cc)
    return configs

def thread_benchmark_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    question = "THREAD_BENCH"
    for cache_size in args.cache_sizes:
        # 1. RANDOM
        c_rand = cfg(
            question, "Neither (RANDOM)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c_rand["cache_policy"] = "RANDOM"
        configs.append(c_rand)

        # 2. Oracle Full Union
        c_oracle = cfg(
            question, "Oracle Full Union", "predict",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=cache_size, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c_oracle["is_oracle"] = True
        c_oracle["oracle_full_union"] = True
        configs.append(c_oracle)

        # 3. Predictor B=12
        c_pred = cfg(
            question, "Prefetch Only B=12", "predict",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=12, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        configs.append(c_pred)

        # 4. Cache Cond Only lambda=1 (J=6)
        c_cc = cfg(
            question, "Cache-Cond Only lambda=1.0", "cached",
            cache_size, 1.0, 6,
            lookahead=1, prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        configs.append(c_cc)

        # 5. Both lambda=1 B=12 (J=6)
        c_both = cfg(
            question, "Both lambda=1.0 B=12", "predict",
            cache_size, 1.0, 6,
            lookahead=1, prefetch_budget=12, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        configs.append(c_both)
        
    return configs

def meeting_sweep_sequential_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    question = "MEETING_SEQ"
    for cache_size in [8, 16, 24]:
        c_rand = cfg(
            question, "Neither (RANDOM)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c_rand["cache_policy"] = "RANDOM"
        configs.append(c_rand)
    return configs

def meeting_sweep_parallel_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    question = "MEETING_PAR"
    
    grid = {
        8: [(1, [2, 4, 6, 8])],
        16: [(1, [4, 8, 12, 16]), (2, [4, 8, 12, 16])],
        24: [(1, [6, 12, 18, 24]), (2, [6, 12, 18, 24]), (3, [6, 12, 18, 24])]
    }
    
    for cache_size in [8, 16, 24]:
        c_rand = cfg(
            question, "Neither (RANDOM)", "cached",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=None, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c_rand["cache_policy"] = "RANDOM"
        configs.append(c_rand)
        
        c_oracle = cfg(
            question, "Oracle Full Union", "predict",
            cache_size, 0.0, 0,
            lookahead=1, prefetch_budget=cache_size, prefetch_threshold=0.0,
            mode_perplexity=False, mode_generate=True,
        )
        c_oracle["is_oracle"] = True
        c_oracle["oracle_full_union"] = True
        configs.append(c_oracle)
        
        if cache_size in grid:
            for la, budgets in grid[cache_size]:
                for b in budgets:
                    c_pred = cfg(
                        question, f"Predictor LA={la} B={b}", "predict",
                        cache_size, 0.0, 0,
                        lookahead=la, prefetch_budget=b, prefetch_threshold=0.0,
                        mode_perplexity=False, mode_generate=True,
                    )
                    configs.append(c_pred)
                    
    return configs

def sec3_3_2_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    question = "SEC3_3_2"
    for cache_size in args.cache_sizes:
        # # LRU
        # c_lru = cfg(question, "Neither (LRU)", "cached", cache_size, 0.0, 0, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
        # c_lru["cache_policy"] = "LRU"
        # configs.append(c_lru)
        
        # RANDOM
        c_rand = cfg(question, "Neither (RANDOM)", "cached", cache_size, 0.0, 0, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
        c_rand["cache_policy"] = "RANDOM"
        configs.append(c_rand)

        # Cache-Cond Only lambda=1.0 FN=4, 5, 6
        for j in [6]:
            c_cc = cfg(question, f"Cache-Cond (J={j})", "cached", cache_size, 1.0, j, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
            configs.append(c_cc)
    return configs


def cc_prompt_compare_configs(
    args: argparse.Namespace,
    min_cache_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Greedy generations: cache-cond at λ=1 for each forced-top-J in ``--cc-forced-top-ns``.

    Default: J=5, J=6, J=0 at T=0. Collects WikiText-103 perplexity plus generation perplexity
    on custom prompt continuations. Responses go to ``--out-dir/generations.md``.
    """
    configs: List[Dict[str, Any]] = []
    question = "CC_PROMPT_COMPARE"
    cc_js = list(getattr(args, "cc_forced_top_ns", [5, 6, 0]))
    temperatures = list(getattr(args, "cc_temperatures", [0.0]))
    for cache_size in args.cache_sizes:
        for temperature in temperatures:
            common = dict(
                lookahead=1,
                prefetch_budget=None,
                prefetch_threshold=0.0,
                mode_perplexity=True,
                mode_generate=True,
                mode_generation_perplexity=True,
                temperature=float(temperature),
            )
            t_label = f"T={temperature:g}"
            for cc_j in cc_js:
                configs.append(cfg(
                    question, f"Cache-Cond (J={cc_j}, λ=1, {t_label})", "cached",
                    cache_size, 1.0, int(cc_j),
                    **common,
                ))
    return configs




def _cfg_key_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def cfg_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    prob_mass_threshold = config.get("prob_mass_threshold", config.get("forced_top_p", -1.0))
    prob_mass_threshold = _cfg_key_optional(prob_mass_threshold)
    if prob_mass_threshold is None:
        prob_mass_threshold = -1.0
    lookahead = config.get("lookahead")
    stride = _cfg_key_optional(config.get("predictor_stride"))
    if stride is None:
        stride = lookahead
    return (
        config["question"],
        str(config.get("label", "")),
        config["backend"],
        config["cache_size"],
        config["lambda_val"],
        config["forced_top_n"],
        prob_mass_threshold,
        lookahead,
        _cfg_key_optional(config.get("prefetch_budget")),
        stride,
        config.get("mode_perplexity"),
        config.get("mode_generate"),
        _cfg_key_optional(config.get("temperature")),
        config.get("bookend_pass") or "main",
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
        "gen_perplexity_std",
        "prompt_gen_perplexity",
        "tokens_per_second",
        "tokens_per_second_std",
        "hit_rate_pct",
        "pred_hit_rate_pct",
        "pred_hit_rate_routed_forced_n_pct",
        "pred_hit_rate_routed_topk_pct",
        "pred_requested_rate_forced_n_pct",
        "pred_requested_rate_topk_pct",
        "pred_window_recall_pct",
        "pred_window_precision_pct",
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
        "pred_hits_window_recall",
        "pred_total_window_recall",
        "pred_hits_window_precision",
        "pred_total_window_precision",
        "stall_loads",
        "prefetch_loads",
        "prefetch_hits_ready",
        "prefetch_hits_wait",
        "prefetch_ticks_skipped",
        "ref_tokens",
    ):
        values = [int(m[key]) for m in good if m.get(key) is not None]
        if values:
            result[key] = sum(values)

    ref_nll = [float(m["ref_nll_sum"]) for m in good if m.get("ref_nll_sum") is not None]
    if ref_nll and result.get("ref_tokens"):
        result["ref_nll_sum"] = sum(ref_nll)
        result["ref_perplexity"] = math.exp(result["ref_nll_sum"] / float(result["ref_tokens"]))

    return result


def _prompt_text_from_obj(prompt: Any) -> str:
    if isinstance(prompt, dict):
        return str(prompt.get("text") or "")
    return str(prompt or "")


def _append_generations_md(
    out: str,
    *,
    out_dir: str,
    label: str,
    cache_size: int,
    forced_top_n: int,
    lambda_val: float,
    temperature: float,
    lookahead: Optional[int],
    batch_idx: int = 0,
    prompt: Any = None,
    predictor_tag: str = "",
    prefetch_budget: Optional[int] = None,
    domain: Optional[str] = None,
) -> None:
    """Append prompt + response blocks for manual verification.

    Writes:
      - generations.md  (human-readable)
      - transcripts.jsonl (one JSON object per prompt/response pair)
    """
    matches = re.findall(r"Generated text only:\n={60}\n(.*?)\n={60}", out, re.DOTALL)
    if not matches:
        return
    os.makedirs(out_dir, exist_ok=True)
    gen_file = os.path.join(out_dir, "generations.md")
    jsonl_file = os.path.join(out_dir, "transcripts.jsonl")
    prompt_text = _prompt_text_from_obj(prompt)
    gold = None
    metric = None
    prompt_idx = None
    if isinstance(prompt, dict):
        gold = prompt.get("gold")
        metric = prompt.get("metric")
        prompt_idx = prompt.get("idx")
        domain = domain or prompt.get("domain")
    # Per-prompt metrics only when this output covers a single prompt (cold path).
    per_prompt_metrics: Dict[str, Optional[float]] = {}
    if len(matches) == 1:
        parsed = parse_output(out)
        for key in ("tokens_per_second", "ref_perplexity", "ref_tokens", "prompt_gen_perplexity"):
            if parsed.get(key) is not None:
                per_prompt_metrics[key] = parsed[key]

    with open(gen_file, "a", encoding="utf-8") as f, open(jsonl_file, "a", encoding="utf-8") as jf:
        for idx, gen_text in enumerate(matches):
            gen_text = gen_text.strip()
            ordinal = batch_idx + idx + 1
            f.write(f"### Prompt {ordinal}\n")
            f.write(f"**Label:** {label}\n")
            if domain:
                f.write(f"**Domain:** {domain}\n")
            if predictor_tag:
                f.write(f"**Predictor:** {predictor_tag}\n")
            f.write(
                f"**Config:** Cache={cache_size} | Top-J={forced_top_n} | "
                f"Lam={lambda_val} | T={temperature:g} | Lookahead={lookahead} | "
                f"B={prefetch_budget}\n\n"
            )
            f.write("**Prompt:**\n")
            f.write(f"```text\n{prompt_text}\n```\n\n")
            f.write("**Response:**\n")
            f.write(f"```text\n{gen_text}\n```\n\n")
            if gold is not None:
                gold_s = gold if isinstance(gold, str) else json.dumps(gold, ensure_ascii=False)
                f.write("**Gold:**\n")
                f.write(f"```text\n{gold_s}\n```\n\n")
            if per_prompt_metrics:
                f.write(
                    "**Metrics:** "
                    + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in per_prompt_metrics.items())
                    + "\n\n"
                )
            f.write("---\n\n")

            record = {
                "prompt_ordinal": ordinal,
                "prompt_idx": prompt_idx if prompt_idx is not None else batch_idx + idx,
                "domain": domain,
                "label": label,
                "predictor_tag": predictor_tag or None,
                "cache_size": cache_size,
                "forced_top_n": forced_top_n,
                "lambda_val": lambda_val,
                "temperature": temperature,
                "lookahead": lookahead,
                "prefetch_budget": prefetch_budget,
                "metric": metric,
                "prompt": prompt_text,
                "response": gen_text,
                "gold": gold,
                **per_prompt_metrics,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    predictor_stride = config.get("predictor_stride")
    predictor_tag = config.get("predictor_tag", "")
    run_temperature = (
        float(config["temperature"])
        if config.get("temperature") is not None
        else float(args.temperature)
    )
    pm_str = f" PM={prob_mass_threshold}" if prob_mass_threshold >= 0.0 else ""
    tag_str = f" predictor={predictor_tag}" if predictor_tag else ""
    stride_str = ""
    if predictor_stride is not None and lookahead is not None and int(predictor_stride) != int(lookahead):
        stride_str = f" stride={predictor_stride}"
    print(
        f"\n{'=' * 18} {config['question']} | {config['label']} | "
        f"C={cache_size} λ={lambda_val} FN={forced_top_n} T={run_temperature:g}{pm_str}"
        f"{f' LA={lookahead}{stride_str} B={budget}' if lookahead is not None else ''}{tag_str} "
        f"[{run_idx + 1}/{total_runs}] {'=' * 18}",
        flush=True,
    )
    if run_idx == 0:
        print(
            f"[sweep] Each config is a **new** {args.model} subprocess (full MoE load + kernels). "
            "Long silence after startup logs is often weight load / compile, not a hang.",
            flush=True,
        )

    # Use per-config predictor_base_dir override if present (multi-predictor comparison)
    effective_predictor_base_dir = config.get("predictor_base_dir", args.predictor_base_dir)
    predictor_path = ""
    predictor_type = config.get("predictor_type")
    if config["backend"] == "predict" and not config.get("is_oracle", False) and predictor_type != "gating":
        assert lookahead is not None
        predictor_path = predictor_path_for_lookahead(effective_predictor_base_dir, lookahead)

    def run_pass(*, mode_perplexity: bool, mode_generate: bool, one_prompt_path: str, batch_idx: Optional[int] = None) -> Optional[Dict[str, Optional[float]]]:
        cmd = build_model_cmd(
            model=args.model,
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
            temperature=run_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            mode_generate=mode_generate,
            mode_wikitext_perplexity=mode_perplexity,
            generation_perplexity=mode_generate and bool(config.get("mode_generation_perplexity", False)),
            predictor_device=args.predictor_device,
            expert_weights_dir=getattr(args, "expert_weights_dir", None),
            predictor_stride=predictor_stride,
            disable_measurement=getattr(args, "disable_measurement", False),
            cache_policy=config.get("cache_policy"),
            oracle_trace_path=resolve_oracle_trace_path(args, batch_idx if batch_idx is not None else 0) if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
            oracle_lookahead=lookahead if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
            oracle_full_union=config.get("oracle_full_union", False),
            oracle_routing_agreement=config.get("oracle_routing_agreement"),
            oracle_noise_seed=getattr(args, "oracle_noise_seed", None),
            predictor_type=predictor_type,
            gating_lookahead=config.get("gating_lookahead"),
            prefetch_non_evicting=bool(config.get("prefetch_non_evicting", False)),
            gating_score_percentile=config.get("gating_score_percentile"),
            speculative_cache_fraction=config.get("speculative_cache_fraction"),
        )

        out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
        if out is None:
            return None
            
        if mode_generate and out:
            matches = re.findall(r"Generated text only:\n={60}\n(.*?)\n={60}", out, re.DOTALL)
            # One subprocess may cover many prompts; pair each gen with its prompt.
            if matches:
                for i, gen_text in enumerate(matches):
                    prompt_obj = prompts[i] if i < len(prompts) else (
                        prompts[batch_idx] if batch_idx is not None and batch_idx < len(prompts) else None
                    )
                    # Reconstruct a tiny fake "out" with one block so helper stays simple
                    fake_out = (
                        "Generated text only:\n"
                        + ("=" * 60) + "\n"
                        + gen_text
                        + "\n" + ("=" * 60)
                    )
                    _append_generations_md(
                        fake_out,
                        out_dir=getattr(args, "out_dir", "."),
                        label=config.get("label", ""),
                        cache_size=cache_size,
                        forced_top_n=forced_top_n,
                        lambda_val=lambda_val,
                        temperature=run_temperature,
                        lookahead=lookahead,
                        batch_idx=(batch_idx if batch_idx is not None else 0) + i,
                        prompt=prompt_obj,
                        predictor_tag=str(predictor_tag or ""),
                        prefetch_budget=budget,
                        domain=getattr(args, "domain", None),
                    )
                    
        return parse_output(out)

    merged: Dict[str, Optional[float]] = {
        "gen_perplexity": None,
        "gen_perplexity_std": None,
        "prompt_gen_perplexity": None,
        "prompt_gen_perplexity_std": None,
        "tokens_per_second": None,
        "tokens_per_second_std": None,
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
        "prefetch_hits_ready": None,
        "prefetch_hits_wait": None,
        "prefetch_ticks_skipped": None,
        "avg_ms_per_expert_load": None,
    }

    def merge_metrics(metrics: Optional[Dict[str, Optional[float]]]) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if value is not None:
                merged[key] = value

    if not args.cold_per_prompt:
        want_ppl = config.get("mode_perplexity", False)
        want_gen = config.get("mode_generate", False)
        if want_ppl:
            merge_metrics(run_pass(mode_perplexity=True, mode_generate=False, one_prompt_path=prompts_file, batch_idx=0))
        if want_gen:
            merge_metrics(run_pass(mode_perplexity=False, mode_generate=True, one_prompt_path=prompts_file, batch_idx=0))
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
                        model=args.model,
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
                        temperature=run_temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        mode_generate=False,
                        mode_wikitext_perplexity=True,
                        predictor_device=args.predictor_device,
                        expert_weights_dir=getattr(args, "expert_weights_dir", None),
                        predictor_stride=predictor_stride,
                        disable_measurement=getattr(args, "disable_measurement", False),
                        cache_policy=config.get("cache_policy"),
                        oracle_trace_path=resolve_oracle_trace_path(args, batch_idx if batch_idx is not None else 0) if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
                        oracle_lookahead=lookahead if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
                        oracle_full_union=config.get("oracle_full_union", False),
                        oracle_routing_agreement=config.get("oracle_routing_agreement"),
                        oracle_noise_seed=getattr(args, "oracle_noise_seed", None),
                        predictor_type=predictor_type,
                        gating_lookahead=config.get("gating_lookahead"),
                        prefetch_non_evicting=bool(config.get("prefetch_non_evicting", False)),
                        gating_score_percentile=config.get("gating_score_percentile"),
                        speculative_cache_fraction=config.get("speculative_cache_fraction"),
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

        if config.get("mode_generate", False) and not config.get("mode_perplexity", False):
            outputs = []
            want_gen_ppl = bool(config.get("mode_generation_perplexity", False))
            for batch_idx, prompt in enumerate(prompts):
                if args.drop_page_cache_between_prompts:
                    drop_page_cache()
                fd, one_prompt_path = tempfile.mkstemp(suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump([prompt], f)
                    cmd = build_model_cmd(
                        model=args.model,
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
                        temperature=run_temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        mode_generate=True,
                        mode_wikitext_perplexity=False,
                        generation_perplexity=want_gen_ppl,
                        predictor_device=args.predictor_device,
                        expert_weights_dir=getattr(args, "expert_weights_dir", None),
                        predictor_stride=predictor_stride,
                        disable_measurement=getattr(args, "disable_measurement", False),
                        cache_policy=config.get("cache_policy"),
                        oracle_trace_path=resolve_oracle_trace_path(args, batch_idx if batch_idx is not None else 0) if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
                        oracle_lookahead=lookahead if config.get("backend") == "predict" and config.get("is_oracle", False) else None,
                        oracle_full_union=config.get("oracle_full_union", False),
                        oracle_routing_agreement=config.get("oracle_routing_agreement"),
                        oracle_noise_seed=getattr(args, "oracle_noise_seed", None),
                        predictor_type=predictor_type,
                        gating_lookahead=config.get("gating_lookahead"),
                        prefetch_non_evicting=bool(config.get("prefetch_non_evicting", False)),
                        gating_score_percentile=config.get("gating_score_percentile"),
                        speculative_cache_fraction=config.get("speculative_cache_fraction"),
                    )
                    out = run_subprocess(cmd, timeout=args.subprocess_timeout, log_file=args.log_file)
                    if out:
                        outputs.append(out)
                        _append_generations_md(
                            out,
                            out_dir=getattr(args, "out_dir", "."),
                            label=config.get("label", ""),
                            cache_size=cache_size,
                            forced_top_n=forced_top_n,
                            lambda_val=lambda_val,
                            temperature=run_temperature,
                            lookahead=lookahead,
                            batch_idx=batch_idx,
                            prompt=prompt,
                            predictor_tag=str(predictor_tag or ""),
                            prefetch_budget=budget,
                            domain=getattr(args, "domain", None),
                        )
                finally:
                    try:
                        os.remove(one_prompt_path)
                    except OSError:
                        pass
            merge_metrics(aggregate_cold_prompt_metrics(outputs))

    if config["question"] == "CC_PROMPT_COMPARE":
        gen_file = os.path.join(getattr(args, "out_dir", "."), "generations.md")
        with open(gen_file, "a", encoding="utf-8") as f:
            f.write(f"## {config.get('label', '')}\n\n")
            if merged.get("gen_perplexity") is not None:
                f.write(f"- **WikiText-103 perplexity:** {merged['gen_perplexity']:.4f}\n")
            if merged.get("prompt_gen_perplexity") is not None:
                ppl_line = f"- **Custom-prompt generation perplexity:** {merged['prompt_gen_perplexity']:.4f}"
                if merged.get("prompt_gen_perplexity_std") is not None:
                    ppl_line += f" (σ={merged['prompt_gen_perplexity_std']:.4f})"
                f.write(ppl_line + "\n")
            if merged.get("tokens_per_second") is not None:
                f.write(f"- **TPS:** {merged['tokens_per_second']:.2f}\n")
            f.write("\n---\n\n")

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
        "predictor_stride": predictor_stride if predictor_stride is not None else lookahead,
        "prefetch_budget": budget,
        "predictor_tag": predictor_tag,
        "oracle_routing_agreement": config.get("oracle_routing_agreement"),
        "mode_perplexity": config.get("mode_perplexity", False),
        "mode_generate": config.get("mode_generate", False),
        "temperature": run_temperature,
        "bookend_pass": config.get("bookend_pass", "main"),
        **merged,
    }


# Using the exact style provided from plot_ablation.py
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
}
# Okabe-Ito-based palette: colorblind-safe and high-contrast on white.
ACCENT = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#117733"]
LOOKAHEAD_COLORS = {1: "#0072B2", 3: "#009E73", 5: "#E69F00", 8: "#D55E00", 12: "#CC79A7", 16: "#332288"}
BUDGET_MARKERS = {8: "o", 16: "s", 32: "^", 48: "D"}
LRU_COLOR = "#333333"
LRU_STYLE = dict(color=LRU_COLOR, linestyle="--", linewidth=2.5, zorder=10)


def lookahead_color(lookahead: int) -> str:
    return LOOKAHEAD_COLORS.get(int(lookahead), "#80cbc4")


def budget_marker(budget: int) -> str:
    return BUDGET_MARKERS.get(int(budget), "x")


def style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()


def base_lookup(df_base: Optional[pd.DataFrame], metric: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if df_base is None:
        return result
    for _, row in df_base[df_base[metric].notna()].iterrows():
        result[int(row["cache_size"])] = float(row[metric])
    return result


# Set in generate_all_plots(); used by save_plot() for figure footers.
_current_plot_command: Optional[str] = None


def format_plot_command_caption(command: str, max_width: int = 132) -> str:
    cmd = " ".join(str(command).split())
    if len(cmd) > 800:
        cmd = cmd[:797] + "..."
    return "\n".join(textwrap.wrap(f"CMD: {cmd}", width=max_width))


def read_plot_command(out_dir: str) -> Optional[str]:
    for name in ("command.txt", "sweep_command.txt"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                return text
    return None


def write_plot_command_file(out_dir: str, command: str, *, overwrite: bool = False) -> None:
    """Persist the launcher command for plot footers (see read_plot_command)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "command.txt")
    if not overwrite and os.path.isfile(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(command.strip() + "\n")
    sweep_path = os.path.join(out_dir, "sweep_command.txt")
    with open(sweep_path, "w", encoding="utf-8") as f:
        f.write(command.strip() + "\n")


def add_plot_command_footer(fig: plt.Figure, command: Optional[str]) -> None:
    # Command footer disabled to match thesis styling
    return


def save_plot(fig: plt.Figure, out_dir: str, filename: str, *, command_caption: Optional[str] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cmd = command_caption
    if cmd is None:
        cmd = _current_plot_command or read_plot_command(out_dir)
    add_plot_command_footer(fig, cmd)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}", flush=True)


def _recall_metric_name(df: pd.DataFrame) -> str:
    if "pred_hit_rate_routed_topk_pct" in df.columns and df["pred_hit_rate_routed_topk_pct"].notna().any():
        return "pred_hit_rate_routed_topk_pct"
    return "pred_hit_rate_pct"


def _cache_cond_rows(sub: pd.DataFrame, lambda_val: float, metric_col: str) -> pd.DataFrame:
    """Cache-Cond rows for plotting; budget-independent (dedupe legacy per-budget duplicates)."""
    prefix = f"Cache-Cond Only λ={lambda_val}"
    return (
        sub[sub["label"].str.startswith(prefix, na=False)]
        .dropna(subset=[metric_col, "lookahead"])
        .sort_values("lookahead")
        .drop_duplicates(subset=["lookahead"], keep="first")
    )


def _draw_prefetch_lookahead_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    cache_size: int,
    *,
    metric_col: str,
    ylabel: str,
    lookaheads_sub: List[float],
    pref_mask: pd.Series,
    budgets: List[float],
    n_budgets: int,
    lru_reference_by_la: Optional[Dict[float, float]] = None,
    annotate_speedup: bool = False,
) -> None:
    """Shared lookahead x-axis panel for prefetch curves (+ optional LRU reference)."""
    budget_cmap = plt.cm.Blues
    lru_hr_by_la: Dict[float, float] = dict(lru_reference_by_la or {})

    if metric_col == "_tps_speedup":
        ax.axhline(1.0, **LRU_STYLE, label="LRU (1.00×)")
    elif metric_col in ("hit_rate_pct", "tokens_per_second"):
        lru_sub = (
            sub[sub["label"] == "Neither (LRU)"]
            .dropna(subset=[metric_col, "lookahead"])
            .sort_values("lookahead")
        )
        if not lru_sub.empty:
            ax.plot(
                lru_sub["lookahead"],
                lru_sub[metric_col],
                marker="D",
                **LRU_STYLE,
                label="LRU Baseline",
            )
            for _, r in lru_sub.iterrows():
                lru_hr_by_la[float(r["lookahead"])] = float(r[metric_col])

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
            s_b["lookahead"],
            s_b[metric_col],
            marker="^",
            color=color,
            linewidth=2,
            label=f"Prefetch B={int(budget)} ({frac:.0%})",
        )
        if annotate_speedup and metric_col == "_tps_speedup":
            for _, r in s_b.iterrows():
                val = float(r[metric_col])
                ax.annotate(
                    f"×{val:.2f}",
                    xy=(float(r["lookahead"]), val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                    alpha=0.9,
                )
        elif metric_col == "hit_rate_pct" and lru_hr_by_la:
            for _, r in s_b.iterrows():
                la = float(r["lookahead"])
                val = float(r[metric_col])
                lru = lru_hr_by_la.get(la)
                if lru is not None:
                    ax.annotate(
                        f"{val - lru:+.1f}pp",
                        xy=(la, val),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=color,
                        fontweight="bold",
                        alpha=0.9,
                    )

    ax.set_xticks(lookaheads_sub)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    if metric_col == "hit_rate_pct":
        ax.set_ylim(top=min(100.0, max(ax.get_ylim()[1], 1.0) + 5))
    style_ax(ax, ylabel, "Lookahead Depth (Tokens)", ylabel)


def have_plot_deps() -> bool:
    return plt is not None and np is not None
















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
        metrics.append(("tokens_per_second", "TPS Speedup (vs LRU) (↑ better)"))
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

                # Cache-Cond (once per λ; budget-independent) and Both (per budget)
                for l_idx, lambda_val in enumerate(lambdas):
                    cc_color   = ["#ffcc80", "#ff9800", "#e65100"][l_idx % 3]
                    both_color = ["#a5d6a7", "#4caf50", "#1b5e20"][l_idx % 3]
                    s_cc = _cache_cond_rows(sub, lambda_val, metric_col)
                    if not s_cc.empty:
                        ax.plot(
                            s_cc["lookahead"], s_cc[metric_col],
                            marker="o", color=cc_color, linewidth=1.5, linestyle=":",
                            label=f"Cache-Cond λ={lambda_val}",
                        )
                        _annotate(s_cc, cc_color, above=False)
                    both_mask = sub["label"].str.startswith(f"Both λ={lambda_val}", na=False)
                    for budget in sorted(sub.loc[both_mask, "prefetch_budget"].dropna().unique(), key=float):
                        frac = budget / cache_size if cache_size else 0
                        s_both = (
                            sub[both_mask & (sub["prefetch_budget"] == budget)]
                            .dropna(subset=[metric_col])
                            .sort_values("lookahead")
                        )
                        if s_both.empty:
                            continue
                        ax.plot(
                            s_both["lookahead"], s_both[metric_col],
                            marker="s", color=both_color, linewidth=1.5, linestyle=":",
                            label=f"Both λ={lambda_val} B={int(budget)} ({frac:.0%})",
                        )
                        _annotate(s_both, both_color, above=True)

                ax.set_xticks(lookaheads_sub)
                if is_hitrate:
                    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(100, ax.get_ylim()[1] + 5))
                style_ax(ax, f"Cache = {cache_size} Experts/Layer", "Lookahead Depth (Tokens)", ylabel)

        plt.tight_layout()
        save_plot(fig, out_dir, f"custom_1_16_summary_{ts}.png")




def plot_hit_rates(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Plot cache/predictor rates vs lookahead depth (one figure per cache size)."""
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
    recall_col = _recall_metric_name(df)

    for cache_size in cache_sizes:
        sub = df[df["cache_size"] == cache_size]
        lookaheads_sub = sorted(sub["lookahead"].dropna().unique())
        pref_mask = sub["label"].str.startswith("Prefetch Only", na=False)
        budgets = sorted(sub.loc[pref_mask, "prefetch_budget"].dropna().unique(), key=float)
        n_budgets = max(len(budgets), 1)

        hr_metrics: List[Tuple[str, str]] = []
        if has_cache_hr:
            hr_metrics.append(("hit_rate_pct", "Cache Hit Rate (%)"))
        if has_pred_hr:
            hr_metrics.append((recall_col, "Predictor Recall on Routed Top-K (%)"))
        if has_pred_precision:
            hr_metrics.append(("pred_requested_rate_topk_pct", "Predicted Experts Requested by Router (%)"))

        ncols = len(hr_metrics)
        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
            fig.suptitle(
                f"Hit Rates vs Lookahead Depth — Cache = {cache_size} Experts/Layer",
                fontsize=13,
                fontweight="bold",
                y=1.03,
            )
            for col, (metric_col, panel_ylabel) in enumerate(hr_metrics):
                _draw_prefetch_lookahead_panel(
                    axes[0][col],
                    sub,
                    int(cache_size),
                    metric_col=metric_col,
                    ylabel=panel_ylabel,
                    lookaheads_sub=lookaheads_sub,
                    pref_mask=pref_mask,
                    budgets=budgets,
                    n_budgets=n_budgets,
                )
            plt.tight_layout()
            save_plot(fig, out_dir, f"hit_rates_C{cache_size}_{ts}.png")




def plot_advisor_dashboard(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Single figure: raw TPS, recall, cache hit rate, and precision vs lookahead."""
    df = df_custom.copy()
    if df.empty:
        return
    df = df.sort_values("lookahead")

    has_tps = df["tokens_per_second"].notna().any()
    recall_col = _recall_metric_name(df)
    has_recall = df[recall_col].notna().any() if recall_col in df.columns else False
    has_cache_hr = df["hit_rate_pct"].notna().any()
    has_precision = df["pred_requested_rate_topk_pct"].notna().any()
    if not (has_tps or has_recall or has_cache_hr or has_precision):
        return

    metrics: List[Tuple[str, str]] = []
    if has_tps:
        metrics.append(("tokens_per_second", "TPS Speedup (vs LRU)"))
    if has_cache_hr:
        metrics.append(("hit_rate_pct", "Cache Hit Rate (%)"))
    if has_recall:
        metrics.append((recall_col, "Predictor Recall (%)"))
    if has_precision:
        metrics.append(("pred_requested_rate_topk_pct", "Precision (%)"))

    cache_sizes = sorted(df["cache_size"].dropna().unique(), key=int)
    for cache_size in cache_sizes:
        sub = df[df["cache_size"] == cache_size].copy()
        lookaheads_sub = sorted(sub["lookahead"].dropna().unique())
        pref_mask = sub["label"].str.startswith("Prefetch Only", na=False)
        budgets = sorted(sub.loc[pref_mask, "prefetch_budget"].dropna().unique(), key=float)
        n_budgets = max(len(budgets), 1)

        ncols = len(metrics)
        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.2), squeeze=False)
            fig.suptitle(
                f"Prefetch Advisor Dashboard — Cache = {cache_size} Experts/Layer",
                fontsize=13,
                fontweight="bold",
                y=1.03,
            )
            for col, (metric_col, panel_ylabel) in enumerate(metrics):
                _draw_prefetch_lookahead_panel(
                    axes[0][col],
                    sub,
                    int(cache_size),
                    metric_col=metric_col,
                    ylabel=panel_ylabel,
                    lookaheads_sub=lookaheads_sub,
                    pref_mask=pref_mask,
                    budgets=budgets,
                    n_budgets=n_budgets,
                )
            plt.tight_layout()
            save_plot(fig, out_dir, f"advisor_dashboard_C{cache_size}_{ts}.png")


def plot_precision_recall_frontier(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Plot predictor precision-recall tradeoff for each cache size.

    X-axis: predicted experts requested by router (precision-like) or window precision
    Y-axis: routed experts found in predictions (recall-like) or window recall
    Color: tokens/sec (performance overlay)
    """
    df = df_custom.copy()
    if df.empty:
        return

    # Prefer window-aware precision/recall if available, fallback to topk metrics or legacy hit rate
    if "pred_window_recall_pct" in df.columns and df["pred_window_recall_pct"].notna().any():
        recall_col = "pred_window_recall_pct"
        precision_col = "pred_window_precision_pct"
        recall_label = "Window Recall, ρ  (%)"
        precision_label = "Window Precision, π  (%)"
    else:
        if "pred_hit_rate_routed_topk_pct" not in df.columns or df["pred_hit_rate_routed_topk_pct"].isna().all():
            df["pred_hit_rate_routed_topk_pct"] = df.get("pred_hit_rate_pct")
        if "pred_requested_rate_topk_pct" not in df.columns:
            df["pred_requested_rate_topk_pct"] = np.nan
        recall_col = "pred_hit_rate_routed_topk_pct"
        precision_col = "pred_requested_rate_topk_pct"
        recall_label = "Routed Experts Found in Predictions (%)  [recall-like]"
        precision_label = "Predicted Experts Requested by Router (%)  [precision-like]"

    # Find LRU baseline rows from the unfiltered custom dataframe to compute normalization factor
    lru_mask = (df_custom["label"] == "Neither (LRU)") | (df_custom["label"] == "LRU Baseline")
    if "backend" in df_custom.columns:
        lru_mask |= (df_custom["backend"] == "cached") & (df_custom.get("lambda_val", 0.0).fillna(0.0) == 0.0)
    df_lru_all = df_custom[lru_mask]

    # Frontier only meaningful on predictor rows with both metrics present.
    pred_mask = df["backend"] == "predict"
    need_cols = [precision_col, recall_col]
    df = df[pred_mask].dropna(subset=need_cols)
    if df.empty:
        return

    cache_sizes = sorted(df["cache_size"].dropna().unique(), key=int)
    for cache_size in cache_sizes:
        sub = df[df["cache_size"] == cache_size].copy()
        if sub.empty:
            continue

        # Compute average LRU tokens per second for this cache size to normalize the color axis
        sub_lru = df_lru_all[df_lru_all["cache_size"] == cache_size]
        lru_tps = sub_lru["tokens_per_second"].dropna().mean() if not sub_lru.empty else None

        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            cmap = plt.cm.viridis
            has_tps = sub["tokens_per_second"].notna().any()
            # Calculate marker size based on prefetch budget amortized over lookahead (B / N)
            la_series = pd.to_numeric(sub.get("lookahead", 1), errors="coerce").fillna(1).replace(0, 1)
            b_series = pd.to_numeric(sub.get("prefetch_budget", 0), errors="coerce").fillna(0)
            amortized_budget = b_series / la_series
            marker_sizes = np.clip(amortized_budget * 50, 20, 400)

            if has_tps:
                cvals = sub["tokens_per_second"].astype(float)
                if lru_tps and lru_tps > 0:
                    cvals = cvals / lru_tps
                    cbar_label = "Relative Throughput (LRU = 1.00×) (↑ better)"
                else:
                    cbar_label = "TPS Speedup (vs LRU) (↑ better)"
                sc = ax.scatter(
                    sub[precision_col],
                    sub[recall_col],
                    c=cvals,
                    cmap=cmap,
                    s=70,
                    alpha=0.9,
                    edgecolors="#111111",
                    linewidths=0.5,
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(cbar_label)
            else:
                ax.scatter(
                    sub[precision_col],
                    sub[recall_col],
                    color="#4fc3f7",
                    s=70,
                    alpha=0.9,
                    edgecolors="#111111",
                    linewidths=0.5,
                )

            # Annotate each point with lookahead / budget for quick filtering.
            for _, r in sub.iterrows():
                la = int(r["lookahead"]) if pd.notna(r.get("lookahead")) else -1
                b = int(r["prefetch_budget"]) if pd.notna(r.get("prefetch_budget")) else -1
                label = f"LA{la}/B{b}"
                ax.annotate(
                    label,
                    (float(r[precision_col]), float(r[recall_col])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    alpha=0.9,
                )

            ax.set_xlim(left=max(0.0, ax.get_xlim()[0]), right=min(100.0, max(100.0, ax.get_xlim()[1])))
            ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]), top=min(100.0, max(100.0, ax.get_ylim()[1])))
            style_ax(
                ax,
                f"Precision vs Recall Frontier — Cache = {cache_size}",
                precision_label,
                recall_label,
            )
            plt.tight_layout()
            save_plot(fig, out_dir, f"precision_recall_frontier_C{cache_size}_{ts}.png")


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
        metrics_to_plot.append(("tokens_per_second", "TPS Speedup (vs LRU)", True))

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
            ax1.set_ylabel("TPS Speedup (vs LRU) (↑ better)")
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
                    edgecolors="#333333",
                    linewidths=0.6,
                    label=lbl,
                    zorder=3,
                )
            ax.set_xlabel("Generation perplexity (↓ better)")
            ax.set_ylabel("TPS Speedup (vs LRU) (↑ better)")
            ax.set_title("PPL vs TPS tradeoff (LAMBDA_FN_SWEEP)", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.5)
            ax.legend(fontsize=8, loc="best")
            plt.tight_layout()
            save_plot(fig2, out_dir, f"lambda_fn_sweep_ppl_vs_tps_{ts}.png")


def plot_prefetch_speedup_attribution(df_custom: pd.DataFrame, out_dir: str, ts: str) -> None:
    """Prefetch-only rows: relate decode TPS to stall loads and predictor recall/precision.

    Recall-like: ``pred_hit_rate_routed_topk_pct`` (else ``pred_hit_rate_pct``): routed experts
    covered by the predictor prefetch set.

    Precision-like: ``pred_requested_rate_topk_pct``: predicted experts that the router actually
    requested (when the model prints that line).

    Lower ``stall_loads`` with higher TPS indicates speedup from hiding expert-load latency.
    """
    lam = pd.to_numeric(df_custom.get("lambda_val", 0), errors="coerce").fillna(0.0)
    lab = df_custom["label"].astype(str)
    mask = (
        (df_custom.get("backend", "").astype(str) == "predict")
        & (lam == 0.0)
        & lab.str.startswith("Prefetch Only", na=False)
    )
    sub = df_custom.loc[mask].copy()
    if sub.empty:
        return

    sub["tokens_per_second"] = pd.to_numeric(sub["tokens_per_second"], errors="coerce")
    sub["stall_loads"] = pd.to_numeric(sub["stall_loads"], errors="coerce")
    sub["prefetch_loads"] = pd.to_numeric(sub["prefetch_loads"], errors="coerce")
    sub["lookahead"] = pd.to_numeric(sub["lookahead"], errors="coerce")

    recall = pd.to_numeric(sub.get("pred_hit_rate_routed_topk_pct"), errors="coerce")
    if recall.isna().all():
        recall = pd.to_numeric(sub.get("pred_hit_rate_pct"), errors="coerce")
    precision = pd.to_numeric(sub.get("pred_requested_rate_topk_pct"), errors="coerce")

    plot_df = sub.assign(_recall=recall, _precision=precision).dropna(subset=["tokens_per_second", "stall_loads"])
    if plot_df.empty:
        return

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), squeeze=False)
        ax0, ax1 = axes[0]

        r0 = plot_df["_recall"].dropna()
        if len(r0) > 0:
            sc0 = ax0.scatter(
                plot_df["stall_loads"],
                plot_df["tokens_per_second"],
                c=plot_df["_recall"],
                cmap="viridis",
                s=np.clip(plot_df["prefetch_budget"].fillna(8) * 2.5, 25, 120),
                alpha=0.88,
                edgecolors="#333333",
                linewidths=0.25,
                vmin=float(r0.min()),
                vmax=float(r0.max()),
            )
            ax0.set_xlabel("Stall loads (blocking on-demand expert loads)")
            ax0.set_ylabel("TPS Speedup (vs LRU)")
            ax0.set_title("Decode TPS vs stalls\ncolour = predictor recall (%)", fontsize=11, fontweight="bold")
            ax0.grid(True, alpha=0.45)
            fig.colorbar(sc0, ax=ax0, shrink=0.82, label="Recall (%)")
        else:
            ax0.scatter(
                plot_df["stall_loads"],
                plot_df["tokens_per_second"],
                c="#4fc3f7",
                s=np.clip(plot_df["prefetch_budget"].fillna(8) * 2.5, 25, 120),
                alpha=0.88,
                edgecolors="#333333",
                linewidths=0.25,
            )
            ax0.set_xlabel("Stall loads (blocking on-demand expert loads)")
            ax0.set_ylabel("TPS Speedup (vs LRU)")
            ax0.set_title("Decode TPS vs stalls\n(recall metrics missing in CSV)", fontsize=11, fontweight="bold")
            ax0.grid(True, alpha=0.45)

        p1 = plot_df["_precision"].dropna()
        if p1.empty:
            ax1.text(0.5, 0.5, "No precision metric\n(pred_requested_rate_topk_pct)", ha="center", va="center", transform=ax1.transAxes, fontsize=11)
            ax1.set_axis_off()
        else:
            sc1 = ax1.scatter(
                plot_df["stall_loads"],
                plot_df["tokens_per_second"],
                c=plot_df["_precision"],
                cmap="magma",
                s=np.clip(plot_df["prefetch_budget"].fillna(8) * 2.5, 25, 120),
                alpha=0.88,
                edgecolors="#333333",
                linewidths=0.25,
                vmin=p1.min(),
                vmax=p1.max(),
            )
            ax1.set_xlabel("Stall loads (blocking on-demand expert loads)")
            ax1.set_ylabel("TPS Speedup (vs LRU)")
            ax1.set_title("Decode TPS vs stalls\ncolour = predictor precision (%)", fontsize=11, fontweight="bold")
            ax1.grid(True, alpha=0.45)
            fig.colorbar(sc1, ax=ax1, shrink=0.82, label="Precision (%)")

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#4fc3f7",
                markersize=6,
                linestyle="None",
                label="marker size ∝ prefetch B",
            ),
        ]
        ax0.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.35)
        if not p1.empty:
            ax1.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.35)

        fig.suptitle("Where does prefetch speedup come from? (λ=0 prefetch-only)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_plot(fig, out_dir, f"prefetch_speedup_attribution_{ts}.png")




def _normalize_tps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tokens_per_second" not in df.columns:
        return df
    
    lru_mask = (df["label"].astype(str).str.contains("LRU", na=False))
    if "backend" in df.columns:
        lru_mask |= (df["backend"] == "cached") & (pd.to_numeric(df.get("lambda_val", 0.0), errors="coerce").fillna(0.0) == 0.0)
    
    df_lru = df[lru_mask]
    
    lru_map = {}
    for c in df_lru["cache_size"].dropna().unique():
        sub = df_lru[df_lru["cache_size"] == c]
        m = sub["tokens_per_second"].dropna().mean()
        if pd.notna(m) and m > 0:
            lru_map[c] = m
            
    def norm(r):
        tps = r.get("tokens_per_second")
        if pd.isna(tps): return tps
        ref = lru_map.get(r.get("cache_size"))
        return float(tps) / ref if ref else tps

    df["tokens_per_second"] = df.apply(norm, axis=1)
    return df


def _policy_label(label: str, lambda_val: float) -> Optional[str]:
    s = str(label)
    if s == "Neither (LRU)":
        return "LRU"
    if s.startswith("Prefetch Only"):
        return "Prefetch" if float(lambda_val) == 0.0 else None
    if s.startswith("Cache-Cond Only"):
        return "Cache-Cond"
    if s.startswith("Both"):
        return "Hybrid"
    return None


def _add_policy_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["policy"] = [
        _policy_label(l, lam)
        for l, lam in zip(out.get("label", ""), out.get("lambda_val", 0))
    ]
    return out


def _csv_row_identity(row: pd.Series) -> Tuple[Any, ...]:
    """Stable row key for CSV merge/dedupe (Series winners vs dict CSV rows)."""
    b = row.get("prefetch_budget")
    bookend = row.get("bookend_pass")
    if pd.isna(bookend) or bookend == "":
        bookend = "main"
    la = row.get("lookahead")
    if la is None or pd.isna(la):
        la_norm = None
    else:
        la_norm = int(la)
    if b is None or pd.isna(b):
        b_norm = None
    else:
        b_norm = float(b)
    return (
        int(row.get("cache_size")),
        str(row.get("label", "")),
        str(row.get("backend", "")),
        float(row.get("lambda_val", 0.0)),
        int(row.get("forced_top_n", 0) or 0),
        la_norm,
        b_norm,
        str(bookend),
    )


def _analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop end-of-cache verification bookends from primary analysis."""
    if "bookend_pass" not in df.columns:
        return df
    bp = df["bookend_pass"].fillna("main").astype(str)
    return df.loc[~bp.eq("end")].copy()


def _dedupe_prefer_newest_csv_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "row_timestamp" not in df.columns:
        return df
    out = df.copy()
    out["_ts"] = pd.to_datetime(out["row_timestamp"], errors="coerce")
    out["_key"] = out.apply(_csv_row_identity, axis=1)
    idx = out.groupby("_key", sort=False)["_ts"].idxmax()
    return out.loc[idx].drop(columns=["_ts", "_key"], errors="ignore")


def _best_row_for_policy(sub: pd.DataFrame, policy: str) -> Optional[pd.Series]:
    rows = sub.loc[sub["policy"] == policy]
    rows = rows[rows["tokens_per_second"].notna()]
    if rows.empty:
        return None
    return rows.loc[rows["tokens_per_second"].idxmax()]


def _best_hybrid_lookahead(sub_c: pd.DataFrame) -> Optional[int]:
    lookaheads = sorted(sub_c["lookahead"].dropna().unique(), key=int)
    if not lookaheads:
        return None
    best_la = int(lookaheads[0])
    best_score = float("-inf")
    for la in lookaheads:
        sub_la = sub_c[sub_c["lookahead"] == la]
        both = _best_row_for_policy(sub_la, "Hybrid")
        score = float(both["tokens_per_second"]) if both is not None else float("-inf")
        if score > best_score:
            best_score = score
            best_la = int(la)
    return best_la


def _lru_baseline_row_for_ppl(sub_c: pd.DataFrame) -> Optional[pd.Series]:
    """LRU baseline per cache size (start bookend); λ=0, no forced-top-n."""
    rows = sub_c[sub_c["policy"] == "LRU"]
    if rows.empty:
        return None
    if "bookend_pass" in rows.columns:
        start = rows[rows["bookend_pass"].fillna("main").astype(str) == "start"]
        if not start.empty:
            rows = start
    if "row_timestamp" in rows.columns and rows["row_timestamp"].notna().any():
        return rows.sort_values("row_timestamp", ascending=False).iloc[0]
    return rows.iloc[0]


def select_ppl_winner_rows(df: pd.DataFrame) -> List[pd.Series]:
    """LRU + best Cache-Cond + Hybrid per C (same configs as the unified LFRU plot)."""
    q = df.get("question", pd.Series("", index=df.index)).astype(str)
    custom = df.loc[q.str.contains("CUSTOM_1_16", na=False)].copy()
    custom = _analysis_rows(custom)
    custom = _dedupe_prefer_newest_csv_rows(_add_policy_column(custom))
    if custom.empty:
        return []

    winners: List[pd.Series] = []
    for c in sorted(custom["cache_size"].dropna().unique(), key=int):
        sub_c = custom[custom["cache_size"] == c]
        lru = _lru_baseline_row_for_ppl(sub_c)
        if lru is not None:
            winners.append(lru)
        best_la = _best_hybrid_lookahead(sub_c)
        if best_la is None:
            continue
        sub_la = sub_c[sub_c["lookahead"] == best_la]
        for policy in ("Cache-Cond", "Hybrid"):
            scope = sub_c if policy == "Cache-Cond" else sub_la
            row = _best_row_for_policy(scope, policy)
            if row is not None:
                winners.append(row)
    return winners


def _row_to_ppl_config(
    row: pd.Series,
    *,
    non_baseline_cache_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a WikiText-103 PPL subprocess config.

    - LRU + Cache-Cond: ``cached`` backend (sec4 ``lambda_fn_sweep`` method, LRU eviction).
    - Hybrid: ``predict`` backend with winner LA/B/LFRU. Requires per-token lambda
      mask during chunked prefill (predict backend; fixed to match cached sec4 path).
      Prefetch does not affect teacher-forced PPL; expect Hybrid PPL ≈ Cache-Cond PPL
      (routing cost), not LRU.
    """
    pm = row.get("prob_mass_threshold", row.get("forced_top_p", -1.0))
    if pd.isna(pm):
        pm = -1.0
    label = str(row.get("label", ""))
    policy = _policy_label(label, row.get("lambda_val", 0.0))

    stride = row.get("predictor_stride")
    if pd.isna(stride):
        stride = row.get("lookahead")
    lookahead = row.get("lookahead")
    if pd.notna(lookahead):
        lookahead = int(lookahead)
    else:
        lookahead = None

    if policy == "LRU":
        return {
            "question": row["question"],
            "label": label,
            "backend": "cached",
            "cache_size": int(row["cache_size"]),
            "lambda_val": 0.0,
            "forced_top_n": 0,
            "prob_mass_threshold": -1.0,
            "forced_top_p": -1.0,
            "lookahead": lookahead,
            "prefetch_budget": None,
            "predictor_stride": None,
            "mode_perplexity": True,
            "mode_generate": False,
            "cache_policy": "LRU",
        }

    if policy == "Cache-Cond":
        return {
            "question": row["question"],
            "label": label,
            "backend": "cached",
            "cache_size": int(row["cache_size"]),
            "lambda_val": float(row["lambda_val"]),
            "forced_top_n": int(row.get("forced_top_n", 0) or 0),
            "prob_mass_threshold": float(pm),
            "forced_top_p": float(pm),
            "lookahead": lookahead,
            "prefetch_budget": None,
            "predictor_stride": None,
            "mode_perplexity": True,
            "mode_generate": False,
            "cache_policy": "LRU",
        }

    if policy == "Hybrid":
        evict = (non_baseline_cache_policy or "LFRU").upper()
        budget = row.get("prefetch_budget")
        return {
            "question": row["question"],
            "label": label,
            "backend": "predict",
            "cache_size": int(row["cache_size"]),
            "lambda_val": float(row["lambda_val"]),
            "forced_top_n": int(row.get("forced_top_n", 0) or 0),
            "prob_mass_threshold": float(pm),
            "forced_top_p": float(pm),
            "lookahead": lookahead,
            "prefetch_budget": None if pd.isna(budget) else int(budget),
            "predictor_stride": None if pd.isna(stride) else int(stride),
            "mode_perplexity": True,
            "mode_generate": False,
            "cache_policy": evict,
        }

    # Fallback: mirror row backend (should not happen for winner selection).
    return {
        "question": row["question"],
        "label": label,
        "backend": row["backend"],
        "cache_size": int(row["cache_size"]),
        "lambda_val": float(row["lambda_val"]),
        "forced_top_n": int(row.get("forced_top_n", 0) or 0),
        "prob_mass_threshold": float(pm),
        "forced_top_p": float(pm),
        "lookahead": lookahead,
        "prefetch_budget": None if pd.isna(row.get("prefetch_budget")) else int(row["prefetch_budget"]),
        "predictor_stride": None if pd.isna(stride) else int(stride),
        "mode_perplexity": True,
        "mode_generate": False,
        "cache_policy": "LRU",
    }


_PPL_POLICY_CHOICES = frozenset({"lru", "cache-cond", "hybrid"})

_PPL_POLICY_ALIASES = {
    "lru": "lru",
    "cache-cond": "cache-cond",
    "cachecond": "cache-cond",
    "hybrid": "hybrid",
}


def _normalize_ppl_policy_name(policy: Optional[str]) -> Optional[str]:
    if policy is None:
        return None
    key = str(policy).lower().replace("_", "-")
    if key == "cache cond":
        key = "cache-cond"
    return _PPL_POLICY_ALIASES.get(key)


def _parse_ppl_policies(raw: Optional[Sequence[str]]) -> Optional[frozenset]:
    """Normalize --ppl-policies values (e.g. hybrid or cache-cond,hybrid)."""
    if not raw:
        return None
    out: set = set()
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            norm = _normalize_ppl_policy_name(part)
            if norm is None:
                raise ValueError(
                    f"unknown --ppl-policies entry {part!r}; "
                    f"choose from {sorted(_PPL_POLICY_CHOICES)}"
                )
            out.add(norm)
    return frozenset(out) if out else None


def run_ppl_winners_only(args: argparse.Namespace) -> int:
    """Fill wikitext PPL on LRU + best Cache-Cond (cached) + Hybrid (predict) per cache size."""
    csv_file = args.csv_file or os.path.join(args.out_dir, f"sweep_{TS}.csv")
    if not os.path.exists(csv_file):
        print(f"[sweep] --ppl-winners-only: CSV not found: {csv_file}", flush=True)
        return 1

    try:
        ppl_policies = _parse_ppl_policies(getattr(args, "ppl_policies", None))
    except ValueError as e:
        print(f"[sweep] --ppl-winners-only: {e}", flush=True)
        return 1

    df = pd.read_csv(csv_file)
    winner_rows = select_ppl_winner_rows(df)
    if ppl_policies is not None:
        winner_rows = [
            r for r in winner_rows
            if _normalize_ppl_policy_name(
                _policy_label(str(r.get("label", "")), r.get("lambda_val", 0.0))
            ) in ppl_policies
        ]
    if not winner_rows:
        msg = "[sweep] --ppl-winners-only: no winner rows found in CSV."
        if ppl_policies is not None:
            msg += f" (filtered to policies: {sorted(ppl_policies)})"
        print(msg, flush=True)
        return 1

    existing = df.to_dict(orient="records")
    by_key = {_csv_row_identity(pd.Series(r)): i for i, r in enumerate(existing)}

    pending: List[Tuple[pd.Series, Dict[str, Any]]] = []
    for row in winner_rows:
        key = _csv_row_identity(row)
        idx = by_key.get(key)
        if idx is None:
            print(f"[sweep] WARNING: winner row missing from CSV: {key}", flush=True)
            continue
        cur = existing[idx]
        ppl = cur.get("gen_perplexity")
        if (
            not args.retry_failed
            and not getattr(args, "ppl_overwrite", False)
            and ppl is not None
            and not (isinstance(ppl, float) and math.isnan(ppl))
        ):
            print(
                f"[sweep] Skipping PPL (already set): C={row['cache_size']} "
                f"{row['label']} LA={row.get('lookahead')}",
                flush=True,
            )
            continue
        pending.append((
            row,
            _row_to_ppl_config(
                row,
                non_baseline_cache_policy=getattr(args, "non_baseline_cache_policy", None),
            ),
        ))

    if not pending:
        print("[sweep] --ppl-winners-only: all winner rows already have gen_perplexity.", flush=True)
        return 0

    policy_note = (
        f"policies={sorted(ppl_policies)}"
        if ppl_policies is not None
        else "LRU+Cache-Cond on cached (sec4), Hybrid on predict"
    )
    print(
        f"[sweep] --ppl-winners-only: {len(pending)} PPL run(s) — {policy_note}.",
        flush=True,
    )
    for _row, cfg in pending:
        budget_str = f" B={cfg['prefetch_budget']}" if cfg.get("prefetch_budget") is not None else ""
        la_str = f" LA={cfg['lookahead']}" if cfg.get("lookahead") is not None else ""
        print(
            f"  C={cfg['cache_size']} backend={cfg['backend']} "
            f"{cfg['label']}{la_str}{budget_str} policy={cfg.get('cache_policy', '')}",
            flush=True,
        )

    prompts = load_prompts(args)
    fd, prompts_file = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(prompts, f)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if args.drop_page_cache_before_first_run:
        drop_page_cache()

    try:
        for idx, (winner_row, config) in enumerate(pending):
            if idx > 0 and args.drop_page_cache_between_runs:
                drop_page_cache()
            key = _csv_row_identity(winner_row)
            row_idx = by_key.get(key)
            if row_idx is None:
                print(
                    f"[sweep] WARNING: winner row missing from CSV during PPL merge: {key}",
                    flush=True,
                )
                continue
            ppl_row = execute_comprehensive_run(config, prompts_file, prompts, args, idx, len(pending))
            merged = dict(existing[row_idx])
            for col in ("gen_perplexity", "gen_perplexity_std"):
                if ppl_row.get(col) is not None:
                    merged[col] = ppl_row[col]
            merged["ppl_run_id"] = run_id
            merged["ppl_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            existing[row_idx] = merged
            pd.DataFrame(existing).to_csv(csv_file, index=False)
    finally:
        try:
            os.remove(prompts_file)
        except OSError:
            pass

    print(f"[sweep] Updated PPL in {csv_file}", flush=True)
    return 0


def generate_all_plots(
    df: pd.DataFrame,
    out_dir: str,
    ts: str,
    *,
    plot_command: Optional[str] = None,
) -> None:
    df = _normalize_tps(df)
    global _current_plot_command
    if not have_plot_deps():
        raise RuntimeError("Plot dependencies are unavailable. Install matplotlib and numpy to generate plots.")
    os.makedirs(out_dir, exist_ok=True)
    _current_plot_command = plot_command or read_plot_command(out_dir)
    df_lambda_fn = df[df["question"] == "LAMBDA_FN_SWEEP"].copy()
    if not df_lambda_fn.empty:
        plot_lambda_fn_sweep(df, out_dir, ts)
    df_custom = df[df["question"].isin(["CUSTOM_1_16", "CUSTOM_1_16_NO_PPL"])].copy()
    if not df_custom.empty:
        plot_custom_1_16(df_custom, out_dir, ts)
        plot_advisor_dashboard(df_custom, out_dir, ts)
        plot_hit_rates(df_custom, out_dir, ts)
        plot_prefetch_speedup_attribution(df_custom, out_dir, ts)
        plot_precision_recall_frontier(df_custom, out_dir, ts)
        plot_predictor_comparison(df_custom, out_dir, ts)




def run_comprehensive_mode(args: argparse.Namespace) -> int:
    if pd is None:
        print("[sweep] Comprehensive mode requires pandas in the active Python environment.", flush=True)
        return 2

    if args.subprocess_timeout is None:
        # Scale with both max_new_tokens and num_prompts.
        # Assume worst-case ~3 TPS under heavy prefetch, plus 600s init overhead.
        num_p = max(1, getattr(args, "num_prompts", 1))
        args.subprocess_timeout = max(900, num_p * args.max_new_tokens * 2 + 600)

    if args.model == "mixtral" and args.constraint_expert_reuse_csv == DEFAULT_QWEN_REUSE_CSV:
        args.constraint_expert_reuse_csv = ""
        print(
            "[sweep] Mixtral: disabling Qwen expert-reuse cache/lookahead constraint "
            "(pass --constraint-expert-reuse-csv to override).",
            flush=True,
        )

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
        plot_cmd = read_plot_command(plot_dir) or " ".join(sys.argv)
        generate_all_plots(df, plot_dir, TS, plot_command=plot_cmd)
        if args.regime_report:
            print_decode_regime_reports_from_df(df)
        return 0

    if getattr(args, "ppl_winners_only", False):
        return run_ppl_winners_only(args)

    os.makedirs(plot_dir, exist_ok=True)
    if args.sweep_question == "cc_prompt_compare":
        gen_file = os.path.join(plot_dir, "generations.md")
        with open(gen_file, "w", encoding="utf-8") as f:
            f.write("# Cache-conditional vs unbiased router — prompt response compare\n\n")
            f.write(f"Run: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")
            f.write(
                "Perplexity: `gen_perplexity` = WikiText-103 (CSV); "
                "`prompt_gen_perplexity` = token-weighted NLL on sampled/greedy continuations "
                "of your custom prompts.\n\n"
            )
    write_plot_command_file(plot_dir, " ".join(sys.argv))
    csv_parent = os.path.dirname(os.path.abspath(csv_file))
    if csv_parent:
        os.makedirs(csv_parent, exist_ok=True)

    min_cache_map = load_min_cache_per_lookahead(args.constraint_expert_reuse_csv)

    # Multi-predictor support: if --predictor-base-dirs is provided with >1 dir,
    # generate configs for each predictor with a tag so the CSV tracks which
    # predictor produced each row.  The comparison plot overlays them.
    predictor_dirs = getattr(args, "predictor_base_dirs", None) or []
    multi_predictor = len(predictor_dirs) > 1
    predictor_tags = getattr(args, "predictor_tags", None) or [
        os.path.basename(os.path.normpath(d)) for d in predictor_dirs
    ]
    if len(predictor_tags) != len(predictor_dirs):
        raise SystemExit("--predictor-tags must have one tag per --predictor-base-dirs entry")

    def _make_custom_configs(fn, question_name):
        if not multi_predictor:
            return fn(args, min_cache_map=min_cache_map)
        cfgs = []
        for pd_dir, tag in zip(predictor_dirs, predictor_tags):
            cfgs.extend(fn(args, predictor_base_dir=pd_dir, predictor_tag=tag, min_cache_map=min_cache_map))
        # Shared LRU only — never append the full untagged grid (predict rows would
        # fall back to --predictor-base-dir default, not the A/B dirs above).
        if fn is custom_1_16_no_ppl_configs:
            mode_ppl = False
        elif fn is custom_1_16_configs:
            mode_ppl = getattr(args, "custom_include_ppl", False)
        else:
            mode_ppl = False
        lru_only: List[Dict[str, Any]] = []
        _append_lru_baselines_once_per_cache(
            lru_only,
            question=question_name,
            cache_sizes=args.cache_sizes,
            args=args,
            min_cache_map=min_cache_map,
            mode_perplexity=mode_ppl,
            predictor_tag="",
        )
        cfgs.extend(lru_only)
        return cfgs

    if getattr(args, "random_baseline_only", False):
        configs = random_baseline_configs(args, min_cache_map)
    else:
        question_builders = {
            "custom_1_16": lambda: _make_custom_configs(custom_1_16_configs, "CUSTOM_1_16"),
            "custom_1_16_no_ppl": lambda: _make_custom_configs(custom_1_16_no_ppl_configs, "CUSTOM_1_16_NO_PPL"),
            "lambda_fn_sweep": lambda: lambda_fn_sweep_configs(args, min_cache_map),
            "oracle_baseline_sweep": lambda: _make_custom_configs(oracle_baseline_sweep_configs, "ORACLE_BASELINE_SWEEP"),
            "sec3_3_2_cache_cond": lambda: sec3_3_2_configs(args, min_cache_map),
            "cc_prompt_compare": lambda: cc_prompt_compare_configs(args, min_cache_map),
            "unified_sweep": lambda: unified_sweep_configs(args, min_cache_map),
            "thread_benchmark": lambda: thread_benchmark_configs(args, min_cache_map),
            "meeting_sweep_sequential": lambda: meeting_sweep_sequential_configs(args, min_cache_map),
            "meeting_sweep_parallel": lambda: meeting_sweep_parallel_configs(args, min_cache_map),
            "paper_baseline": lambda: _make_custom_configs(paper_baseline_sweep_configs, "PAPER_BASELINE"),
            "gating_vs_ml": lambda: _make_custom_configs(gating_vs_ml_sweep_configs, "GATING_VS_ML"),
            "gating_budget_sweep": lambda: _make_custom_configs(gating_budget_sweep_configs, "GATING_BUDGET_SWEEP"),
            "gating_cc_sweep": lambda: _make_custom_configs(gating_cc_sweep_configs, "GATING_CC_SWEEP"),
            "lru_vs_gating": lambda: _make_custom_configs(lru_vs_gating_sweep_configs, "LRU_VS_GATING"),
        }
        configs = question_builders[args.sweep_question]()
        # When --prefetch-only without --skip-baselines, also include RANDOM/LRU baselines.
        if getattr(args, "prefetch_only", False) and not getattr(args, "skip_baselines", False):
            configs.extend(random_baseline_configs(args, min_cache_map))
            _append_lru_baselines_once_per_cache(
                configs,
                question="CUSTOM_1_16_NO_PPL",
                cache_sizes=args.cache_sizes,
                args=args,
                min_cache_map=min_cache_map,
                mode_perplexity=False,
                predictor_tag="",
            )
    configs = apply_row_selection_flags(configs, args)
    apply_non_baseline_cache_policy(configs, getattr(args, "non_baseline_cache_policy", None))
    if getattr(args, "enable_generation_perplexity", False):
        for c in configs:
            if c.get("mode_generate", False):
                c["mode_generation_perplexity"] = True

    append_existing = getattr(args, "append", False)
    merge_existing = (
        args.retry_failed
        or getattr(args, "random_baseline_only", False)
        or append_existing
    )
    existing_rows: List[Dict[str, Any]] = []
    done_keys: set[Tuple[Any, ...]] = set()
    if merge_existing and args.csv_file and os.path.exists(args.csv_file):
        df_existing = pd.read_csv(args.csv_file)
        for _, row in df_existing.iterrows():
            row_dict = row.to_dict()
            existing_rows.append(row_dict)
            if row_has_data(row_dict):
                done_keys.add(cfg_key(row_dict))
        if args.retry_failed or append_existing:
            before = len(configs)
            configs = [c for c in configs if cfg_key(c) not in done_keys]
            skipped = before - len(configs)
            if args.retry_failed:
                print(
                    f"[sweep] --retry-failed: {skipped} configs already done, "
                    f"{len(configs)} to re-run.",
                    flush=True,
                )
            else:
                print(
                    f"[sweep] --append: keeping {len(existing_rows)} existing row(s), "
                    f"skipping {skipped} already-done, adding {len(configs)} new.",
                    flush=True,
                )

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
    print(f"[sweep] Question   : {args.sweep_question}", flush=True)
    print(f"[sweep] CSV output : {csv_file}", flush=True)
    print(f"[sweep] Timeout    : {args.subprocess_timeout}s per run", flush=True)
    print("-" * 80, flush=True)

    if getattr(args, "list_configs", False):
        for c in expanded_runs:
            print(
                f"[config] {c['label']} | backend={c['backend']} C={c['cache_size']} "
                f"λ={c['lambda_val']} J={c['forced_top_n']} LA={c.get('lookahead')} "
                f"B={c.get('prefetch_budget')} tag={c.get('predictor_tag') or '-'} "
                f"policy={c.get('cache_policy') or '-'} type={c.get('predictor_type') or '-'} "
                f"oracle={bool(c.get('is_oracle'))}",
                flush=True,
            )
        return

    prompts = load_prompts(args)
    print(f"[sweep] Loaded {len(prompts)} prompt(s), dataset={args.dataset}", flush=True)
    fd, prompts_file = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(prompts, f)

    results = list(existing_rows) if merge_existing else []
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
            domain = getattr(args, "domain", None)
            if domain:
                row["domain"] = domain
            elif prompts and isinstance(prompts[0], dict) and prompts[0].get("domain"):
                row["domain"] = prompts[0]["domain"]
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
    if getattr(args, "random_baseline_only", False) or args.sweep_question == "cc_prompt_compare":
        if args.sweep_question == "cc_prompt_compare":
            gen_file = os.path.join(plot_dir, "generations.md")
            print(f"[sweep] Prompt responses saved to {gen_file}", flush=True)
        else:
            print("[sweep] Skipping plots (--random-baseline-only append mode).", flush=True)
    else:
        print("\n[sweep] Generating plots...", flush=True)
        try:
            plot_cmd = read_plot_command(plot_dir) or " ".join(sys.argv)
            generate_all_plots(df, plot_dir, TS, plot_command=plot_cmd)
        except Exception as e:
            if not have_plot_deps():
                print("[sweep] Skipping plots because matplotlib/numpy are unavailable in this Python.", flush=True)
            else:
                import traceback

                print(f"[sweep] Error during plotting: {e}", flush=True)
                traceback.print_exc()
    if args.regime_report:
        print_decode_regime_reports_from_df(df)
    print("[sweep] Done!", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        "--sweep-question",
        choices=[
            "custom_1_16",
            "custom_1_16_no_ppl",
            "lambda_fn_sweep",
            "oracle_baseline_sweep",
            "sec3_3_2_cache_cond",
            "cc_prompt_compare",
            "unified_sweep",
            "thread_benchmark",
            "meeting_sweep_sequential",
            "meeting_sweep_parallel",
            "paper_baseline",
            "gating_vs_ml",
            "gating_budget_sweep",
            "gating_cc_sweep",
            "lru_vs_gating",
        ],
        default="custom_1_16_no_ppl",
        help="custom_1_16(_no_ppl): LRU/prefetch/cache-cond/hybrid over lookahead x budget "
             "(with/without a perplexity pass). lambda_fn_sweep: cache-conditional routing "
             "forced-top-J vs probability-mass threshold. cc_prompt_compare: greedy generations "
             "for cache-cond λ=1 at each J in --cc-forced-top-ns (default: 5 6 0). oracle_baseline_sweep: "
             "Oracle Top-B vs LRU vs RANDOM over cache size with coupled lookahead. "
             "gating_vs_ml: cross-layer gating vs ML prefetch-only with matched budget fraction "
             "(ML uses full sweep budget; gating capped at routed top-k, speculative pool sized to cap). "
             "gating_budget_sweep: paper gating-heuristic prefetch only over --prefetch-budgets. "
             "gating_cc_sweep: gating prefetch + cache-conditional routing (λ=1, --routing-bias-top-ns). "
             "lru_vs_gating: straight LRU (cached) vs LRU+gating prefetch over --prefetch-budgets.",
    )
    p.add_argument(
        "--prefetch-non-evicting",
        action="store_true",
        default=False,
        help="gating rows: only prefetch into empty cache slots (default: off; use LRU eviction).",
    )
    p.add_argument(
        "--no-prefetch-non-evicting",
        action="store_false",
        dest="prefetch_non_evicting",
        help="Allow gating prefetch to evict LRU victims (not paper-faithful).",
    )

    p.add_argument("--model", choices=["qwen", "mixtral"], default="qwen")
    p.add_argument("--cache-sizes", type=int, nargs="+", default=[8], help="Cache sizes to sweep.")
    p.add_argument("--prefetch-budgets", type=int, nargs="+", default=[16, 32], help="Explicit prefetch budgets when --custom-explicit-prefetch-budgets is set.")
    p.add_argument(
        "--gating-score-percentiles",
        type=float,
        nargs="+",
        default=[],
        help="Gating score-based prefetch: prefetch all experts with softmax score >= this percentile "
             "(Zhu et al. / SpecMD). E.g. 0.8 for 80th percentile. Adaptive count vs fixed top-B.",
    )
    p.add_argument("--prefetch-thresholds", type=float, nargs="+", default=[], help="Probability thresholds to sweep for threshold-based routing.")
    p.add_argument("--prefetch-threshold", type=float, default=0.0, help="Probability threshold (0.0 to 1.0) to filter experts in the predictor backend.")
    p.add_argument("--budget-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="Budget as a fraction of cache size for custom_1_16/lambda_fn/top_p sweeps. "
                        "Sweeps the sweet spot between cache thrashing (high fraction) and under-utilisation (low fraction). "
                        "With --custom-explicit-prefetch-budgets, unions these fractions with --prefetch-budgets.")
    p.add_argument(
        "--no-budget-fractions",
        action="store_true",
        help="Ignore --budget-fractions so only --prefetch-budgets are used "
             "(with --custom-explicit-prefetch-budgets).",
    )
    p.add_argument(
        "--custom-explicit-prefetch-budgets",
        action="store_true",
        help="CUSTOM_1_16 family: use --prefetch-budgets values (clamped to 1..cache_size) instead of --budget-fractions.",
    )
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="Lambda sweep for comprehensive mode.")
    p.add_argument("--lookaheads", type=int, nargs="+", default=None, help="Lookahead depths for comprehensive mode.")
    p.add_argument(
        "--predictor-base-dir",
        type=str,
        default="/home/michael/heteroPredict/trainingData/qwen3_30b/transformer_final_pfill_markov_emb",
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
    p.add_argument(
        "--predictor-tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for --predictor-base-dirs (same order). Default: each directory's basename.",
    )
    p.add_argument("--predict-layers", type=int, nargs="*", default=None)
    p.add_argument("--predictor-device", type=str, default="cpu")
    p.add_argument("--routing-bias-top-n", type=int, default=6, dest="routing_bias_top_n",
                   help="Forced top-N for Cache-Cond and Both configs (biases router toward cached experts). "
                        "Prefetch Only always uses 0 (unbiased router). "
                        "Ignored when --routing-bias-top-ns is set.")
    p.add_argument(
        "--routing-bias-top-ns",
        type=int,
        nargs="+",
        default=None,
        help="CUSTOM_1_16 family: sweep multiple forced-top-J values for Cache-Cond and Hybrid "
             "(e.g. 5 6). Overrides --routing-bias-top-n.",
    )
    p.add_argument("--cache-cond-forced-top-ns", type=int, nargs="+", default=[4, 6, 8],
                   help="lambda_fn_sweep: forced-top-J values to compare.")
    p.add_argument(
        "--cc-temperatures",
        type=float,
        nargs="+",
        default=[0.0],
        help="cc_prompt_compare: sampling temperatures per routing config (default: 0 = greedy).",
    )
    p.add_argument(
        "--cc-forced-top-ns",
        type=int,
        nargs="+",
        default=[5, 6, 0],
        help="cc_prompt_compare: cache-conditional forced-top-J values at λ=1 (default: 5 6 0).",
    )
    p.add_argument(
        "--probability-mass-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
        help="lambda_fn_sweep: cumulative probability-mass thresholds to compare against forced-top-J.",
    )
    p.add_argument("--config-path", type=str, default=None)
    p.add_argument(
        "--dataset",
        choices=["default", "txt", "oracle", *HF_EVAL_CHOICES],
        default="default",
        help="Prompt source. HF catalog uses disjoint eval partitions: official test "
             "splits, or a hash holdout for FineWeb/Orca. 'all' takes n prompts from each. "
             "oracle/txt/default are local.",
    )
    p.add_argument(
        "--dataset-offset",
        type=int,
        default=None,
        help="Override eval-partition skip (never reaches into the train partition).",
    )
    p.add_argument("--prompts-txt", type=str, default=None)
    p.add_argument(
        "--examples-json",
        type=str,
        default=None,
        help="Structured examples JSON from cache_eval_prompts.py "
             "({domain,idx,text,gold,metric}). Preferred over blank-line txt.",
    )
    p.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Stamp this domain name onto every CSV row (per-domain sweeps).",
    )
    p.add_argument(
        "--enable-generation-perplexity",
        action="store_true",
        help="Also score generation perplexity on the sweep prompts "
             "(fills prompt_gen_perplexity; used for wikitext/fineweb/orca).",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to use. With --dataset txt, 0 means all prompts in --prompts-txt.",
    )
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
    p.add_argument(
        "--constraint-expert-reuse-csv",
        type=str,
        default=DEFAULT_QWEN_REUSE_CSV,
        help="Optional expert-reuse CSV used to skip (cache, lookahead) pairs whose cache is too "
             "small to be lossless at that lookahead. Missing file disables the filter.",
    )
    p.add_argument(
        "--cache-lookahead-slack",
        type=int,
        default=0,
        help="Allow (C, N) when C >= required(N) - slack from the expert-reuse CSV (default 0).",
    )
    p.add_argument(
        "--cache-grouped-bookends",
        action="store_true",
        help="CUSTOM_1_16 family: for each cache size run LRU + Cache-Cond (start), all "
             "Prefetch/Hybrid, then LRU + Cache-Cond again (end verification) before the next C.",
    )
    p.add_argument("--csv-file", type=str, default=None, help="CSV path for the sweep / retry / plot-only.")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--log-file", type=str, default=None, help="Append full subprocess logs here.")
    p.add_argument("--subprocess-timeout", type=int, default=None)
    p.add_argument("--custom-include-ppl", action="store_true",
                   help="When using custom_1_16, also run a perplexity pass (separate from the TPS generation pass).")
    p.add_argument(
        "--ppl-on-lambda-policies",
        action="store_true",
        help="custom_1_16(_no_ppl): run wikitext103 perplexity on every Cache-Cond and Hybrid row "
             "(λ>0). Prefer --ppl-winners-only after the TPS sweep for thesis final results.",
    )
    p.add_argument(
        "--ppl-winners-only",
        action="store_true",
        help="Read --csv-file, pick LRU + best Cache-Cond + Hybrid per C. "
             "Cache-Cond uses cached backend (sec4 method); Hybrid uses predict backend "
             "with winner lookahead/budget/LFRU. Merges gen_perplexity into rows.",
    )
    p.add_argument(
        "--ppl-overwrite",
        action="store_true",
        help="With --ppl-winners-only, re-run PPL even when gen_perplexity is already set.",
    )
    p.add_argument(
        "--ppl-policies",
        nargs="+",
        default=None,
        metavar="POLICY",
        help="With --ppl-winners-only, limit to policy subset: lru, cache-cond, hybrid "
             "(e.g. --ppl-policies hybrid). Comma-separated values OK.",
    )
    p.add_argument("--plot-only", action="store_true")
    p.add_argument(
        "--regime-report",
        action="store_true",
        help="After sweep or with --plot-only, print §decode model regime alignment per CSV row.",
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--append",
        action="store_true",
        help="Append new sweep rows to an existing --csv-file instead of overwriting it "
             "(e.g. phase-2 gating after phase-1 custom_1_16_no_ppl).",
    )
    p.add_argument("--repeat-each-config", type=int, default=1)
    p.add_argument("--shuffle-config-order", action="store_true")
    p.add_argument("--drop-page-cache-between-runs", action="store_true")
    p.add_argument("--drop-page-cache-before-first-run", action="store_true")
    p.add_argument("--cold-per-prompt", action="store_true")
    p.add_argument("--list-configs", action="store_true",
                   help="Print the selected configs and exit without running anything.")
    p.add_argument("--drop-page-cache-between-prompts", action="store_true")
    p.add_argument(
        "--expert-weights-dir", type=str, default=None,
        help="Override expert weight directory for MoE layers (Qwen default: "
             f"{DEFAULT_QWEN_EXPERT_WEIGHTS_DIR}). Supports packed (EXPK, 1 "
             "file/expert) and unpacked (9 files/expert) layouts — auto-detected at runtime.",
    )
    p.add_argument(
        "--prefetch-forced-top-ns",
        type=int,
        nargs="*",
        default=None,
        help="CUSTOM_1_16 family only: emit extra Prefetch-Only rows with --forced-top-n J per J "
             "in this list (0 = default unbiased prefetch line). Use e.g. `0 2 3 4 5 6 7 8` to compare "
             "predictor+top-J vs plain prefetch at the same budget.",
    )
    p.add_argument(
        "--disable-measurement",
        action="store_true",
        help="Pass --suppress-predictor-stats to the backend to disable measurement for runtime performance evaluation."
    )
    p.add_argument(
        "--non-baseline-cache-policy",
        type=str,
        default=None,
        help="Cache eviction policy for all configs except the LRU baseline (Neither). "
             "E.g. LFRU for prefetch / cache-cond / hybrid while baseline stays LRU.",
    )
    p.add_argument(
        "--random-baseline-only",
        action="store_true",
        help="Run only Neither (RANDOM) cached λ=0 baselines (one per --cache-sizes). "
             "Merges into an existing --csv-file when present.",
    )
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Drop Neither (LRU/RANDOM) baseline rows. Use when appending to a CSV that "
             "already has baselines from a prior step.",
    )
    p.add_argument(
        "--cache-cond-only",
        action="store_true",
        help="Run only Cache-Cond Only rows (no baselines / prefetch / hybrid).",
    )
    p.add_argument(
        "--both-only",
        action="store_true",
        help="Run only hybrid Both (prefetch + cache-cond) rows.",
    )
    p.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Skip cache-cond and hybrid (Both λ>0) rows. Only runs Prefetch Only rows for "
             "each predictor, plus LRU and RANDOM baselines (unless --skip-baselines).",
    )
    p.add_argument(
        "--lru-only", action="store_true",
        help="oracle_baseline_sweep: run ONLY the LRU baseline, skip everything else."
    )
    p.add_argument(
        "--include-oracle-baselines",
        action="store_true",
        help="Include Oracle (perfect trace) predictor configs.",
    )
    p.add_argument(
        "--include-oracle-full-union",
        action="store_true",
        help="oracle_baseline_sweep: add Oracle Full Union row per cache size (prefetch entire LA window).",
    )
    p.add_argument(
        "--no-actual-predictor",
        action="store_true",
        help="oracle_baseline_sweep: skip running the actual predictor, only run oracle and baselines.",
    )
    p.add_argument(
        "--oracle-full-union-only",
        action="store_true",
        help="oracle_baseline_sweep: emit ONLY the Oracle Full Union row (skip LRU/RANDOM/Top-B/Predictor).",
    )
    p.add_argument(
        "--couple-lookahead-to-cache",
        action="store_true",
        help="oracle_baseline_sweep: use LA=max(1, C//divisor) instead of full --lookaheads grid.",
    )
    p.add_argument(
        "--cache-lookahead-divisor",
        type=int,
        default=8,
        help="With --couple-lookahead-to-cache: lookahead = max(1, cache_size // divisor). Default 8.",
    )
    p.add_argument(
        "--oracle-routing-agreements",
        type=float,
        nargs="+",
        default=None,
        help="oracle_baseline_sweep: draft-model proxy routing agreement fractions "
        "(e.g. 0.875 0.90 1.0). Each value emits a Full Union row per cache/lookahead.",
    )
    p.add_argument(
        "--oracle-noise-seed",
        type=int,
        default=42,
        help="Deterministic seed for noisy oracle expert corruption (default: 42).",
    )
    p.add_argument(
        "--oracle-trace-dir",
        type=str,
        default=None,
        help="Directory containing oracle_trace_*_{idx}.txt files (default: trainingData/wikitext_test_traces).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_comprehensive_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
