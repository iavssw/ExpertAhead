#!/usr/bin/env python3
"""
Thesis results launcher — GPU sweeps for all five results sections.

Each experiment writes under ``final_results_runs/<key>/<timestamp>/`` with a
``command.txt`` log. Sections 2–5 use ``sweep_predict_cached_cache_metrics.py`` and
write ``sweep.csv`` incrementally (resume with ``--retry-failed``). Section 1 uses
``sweep_cache_policy.py`` and writes ``sweep.csv`` after each config (plus policy plots at the end).

Experiments:

``sec1_cache_policy``
    Section 1. Eviction policies (LRU, MRU, LFU, …) × cache size × λ on the
    **cached** Qwen backend (no predictor). Generation only (TPS + hit rate).

``sec2_predictor_effectiveness``
    Sections 2 and 3. Predict vs LRU, lookahead × budget, lossless (no PPL).
    Run before sec4/sec5 so the precision/recall frontier picks ideal (N, B).

``sec4_routing_topj_vs_pm``
    Section 4. Forced top-J vs probability-mass threshold (λ=1, cached).

``sec5_all_methods``
    Section 5. LRU vs Prefetch vs Cache-Cond vs Hybrid (λ∈{0,1}), with PPL.

``final_results_collection``
    Full thesis sweep: C∈{8,16,24,32,40,48,56,64}, all valid lookaheads × budget
    fractions, 10-prompt TPS for all four policies, then a second pass that runs wikitext
    PPL on LRU + best Cache-Cond + Hybrid at each cache size (same selection as the
    unified LFRU plot). Uses LFRU eviction + page-cache drops.

``random_baseline_collection``
    Append-only: one Neither (RANDOM) cached λ=0 row per C (same prompts as the collection).
    Merge into an existing ``sweep.csv`` via ``--csv-file`` / ``--run-dir``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_SWEEP_SCRIPT = os.path.join(SCRIPT_DIR, "sweep_predict_cached_cache_metrics.py")
CACHE_POLICY_SCRIPT = os.path.join(SCRIPT_DIR, "sweep_cache_policy.py")
ROOT_PY = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_PREDICTOR_BASE = (
    "/home/michael/heteroPredict/trainingData/qwen3_30b/final_predictor"
)
DEFAULT_REUSE_CSV = os.path.join(ROOT_PY, "expert_predictor", "expert_reuse_qwen3_30b.csv")

FINALS_LOOKAHEADS = [1]
# sec2: lookaheads with final_predictor checkpoints (skip 5, 8+ for grid size).
SEC2_LOOKAHEADS = [1, 2, 3, 4, 6]
ROUTING_LOOKAHEADS = [1, 4, 8, 16]
FINALS_CACHE_SIZES = [24, 32]
FINAL_COLLECTION_CACHE_SIZES = [8, 16, 24, 32, 40, 48, 56, 64]
FINAL_COLLECTION_NUM_PROMPTS = 10
# sec2: sweep cache sizes to assess predictor effectiveness.
SEC2_CACHE_SIZES = [8, 32]
# Section 1: sweep several cache sizes to compare eviction policies.
SEC1_CACHE_SIZES = [8, 16, 32, 48, 64]
SEC1_POLICIES = ["LRU", "MRU", "LFU", "MFU", "RANDOM", "LFRU", "PREFILL"]
SEC1_LAMBDAS = [0.0]
FINALS_BUDGET_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
FINALS_ROUTING_BIAS_TOP_N = 5
ROUTING_FORCED_TOP_NS = [3, 4, 5, 6, 7]
ROUTING_PM_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]


def _predictor_lookaheads(base_dir: str) -> List[int]:
    """Lookahead depths with a checkpoint directory under *base_dir*."""
    las: List[int] = []
    try:
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            m = re.search(r"_f(\d+)$", entry.name)
            if m:
                las.append(int(m.group(1)))
    except OSError:
        pass
    return sorted(set(las))


def _collection_lookaheads(predictor_base_dir: str) -> List[int]:
    available = set(_predictor_lookaheads(predictor_base_dir))
    return [la for la in FINALS_LOOKAHEADS if la in available] or list(FINALS_LOOKAHEADS)


def _common_sweep_args(lookaheads: List[int], cache_sizes: Optional[List[int]] = None) -> List[str]:
    sizes = cache_sizes if cache_sizes is not None else FINALS_CACHE_SIZES
    return [
        "--model", "qwen",
        "--dataset", "wikitext",
        "--cache-sizes", *[str(x) for x in sizes],
        "--lookaheads", *[str(x) for x in lookaheads],
        "--budget-fractions", *[str(x) for x in FINALS_BUDGET_FRACTIONS],
        "--routing-bias-top-n", str(FINALS_ROUTING_BIAS_TOP_N),
        "--constraint-expert-reuse-csv", DEFAULT_REUSE_CSV,
        "--expert-weights-dir", "/home/michael/heteroPredict/py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed",
    ]


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    script: str
    extra_args: List[str] = field(default_factory=list)
    uses_predictor: bool = True
    sweep_question: str = ""  # informational only (metrics sweeps)


def _experiment_defs() -> Dict[str, Experiment]:
    return {
        "sec1_cache_policy": Experiment(
            key="sec1_cache_policy",
            description="(sec 1) Expert cache eviction policies × cache size — TPS + hit rate (cached Qwen, λ=0).",
            script=CACHE_POLICY_SCRIPT,
            uses_predictor=False,
            extra_args=[
                "--model", "qwen",
                "--backend", "cached",
                "--dataset", "wikitext",
                "--policies", *SEC1_POLICIES,
                "--cache-sizes", *[str(x) for x in SEC1_CACHE_SIZES],
                "--lambdas", *[str(x) for x in SEC1_LAMBDAS],
                "--mode", "generation",
            ],
        ),
        "sec2_predictor_effectiveness": Experiment(
            key="sec2_predictor_effectiveness",
            description="(sec 2+3) Predict vs LRU, lookahead × budget, lossless (no PPL).",
            script=METRICS_SWEEP_SCRIPT,
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                *_common_sweep_args(SEC2_LOOKAHEADS, SEC2_CACHE_SIZES),
                "--sweep-question", "custom_1_16_no_ppl",
                "--lambdas", "0",
                "--num-prompts", "8"
            ],
        ),
        "sec4_routing_topj_vs_pm": Experiment(
            key="sec4_routing_topj_vs_pm",
            description="(sec 4) Cache-conditional routing: forced top-J vs probability mass.",
            script=METRICS_SWEEP_SCRIPT,
            sweep_question="lambda_fn_sweep",
            extra_args=[
                *_common_sweep_args(ROUTING_LOOKAHEADS),
                "--sweep-question", "lambda_fn_sweep",
                "--lambdas", "1",
                "--cache-cond-forced-top-ns", *[str(x) for x in ROUTING_FORCED_TOP_NS],
                "--probability-mass-thresholds", *[str(x) for x in ROUTING_PM_THRESHOLDS],
            ],
        ),
        "sec5_all_methods": Experiment(
            key="sec5_all_methods",
            description="(sec 5) Four-way LRU / Prefetch / Cache-Cond / Hybrid without PPL.",
            script=METRICS_SWEEP_SCRIPT,
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                *_common_sweep_args(FINALS_LOOKAHEADS),
                "--sweep-question", "custom_1_16_no_ppl",
                "--lambdas", "0", "1",
                "--cache-cond-forced-top-ns", "5",
            ],
        ),
        "sec5_perplexity_drop": Experiment(
            key="sec5_perplexity_drop",
            description="(sec 5) Qualify perplexity drop on specific custom prompts (requires sec5_ppl_prompts.txt).",
            script=METRICS_SWEEP_SCRIPT,
            sweep_question="lambda_fn_sweep",
            extra_args=[
                "--model", "qwen",
                "--dataset", "txt",
                "--prompts-txt", os.path.join(ROOT_PY, "sec5_ppl_prompts.txt"),
                "--cache-sizes", "8", "24", "40", "56",
                "--lookaheads", "1",
                "--budget-fractions", "1.0",
                "--routing-bias-top-n", "5",
                "--constraint-expert-reuse-csv", DEFAULT_REUSE_CSV,
                "--sweep-question", "lambda_fn_sweep",
                "--lambdas", "1",
                "--cache-cond-forced-top-ns", "5", "8",
                "--ppl-on-lambda-policies",
                "--disable-measurement",
                "--temperature", "0.0",
                "--top-k", "50",
            ],
        ),
        "oracle_vs_actual": Experiment(
            key="oracle_vs_actual",
            description="(sec 5) Compare Oracle vs Actual Predictor vs LRU/RANDOM baseline.",
            script=METRICS_SWEEP_SCRIPT,
            uses_predictor=True,
            sweep_question="oracle_baseline_sweep",
            extra_args=[
                "--model", "qwen",
                "--dataset", "oracle",
                "--cache-sizes", "8", "24", "40", "56",
                "--lookaheads", "1", "5",
                "--budget-fractions", "1.0",
                "--constraint-expert-reuse-csv", DEFAULT_REUSE_CSV,
                "--sweep-question", "oracle_baseline_sweep",
                "--disable-measurement",
                "--temperature", "0.0",
            ],
        ),
        "final_results_collection": Experiment(
            key="final_results_collection",
            description=(
                "Full four-policy TPS sweep across C={8..64}, then wikitext PPL: LRU+Cache-Cond "
                "on cached backend (sec4), Hybrid on predict (LFRU, winner LA/B)."
            ),
            script=METRICS_SWEEP_SCRIPT,
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                "--model", "qwen",
                "--dataset", "wikitext",
                "--cache-sizes", *[str(x) for x in FINAL_COLLECTION_CACHE_SIZES],
                "--budget-fractions", *[str(x) for x in FINALS_BUDGET_FRACTIONS],
                "--routing-bias-top-n", str(FINALS_ROUTING_BIAS_TOP_N),
                "--constraint-expert-reuse-csv", DEFAULT_REUSE_CSV,
                "--cache-lookahead-slack", "5",
                "--cache-grouped-bookends",
                "--sweep-question", "custom_1_16_no_ppl",
                "--lambdas", "1",
                "--non-baseline-cache-policy", "LFRU",
                "--disable-measurement",
                "--drop-page-cache-before-first-run",
                "--drop-page-cache-between-runs",
            ],
        ),
        "random_baseline_collection": Experiment(
            key="random_baseline_collection",
            description=(
                "RANDOM eviction baseline (λ=0, cached) at each C={8..64}; appends to an "
                "existing collection sweep.csv (8 runs, same wikitext prompts)."
            ),
            script=METRICS_SWEEP_SCRIPT,
            uses_predictor=False,
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                "--model", "qwen",
                "--dataset", "wikitext",
                "--cache-sizes", *[str(x) for x in FINAL_COLLECTION_CACHE_SIZES],
                "--constraint-expert-reuse-csv", DEFAULT_REUSE_CSV,
                "--cache-lookahead-slack", "5",
                "--random-baseline-only",
                "--disable-measurement",
                "--drop-page-cache-before-first-run",
                "--drop-page-cache-between-runs",
            ],
        ),
    }


def _sweep_python() -> str:
    """Python for sweep subprocesses — prefer active venv (rocmPytorch) over system python3."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = os.path.join(venv, "bin", "python3")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def list_experiments() -> None:
    print("Thesis GPU sweeps:\n")
    for e in _experiment_defs().values():
        q = f"  sweep_question={e.sweep_question}\n" if e.sweep_question else ""
        print(f"{e.key}\n  {e.description}\n  script={os.path.basename(e.script)}\n{q}")


def _metrics_sweep_cmd(
    exp: Experiment,
    *,
    extra_args: List[str],
    run_dir: str,
    csv_path: str,
    log_path: str,
    predictor_base_dir: str,
    num_prompts: int,
    prompt_max_chars: int,
    max_new_tokens: int,
    temperature: float,
    subprocess_timeout: Optional[int],
    retry_failed: bool,
    plot_only: bool,
    disable_measurement: bool,
    ppl_winners_only: bool,
    ppl_overwrite: bool,
    ppl_policies: Optional[List[str]],
    extra_forward: List[str],
) -> List[str]:
    cmd: List[str] = [_sweep_python(), exp.script, *extra_args]
    cmd.extend([
        "--num-prompts", str(num_prompts),
        "--prompt-max-chars", str(prompt_max_chars),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--out-dir", run_dir,
        "--csv-file", csv_path,
        "--log-file", log_path,
    ])
    if exp.uses_predictor:
        cmd.extend(["--predictor-base-dir", predictor_base_dir])
    if subprocess_timeout is not None:
        cmd.extend(["--subprocess-timeout", str(subprocess_timeout)])
    if retry_failed:
        cmd.append("--retry-failed")
    if plot_only:
        cmd.append("--plot-only")
    if disable_measurement:
        cmd.append("--disable-measurement")
    if ppl_winners_only:
        cmd.append("--ppl-winners-only")
    if ppl_overwrite:
        cmd.append("--ppl-overwrite")
    if ppl_policies:
        cmd.extend(["--ppl-policies", *ppl_policies])
    cmd.extend(extra_forward)
    return cmd


def run_one(
    exp_key: str,
    *,
    predictor_base_dir: str,
    out_root: str,
    num_prompts: int,
    prompt_max_chars: int,
    max_new_tokens: int,
    temperature: float,
    subprocess_timeout: Optional[int],
    dry_run: bool,
    retry_failed: bool,
    plot_only: bool,
    disable_measurement: bool,
    extra_forward: List[str],
    skip_ppl_phase: bool = False,
    ppl_winners_only: bool = False,
    ppl_overwrite: bool = False,
    ppl_policies: Optional[List[str]] = None,
    run_dir_override: Optional[str] = None,
    csv_file_override: Optional[str] = None,
) -> int:
    defs = _experiment_defs()
    if exp_key not in defs:
        print(f"Unknown experiment {exp_key!r}. Use --list.", file=sys.stderr)
        return 2

    exp = defs[exp_key]
    two_phase = exp_key == "final_results_collection"
    collection_family = exp_key in ("final_results_collection", "random_baseline_collection")

    extra_args = list(exp.extra_args)
    if exp_key == "final_results_collection":
        las = _collection_lookaheads(predictor_base_dir)
        extra_args = ["--lookaheads", *[str(x) for x in las], *extra_args]
    if collection_family and num_prompts == 100:
        num_prompts = FINAL_COLLECTION_NUM_PROMPTS

    if run_dir_override:
        run_dir = run_dir_override
    elif ppl_winners_only and csv_file_override:
        run_dir = os.path.dirname(os.path.abspath(csv_file_override))
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_root, exp_key, ts)
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    # Respect caller env (e.g. bash A/B for parallel vs sequential expert pread).
    env.setdefault("HETEROPREDICT_SEQUENTIAL_EXPERT_IO", "0")

    if exp.script == CACHE_POLICY_SCRIPT:
        prefix = os.path.join(run_dir, "sweep")
        cmd: List[str] = [_sweep_python(), exp.script, *exp.extra_args]
        cmd.extend([
            "--num-prompts", str(num_prompts),
            "--max-new-tokens", str(max_new_tokens),    
            "--prompt-max-chars", str(prompt_max_chars),
            "--output-prefix", prefix,
        ])
        if subprocess_timeout is not None:
            cmd.extend(["--subprocess-timeout", str(subprocess_timeout)])
        else:
            auto = max(900, num_prompts * max_new_tokens * 2 + 600)
            cmd.extend(["--subprocess-timeout", str(auto)])
        cmd.extend(extra_forward)
        with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")
        print("RUN_DIR:", run_dir)
        print("CMD:", " ".join(cmd))
        if dry_run:
            return 0
        return subprocess.call(cmd, env=env)

    csv_path = csv_file_override or os.path.join(run_dir, "sweep.csv")
    log_path = os.path.join(run_dir, "run.log")

    phase1_cmd: Optional[List[str]] = None
    if not ppl_winners_only and not (two_phase and skip_ppl_phase):
        phase1_cmd = _metrics_sweep_cmd(
            exp,
            extra_args=extra_args,
            run_dir=run_dir,
            csv_path=csv_path,
            log_path=log_path,
            predictor_base_dir=predictor_base_dir,
            num_prompts=num_prompts,
            prompt_max_chars=prompt_max_chars,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            subprocess_timeout=subprocess_timeout,
            retry_failed=retry_failed,
            plot_only=plot_only,
            disable_measurement=disable_measurement,
            ppl_winners_only=False,
            ppl_overwrite=False,
            ppl_policies=None,
            extra_forward=extra_forward,
        )

    phase2_cmd: Optional[List[str]] = None
    if two_phase and (ppl_winners_only or not skip_ppl_phase):
        phase2_cmd = _metrics_sweep_cmd(
            exp,
            extra_args=extra_args,
            run_dir=run_dir,
            csv_path=csv_path,
            log_path=log_path,
            predictor_base_dir=predictor_base_dir,
            num_prompts=num_prompts,
            prompt_max_chars=prompt_max_chars,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            subprocess_timeout=subprocess_timeout,
            retry_failed=retry_failed,
            plot_only=False,
            disable_measurement=disable_measurement,
            ppl_winners_only=True,
            ppl_overwrite=ppl_overwrite,
            ppl_policies=ppl_policies,
            extra_forward=extra_forward,
        )

    with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as f:
        if phase1_cmd:
            f.write("# Phase 1 — TPS sweep\n")
            f.write(" ".join(phase1_cmd) + "\n")
        if phase2_cmd:
            f.write("# Phase 2 — PPL on LRU + best Cache-Cond/Hybrid per cache size\n")
            f.write(" ".join(phase2_cmd) + "\n")

    print("RUN_DIR:", run_dir)
    if phase1_cmd:
        print("PHASE 1 (TPS):", " ".join(phase1_cmd))
    if phase2_cmd:
        print("PHASE 2 (PPL LRU + winners):", " ".join(phase2_cmd))
    if dry_run:
        return 0

    if phase1_cmd:
        rc = subprocess.call(phase1_cmd, env=env)
        if rc != 0:
            return rc
    if phase2_cmd:
        if not os.path.isfile(csv_path):
            print(f"Phase 2 skipped: CSV not found at {csv_path}", file=sys.stderr)
            return 1
        return subprocess.call(phase2_cmd, env=env)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument(
        "--out-root",
        type=str,
        default=os.path.join(SCRIPT_DIR, "final_results_runs"),
    )
    p.add_argument("--predictor-base-dir", type=str, default=DEFAULT_PREDICTOR_BASE)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--prompt-max-chars", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--subprocess-timeout", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--disable-measurement", action="store_true", help="Pass --disable-measurement to the sweep script.")
    p.add_argument(
        "--skip-ppl-phase",
        action="store_true",
        help="final_results_collection: run TPS sweep only (skip phase-2 PPL on winners).",
    )
    p.add_argument(
        "--ppl-winners-only",
        action="store_true",
        help="final_results_collection: run only phase-2 PPL on LRU + best Cache-Cond/Hybrid "
             "per C (requires --csv-file).",
    )
    p.add_argument(
        "--ppl-overwrite",
        action="store_true",
        help="With --ppl-winners-only, re-measure PPL even when gen_perplexity is already set.",
    )
    p.add_argument(
        "--ppl-policies",
        nargs="+",
        default=None,
        metavar="POLICY",
        help="With --ppl-winners-only, limit to lru, cache-cond, and/or hybrid "
             "(e.g. --ppl-policies hybrid --ppl-overwrite).",
    )
    p.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Existing sweep.csv for --ppl-winners-only, random_baseline_collection append, or resume.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory (defaults to parent of --csv-file).",
    )
    p.add_argument("forwarded", nargs="*", help="Extra args for the sweep script.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.list:
        list_experiments()
        return 0
    if not args.experiment:
        print("Specify --experiment <key> or use --list.", file=sys.stderr)
        return 2
    return run_one(
        args.experiment,
        predictor_base_dir=args.predictor_base_dir,
        out_root=args.out_root,
        num_prompts=args.num_prompts,
        prompt_max_chars=args.prompt_max_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        subprocess_timeout=args.subprocess_timeout,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        plot_only=args.plot_only,
        disable_measurement=args.disable_measurement,
        extra_forward=list(args.forwarded),
        skip_ppl_phase=args.skip_ppl_phase,
        ppl_winners_only=args.ppl_winners_only,
        ppl_overwrite=args.ppl_overwrite,
        ppl_policies=args.ppl_policies,
        run_dir_override=args.run_dir,
        csv_file_override=args.csv_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
