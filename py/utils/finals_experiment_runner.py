#!/usr/bin/env python3
"""
Final-results experiment launcher — **nine** GPU sweeps (+ optional ``final_analysis_readme``).

Expert-reuse / motivation figures come from ``py/expert_predictor/analyze_expert_reuse.py``, not here.

Invokes ``sweep_predict_cached_cache_metrics.py`` with fixed argument bundles.
Comprehensive sweeps write ``sweep.csv`` after each **config row**; use ``--retry-failed``
with a stable ``--csv-file`` to resume.

GPU experiments (sweeps)
------------------------
1. ``predictor_architecture_ablation`` — multi-predictor A/B (needs ``--predictor-base-dirs``).
2. ``decode_throughput_and_pareto`` — LRU vs prefetch-only + gen PPL + TPS; same CSV also
   yields ``precision_recall_frontier_*`` plots (Pareto-style overlay).
2b. ``decode_throughput_and_pareto_advisor`` — same decode grid as (2) but **no gen PPL pass**;
   CSV still has **predictor recall/precision** and **stall/prefetch load** stats for advisor-facing
   speedup attribution (see ``analyze_predictor_speedup_attribution.py``).
3. ``all_methods_decode`` — LRU + prefetch + cache-cond + hybrid, gen PPL.
4. ``routing_topj_vs_probability_mass`` — ``lambda_fn_sweep`` (top-j vs PM vs LRU, TPS+PPL).
5. ``predictor_on_topj_prefetch`` — predict + prefetch-only, sweep ``J`` in ``{0,2..8}``.
6. ``probability_mass_calibration_ppl`` — PPL-only vs PM (``probability_mass_calibration``).
7. ``cache_budget_tradeoff`` — dense explicit prefetch budgets × cache sizes × lookaheads (λ=0);
   post-process with ``analyze_cache_prefetch_tradeoff.py`` for Pareto picks.
8. ``precision_recall_cache_lookahead_prefetch`` — same **grid** as (7) but ``custom_1_16_no_ppl``
9. ``april_4way_replication`` — April 2026 coupled (C, LA, B) grid + combined PR/TPS/hit plots
   (decode only, no gen-PPL pass). CSV includes ``pred_hit_rate_routed_topk_pct`` (recall-like) and
   ``pred_requested_rate_topk_pct`` (precision-like); plots include ``advisor_dashboard_C*.png``
   (raw TPS, recall, cache hit, precision), ``precision_recall_frontier_C*.png``, and
   ``hit_rates_C*.png`` per cache size. Figures embed the launcher command from ``command.txt``.

(Experiments 1–7 are the original numbered set; (2b) is the fast decode variant; (8) is the PR-focused
decode-only grid matching (7) without gen PPL.)

Slide-style Pareto (precision/recall frontier) is produced for CUSTOM_1_16 family sweeps; experiment **(2)**
also includes gen PPL in the CSV when ``--custom-include-ppl`` is on.

Optional (no sweep): ``final_analysis_readme`` — writes ``FINAL_ANALYSIS.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_SCRIPT = os.path.join(SCRIPT_DIR, "sweep_predict_cached_cache_metrics.py")

DEFAULT_PREDICTOR_BASE = (
    "/home/michael/heteroPredict/trainingData/qwen3_30b/final_multi_input_model"
)
ROOT_PY = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_REUSE_CSV = os.path.join(ROOT_PY, "expert_predictor", "expert_reuse_qwen3_30b.csv")

FINALS_LOOKAHEADS = [1, 2, 3, 4, 6, 8, 10, 12, 16]
# Subset for advisor-style decode+PPL runs (~12–16h vs multi-day full grid).
ADVISOR_DECODE_LOOKAHEADS = [1, 2, 4, 8, 16]
FINALS_CACHE_SIZES = [24, 32, 48]
FINALS_TOP_J = [2, 3, 4, 5, 6, 7, 8]
FINALS_PM = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FINALS_BUDGET_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
PREFETCH_J_SWEEP = [0, 2, 3, 4, 5, 6, 7, 8]

# Wider cache grid + explicit prefetch B list for cache–budget tradeoff / Pareto analysis.
CACHE_BUDGET_TRADEOFF_CACHE = [16, 20, 24, 28, 32, 36, 40, 48]
CACHE_BUDGET_TRADEOFF_LOOKAHEADS = [1, 2, 4, 8, 16]
CACHE_BUDGET_TRADEOFF_PREFETCH = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]


def _shared_sweep_args(lookaheads: Optional[List[int]] = None) -> List[str]:
    las = lookaheads if lookaheads is not None else FINALS_LOOKAHEADS
    return [
        "--model",
        "qwen",
        "--dataset",
        "wikitext",
        "--cache-sizes",
        *[str(x) for x in FINALS_CACHE_SIZES],
        "--lookaheads",
        *[str(x) for x in las],
        "--constraint-expert-reuse-csv",
        DEFAULT_REUSE_CSV,
        "--routing-bias-top-n",
        "8",
        "--budget-fractions",
        *[str(x) for x in FINALS_BUDGET_FRACTIONS],
    ]


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    sweep_question: str
    extra_args: List[str]
    # When set, run_one uses these instead of CLI --num-prompts / --max-new-tokens / --prompt-max-chars.
    pinned_num_prompts: Optional[int] = None
    pinned_max_new_tokens: Optional[int] = None
    pinned_prompt_max_chars: Optional[int] = None


def _experiment_defs() -> Dict[str, Experiment]:
    sh = _shared_sweep_args()
    return {
        "predictor_architecture_ablation": Experiment(
            key="predictor_architecture_ablation",
            description="(1) Optional A/B: ``--predictor-base-dirs`` (≥2) + CUSTOM_1_16 + gen PPL.",
            sweep_question="custom_1_16",
            extra_args=[
                *sh,
                "--sweep-question",
                "custom_1_16",
                "--lambdas",
                "0",
                "1",
                "--custom-include-ppl",
            ],
        ),
        "decode_throughput_and_pareto": Experiment(
            key="decode_throughput_and_pareto",
            description="(2) LRU vs prefetch-only + gen PPL/TPS; PR frontier plots from same CSV.",
            sweep_question="custom_1_16",
            extra_args=[
                *sh,
                "--sweep-question",
                "custom_1_16",
                "--lambdas",
                "0",
                "--custom-include-ppl",
            ],
        ),
        "decode_throughput_and_pareto_advisor": Experiment(
            key="decode_throughput_and_pareto_advisor",
            description="(2b) LRU vs prefetch-only (λ=0), decode TPS + predictor **recall/precision** "
            "columns and **stall/prefetch load** counters (no gen PPL pass). Post-run: "
            "``python py/utils/analyze_predictor_speedup_attribution.py --csv …/sweep.csv``. "
            "Plots include ``prefetch_speedup_attribution_*.png`` (TPS vs stalls). "
            "Pinned: 6 prompts, 128 max-new-tokens, 2048 prompt chars; fewer lookaheads than (2).",
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                *_shared_sweep_args(ADVISOR_DECODE_LOOKAHEADS),
                "--sweep-question",
                "custom_1_16_no_ppl",
                "--lambdas",
                "0",
            ],
            pinned_num_prompts=6,
            pinned_max_new_tokens=128,
            pinned_prompt_max_chars=2048,
        ),
        "all_methods_decode": Experiment(
            key="all_methods_decode",
            description="(3) LRU + prefetch + cache-cond + hybrid (λ∈{0,1}), gen PPL.",
            sweep_question="custom_1_16",
            extra_args=[
                *sh,
                "--sweep-question",
                "custom_1_16",
                "--lambdas",
                "0",
                "1",
                "--custom-include-ppl",
            ],
        ),
        "routing_topj_vs_probability_mass": Experiment(
            key="routing_topj_vs_probability_mass",
            description="(4) lambda_fn_sweep: top-j (2..8) vs PM (0.2..0.9) vs LRU; TPS + gen PPL.",
            sweep_question="lambda_fn_sweep",
            extra_args=[
                *sh,
                "--sweep-question",
                "lambda_fn_sweep",
                "--lambdas",
                "1",
                "--budget-fractions",
                "1.0",
                "--cache-cond-forced-top-ns",
                *[str(x) for x in FINALS_TOP_J],
                "--probability-mass-thresholds",
                *[str(x) for x in FINALS_PM],
            ],
        ),
        "predictor_on_topj_prefetch": Experiment(
            key="predictor_on_topj_prefetch",
            description="(5) Predict + prefetch-only: sweep J in {0,2..8} via --prefetch-forced-top-ns.",
            sweep_question="custom_1_16",
            extra_args=[
                *sh,
                "--sweep-question",
                "custom_1_16",
                "--lambdas",
                "0",
                "--custom-include-ppl",
                "--prefetch-forced-top-ns",
                *[str(x) for x in PREFETCH_J_SWEEP],
            ],
        ),
        "probability_mass_calibration_ppl": Experiment(
            key="probability_mass_calibration_ppl",
            description="(6) PPL-only vs PM (probability_mass_calibration); pairs with routing_topj_vs_probability_mass.",
            sweep_question="probability_mass_calibration",
            extra_args=[
                *sh,
                "--sweep-question",
                "probability_mass_calibration",
                "--lambdas",
                "1",
                "--budget-fractions",
                "1.0",
                "--probability-mass-thresholds",
                *[str(x) for x in FINALS_PM],
            ],
        ),
        "cache_budget_tradeoff": Experiment(
            key="cache_budget_tradeoff",
            description="(7) Explicit prefetch B grid × cache sizes × lookaheads (λ=0 prefetch-only + PPL).",
            sweep_question="custom_1_16",
            extra_args=[
                "--model",
                "qwen",
                "--dataset",
                "wikitext",
                "--cache-sizes",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_CACHE],
                "--lookaheads",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_LOOKAHEADS],
                "--constraint-expert-reuse-csv",
                DEFAULT_REUSE_CSV,
                "--routing-bias-top-n",
                "8",
                "--custom-explicit-prefetch-budgets",
                "--prefetch-budgets",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_PREFETCH],
                "--sweep-question",
                "custom_1_16",
                "--lambdas",
                "0",
                "--custom-include-ppl",
            ],
        ),
        "precision_recall_cache_lookahead_prefetch": Experiment(
            key="precision_recall_cache_lookahead_prefetch",
            description="(8) Same explicit B×cache×lookahead grid as (7), decode-only (no gen PPL). "
            "Best for ``advisor_dashboard_C*.png`` (raw TPS, recall, cache hit, precision), "
            "``precision_recall_frontier_C*.png``, and ``hit_rates_C*.png``; use (7) if you also "
            "need gen perplexity per row.",
            sweep_question="custom_1_16_no_ppl",
            extra_args=[
                "--model",
                "qwen",
                "--dataset",
                "wikitext",
                "--cache-sizes",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_CACHE],
                "--lookaheads",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_LOOKAHEADS],
                "--constraint-expert-reuse-csv",
                DEFAULT_REUSE_CSV,
                "--routing-bias-top-n",
                "8",
                "--custom-explicit-prefetch-budgets",
                "--prefetch-budgets",
                *[str(x) for x in CACHE_BUDGET_TRADEOFF_PREFETCH],
                "--sweep-question",
                "custom_1_16_no_ppl",
                "--lambdas",
                "0",
            ],
        ),
        "april_4way_replication": Experiment(
            key="april_4way_replication",
            description="(9) April ``lookahead_4way_20prompts`` coupled grid: "
            "(C,LA,B) = (9,1,8)…(50,16,42), forced_top_n=6, 20 prompts. "
            "Plots: ``april_4way_dashboard_*.png``, ``april_4way_precision_recall_*.png``, "
            "``predictor_speedup_attribution.csv``.",
            sweep_question="april_4way_no_ppl",
            extra_args=[
                "--model",
                "qwen",
                "--dataset",
                "wikitext",
                "--constraint-expert-reuse-csv",
                DEFAULT_REUSE_CSV,
                "--april-forced-top-n",
                "6",
                "--sweep-question",
                "april_4way_no_ppl",
            ],
            pinned_num_prompts=20,
            pinned_max_new_tokens=128,
            pinned_prompt_max_chars=4096,
        ),
    }


def list_experiments() -> None:
    print("GPU sweeps (motivation / expert reuse: use py/expert_predictor/analyze_expert_reuse.py):\n")
    for e in _experiment_defs().values():
        print(f"{e.key}\n  {e.description}\n  sweep_question={e.sweep_question}\n")
    print(
        "final_analysis_readme\n"
        "  Optional: write FINAL_ANALYSIS.md (select best point at 1/2/5/10% degradation).\n"
        "  (no sweep)\n"
    )


def _write_final_analysis_readme(run_dir: str) -> None:
    path = os.path.join(run_dir, "FINAL_ANALYSIS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(
                """\
                # Final analysis (per cache size C, degradation budget Y)

                For Y ∈ {1, 2, 5, 10}% relative increase in ``gen_perplexity`` vs LRU at the same C:

                1. Fix a baseline LRU row (same ``prompt_hash``, same decode settings).
                2. For each candidate row: ``pct = 100 * (ppl - ppl_LRU) / ppl_LRU``.
                3. Among rows with ``pct <= Y``, maximize ``tokens_per_second`` (tie-break: lower
                   prefetch budget, then lower lookahead).

                Map C to approximate RAM using measured bytes per expert × experts cached × layers.
                """
            )
        )


def run_one(
    exp_key: str,
    *,
    predictor_base_dir: str,
    predictor_base_dirs: Optional[List[str]],
    out_root: str,
    num_prompts: int,
    prompt_max_chars: int,
    max_new_tokens: int,
    temperature: float,
    subprocess_timeout: Optional[int],
    dry_run: bool,
    retry_failed: bool,
    plot_only: bool,
    extra_forward: List[str],
) -> int:
    if exp_key == "final_analysis_readme":
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_root, exp_key, ts)
        os.makedirs(run_dir, exist_ok=True)
        _write_final_analysis_readme(run_dir)
        print("Wrote", os.path.join(run_dir, "FINAL_ANALYSIS.md"))
        return 0

    defs = _experiment_defs()
    if exp_key not in defs:
        print(f"Unknown experiment {exp_key!r}. Use --list.", file=sys.stderr)
        return 2

    if exp_key == "predictor_architecture_ablation":
        if not predictor_base_dirs or len(predictor_base_dirs) < 2:
            print(
                "predictor_architecture_ablation needs:\n"
                "  --predictor-base-dirs /path/A /path/B",
                file=sys.stderr,
            )
            return 2

    exp = defs[exp_key]
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, exp_key, ts)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "sweep.csv")

    extra = list(exp.extra_args)
    if exp_key == "predictor_architecture_ablation":
        extra.extend(["--predictor-base-dirs", *predictor_base_dirs])
    else:
        extra.extend(["--predictor-base-dir", predictor_base_dir])

    num_p = exp.pinned_num_prompts if exp.pinned_num_prompts is not None else num_prompts
    max_nt = exp.pinned_max_new_tokens if exp.pinned_max_new_tokens is not None else max_new_tokens
    pmc = exp.pinned_prompt_max_chars if exp.pinned_prompt_max_chars is not None else prompt_max_chars
    if exp.pinned_num_prompts is not None:
        print(
            f"[finals] Pinned workload for {exp_key!r}: "
            f"--num-prompts {num_p} --max-new-tokens {max_nt} --prompt-max-chars {pmc}",
            flush=True,
        )

    cmd: List[str] = [
        sys.executable,
        SWEEP_SCRIPT,
        *extra,
        "--num-prompts",
        str(num_p),
        "--prompt-max-chars",
        str(pmc),
        "--max-new-tokens",
        str(max_nt),
        "--temperature",
        str(temperature),
        "--out-dir",
        run_dir,
        "--csv-file",
        csv_path,
    ]
    if subprocess_timeout is not None:
        cmd.extend(["--subprocess-timeout", str(subprocess_timeout)])
    if retry_failed:
        cmd.append("--retry-failed")
    if plot_only:
        cmd.append("--plot-only")
    cmd.extend(extra_forward)

    with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    print("RUN_DIR:", run_dir)
    print("CMD:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument(
        "--out-root",
        type=str,
        default=os.path.join(SCRIPT_DIR, "final_results_runs"),
    )
    p.add_argument("--predictor-base-dir", type=str, default=DEFAULT_PREDICTOR_BASE)
    p.add_argument(
        "--predictor-base-dirs",
        type=str,
        nargs="+",
        default=None,
        help="For predictor_architecture_ablation only (≥2 directories).",
    )
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--prompt-max-chars", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--subprocess-timeout", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--plot-only", action="store_true")
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
        predictor_base_dirs=args.predictor_base_dirs,
        out_root=args.out_root,
        num_prompts=args.num_prompts,
        prompt_max_chars=args.prompt_max_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        subprocess_timeout=args.subprocess_timeout,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        plot_only=args.plot_only,
        extra_forward=list(args.forwarded),
    )


if __name__ == "__main__":
    raise SystemExit(main())
