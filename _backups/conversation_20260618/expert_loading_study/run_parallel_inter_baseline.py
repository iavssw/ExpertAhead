#!/usr/bin/env python3
"""
Run LRU / RANDOM / predict baselines comparing parallel vs sequential inter-expert loads.

Parallel (default): unset HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO
Sequential baseline: HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1

Rebuild after C++ changes (repo root):
  source utils/setup.sh && cmake --build build -j

Usage:
  # Predict + LRU + RANDOM (f1, lookahead 1), parallel vs sequential inter
  python run_parallel_inter_baseline.py --cache-sizes 8 24 --quick

  # f6 horizon predictor (stride 6, higher-B union prefetch)
  python run_parallel_inter_baseline.py --preset f6 --cache-sizes 24 40 --quick --skip-sequential

  # LRU + RANDOM only (cached backend, no predict)
  python run_parallel_inter_baseline.py --baselines-only --cache-sizes 8 24 --quick
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

STUDY_DIR = Path(__file__).parent
REPO_ROOT = STUDY_DIR.parent.parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
SWEEP = REPO_ROOT / "py/utils/sweep_predict_cached_cache_metrics.py"
from io_common import apply_strict_ssd_env, apply_legacy_expert_io_env, apply_modern_expert_io_env, strict_ssd_sweep_flags
DEFAULT_PACKED = REPO_ROOT / "py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed"

PREDICTOR_F1 = (
    REPO_ROOT
    / "trainingData/qwen3_30b/transformer_final_pfill_markov_emb/transformer_eh4_h64_f1"
)
PREDICTOR_F6 = (
    REPO_ROOT
    / "trainingData/qwen3_30b/transformer_final_not_actually/transformer_eh4_h64_f6"
)


@dataclass(frozen=True)
class SweepPreset:
    name: str
    predictor_base_dir: Path
    lookaheads: List[int]
    budget_fractions: Optional[List[float]] = None
    prefetch_budgets: Optional[List[int]] = None
    explicit_budgets: bool = False
    cache_lookahead_slack: int = 0
    default_cache_sizes: Optional[List[int]] = None


PRESETS = {
    "f1": SweepPreset(
        name="f1",
        predictor_base_dir=PREDICTOR_F1,
        lookaheads=[1],
        budget_fractions=[0.5, 1.0],
    ),
    "f6": SweepPreset(
        name="f6",
        predictor_base_dir=PREDICTOR_F6,
        lookaheads=[6],
        # Small B: high π at 6-step horizon; parallel inter-IO makes cheap prefetches viable.
        prefetch_budgets=[6, 8, 10, 12, 15, 18],
        explicit_budgets=True,
        # expert_reuse CSV requires C>=37 at LA=6; slack allows C=24 experiments.
        cache_lookahead_slack=13,
        default_cache_sizes=[24, 40, 48],
    ),
}


def _run_sweep(
    *,
    out_csv: Path,
    cache_sizes: list[int],
    tag: str,
    expert_weights_dir: str,
    predictor_base_dir: str,
    lookaheads: List[int],
    budget_fractions: Optional[List[float]],
    prefetch_budgets: Optional[List[int]],
    explicit_budgets: bool,
    cache_lookahead_slack: int,
    sequential_inter: bool,
    baselines_only: bool,
    predict_only: bool,
    max_new_tokens: int,
    num_prompts: int,
    strict_ssd: bool,
    legacy_io: bool,
) -> int:
    env = os.environ.copy()
    env.setdefault("HSA_ENABLE_SDMA", "0")
    apply_strict_ssd_env(env, enabled=strict_ssd)
    if legacy_io:
        apply_legacy_expert_io_env(env, enabled=True, sequential_inter=True)
    else:
        apply_modern_expert_io_env(
            env,
            sequential_inter=sequential_inter,
            sequential_intra=True,
        )

    cmd = [
        sys.executable,
        str(SWEEP),
        "--sweep-question",
        "custom_1_16_no_ppl",
        "--model",
        "qwen",
        "--cache-sizes",
        *[str(c) for c in cache_sizes],
        "--lookaheads",
        *[str(la) for la in lookaheads],
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-prompts",
        str(num_prompts),
        "--csv-file",
        str(out_csv),
        "--out-dir",
        str(out_csv.parent),
        "--log-file",
        str(out_csv.with_suffix(".log")),
        "--expert-weights-dir",
        expert_weights_dir,
        "--predictor-base-dir",
        predictor_base_dir,
        "--cache-lookahead-slack",
        str(cache_lookahead_slack),
    ]

    if explicit_budgets:
        cmd.append("--custom-explicit-prefetch-budgets")
        cmd.append("--prefetch-budgets")
        cmd.extend(str(b) for b in (prefetch_budgets or []))
    else:
        cmd.append("--budget-fractions")
        cmd.extend(str(f) for f in (budget_fractions or [0.5, 1.0]))

    if baselines_only:
        cmd.append("--lru-random-baselines-only")
    else:
        cmd.append("--prefetch-only")

    cmd.extend(strict_ssd_sweep_flags(enabled=strict_ssd))

    label = "legacy_io" if legacy_io else ("sequential_inter" if sequential_inter else "parallel_inter")
    print(f"\n{'='*72}\n  {tag}  inter_expert={label}\n  CSV: {out_csv}\n{'='*72}")
    (out_csv.parent / "sweep_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="LRU/RANDOM/predict sweep: parallel vs sequential inter-expert I/O")
    p.add_argument("--preset", choices=sorted(PRESETS), default="f1", help="Predictor + lookahead/budget defaults")
    p.add_argument("--cache-sizes", type=int, nargs="+", default=None)
    p.add_argument("--expert-weights-dir", type=str, default=str(DEFAULT_PACKED))
    p.add_argument("--predictor-base-dir", type=str, default=None, help="Override preset predictor path")
    p.add_argument("--lookaheads", type=int, nargs="+", default=None, help="Override preset lookaheads")
    p.add_argument(
        "--budget-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Override preset budget fractions (f1-style)",
    )
    p.add_argument(
        "--prefetch-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Override preset explicit prefetch budgets (f6-style)",
    )
    p.add_argument(
        "--cache-lookahead-slack",
        type=int,
        default=None,
        help="Allow C >= required(N)-slack from expert-reuse CSV",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="40 tokens, 1 prompt")
    p.add_argument("--num-prompts", type=int, default=None, help="Override prompt count (default: 1 quick / 3 full)")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Override decode length (default: 40 quick / 80 full)")
    p.add_argument("--baselines-only", action="store_true", help="Cached LRU + RANDOM only (no predict rows)")
    p.add_argument(
        "--predict-only",
        action="store_true",
        help="Same as default: Prefetch Only predict rows plus LRU/RANDOM baselines (no hybrid λ rows)",
    )
    p.add_argument("--skip-sequential", action="store_true", help="Skip sequential-inter baseline")
    p.add_argument(
        "--legacy-io",
        action="store_true",
        help="Jun-17 pre-batch I/O (HETEROPREDICT_LEGACY_EXPERT_IO=1). Skips parallel-inter A/B.",
    )
    p.add_argument(
        "--strict-ssd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="O_DIRECT-only loads + drop page cache between sweep configs (default: on)",
    )
    args = p.parse_args()

    preset = PRESETS[args.preset]
    cache_sizes = args.cache_sizes or preset.default_cache_sizes or [8, 24]
    predictor_base_dir = args.predictor_base_dir or str(preset.predictor_base_dir)
    lookaheads = args.lookaheads or preset.lookaheads
    cache_slack = (
        args.cache_lookahead_slack
        if args.cache_lookahead_slack is not None
        else preset.cache_lookahead_slack
    )

    if args.budget_fractions is not None:
        budget_fractions = args.budget_fractions
        prefetch_budgets = None
        explicit_budgets = False
    elif args.prefetch_budgets is not None:
        budget_fractions = None
        prefetch_budgets = args.prefetch_budgets
        explicit_budgets = True
    elif preset.explicit_budgets:
        budget_fractions = None
        prefetch_budgets = preset.prefetch_budgets
        explicit_budgets = True
    else:
        budget_fractions = preset.budget_fractions
        prefetch_budgets = None
        explicit_budgets = False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preset_tag = args.preset if not args.baselines_only else "baselines"
    out_dir = Path(args.out_dir) if args.out_dir else STUDY_DIR / "results" / f"parallel_inter_{preset_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = "baselines" if args.baselines_only else f"{preset_tag}_predict"
    if args.legacy_io:
        inter_modes = [(True, True)]
    else:
        inter_modes = [(False, False)] if args.skip_sequential else [(False, False), (True, False)]

    print(
        f"Preset={args.preset}  predictor={predictor_base_dir}\n"
        f"  lookaheads={lookaheads}  cache_sizes={cache_sizes}  slack={cache_slack}\n"
        f"  budgets={'explicit ' + str(prefetch_budgets) if explicit_budgets else 'fractions ' + str(budget_fractions)}\n"
        f"  strict_ssd={args.strict_ssd}  legacy_io={args.legacy_io}",
        flush=True,
    )

    rc = 0
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (40 if args.quick else 80)
    num_prompts = args.num_prompts if args.num_prompts is not None else (1 if args.quick else 3)
    for sequential, legacy in inter_modes:
        suffix = "legacy" if legacy else ("sequential" if sequential else "parallel")
        csv_path = out_dir / f"sweep_{run_tag}_{suffix}.csv"
        step_rc = _run_sweep(
            out_csv=csv_path,
            cache_sizes=cache_sizes,
            tag=run_tag,
            expert_weights_dir=args.expert_weights_dir,
            predictor_base_dir=predictor_base_dir,
            lookaheads=lookaheads,
            budget_fractions=budget_fractions,
            prefetch_budgets=prefetch_budgets,
            explicit_budgets=explicit_budgets,
            cache_lookahead_slack=cache_slack,
            sequential_inter=sequential,
            baselines_only=args.baselines_only,
            predict_only=not args.baselines_only,
            max_new_tokens=max_new_tokens,
            num_prompts=num_prompts,
            strict_ssd=args.strict_ssd,
            legacy_io=legacy,
        )
        rc = rc or step_rc

    print(f"\nDone. Results in {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
