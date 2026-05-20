"""
main.py — Entry point for the MoE SSD-prefetch paper modeling framework.

Generates 4 publication-quality figures + CSV exports:
  Fig 1: TPS vs Cache Hit Rate          (SSD bottleneck, no predictor)
  Fig 2: TPS vs Predictor Recall        (how predictor helps, by lookahead)
  Fig 3: Token Latency Breakdown        (compute / hidden SSD / blocking SSD vs recall)
  Fig 4: Strategy Comparison            (sync vs async prefetch, by hit rate)

Metric names are aligned with sweep_predict_cached_cache_metrics.py:
  hit_rate_pct         ←→  cache_hit_rate
  pred_hit_rate_routed_forced_n_pct  ←→  recall (x-axis in Fig 2/3)
  tokens_per_second    ←→  tps
  stall_loads          ←→  blocking_invocations
  prefetch_loads       ←→  hidden_invocations
  avg_ms_per_expert_load  ←→  t_ssd_ms

Usage:
  python main.py [--out-dir /path/to/output]

Output:
  <out_dir>/figures/fig{1,2,3,4}_*.{png,pdf}
  <out_dir>/csv/fig{1,2,3,4}_*.csv
  <out_dir>/csv/cache_size_sweep.csv   (sanity check: hit_rate vs cache_size)
  <out_dir>/csv/assumptions.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from params import MODEL, HW, CACHE_SIZES_TO_SWEEP
from assumptions import DEFAULT as ASSUMPTIONS
from model import (
    sweep_cache_hit_rate,
    sweep_cache_size,
    sweep_recall_vs_tps,
    sweep_strategies,
)
from figures import (
    fig1_tps_vs_cache_hit_rate,
    fig2_recall_vs_tps,
    fig3_tps_vs_cache_size,
    fig8_strategy_comparison,
)


# ---------------------------------------------------------------------------
# Sweep resolution
# ---------------------------------------------------------------------------
CACHE_HIT_RATES  = np.linspace(0.0, 1.0, 200)
RECALLS_50_100   = np.linspace(0.50, 1.0, 100)
RECALLS_0_100    = np.linspace(0.00, 1.0, 200)
LOOKAHEAD_DEPTHS = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# All modeling assumptions come from assumptions.py — no magic numbers here.
# ---------------------------------------------------------------------------
A = ASSUMPTIONS

FIXED_CACHE_HIT_RATE = A.fixed_cache_hit_rate   # 0.72  (measured LRU cache=24)
FIXED_PRECISION      = A.fixed_precision         # 0.85  (estimate; low sensitivity)
FIXED_RECALL         = A.fixed_recall            # 0.80  (from benchmark logs)
FIXED_LOOKAHEAD      = A.fixed_lookahead         # 2     (matches benchmark config)
FIXED_REUSE          = A.reuse_factor            # 2.0   (estimate)
FIXED_OVERLAP_EFF    = A.overlap_efficiency      # 0.75  (conservative estimate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, csv_dir: str, stem: str):
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(csv_dir, f"{stem}.csv")
    df.to_csv(path, index=False)
    print(f"  CSV  {path}  ({len(df)} rows x {len(df.columns)} cols)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MoE SSD-prefetch paper figures (4 figures)")
    parser.add_argument("--out-dir", default="paper_output",
                        help="Root output directory (default: ./paper_output)")
    args = parser.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    csv_dir = os.path.join(args.out_dir, "csv")

    print(f"\n{'='*60}")
    print(f"  MoE SSD-Prefetch Paper -- Modeling Framework")
    print(f"  Output: {os.path.abspath(args.out_dir)}")
    print(f"{'='*60}")
    print(f"  Model     : {MODEL.name}")
    print(f"  T_compute : {HW.t_compute_ms} ms/token  ({HW.tps_compute:.1f} TPS ceiling)")
    print(f"  T_ssd     : {HW.t_ssd_ms} ms/expert  ({HW.ssd_throughput_gbps:.2f} GB/s)")
    print(f"  T_h2d     : {HW.t_h2d_ms} ms/expert  ({HW.h2d_throughput_gbps:.0f} GB/s)")
    print(f"{'='*60}\n")

    # ── Print and export assumptions ─────────────────────────────────────
    A.print_summary()
    assumptions_df = pd.DataFrame([
        {"assumption": "reuse_factor",             "value": A.reuse_factor,
         "source": "estimate -- avg tokens an expert is reused after one load",
         "sensitivity": "HIGH"},
        {"assumption": "fixed_recall",             "value": A.fixed_recall,
         "source": "benchmark logs: pred_hit_rate_routed_forced_n_pct 68-78%",
         "sensitivity": "HIGH"},
        {"assumption": "fixed_precision",          "value": A.fixed_precision,
         "source": "not directly measured; low sensitivity",
         "sensitivity": "LOW"},
        {"assumption": "fixed_cache_hit_rate",     "value": A.fixed_cache_hit_rate,
         "source": "measured: cache=24/layer, hit_rate_pct ~71-72%",
         "sensitivity": "MEDIUM"},
        {"assumption": "fixed_lookahead",          "value": A.fixed_lookahead,
         "source": "matches benchmark --predictor-lookahead 2 (eh1_h32_f2)",
         "sensitivity": "HIGH"},
        {"assumption": "routing_hit_rate_boost_pp","value": A.routing_hit_rate_boost_pp,
         "source": "ESTIMATE ONLY -- replace with measured lambda-sweep result",
         "sensitivity": "MEDIUM"},
        {"assumption": "overlap_efficiency",       "value": A.overlap_efficiency,
         "source": "ESTIMATE -- fraction of SSD bandwidth usable during compute",
         "sensitivity": "HIGH"},
    ])
    export_csv(assumptions_df, csv_dir, "assumptions")

    # ── Sanity-check table: cache_size -> hit_rate, demand, capacity ──────
    print("\n-- Cache size -> model metrics (sanity check) --")
    df_cs = sweep_cache_size(
        cache_sizes=CACHE_SIZES_TO_SWEEP,
        recall=FIXED_RECALL,
        precision=FIXED_PRECISION,
        lookahead_depth=FIXED_LOOKAHEAD,
        reuse_factor=FIXED_REUSE,
        overlap_efficiency=FIXED_OVERLAP_EFF,
    )
    export_csv(df_cs, csv_dir, "cache_size_sweep")
    cols = ["cache_size_per_layer", "cache_hit_rate", "demand_misses",
            "max_hidden_loads", "overlap_deficit", "sync_tps", "predict_tps"]
    print(df_cs[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── Figure 1: TPS vs Cache Hit Rate ───────────────────────────────────
    # Shows the raw SSD bottleneck with no predictor.
    # x: hit_rate_pct    y: tokens_per_second
    print("\nFig 1: TPS vs Cache Hit Rate (SSD bottleneck, no predictor)")
    df1 = sweep_cache_hit_rate(CACHE_HIT_RATES)
    export_csv(df1, csv_dir, "fig1_tps_vs_hit_rate")
    fig1_tps_vs_cache_hit_rate(df1, fig_dir)

    # ── Figure 2: TPS vs Predictor Recall ─────────────────────────────────
    # Shows how recall (pred_hit_rate_routed_forced_n_pct) drives TPS.
    # Multiple lookahead curves show overlap capacity vs demand.
    # x: recall    y: tokens_per_second    curves: lookahead depth
    print("\nFig 2: TPS vs Predictor Recall (by lookahead depth)")
    df2 = sweep_recall_vs_tps(
        recalls=RECALLS_50_100,
        lookahead_depths=LOOKAHEAD_DEPTHS,
        cache_hit_rate=FIXED_CACHE_HIT_RATE,
        precision=FIXED_PRECISION,
        reuse_factor=FIXED_REUSE,
        overlap_efficiency=FIXED_OVERLAP_EFF,
    )
    export_csv(df2, csv_dir, "fig2_tps_vs_recall")
    fig2_recall_vs_tps(df2, fig_dir)

    # ── Figure 3: TPS vs Cache Capacity ──────────────────────────────
    # Thesis point (iv): cache capacity impact.
    # Uses the calibrated working-set model: hit_rate = 1 - exp(-C / 18.8)
    print("\nFig 3: TPS vs Cache Capacity")
    fig3_tps_vs_cache_size(df_cs, fig_dir)   # reuse the sanity-check sweep

    # ── Figure 4: Strategy Comparison ─────────────────────────────────────
    # Compares sync on-demand vs async prefetch (predictor) across hit rates.
    # x: hit_rate_pct    y: tokens_per_second    curves: strategy
    print("\nFig 4: Strategy Comparison (sync vs async prefetch)")
    df4 = sweep_strategies(
        cache_hit_rates=CACHE_HIT_RATES,
        recall=FIXED_RECALL,
        precision=FIXED_PRECISION,
        lookahead_depth=FIXED_LOOKAHEAD,
        reuse_factor=FIXED_REUSE,
        overlap_efficiency=FIXED_OVERLAP_EFF,
    )
    export_csv(df4, csv_dir, "fig4_strategy_comparison")
    fig8_strategy_comparison(df4, fig_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    raw_cap = FIXED_LOOKAHEAD * HW.t_compute_ms / HW.t_ssd_ms
    eff_cap = FIXED_OVERLAP_EFF * raw_cap
    experts = MODEL.active_k * MODEL.num_layers
    demand  = experts * (1.0 - FIXED_CACHE_HIT_RATE)
    unique  = demand / max(FIXED_REUSE, 1.0)
    deficit = eff_cap - unique
    status  = f"surplus of {deficit:.0f}" if deficit >= 0 else f"DEFICIT of {-deficit:.0f}"

    print(f"\n{'='*60}")
    print(f"  KEY MODEL METRICS  (lookahead={FIXED_LOOKAHEAD}, "
          f"overlap_eff={FIXED_OVERLAP_EFF:.0%})")
    print(f"{'='*60}")
    print(f"  Demand misses/token  : {demand:.0f} experts  "
          f"(hit_rate_pct={FIXED_CACHE_HIT_RATE*100:.0f}%)")
    print(f"  Unique future misses : {unique:.0f} experts  "
          f"(reuse_factor={FIXED_REUSE})")
    print(f"  Raw overlap capacity : {raw_cap:.0f} experts  "
          f"(= {FIXED_LOOKAHEAD} x {HW.t_compute_ms}ms / {HW.t_ssd_ms}ms)")
    print(f"  Effective capacity   : {eff_cap:.0f} experts  "
          f"(= {FIXED_OVERLAP_EFF:.0%} x {raw_cap:.0f})")
    print(f"  Budget vs demand     : {status}")
    print(f"\n  Figures : {os.path.abspath(fig_dir)}")
    print(f"  CSVs    : {os.path.abspath(csv_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
