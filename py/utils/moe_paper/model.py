"""
model.py — Overlap-aware analytical model for SSD-backed MoE decode.

Core insight: the predictor only helps if the overlap budget is large enough
to hide future SSD loads behind ongoing compute.

  overlap_budget   = lookahead_depth × T_compute
  max_hidden_loads = overlap_budget  / T_ssd

Unique future misses ≤ max_hidden_loads  →  no stall
Unique future misses  > max_hidden_loads  →  pipeline collapse

Precision = fraction of prefetched experts that are actually used
Recall    = fraction of needed future experts successfully prefetched
            (recall matters far more: false negatives → blocking stalls)

All output units:
  - latencies : ms
  - TPS       : tokens/second
  - speedup   : relative to no-cache synchronous baseline
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from params import MODEL, HW, ModelParams, HardwareParams


# ---------------------------------------------------------------------------
# Token-level latency model
# ---------------------------------------------------------------------------

@dataclass
class TokenLatency:
    """Decomposed latency for one decode token."""
    compute_ms: float
    hidden_ssd_ms: float    # SSD loads overlapped behind compute — no stall
    blocking_ssd_ms: float  # SSD loads that must be waited for — stall
    total_ms: float
    tps: float
    speedup_vs_sync: float  # relative to all-miss synchronous baseline


def _clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))


def token_latency(
    *,
    recall: float,
    precision: float,
    lookahead_depth: int,
    cache_hit_rate: float,
    reuse_factor: float = 2.0,
    overlap_efficiency: float = 0.75,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> TokenLatency:
    """
    Compute per-token latency given predictor quality and hardware parameters.

    Overlap model (serial SSD bandwidth limit):
      Theoretical max hidden loads = (lookahead_depth × T_compute) / T_ssd
      Effective max hidden loads   = overlap_efficiency × theoretical max

      At lookahead=1: theoretical = 49/0.834 ≈ 59 experts
      At lookahead=2: theoretical = 98/0.834 ≈ 117 experts
      Demand misses at 72% hit rate = 384 × 0.28 ≈ 107 experts

      So overlap is a genuine constraint: at lookahead=1 only ~44 experts
      (with overlap_efficiency=0.75) can be hidden out of 107 demanded —
      the rest are blocking stalls.

    Args:
        recall:             Fraction [0,1] of future-needed experts correctly prefetched.
        precision:          Fraction [0,1] of prefetched experts that are actually needed.
                            Imprecise prefetches waste overlap budget on useless loads.
        lookahead_depth:    Tokens predicted ahead; sets the overlap window size.
        cache_hit_rate:     LRU hit rate; use natural_hit_rate(cache_size) from params.py
                            for a physics-grounded value, or set directly for sweeps.
        reuse_factor:       Average future tokens that reuse a given expert after one load.
                            Reduces the count of *unique* loads the predictor must cover.
        overlap_efficiency: Fraction of theoretical bandwidth usable for background prefetch.
                            Accounts for thread pool overhead, O_DIRECT seek costs, PCIe
                            contention. Default 0.75 is conservative.
        hw:                 HardwareParams instance.
        model:              ModelParams instance.

    Returns:
        TokenLatency with full decomposition: compute / hidden SSD / blocking SSD.
    """
    recall             = _clip(recall)
    precision          = _clip(precision)
    cache_hit_rate     = _clip(cache_hit_rate)
    overlap_efficiency = _clip(overlap_efficiency, 0.0, 1.0)

    experts_per_token = model.active_k * model.num_layers  # 384

    # ── Baseline for speedup denominator ─────────────────────────────────
    # All-miss synchronous: every expert load is a blocking stall.
    t_sync_baseline_ms = hw.t_compute_ms + experts_per_token * hw.t_ssd_ms

    # ── Overlap budget (bandwidth-limited) ────────────────────────────────
    # During lookahead_depth compute steps, the SSD can transfer at most
    # (lookahead_depth × T_compute / T_ssd) experts — this is the SERIAL
    # bandwidth ceiling, not a parallelism assumption.
    # overlap_efficiency < 1 reduces usable capacity for reasons documented
    # in assumptions.py.
    raw_overlap_capacity = lookahead_depth * hw.t_compute_ms / hw.t_ssd_ms
    max_hidden_loads     = overlap_efficiency * raw_overlap_capacity

    # ── Demand misses ─────────────────────────────────────────────────────
    demand_misses = experts_per_token * (1.0 - cache_hit_rate)

    # ── Unique future experts the predictor must cover ────────────────────
    # reuse_factor > 1 means the same expert fires multiple times in the
    # lookahead window; one prefetch covers all those invocations.
    unique_future_misses = demand_misses / max(reuse_factor, 1.0)

    # ── Predictor coverage ────────────────────────────────────────────────
    # Recall: fraction of unique_future_misses correctly identified and prefetched.
    # False negatives (1 - recall) are never prefetched → always blocking.
    correctly_prefetched = recall * unique_future_misses
    false_negatives      = (1.0 - recall) * unique_future_misses  # always block

    # Precision: fraction of prefetch slots that are actually needed.
    # False positives (1 - precision) consume overlap budget without benefit.
    # Effective useful capacity = max_hidden_loads × precision
    # (for every useful load the prefetcher issues 1/precision total loads)
    useful_overlap_capacity = max_hidden_loads * precision

    # Experts we can actually hide = min(what predictor provides, useful budget)
    hidden_unique      = min(correctly_prefetched, useful_overlap_capacity)
    overflow_unique    = max(0.0, correctly_prefetched - useful_overlap_capacity)

    # Blocking = false negatives + overflow beyond budget, scaled back to invocations
    blocking_unique     = false_negatives + overflow_unique
    blocking_invocations = blocking_unique * reuse_factor
    hidden_invocations   = hidden_unique   * reuse_factor

    # Sanity clamp: totals can't exceed demand misses
    blocking_invocations = min(blocking_invocations, demand_misses)
    hidden_invocations   = min(hidden_invocations, demand_misses - blocking_invocations)

    hidden_ms   = hidden_invocations   * hw.t_ssd_ms
    blocking_ms = blocking_invocations * hw.t_ssd_ms

    # Token latency: compute + any blocking stalls
    # Hidden loads run in the background and do not add to token latency.
    total_ms = hw.t_compute_ms + blocking_ms
    tps      = 1000.0 / total_ms if total_ms > 0 else 0.0
    speedup  = t_sync_baseline_ms / total_ms if total_ms > 0 else 0.0

    return TokenLatency(
        compute_ms=hw.t_compute_ms,
        hidden_ssd_ms=hidden_ms,
        blocking_ssd_ms=blocking_ms,
        total_ms=total_ms,
        tps=tps,
        speedup_vs_sync=speedup,
    )


# ---------------------------------------------------------------------------
# Sweep helpers — return DataFrames ready for CSV export
# ---------------------------------------------------------------------------

def sweep_cache_hit_rate(
    hit_rates: np.ndarray,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """
    Fig 1: TPS vs cache hit rate, no predictor (synchronous on-demand loads for misses).
    Shows the raw cost of SSD misses without any prefetching.
    Anchor: hit=1.0 → base TPS (20); hit≈0.72 → cached LRU measured TPS.
    NOTE: this sweep has no predictor. overlap_efficiency is not used here.
    """
    rows = []
    experts = model.active_k * model.num_layers
    t_sync_baseline = hw.t_compute_ms + experts * hw.t_ssd_ms

    for hr in hit_rates:
        hr = float(np.clip(hr, 0, 1))
        misses = experts * (1.0 - hr)
        blocking_ms = misses * hw.t_ssd_ms
        total_ms = hw.t_compute_ms + blocking_ms
        tps = 1000.0 / total_ms
        rows.append(dict(
            cache_hit_rate=hr,
            demand_misses=misses,
            blocking_ssd_ms=blocking_ms,
            compute_ms=hw.t_compute_ms,
            token_latency_ms=total_ms,
            tps=tps,
            speedup=t_sync_baseline / total_ms,
        ))
    return pd.DataFrame(rows)


def sweep_cache_size(
    cache_sizes: List[int],
    recall: float = 0.80,
    precision: float = 0.85,
    lookahead_depth: int = 2,
    reuse_factor: float = 2.0,
    overlap_efficiency: float = 0.75,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """
    Sweep per-layer cache size, deriving hit rate from the calibrated
    exponential working-set model (natural_hit_rate in params.py).

    Compares:
      - Synchronous baseline (no predictor)
      - Async prefetch (with predictor at given recall/precision/lookahead)

    This directly answers: "how much does cache size matter and how much
    does the predictor help on top of a given cache?"
    """
    from params import natural_hit_rate as nhr
    experts = model.active_k * model.num_layers
    t_sync_all_miss = hw.t_compute_ms + experts * hw.t_ssd_ms
    rows = []
    for cs in cache_sizes:
        hr = nhr(cs)
        demand_misses = experts * (1.0 - hr)

        # Synchronous (no predictor): all misses are blocking
        t_sync = hw.t_compute_ms + demand_misses * hw.t_ssd_ms
        tps_sync = 1000.0 / t_sync

        # With predictor
        lt = token_latency(
            recall=recall,
            precision=precision,
            lookahead_depth=lookahead_depth,
            cache_hit_rate=hr,
            reuse_factor=reuse_factor,
            overlap_efficiency=overlap_efficiency,
            hw=hw, model=model,
        )

        raw_capacity = lookahead_depth * hw.t_compute_ms / hw.t_ssd_ms
        max_hidden = overlap_efficiency * raw_capacity

        rows.append(dict(
            cache_size_per_layer=cs,
            cache_hit_rate=hr,
            demand_misses=demand_misses,
            unique_future_misses=demand_misses / max(reuse_factor, 1.0),
            raw_overlap_capacity=raw_capacity,
            max_hidden_loads=max_hidden,
            overlap_deficit=max(0.0, demand_misses / max(reuse_factor, 1.0) - max_hidden),
            # Sync baseline
            sync_token_latency_ms=t_sync,
            sync_tps=tps_sync,
            sync_speedup=t_sync_all_miss / t_sync,
            # With predictor
            predict_hidden_ms=lt.hidden_ssd_ms,
            predict_blocking_ms=lt.blocking_ssd_ms,
            predict_token_latency_ms=lt.total_ms,
            predict_tps=lt.tps,
            predict_speedup=lt.speedup_vs_sync,
            # Parameters
            recall=recall,
            precision=precision,
            lookahead_depth=lookahead_depth,
            reuse_factor=reuse_factor,
            overlap_efficiency=overlap_efficiency,
        ))
    return pd.DataFrame(rows)


def sweep_recall_vs_tps(
    recalls: np.ndarray,
    lookahead_depths: List[int],
    cache_hit_rate: float = 0.72,
    precision: float = 0.85,
    reuse_factor: float = 2.0,
    overlap_efficiency: float = 0.75,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """Fig 2: TPS vs recall for multiple lookahead depths."""
    rows = []
    for la in lookahead_depths:
        raw_cap = la * hw.t_compute_ms / hw.t_ssd_ms
        for r in recalls:
            lt = token_latency(
                recall=float(r),
                precision=precision,
                lookahead_depth=la,
                cache_hit_rate=cache_hit_rate,
                reuse_factor=reuse_factor,
                overlap_efficiency=overlap_efficiency,
                hw=hw, model=model,
            )
            rows.append(dict(
                recall=float(r),
                precision=precision,
                lookahead_depth=la,
                cache_hit_rate=cache_hit_rate,
                reuse_factor=reuse_factor,
                overlap_efficiency=overlap_efficiency,
                raw_overlap_capacity=raw_cap,
                max_hidden_loads=overlap_efficiency * raw_cap,
                compute_ms=lt.compute_ms,
                hidden_ssd_ms=lt.hidden_ssd_ms,
                blocking_ssd_ms=lt.blocking_ssd_ms,
                token_latency_ms=lt.total_ms,
                tps=lt.tps,
                speedup=lt.speedup_vs_sync,
            ))
    return pd.DataFrame(rows)


def sweep_precision_recall(
    recalls: np.ndarray,
    precisions: List[float],
    lookahead_depth: int = 2,
    cache_hit_rate: float = 0.72,
    reuse_factor: float = 2.0,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """Fig 3: Precision vs recall sensitivity — shows recall dominates."""
    rows = []
    for p in precisions:
        for r in recalls:
            lt = token_latency(
                recall=float(r),
                precision=float(p),
                lookahead_depth=lookahead_depth,
                cache_hit_rate=cache_hit_rate,
                reuse_factor=reuse_factor,
                hw=hw, model=model,
            )
            rows.append(dict(
                recall=float(r),
                precision=float(p),
                lookahead_depth=lookahead_depth,
                cache_hit_rate=cache_hit_rate,
                overlap_budget_ms=lookahead_depth * hw.t_compute_ms,
                max_hidden_loads=lookahead_depth * hw.t_compute_ms / hw.t_ssd_ms,
                compute_ms=lt.compute_ms,
                hidden_ssd_ms=lt.hidden_ssd_ms,
                blocking_ssd_ms=lt.blocking_ssd_ms,
                token_latency_ms=lt.total_ms,
                tps=lt.tps,
                speedup=lt.speedup_vs_sync,
            ))
    return pd.DataFrame(rows)


def sweep_lookahead_overlap(
    lookaheads: np.ndarray,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """Fig 4: Lookahead depth vs overlap capacity and max hidden loads."""
    rows = []
    for la in lookaheads:
        la = float(la)
        budget_ms = la * hw.t_compute_ms
        max_hidden = budget_ms / hw.t_ssd_ms
        rows.append(dict(
            lookahead_depth=la,
            overlap_budget_ms=budget_ms,
            max_hidden_loads=max_hidden,
            t_compute_ms=hw.t_compute_ms,
            t_ssd_ms=hw.t_ssd_ms,
        ))
    return pd.DataFrame(rows)


def sweep_latency_breakdown(
    recalls: np.ndarray,
    lookahead_depth: int = 2,
    cache_hit_rate: float = 0.72,
    precision: float = 0.85,
    reuse_factor: float = 2.0,
    overlap_efficiency: float = 0.75,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """Stacked latency breakdown (compute / hidden SSD / blocking SSD) vs recall."""
    rows = []
    for r in recalls:
        lt = token_latency(
            recall=float(r),
            precision=precision,
            lookahead_depth=lookahead_depth,
            cache_hit_rate=cache_hit_rate,
            reuse_factor=reuse_factor,
            overlap_efficiency=overlap_efficiency,
            hw=hw, model=model,
        )
        rows.append(dict(
            recall=float(r),
            precision=precision,
            lookahead_depth=lookahead_depth,
            overlap_efficiency=overlap_efficiency,
            compute_ms=lt.compute_ms,
            hidden_ssd_ms=lt.hidden_ssd_ms,
            blocking_ssd_ms=lt.blocking_ssd_ms,
            token_latency_ms=lt.total_ms,
            tps=lt.tps,
        ))
    return pd.DataFrame(rows)


def sweep_pipeline_collapse(
    unique_misses: np.ndarray,
    lookahead_depth: int = 2,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """
    Fig 6: Pipeline collapse — TPS vs unique_future_misses relative to max_hidden_loads.
    Shows the sharp knee when unique_future_misses > max_hidden_loads.
    """
    budget_ms = lookahead_depth * hw.t_compute_ms
    max_hidden = budget_ms / hw.t_ssd_ms
    experts = model.active_k * model.num_layers
    t_sync_baseline = hw.t_compute_ms + experts * hw.t_ssd_ms

    rows = []
    for ufm in unique_misses:
        ufm = float(ufm)
        hidden = min(ufm, max_hidden)
        blocking = max(0.0, ufm - max_hidden)
        blocking_ms = blocking * hw.t_ssd_ms
        total_ms = hw.t_compute_ms + blocking_ms
        tps = 1000.0 / total_ms
        rows.append(dict(
            unique_future_misses=ufm,
            max_hidden_loads=max_hidden,
            normalized_misses=ufm / max_hidden if max_hidden > 0 else float('inf'),
            hidden_loads=hidden,
            blocking_loads=blocking,
            hidden_ssd_ms=hidden * hw.t_ssd_ms,
            blocking_ssd_ms=blocking_ms,
            token_latency_ms=total_ms,
            tps=tps,
            speedup=t_sync_baseline / total_ms,
            lookahead_depth=lookahead_depth,
        ))
    return pd.DataFrame(rows)


def sweep_reuse_sensitivity(
    recalls: np.ndarray,
    reuse_factors: Dict[str, float],
    lookahead_depth: int = 2,
    cache_hit_rate: float = 0.72,
    precision: float = 0.85,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """Fig 7: Reuse sensitivity — high-reuse (Qwen-like) vs low-reuse (Mixtral-like)."""
    rows = []
    for name, rf in reuse_factors.items():
        for r in recalls:
            lt = token_latency(
                recall=float(r),
                precision=precision,
                lookahead_depth=lookahead_depth,
                cache_hit_rate=cache_hit_rate,
                reuse_factor=rf,
                hw=hw, model=model,
            )
            rows.append(dict(
                reuse_scenario=name,
                reuse_factor=rf,
                recall=float(r),
                precision=precision,
                lookahead_depth=lookahead_depth,
                compute_ms=lt.compute_ms,
                hidden_ssd_ms=lt.hidden_ssd_ms,
                blocking_ssd_ms=lt.blocking_ssd_ms,
                token_latency_ms=lt.total_ms,
                tps=lt.tps,
                speedup=lt.speedup_vs_sync,
            ))
    return pd.DataFrame(rows)


def sweep_strategies(
    cache_hit_rates: np.ndarray,
    recall: float = 0.80,
    precision: float = 0.85,
    lookahead_depth: int = 2,
    reuse_factor: float = 2.0,
    overlap_efficiency: float = 0.75,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> pd.DataFrame:
    """
    Compare two strategies across cache hit rates:
      1. Synchronous on-demand (no prefetch, no predictor)
      2. Async prefetch with predictor (recall/precision/lookahead)
    The routing boost (+pp) for a third curve uses assumptions.routing_hit_rate_boost_pp.
    """
    from assumptions import DEFAULT as A
    experts = model.active_k * model.num_layers
    t_sync_baseline = hw.t_compute_ms + experts * hw.t_ssd_ms
    rows = []
    for hr in cache_hit_rates:
        hr = float(np.clip(hr, 0, 1))

        # Strategy 1: synchronous on-demand (stall_loads = all misses)
        misses = experts * (1.0 - hr)
        t_sync = hw.t_compute_ms + misses * hw.t_ssd_ms
        tps_sync = 1000.0 / t_sync

        # Strategy 2: async prefetch with predictor
        lt_async = token_latency(
            recall=recall, precision=precision,
            lookahead_depth=lookahead_depth,
            cache_hit_rate=hr,
            reuse_factor=reuse_factor,
            overlap_efficiency=overlap_efficiency,
            hw=hw, model=model,
        )

        # Strategy 3: async prefetch + cache-aware routing (+pp hit rate boost)
        hr_boosted = min(hr + A.routing_hit_rate_boost_pp, 1.0)
        lt_boost = token_latency(
            recall=recall, precision=precision,
            lookahead_depth=lookahead_depth,
            cache_hit_rate=hr_boosted,
            reuse_factor=reuse_factor,
            overlap_efficiency=overlap_efficiency,
            hw=hw, model=model,
        )

        rows.append(dict(
            cache_hit_rate=hr,
            overlap_efficiency=overlap_efficiency,
            # sync
            sync_token_latency_ms=t_sync,
            sync_tps=tps_sync,
            sync_speedup=t_sync_baseline / t_sync,
            sync_stall_loads=misses,
            # async prefetch
            async_token_latency_ms=lt_async.total_ms,
            async_tps=lt_async.tps,
            async_speedup=lt_async.speedup_vs_sync,
            async_blocking_ms=lt_async.blocking_ssd_ms,
            async_hidden_ms=lt_async.hidden_ssd_ms,
            # async + routing
            routing_token_latency_ms=lt_boost.total_ms,
            routing_tps=lt_boost.tps,
            routing_speedup=lt_boost.speedup_vs_sync,
            routing_effective_hit_rate=hr_boosted,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional: timeline / event trace export
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    name: str       # e.g. "compute", "ssd_load", "h2d"
    start_ms: float
    end_ms: float
    token_idx: int
    expert_idx: Optional[int] = None
    kind: str = "compute"   # "compute" | "ssd" | "h2d" | "blocked"


def generate_timeline_trace(
    num_tokens: int,
    recall: float,
    lookahead_depth: int,
    cache_hit_rate: float = 0.72,
    reuse_factor: float = 2.0,
    hw: HardwareParams = HW,
    model: ModelParams = MODEL,
) -> List[TraceEvent]:
    """
    Generate a simplified event trace for num_tokens of decode.
    Returns a list of TraceEvents suitable for Gantt chart rendering.

    NOTE: This is a simplified model — it does not simulate per-expert
    scheduling, only aggregate compute/overlap/blocking intervals per token.
    """
    events: List[TraceEvent] = []
    t = 0.0
    experts = model.active_k * model.num_layers
    max_hidden = lookahead_depth * hw.t_compute_ms / hw.t_ssd_ms
    demand_misses = experts * (1.0 - cache_hit_rate)
    unique_future = demand_misses / max(reuse_factor, 1.0)
    hidden_count = min(recall * unique_future, max_hidden)
    blocking_count = max(0.0, demand_misses - hidden_count * reuse_factor
                         + (1.0 - recall) * unique_future * reuse_factor)
    blocking_count = min(blocking_count, demand_misses)

    for tok_idx in range(num_tokens):
        # Compute span
        events.append(TraceEvent(
            name="compute", start_ms=t, end_ms=t + hw.t_compute_ms,
            token_idx=tok_idx, kind="compute",
        ))
        t_compute_end = t + hw.t_compute_ms

        # Hidden prefetch loads (overlapped with compute of next token)
        # Model as running in parallel from t to t + overlap_budget
        hidden_end = t + hidden_count * hw.t_ssd_ms
        if hidden_count > 0:
            events.append(TraceEvent(
                name="ssd_prefetch", start_ms=t, end_ms=hidden_end,
                token_idx=tok_idx, kind="ssd",
            ))

        # Blocking loads (after compute, stall the next token)
        if blocking_count > 0:
            blocking_start = t_compute_end
            blocking_end = blocking_start + blocking_count * hw.t_ssd_ms
            events.append(TraceEvent(
                name="ssd_blocking", start_ms=blocking_start, end_ms=blocking_end,
                token_idx=tok_idx, kind="blocked",
            ))
            t = blocking_end
        else:
            t = t_compute_end

    return events


def trace_to_dataframe(events: List[TraceEvent]) -> pd.DataFrame:
    return pd.DataFrame([
        dict(name=e.name, start_ms=e.start_ms, end_ms=e.end_ms,
             duration_ms=e.end_ms - e.start_ms,
             token_idx=e.token_idx, kind=e.kind)
        for e in events
    ])
