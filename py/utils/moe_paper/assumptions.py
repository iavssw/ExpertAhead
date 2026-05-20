"""
assumptions.py — Central registry of all modeling assumptions.

Every non-trivial constant used by model.py and main.py lives here with:
  - value
  - unit / range
  - justification / source
  - sensitivity flag (HIGH = changing this meaningfully shifts conclusions)

Import ModelAssumptions.DEFAULT for a single canonical config,
or construct a custom one for sensitivity sweeps.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelAssumptions:
    """
    All tunable modeling assumptions in one place.
    Every field has a docstring that explains what it means and why the
    default value was chosen.
    """

    # ------------------------------------------------------------------
    # Expert reuse locality
    # ------------------------------------------------------------------
    reuse_factor: float = 2.0
    """
    Average number of future decode tokens that will invoke the same expert
    after it has been loaded into VRAM once.

    Intuition: if an expert is "hot" for 2 consecutive tokens, one SSD load
    amortises over 2 invocations, halving the unique miss count.

    Default = 2.0
    Justification: approximate estimate based on typical autoregressive
    generation patterns. Qwen3 has 128 experts/layer with only 8 active,
    creating high sparsity, but topic-consistent text tends to activate the
    same specialist experts repeatedly over short windows.
    In practice reuse varies with prompt style and layer depth.

    Sensitivity: HIGH — this directly sets unique_future_experts
    = demand_misses / reuse_factor, which gates all overlap calculations.
    A reuse_factor of 1.0 (no reuse) is the pessimistic bound;
    3–4 is a plausible optimistic bound for focused factual prompts.

    How to measure: count expert_id repetitions across consecutive tokens
    in the benchmark log's "Predictor TopK matches" output.
    Empirical measurement from your system would replace this estimate.
    """

    # ------------------------------------------------------------------
    # Fixed predictor recall used as the "operating point" for single-
    # variable sweeps (figs 1, 4, 5, 8).  Figs 2 and 3 sweep this value.
    # ------------------------------------------------------------------
    fixed_recall: float = 0.80
    """
    Predictor recall used when recall is NOT the x-axis variable.

    Default = 0.80  (80%)
    Justification: approximately consistent with the benchmark logs which
    show Predictor RoutedForcedN HitRate in the 68–78% range per layer.
    We round up slightly to 80% to represent a well-tuned predictor.

    Sensitivity: HIGH for TPS curves. All figures that don't sweep recall
    inherit this value directly.
    """

    # ------------------------------------------------------------------
    # Fixed predictor precision
    # ------------------------------------------------------------------
    fixed_precision: float = 0.85
    """
    Fraction of prefetched experts that are actually needed (1 - FPR).

    Default = 0.85  (85%)
    Justification: not directly measured, but precision is a secondary
    concern in our model — false positives mainly waste prefetch bandwidth
    rather than causing blocking stalls.  A reasonable, slightly optimistic
    estimate for a trained predictor.

    Sensitivity: LOW. Fig 3 demonstrates that varying precision from
    50%→100% shifts TPS far less than the same variation in recall.
    The precision effect enters only via the wasted_ratio budget correction:
      effective_budget = max_hidden / (1 + 1/precision - 1)
    At precision=0.85 the budget shrinks by ~18%; at precision=0.5 by ~100%.
    """

    # ------------------------------------------------------------------
    # Fixed cache hit rate (LRU baseline)
    # ------------------------------------------------------------------
    fixed_cache_hit_rate: float = 0.72
    """
    Base LRU cache hit rate, used as the "operating point" for sweeps that
    don't vary the cache hit rate (figs 2, 3, 5, 7).

    Default = 0.72  (72%)
    Justification: directly from benchmark logs:
      "Layer 46 Cache Stats: HitRate=71.85%"
      "Layer 47 Cache Stats: HitRate=65.45%"
    We use the higher-layer value (~72%) as representative.
    This corresponds to cache_size=24 experts/layer in the benchmarks.

    Sensitivity: MEDIUM. Figures 1 and 8 sweep this; others fix it here.
    """

    # ------------------------------------------------------------------
    # Fixed lookahead depth
    # ------------------------------------------------------------------
    fixed_lookahead: int = 2
    """
    Number of future tokens the predictor prefetches for.

    Default = 2
    Justification: matches the benchmark configuration
    (--predictor-lookahead 2, subdirectory eh1_h32_f2).

    Sensitivity: HIGH for overlap capacity (fig 4). Fig 2 sweeps this.
    """

    # ------------------------------------------------------------------
    # Cache-aware routing boost (fig 8 / strategy comparison)
    # ------------------------------------------------------------------
    routing_hit_rate_boost_pp: float = 0.10
    """
    Percentage-point increase in cache hit rate assumed for the
    "async prefetch + cache-aware routing" strategy in Fig 8.

    Default = 0.10  (+10 pp, e.g. 72% → 82%)
    Justification: this is a modeling assumption, NOT a measured value.
    Cache-aware routing (λ-biased routing) is known to improve hit rate,
    but the exact magnitude depends on λ and workload.  +10 pp is a
    plausible conservative estimate for λ ≈ 0.5–1.0 based on prior
    sweep experiments in sweep_predict_cached_cache_metrics.py.

    Sensitivity: MEDIUM for fig 8 gap between async and routing curves.
    This value should be replaced with a measured sweep result when available.
    """

    # ------------------------------------------------------------------
    # Overlap efficiency: fraction of theoretical SSD bandwidth usable
    # for background prefetch during a compute step.
    # ------------------------------------------------------------------
    overlap_efficiency: float = 0.75
    """
    Fraction of the theoretical overlap budget that is practically usable
    for hiding SSD loads behind compute.

    Theoretical max hidden loads (serial SSD model):
      max_hidden = (lookahead_depth × T_compute) / T_ssd
               = (2 × 49 ms) / 0.834 ms ≈ 117 experts   [lookahead=2]
               = (1 × 49 ms) / 0.834 ms ≈  59 experts   [lookahead=1]

    Effective max hidden loads with efficiency factor:
      effective_max_hidden = overlap_efficiency × max_hidden

    Why it is < 1.0:
      - The background thread pool (ThreadedTorchScriptPredictor) has
        scheduling overhead: dispatching, queue management, wakeup latency.
      - O_DIRECT reads require aligned buffers and file open/seek per expert;
        this adds per-read fixed cost beyond pure data transfer time.
      - PCIe bandwidth is shared with the attention weight transfers during
        the forward pass; prefetch contention reduces available throughput.
      - The SSD scheduler may reorder or delay small reads internally.

    Concrete consequence:
      At lookahead=1, even with overlap_efficiency=1.0, max_hidden≈59 but
      demand misses≈107 (at 72% hit rate), so >40% of loads are blocking.
      With overlap_efficiency=0.75: effective_max_hidden≈44 → ~59% blocking.
      This is the fundamental reason SSD-backed inference is latency-limited
      at the cache sizes currently benchmarked.

    Default = 0.75  (conservative — 75% of bandwidth usable for prefetch)
    Sensitivity: HIGH. Setting this to 1.0 is optimistic and overstates
    the benefit of prefetching. The true value depends on the thread pool
    implementation and SSD characteristics and should ideally be measured
    directly (e.g., by comparing prewarm_experts() throughput during active
    decode vs. idle).
    """

    # ------------------------------------------------------------------
    # Reuse factor scenarios for the sensitivity study (fig 7)
    # ------------------------------------------------------------------
    reuse_scenarios: Dict[str, float] = field(default_factory=lambda: {
        "High reuse (Qwen-like)":  3.5,
        "Medium reuse":             2.0,
        "Low reuse (Mixtral-like)": 1.2,
    })
    """
    Named reuse factor scenarios used in Fig 7.

    "High reuse (Qwen-like)"   = 3.5
      Rationale: 128 experts, only 8 active, strong topic locality means
      the same small set of experts fires repeatedly over many tokens.

    "Medium reuse"             = 2.0
      Rationale: the DEFAULT assumption above; balanced estimate.

    "Low reuse (Mixtral-like)" = 1.2
      Rationale: Mixtral has 8 experts with 2 active — much smaller pool,
      so distinct experts are selected more frequently per token, reducing
      amortisation. The 1.2 value reflects a mild locality advantage over
      the worst case (1.0 = no reuse at all).
    """

    def print_summary(self):
        """Print a human-readable summary of all assumptions."""
        from params import HW, MODEL
        sep = "=" * 64
        experts = MODEL.active_k * MODEL.num_layers
        demand_misses = experts * (1 - self.fixed_cache_hit_rate)
        max_hidden_raw = self.fixed_lookahead * HW.t_compute_ms / HW.t_ssd_ms
        max_hidden_eff = self.overlap_efficiency * max_hidden_raw
        print(f"\n{sep}")
        print("  MODELING ASSUMPTIONS")
        print(sep)
        print(f"  reuse_factor                : {self.reuse_factor}")
        print(f"    → estimate; avg tokens an expert fires after one SSD load")
        print(f"    → HIGH sensitivity; empirical measurement recommended")
        print()
        print(f"  fixed_recall                : {self.fixed_recall*100:.0f}%")
        print(f"    → from benchmark logs (RoutedForcedN HitRate 68–78%)")
        print()
        print(f"  fixed_precision             : {self.fixed_precision*100:.0f}%")
        print(f"    → not directly measured; LOW sensitivity (see fig 3)")
        print()
        print(f"  fixed_cache_hit_rate        : {self.fixed_cache_hit_rate*100:.0f}%")
        print(f"    → measured: cache=24/layer LRU hit rate ~71–72%")
        print()
        print(f"  fixed_lookahead             : {self.fixed_lookahead} tokens")
        print(f"    → matches benchmark --predictor-lookahead 2 (eh1_h32_f2)")
        print()
        print(f"  overlap_efficiency          : {self.overlap_efficiency*100:.0f}%")
        print(f"    → fraction of SSD bandwidth usable for background prefetch")
        print(f"    → theoretical max hidden: {max_hidden_raw:.0f} experts "
              f"(lookahead={self.fixed_lookahead})")
        print(f"    → effective max hidden:   {max_hidden_eff:.0f} experts "
              f"(= {self.overlap_efficiency:.0%} × {max_hidden_raw:.0f})")
        print(f"    → demand misses/token:    {demand_misses:.0f} experts "
              f"(at {self.fixed_cache_hit_rate*100:.0f}% hit rate)")
        deficit = max_hidden_eff - demand_misses
        status = f"surplus of {deficit:.0f}" if deficit >= 0 else f"DEFICIT of {-deficit:.0f}"
        print(f"    → budget vs demand:       {status} experts")
        print(f"    → ESTIMATE; ideally measured by running prefetch")
        print(f"       throughput during active decode")
        print()
        print(f"  routing_hit_rate_boost_pp   : +{self.routing_hit_rate_boost_pp*100:.0f} pp")
        print(f"    → ESTIMATE only; replace with measured λ-sweep result")
        print()
        print("  reuse_scenarios (fig 7):")
        for name, rf in self.reuse_scenarios.items():
            print(f"    {rf:.1f}  {name}")
        print(sep + "\n")


# Singleton default — import this everywhere
DEFAULT = ModelAssumptions()
