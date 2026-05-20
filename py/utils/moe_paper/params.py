"""
params.py — Measured hardware and model constants for Qwen3-30B-A3B on AMD Ryzen AI Max+ 395.

All timing values are derived directly from benchmark_expert_loading.py measurements.
Edit this file to retune the model if new benchmarks are collected.
"""
import math

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelParams:
    """Architecture constants for Qwen3-30B-A3B AWQ."""
    name: str = "Qwen3-30B-A3B AWQ"
    num_layers: int = 48           # Total transformer layers (all are MoE)
    num_experts: int = 128         # Experts per layer
    active_k: int = 8             # Active experts per token per layer
    expert_invocations_per_token: int = 384   # = active_k * num_layers
    expert_size_mb: float = 2.46  # Measured: packed expert bin ~2.46 MB


@dataclass(frozen=True)
class HardwareParams:
    """Measured latency / bandwidth on the target edge SoC."""
    # --- Fully memory-resident decode (base backend, all experts in GPU VRAM) ---
    t_compute_ms: float = 49.0      # ms/token  (measured, base backend)
    tps_compute: float = 20.4       # tokens/sec ceiling  (= 1000/49)

    # --- SSD → O_DIRECT pread → pinned CPU → H2D DMA (prewarm_experts) ---
    t_ssd_ms: float = 0.834         # ms/expert  (combined SSD read + H2D)
    ssd_throughput_gbps: float = 2.96  # GB/s  (= 2.46 MB / 0.834 ms)

    # --- H2D DMA only (pinned CPU → VRAM, real weights) ---
    t_h2d_ms: float = 0.063         # ms/expert
    h2d_throughput_gbps: float = 39.0  # GB/s  (= 2.46 MB / 0.063 ms)

    # Derived: pure SSD IO latency (without H2D) = t_ssd - t_h2d
    @property
    def t_ssd_io_ms(self) -> float:
        return self.t_ssd_ms - self.t_h2d_ms


# ---------------------------------------------------------------------------
# Benchmark "ground-truth" anchor points
# These are used to annotate figures with real measured data.
# ---------------------------------------------------------------------------

@dataclass
class MeasuredPoint:
    label: str
    cache_hit_rate: float    # fraction [0, 1]
    tps: float
    description: str = ""


MEASURED_POINTS: List[MeasuredPoint] = [
    MeasuredPoint(
        label="All experts in cache",
        cache_hit_rate=1.0,
        tps=1/49*1000,
        description="All experts resident in GPU VRAM; no SSD loads"
    ),
    MeasuredPoint(
        label="Cached LRU (cache=24)",
        cache_hit_rate=0.72,  # approximate from benchmark logs ~71-72%
        tps=7.5,
        description="SSD-backed with LRU cache of 24 experts/layer"
    ),
]

# Singleton instances for import convenience
MODEL   = ModelParams()
HW      = HardwareParams()

# ---------------------------------------------------------------------------
# Cache size → natural hit rate model
# ---------------------------------------------------------------------------
# The relationship between per-layer cache size and LRU hit rate is not linear.
# We use an exponential "working set" model calibrated to the single measured
# ground-truth point: cache_size=24 → hit_rate≈0.72 on the packed benchmark.
#
# Model: hit_rate(C) = 1 - exp(-C / ws)
# where ws (working set size) is solved from the calibration point.
#
# Calibration:
#   0.72 = 1 - exp(-24 / ws)
#   ws = -24 / ln(1 - 0.72) = -24 / ln(0.28) ≈ 18.8 experts
#
# This reflects that a relatively small set of "hot" experts dominates
# activations — the model captures Zipf-like locality without per-expert
# tracking.

_HIT_RATE_CALIBRATION_CACHE = 24     # per-layer cache size (measured)
_HIT_RATE_CALIBRATION_HR    = 0.72   # measured hit rate at that cache size

# Solve for working set size once at import time
_WS = -_HIT_RATE_CALIBRATION_CACHE / math.log(1.0 - _HIT_RATE_CALIBRATION_HR)


def natural_hit_rate(cache_size_per_layer: int) -> float:
    """
    Estimate LRU cache hit rate for a given per-layer cache size.

    Uses an exponential working-set model calibrated to the measured point:
      cache=24 per layer  →  hit_rate ≈ 0.72

    Args:
        cache_size_per_layer: Number of expert weight sets kept in GPU VRAM per layer.

    Returns:
        Estimated hit rate in [0, 1].
    """
    if cache_size_per_layer <= 0:
        return 0.0
    return min(1.0 - math.exp(-cache_size_per_layer / _WS), 1.0)


# Cache sizes to sweep (per-layer expert count in VRAM)
CACHE_SIZES_TO_SWEEP = [4, 8, 12, 16, 24, 32, 48, 64]
