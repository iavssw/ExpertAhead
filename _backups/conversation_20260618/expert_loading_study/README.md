# Expert Loading Microbenchmark Study

Understand whether MoE expert-load latency is limited by **file packing** (one `.bin` per expert vs nine separate tensors) or by **SSD request parallelism** (how many concurrent `O_DIRECT pread` calls we issue).

## Background (current C++ behavior)

| Level | Behavior | Code |
|-------|----------|------|
| **Within one expert** | 6–9 parallel `pread` calls (packed: regions of one file; unpacked: separate files) | `moe_expert_load_{packed,unpacked}.inl` |
| **Across experts** | **Parallel by default** — batch `load_expert_weights()` via `ensure_experts_cached_batch()` / `load_predicted_experts()` | `unified_llm_w4a16.cpp` |
| **Across layers** | Sequential in `prewarm_experts()` | `helper.cpp:1071` |

Set `HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1` to disable parallel preads *within* a single expert (A/B test hook in `moe_expert_io.inl`).

Set `HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1` to disable parallel loads *across* experts (old sequential behavior).

### Strict SSD I/O (study sweeps default)

Study runners enable **`HETEROPREDICT_STRICT_SSD_IO=1`** by default (`--strict-ssd`, use `--no-strict-ssd` to opt out):

- **Packed**: entire expert file via `O_DIRECT` (including unaligned tail via one extra 512-byte block read — no buffered fallback).
- **Unpacked**: `O_DIRECT` into an aligned staging buffer, then `memcpy` into tensor storage (avoids stripping O_DIRECT when GPU buffers are misaligned).
- **`POSIX_FADV_DONTNEED`** after each load to evict any cache pollution.
- **`drop_caches`** before the first sweep config and between configs (needs write access to `/proc/sys/vm/drop_caches`; run with `sudo` or grant capability).

O_DIRECT bypasses the page cache for the read itself, so a RAM cgroup limit is usually unnecessary for these benchmarks. If you still see inflated hit rates from cache warmth, use `--drop-page-cache-between-prompts` on the sweep script or run under a fresh boot.

Each Qwen3-30B-A3B expert is ~2.47 MB (packed and unpacked are the same size).

## Study questions

1. **Packing**: Does `EXPK` single-file layout beat 9-file unpacked layout at equal bytes?
2. **Intra-expert parallelism**: Does parallel pread within one expert improve SSD throughput?
3. **Inter-expert parallelism**: If we issued N expert loads concurrently (not current behavior), would the SSD saturate or scale?

## Scripts

| Script | What it measures | Needs GPU |
|--------|------------------|-----------|
| `microbench_raw_io.py` | Pure `O_DIRECT` SSD reads; intra + inter expert parallelism | No |
| `microbench_cpp_prewarm.py` | Full path via `prewarm_experts()` (SSD + H2D DMA) | Yes |
| `run_study.py` | Runs both benchmarks on packed + unpacked dirs, plots | Optional |
| `run_parallel_inter_baseline.py` | LRU/RANDOM/predict sweep: parallel vs sequential inter-expert | Yes |
| `plot_study.py` | Generates PNGs from CSV results | No |

## Quick start

```bash
cd py/expert_loading_study

# SSD-only microbench (no CUDA required)
python microbench_raw_io.py \
  --bin-dir ../unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed \
  --expert-counts 1 2 4 8 \
  --num-rounds 10 \
  --out-dir results/manual_packed

# Compare packed vs unpacked + generate plots
python run_study.py --num-rounds 10
```

## End-to-end baseline sweep (after rebuild)

```bash
source utils/setup.sh && cmake --build build -j
cd py/expert_loading_study

# LRU + RANDOM only
python run_parallel_inter_baseline.py --baselines-only --cache-sizes 8 24 --quick

# LRU + RANDOM + predict prefetch
python run_parallel_inter_baseline.py --cache-sizes 8 --quick
```

## Interpreting results

**If intra-expert parallel ≫ sequential** but packed ≈ unpacked:
→ Parallel SSD requests matter; packing format is not the bottleneck.

**If inter-expert parallel gives ~N× speedup for N experts**:
→ Parallel inter-expert loading is worthwhile (now enabled by default in C++).

**If inter-expert speedup ≈ 1×** even at N=8:
→ SSD/controller is already saturated by intra-expert parallelism; focus on hit rate / prefetch overlap instead.

## Output plots

- `intra_expert_packing_vs_parallel.png` — sequential vs parallel tensor reads, per format
- `inter_expert_parallelism.png` — total time to load N experts, sequential vs parallel
- `inter_expert_speedup.png` — speedup curve vs batch size
- `cpp_prewarm_intra_parallel.png` — C++ path with `HETEROPREDICT_SEQUENTIAL_EXPERT_IO`

## Related code

- `py/utils/benchmark_expert_loading.py` — end-to-end SSD / H2D / compute breakdown
- `include/unified_llm_w4a16_common/moe_expert_io.inl` — `O_DIRECT` + sequential toggle
