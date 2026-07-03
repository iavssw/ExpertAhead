# Qwen3 Run Config Reference

The Qwen3 driver (`qwen3_30B-A3B_w4a16_model.py`) is invoked with a JSON **run config** file:

```bash
python3 py/unified_llm_w4a16/qwen3_30B-A3B_w4a16_model.py --run-config /path/to/run.json
```

Sweep scripts (`py/utils/sweep_predict_cached_cache_metrics.py`) build temporary run-config JSON files automatically. The fields below are the same whether you hand-author a config or use the sweep harness.

---

## Backends

| `backend` | C++ module | MoE expert cache | Prefetch / predictor |
|-----------|------------|------------------|----------------------|
| `"base"` | `unified_llm_w4a16_base_libtorch` | No (all experts resident) | No |
| `"cached"` | `unified_llm_w4a16_cached_libtorch` | Yes (LRU slots) | No |
| `"predict"` | `unified_llm_w4a16_predict_libtorch` | Yes | Yes (ML, gating, or oracle) |

**Qwen3 30B-A3B defaults:** 128 experts, 8 experts/token, 48 MoE layers.

---

## Quick-start examples

### Straight LRU baseline (no prefetch)

```json
{
  "backend": "cached",
  "device": "cuda",
  "max_cached_experts": 16,
  "cache_policy": "LRU",
  "generate": true,
  "max_new_tokens": 64,
  "temperature": 0.0
}
```

### Cross-layer gating prefetch (fixed top-B)

```json
{
  "backend": "predict",
  "device": "cuda",
  "predictor_type": "gating",
  "max_cached_experts": 16,
  "prefetch_experts_count": 4,
  "speculative_cache_fraction": 0.25,
  "cache_policy": "LRU",
  "generate": true,
  "max_new_tokens": 64,
  "temperature": 0.0
}
```

`speculative_cache_fraction: 0.25` with cache 16 → 12 main LRU slots + 4 speculative-only slots (paper-faithful isolation). Set to `0.0` for legacy unified-pool behavior.

### Gating with score-based (percentile) prefetch

```json
{
  "backend": "predict",
  "predictor_type": "gating",
  "gating_score_percentile": 0.8,
  "max_cached_experts": 16,
  "speculative_cache_fraction": 0.5,
  "generate": true,
  "max_new_tokens": 64
}
```

When `gating_score_percentile > 0`, prefetch count adapts to the router score distribution (all experts with softmax ≥ 80th percentile). `prefetch_experts_count` is ignored for selection in this mode; the speculative pool size caps how many loads are issued.

### ML TorchScript predictor

```json
{
  "backend": "predict",
  "predictor_type": "torchscript",
  "predictor_model": "/path/to/eh1_h32_f4",
  "predictor_lookahead": 4,
  "max_cached_experts": 16,
  "prefetch_experts_count": 4,
  "generate": true,
  "max_new_tokens": 64
}
```

---

## Parameter reference

### Model & weights

| Key | Type | Default | Backends | Description |
|-----|------|---------|----------|-------------|
| `model_path` | string | `"QuixiAI/Qwen3-30B-A3B-AWQ"` | all | HuggingFace model ID or local path for weight loading. |
| `tokenizer_path` | string | `model_path` | all | Tokenizer path (defaults to `model_path`). |
| `expert_weights_dir` | string | `py/unified_llm_w4a16/model_weights/Qwen3-30B-A3B-AWQ_packed` | all | Directory of packed/unpacked MoE expert `.bin` files. Passed to C++ `load_quantized_weights_from_bins`. |
| `config_path` | string | `configs/configs_strixH_qwen3_30B_A3B.json5` | all | NPU/runtime JSON5 config (heterogeneity, debug, dummy weights). For `predict`, may be rewritten when `predictor_device` is set. |
| `device` | string | `"cuda"` | all | `"cuda"` or `"cpu"`. Overridden to CPU if `config_path` sets `"heterogeneity": "cpu"`. |

**`config_path` fields** (not run-config keys, but affect behavior):

| Field | Description |
|-------|-------------|
| `heterogeneity` | `"gpu"` or `"cpu"` |
| `dummy_weights` | Skip real weight load; use random init |
| `debug_verbosity` | C++ debug print level |
| `usePreSavedWeights` | Use pre-baked bin layout |
| `warmup` | Kernel warmup flag |

---

### Expert cache (cached & predict)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_cached_experts` | int | `8` | Expert slots per MoE layer. Qwen sweep scripts use this name; Mixtral scripts use `expert_cache` instead. |
| `cache_policy` | string | `"LRU"` | Eviction policy: `LRU`, `MRU`, `LFU`, `MFU`, `CLOCK`, `RANDOM`, `LFRU`, `PREFILL`. Applied after model init via `set_cache_policy`. |
| `per_layer_cache_sizes` | int[] | `[]` | Optional per-layer cache override (length = num layers). Empty = use `max_cached_experts` for every layer. |
| `prefill_top_n` | int | `0` | Lock top-N prefill experts into cache for the whole decode phase. `0` = disabled. |

On init, `cached` and `predict` backends **prewarm** `max_cached_experts` (or max of `per_layer_cache_sizes`) experts per layer into the main cache slots.

---

### Cache-conditional routing (cached & predict)

These bias the router toward experts already resident in cache. Applied after init via `_apply_routing_and_cache_cli`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lambda_val` | float | `0.0` | Cache-conditional logit bias strength. `0` = standard unbiased routing. Typical experiments: `0.5`–`1.0`. |
| `forced_top_n` | int | `0` | Force the top-N unbiased router experts into the cache mask so they are always preferred when `lambda > 0`. `0` = off. |
| `forced_top_p` | float | `-1.0` | Legacy: force experts whose cumulative softmax mass reaches `p` into the cache mask. `-1` = off. Prefer `mass_threshold_substitution_p`. |
| `mass_threshold_substitution_p` | float | `-1.0` | Probability-mass prefix threshold OR'd into the cache mask (works with `lambda_val > 0`). `-1` = off. |

---

### Predict backend — general

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `predictor_type` | string | `"torchscript"` | `"torchscript"` (ML), `"gating"` (cross-layer router heuristic), or oracle via trace (see below). |
| `prefetch_experts_count` | int | `1` | Prefetch budget **B** (top-B experts per prediction tick). Used by ML predictor and fixed top-B gating. |
| `predict_layers` | int[] | `[]` | Restrict predictor to specific layer indices. Empty = all layers. |
| `per_layer_prefetch_counts` | int[] | `[]` | Per-layer prefetch budget override. Usually set by CSV calibration. |
| `prefetch_threshold` | float | `0.0` | ML predictor only: minimum predicted probability to include an expert. `0` = no threshold. |
| `predictor_device` | string | `"gpu"` | `"gpu"`, `"cpu"`, or `"auto"`. Where TorchScript models run. Written into a temp copy of `config_path` when not `auto`. |
| `predictor_model` | string | `""` | Directory of per-layer TorchScript predictor models (e.g. `eh1_h32_f4`). Not used for `predictor_type: gating`. |
| `predictor_lookahead` | int | `1` | ML predictor token horizon (should match `fN` in model dir name). Set automatically from path when possible. |
| `expert_reuse_csv` | string | `null` | If set with `predictor_model`, calibrates `per_layer_cache_sizes` and `per_layer_prefetch_counts` from reuse statistics. |
| `suppress_predictor_stats` | bool | `false` | Disable predictor accuracy / prefetch counters (sweep uses this as `disable_measurement`). |

---

### Predict backend — gating heuristic

Cross-layer prediction: after layer L's post-attention norm, layer L+1's router runs on that hidden state and prefetches experts asynchronously.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gating_lookahead` | int | `1` | Passed to C++ but **currently unused** (hardwired L→L+1 only). |
| `gating_score_percentile` | float | `0.0` | `0` = fixed top-B (`prefetch_experts_count`). `> 0` (e.g. `0.8`) = prefetch all experts with softmax score ≥ p-th percentile (adaptive count). |
| `speculative_cache_fraction` | float | `0.0` | Fraction of `max_cached_experts` reserved as **speculative-only** slots. `0` = unified pool (legacy). `B/cache_size` matches paper-style separate buffers (e.g. `0.25` for B=4, cache=16). |
| `prefetch_non_evicting` | bool | `false` | When `speculative_cache_fraction == 0`: prefetch only fills empty slots. When fraction > 0: speculative pool is already isolated; this mainly affects legacy mode. |

**Timing (current implementation):** prefetch triggers immediately when `post_normed` is ready (before layer L MoE forward). Prediction + SSD loads run on a background thread overlapping L MoE compute.

---

### Predict backend — oracle baseline

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `oracle_trace_path` | string | `""` | Path to captured oracle trace JSON. Non-empty enables `OracleTracePredictor`. |
| `oracle_lookahead` | int | `0` | Token lookahead for oracle replay. Defaults to `predictor_lookahead` if trace set and this is `0`. |
| `oracle_full_union` | bool | `false` | Prefetch union of all experts in the lookahead window instead of per-step top-B. |

For sweep oracle rows, `oracle_trace_path` is injected automatically; you normally set this only for manual replay.

---

### Generation & evaluation modes

Exactly one primary mode is selected by flags below (checked in `main()` order).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `wikitext103_perplexity` | bool | `false` | Run WikiText-103 sliding-window perplexity (exits before generation). |
| `wikitext103_split` | string | `"test"` | HF dataset split. |
| `wikitext103_max_length` | int | `2048` | Window length. |
| `wikitext103_stride` | int | `2048` | Window stride. |
| `wikitext103_max_windows` | int | `0` | Cap windows (`0` = no cap). |
| `prompt_test` | int | `null` | If set, run fixed-token prompt test mode (token count target). |
| `sweep_prompts_file` | string | `null` | JSON file with a list of prompt strings; runs batch generation + prints cache/predictor stats. Used by sweep harness. |
| `perplexity` | bool | `false` | Single-prompt teacher-forced perplexity on `text`. |
| `generate` | bool | `true` | Autoregressive generation (default path). |
| `text` | string | (built-in default) | Prompt text when not using oracle token IDs or sweep file. |
| `max_new_tokens` | int | `16` | Tokens to generate per prompt. |
| `temperature` | float | `0.0` | Sampling temperature (`0` ≈ greedy). |
| `top_p` | float | `0.9` | Nucleus sampling. |
| `top_k` | int | `50` | Top-k sampling. |
| `generation_perplexity` | bool | `false` | During sweep generation, also report generation NLL / perplexity. |

---

### Oracle trace capture / replay (advanced)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `capture_oracle_trace` | string | `""` | Output path; run once to record routed experts. |
| `oracle_trace` | string | `""` | Input trace for exact prompt token replay during generation. |

---

## Backend constructor mapping (predict)

When `backend == "predict"`, run-config keys map to the C++ `UnifiedLLMW4A16` constructor in this order (after architecture dims):

1. `device`
2. `max_cached_experts`
3. `predictor_model`
4. `config_path` (possibly temp file)
5. `prefetch_experts_count`
6. `predict_layers`
7. `per_layer_cache_sizes`
8. `per_layer_prefetch_counts` (from CSV calibration or `[]`)
9. `oracle_trace_path`
10. `oracle_lookahead`
11. `oracle_full_union`
12. `prefetch_threshold`
13. `predictor_type`
14. `gating_lookahead`
15. `prefetch_non_evicting`
16. `gating_score_percentile`
17. `speculative_cache_fraction`

Post-construct Python calls:

- `set_predictor_lookahead(...)` — gating uses `gating_lookahead`; ML/oracle uses `predictor_lookahead` or `oracle_lookahead`.
- `set_prefetch_non_evicting(true)` if `prefetch_non_evicting`.
- `_apply_routing_and_cache_cli(...)` for lambda, forced routing, cache policy.

---

## Sweep harness notes

`py/utils/sweep_predict_cached_cache_metrics.py` writes run-config JSON via `build_model_cmd()`. Common sweep → run-config mappings:

| Sweep CLI | Run-config key |
|-----------|----------------|
| `--cache-sizes` | `max_cached_experts` |
| `--prefetch-budgets` | `prefetch_experts_count` |
| `--lambdas` | `lambda_val` |
| `--predictor-base-dir` + lookahead | `predictor_model`, `predictor_lookahead` |
| `--sweep-question lru_vs_gating` | `predictor_type: gating`, `speculative_cache_fraction: B/cache_size` |
| `--gating-score-percentiles` | `gating_score_percentile` |
| `--prefetch-non-evicting` | `prefetch_non_evicting` |

Example sweep:

```bash
bash py/utils/run_lru_vs_gating_sweep.sh
# or
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --sweep-question lru_vs_gating \
  --model qwen \
  --cache-sizes 16 \
  --custom-explicit-prefetch-budgets \
  --prefetch-budgets 2 4 6 8 \
  ...
```

---

## Metrics printed at end of run

Relevant for comparing prefetch strategies (parsed by sweep CSV):

| Stat | Meaning |
|------|---------|
| `Cache Hits / Misses / HitRate` | Router served from resident experts vs stall loads |
| `Stall Loads` | SSD loads on the main routing path |
| `Prefetch Loads` | SSD loads issued on speculative path |
| `GatingPredict Recall / Precision` | Cross-layer prediction accuracy vs actual top-8 |
| `GatingPrefetch Delivery` | Routed experts that were prefetched in time |
| `prefetch_dropped_no_victim` | Prefetch skipped (no speculative slot available) |

---

## Build requirement

After changing C++ predict backend code:

```bash
cd src/microbenchmarks && make unified_llm_w4a16_predict_libtorch
```

The shared library is emitted to `py/unified_llm_w4a16/unified_llm_w4a16_predict_libtorch.so`.
