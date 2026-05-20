# Steady-State Decode Model — Study Guide

This README summarizes the analytical model for the thesis section **“Modeling Steady-State Decode.”** It matches how the **predict** and **cached** backends behave in this repo (`unified_llm_w4a16_predict`, sweeps in `sweep_predict_cached_cache_metrics.py` / `finals_experiment_runner.py`).

The Python code in this folder (`model.py`, `params.py`) is a **plotting implementation** of an older variant. When the prose below and the code disagree, **trust this README for the thesis**; update `model.py` after the LaTeX is locked.

---

## 1. Hardware story (read this first)

**Platform:** Unified **system memory** (CPU/GPU shared) + **SSD** for weights that do not fit in the working set.

| Configuration | What is resident | What each decode token pays |
|---------------|------------------|-----------------------------|
| **Base** | All experts in unified memory | **\(T_{\text{ceil}}\)** ≈ 49 ms/token (~20 TPS) — **ceiling** |
| **Cached / predict** | Only **\(C\)** experts/layer in the GPU cache; rest on SSD | Hits + misses + (optional) prefetch |

**Important:** \(T_{\text{ceil}}\) is **not** “pure GEMM.” It is end-to-end MoE decode when every expert is already resident in unified memory, including **unified-memory → VRAM transfer** for the active experts plus compute. That is the **true hardware ceiling** on this machine—not an ideal GPU-only lower bound.

**Expert cache hit** (cached/predict): weights already in the on-GPU cache slot → pay **\(T_{\text{mem}}\)** per invocation (unified-memory → VRAM style transfer), not SSD.

**Expert cache miss:** load from SSD → **\(T_{\text{ssd}}\)** per invocation (O_DIRECT read + H2D in microbench).

---

## 2. Measured constants (Qwen3-30B-A3B, your SoC)

From `benchmark_expert_loading.py` / `params.py` (re-measure after hardware changes):

| Quantity | Symbol | Typical value | Notes |
|----------|--------|---------------|--------|
| Active experts / token / layer | \(K\) | 8 | Confirm against your model config |
| MoE layers | \(L\) | 48 | |
| Invocations per token | \(E_{\text{req}} = K \cdot L\) | **384** | |
| Ceiling latency / token | \(T_{\text{ceil}}\) | **~49 ms** | Base backend; includes transfer + compute |
| SSD miss cost / invocation | \(T_{\text{ssd}}\) | **~0.834 ms** | Stall load path |
| Cache hit transfer / invocation | \(T_{\text{mem}}\) | **~0.063 ms** | No SSD |
| Incremental miss penalty | \(T_{\text{ssd}} - T_{\text{mem}}\) | **~0.771 ms** | Use in incremental formulas |

**Sanity check (all misses, no prefetch):**  
\(49 + 384 \times 0.771 \approx 345\) ms/token ≈ full SSD miss stack (~320 ms from \(384 \times 0.834\)).

**Ratio:** \(T_{\text{ssd}} / T_{\text{mem}} \approx 13\times\) — one SSD miss is far more expensive than serving an expert already resident for transfer.

---

## 3. Symbol table (do not mix these up)

### Per-token expert counts

| Symbol | Definition |
|--------|------------|
| \(E_{\text{req}}\) | Expert **invocations** per decode token (\(K \cdot L\)) |
| \(E_{\text{miss}}\) | Invocations that trigger an SSD load |
| \(E_{\text{hit}}\) | Invocations served from cache without SSD |
| \(h\) | Hit rate: \(E_{\text{hit}} / E_{\text{req}}\) |

\[
E_{\text{miss}} = E_{\text{req}}(1-h), \qquad E_{\text{hit}} = E_{\text{req}}\,h.
\]

### Latencies

| Symbol | Meaning |
|--------|---------|
| \(T_{\text{ceil}}\) | Base decode time per token (ceiling) |
| \(T_{\text{ssd}}\) | Per-invocation cost on **miss** |
| \(T_{\text{mem}}\) | Per-invocation cost on **hit** |

### System knobs (match sweep CSV columns)

| Symbol | Code / sweep | Meaning |
|--------|----------------|---------|
| \(N\) | `lookahead`, `fN` in predictor path | **Horizon** and **predictor invocation stride** (fires every \(N\) tokens) |
| **\(B\)** | `prefetch_budget`, `--prefetch-experts-count` | Max experts to **prefetch per layer** per predictor call (\(B \le C\)) |
| **\(J\)** | `forced_top_n`, `routing_bias_top_n` | **Not prefetch.** Top-\(J\) unbiased experts forced into cache-conditional routing mask |
| \(C\) | `cache_size`, `--max-cached-experts` | Experts per layer in GPU cache |
| \(\rho\) | recall-like stats | Fraction of **needed** future misses the predictor targets |
| \(\pi\) | precision-like stats | Fraction of **prefetched** experts that are actually needed |
| \(\lambda\) | `lambda_val` | Cache-conditional routing bias strength |

---

## 4. Baseline: synchronous decode (no prefetch)

Every miss blocks on SSD. Anchored at the base ceiling so \(h=1\) recovers \(T_{\text{ceil}}\):

\[
\boxed{
T_{\text{token}}^{\text{sync}}
\approx T_{\text{ceil}} + E_{\text{miss}}\,\bigl(T_{\text{ssd}} - T_{\text{mem}}\bigr)
}
\]

Equivalent form (full decomposition):

\[
T_{\text{token}}^{\text{sync}}
\approx T_{\text{ceil}} + E_{\text{hit}}\,T_{\text{mem}} + E_{\text{miss}}\,T_{\text{ssd}}
\quad\text{(if }T_{\text{ceil}}\text{ calibrated at }h=1\text{ only—prefer incremental form above).}
\]

**Throughput:** \(\text{TPS} = 1000 / T_{\text{token}}\).

---

## 5. Cache-conditional routing (uses \(J\), not \(B\))

Routing bias tries to select experts already in the cache. In experiments, the top **\(J\)** unbiased router experts are always kept in the bias mask; remaining slots use λ-biased top-\(K\).

Effective hit rate:

\[
h_{\text{eff}} = \min\bigl(1,\; h_{\text{base}} + \Delta h_{\text{route}}(J,\lambda,\ldots)\bigr).
\]

Substitute \(h_{\text{eff}}\) into \(E_{\text{miss}} = E_{\text{req}}(1 - h_{\text{eff}})\).

\(\Delta h_{\text{route}}\) should come from measured sweeps (λ, \(J\), PM thresholds)—not assumed twice elsewhere.

---

## 6. Predictive prefetch — what the implementation actually does

Read this before the equations.

1. **Predictor runs every \(N\) tokens**, not every token (`lookahead_stride_` in C++ = \(N\)).
2. Model path `eh1_h32_f{N}` was trained to predict experts **\(N\) steps ahead**; stats use **horizon = \(N\)**.
3. On each invocation, **each MoE layer** prefetches at most **\(B\)** experts from the predictor ranking (`prefetch_experts_count`).
4. One global predictor “tick” can start up to **\(L \cdot B\)** SSD loads (all layers, async, sharing SSD/PCIe).
5. If the previous prefetch is still in flight, the next predictor call may be **skipped** (no main-thread stall, but no new prefetch).
6. **Inputs:** embedding, prefill expert distribution, previous-token routing, optional cross-layer features.

**Prefetch is not continuous** between ticks—only reactive LRU/stall loads fill the gap.

---

## 7. Prefetch overlap over one interval of \(N\) tokens

Group decode into intervals of length \(N\) (one predictor burst at the start of each interval).

### Bandwidth capacity (how many loads SSD can hide behind compute)

\[
B_{\text{overlap}} = N \cdot T_{\text{ceil}}, \qquad
M_{\text{cap}} = \frac{B_{\text{overlap}}}{T_{\text{ssd}}} = \frac{N \cdot T_{\text{ceil}}}{T_{\text{ssd}}}.
\]

Example: \(N=2\), \(T_{\text{ceil}}=49\) ms, \(T_{\text{ssd}}=0.834\) ms → \(M_{\text{cap}} \approx 117\) loads (upper bound from time).

### Issue capacity (how many loads the engine will *start*)

\[
M_{\text{issue}} = L \cdot B.
\]

Example: \(L=48\), \(B=18\) → up to 864 loads **issued** per predictor tick (bandwidth usually limits this first).

### Recall \(\rho\) and precision \(\pi\)

- **\(\rho\):** fraction of **needed** misses in the horizon that the predictor identifies. Low \(\rho\) → false negatives → **blocking** stalls.
- **\(\pi\):** fraction of **prefetched** experts that are actually used. Low \(\pi\) → wasted SSD bandwidth; caps useful prefetch at **\(\pi \cdot M_{\text{issue}}\)**.

Let \(U\) = distinct SSD misses the predictor must cover in the interval (\(U \le N \cdot E_{\text{miss}}\) per token if averaged; horizon eval targets the token at \(t+N\)).

\[
M_{\text{hidden}} = \min\bigl(\rho\,U,\; \pi\,L B,\; M_{\text{cap}}\bigr).
\]

### Per-token averages (steady state)

\[
\bar{E}_{\text{hidden}} = \frac{M_{\text{hidden}}}{N}, \qquad
\bar{E}_{\text{block}} = E_{\text{miss}} - \bar{E}_{\text{hidden}}.
\]

---

## 8. Combined model (prefetch + cache-conditional)

Use \(h_{\text{eff}}\) for routing (Section 5), then prefetch (Section 7):

\[
E_{\text{miss}} = E_{\text{req}}\bigl(1 - h_{\text{eff}}\bigr),
\]

\[
\boxed{
T_{\text{token}}
\approx T_{\text{ceil}} + \bar{E}_{\text{block}}\,\bigl(T_{\text{ssd}} - T_{\text{mem}}\bigr)
}
\]

with \(\bar{E}_{\text{block}}\) from Section 7.

**Do not** also add a separate “\(\Delta h_{\text{pred}}\)” **and** full \(\rho,\pi,N,B\) prefetch—prefetch is \(\rho,\pi,N,B\); routing is \(\Delta h_{\text{route}}(J,\lambda)\).

**Do not** use a free-floating \((1-\alpha)\) on all miss time unless you define \(\alpha = \bar{E}_{\text{hidden}}/E_{\text{miss}}\) **after** computing \(M_{\text{hidden}}\) (derived metric, not a separate knob).

---

## 9. What to plot / measure (links to repo)

| Thesis claim | Where it comes from |
|--------------|---------------------|
| Ceiling ~20 TPS | Base run, `T_ceil` in microbench |
| SSD vs hit gap | `benchmark_expert_loading.py --mode all` |
| LRU TPS vs cache size | `cached`, sweep hit rate |
| Prefetch vs recall | `predict`, `pred_hit_rate_*`, fig 2 |
| Prefetch vs precision | `pred_requested_rate_*`, fig 3 |
| Stall vs prefetch loads | `stall_loads`, `prefetch_loads`, `AvgLoadTime` in logs |
| Effect of \(N\) | Sweep `lookahead` / `fN` paths |
| Effect of \(B\) | `prefetch_budget` in CSV |
| Effect of \(J\) | `forced_top_n` on **cache-cond** rows only |
| Effect of \(C\) | `cache_size` |

**Run micro + macro benchmarks:** `run_benchmarks.sh` at repo root (after rebuilding libtorch in `rocmPytorch`).

**Generate model curves (after code updated):** `python main.py` in this directory.

---

## 10. Common mistakes checklist

- [ ] Calling \(T_{\text{ceil}}\) “compute only” — it includes SM→VRAM + compute on base.
- [ ] Using **\(J\)** for prefetch budget — **\(J\)** is forced top-\(J\) routing; use **\(B\)** for prefetch.
- [ ] Assuming predictor runs **every token** — it runs every **\(N\)** tokens.
- [ ] Ignoring **\(L \cdot B\)** issue cap — bandwidth is not the only limit.
- [ ] Double-counting prefetch (as \(\rho\) **and** \(\Delta h_{\text{pred}}\)) **and** routing boost.
- [ ] Adding \(E_{\text{hit}} T_{\text{mem}}\) on top of \(T_{\text{ceil}}\) without careful calibration — use incremental \((T_{\text{ssd}} - T_{\text{mem}})\) form.
- [ ] Confusing OS page cache with **expert cache hit** — hits are GPU cache slots / resident path, not “Linux cached the file.”

---

## 11. One-page cheat sheet

```
E_req = K * L = 384
E_miss = E_req * (1 - h)
E_hit  = E_req * h

T_sync ≈ T_ceil + E_miss * (T_ssd - T_mem)

Over each N tokens:
  M_cap   = N * T_ceil / T_ssd
  M_issue = L * B
  M_hidden = min(rho * U, pi * L * B, M_cap)
  E_hidden_bar = M_hidden / N
  E_block_bar  = E_miss - E_hidden_bar

T_token ≈ T_ceil + E_block_bar * (T_ssd - T_mem)

h_eff = min(1, h_base + Delta_h_route(J, lambda, ...))  # routing only
```

**TPS** = \(1000 / T_{\text{token}}\).

---

## 12. Files in this directory

| File | Role |
|------|------|
| `params.py` | Measured \(T_{\text{ceil}}\), \(T_{\text{ssd}}\), \(T_{\text{mem}}\), \(K\), \(L\) |
| `assumptions.py` | Estimates: reuse, overlap efficiency, \(\Delta h_{\text{route}}\), fixed \(\rho,\pi\) |
| `model.py` | Sweeps for figures (to be aligned with this README) |
| `main.py` | Generates `paper_output/figures` and CSVs |
| `README.md` | This document |

---

## 13. Backend fidelity (verified against C++)

This section maps the README to `unified_llm_w4a16_{base,cached,predict}`. The **steady-state formulas** (Sections 4–8) are an **analytical overlay**—they are not implemented as a runtime model inside the backends.

### `unified_llm_w4a16_base`

| README claim | Code behavior |
|--------------|---------------|
| \(T_{\text{ceil}}\): all experts resident, no SSD on decode | `forward_generation` indexes `gate_up_experts[e]` / `down_experts[e]` directly; no `ensure_expert_cached`, no expert bins at decode time (`unified_llm_w4a16_base/unified_llm_w4a16.cpp`). |
| MoE timing | `total_moe_compute_time_ms_` / `moe_expert_invocations_`; `print_cache_stats` → `print_moe_compute_only`. |
| Weights | Loaded once via safetensors or `load_quantized_weights_from_bins` into per-expert modules (`base/helper.cpp`). |

**Wording note:** “Unified memory ceiling” is the **hardware interpretation** of base runs. In code, weights live in **GPU-resident** expert modules after load, not in a separate OS page-cache path.

### `unified_llm_w4a16_cached`

| README claim | Code behavior |
|--------------|---------------|
| \(C\) experts/layer, LRU (and other policies) | `max_cached_experts_`, `pick_victim()` / `slot_meta_` (`cached/unified_llm_w4a16.cpp`). |
| Sync blocking miss | `ensure_expert_cached` → `load_expert_weights` on main thread; `hipDeviceSynchronize` after load. |
| No prefetch | Comment `// Prefetch removed`; no speculative path. |
| Cache-conditional \(J\), \(\lambda\) | `forced_top_n_`, `forced_top_p_`, `lambda_` bias mask + biased top-\(k\) (same structure as predict). |
| `FORCE_EXPERT_MISS` | **Cached only** — env var forces every access to miss path. **Predict does not have this.** |
| Miss bandwidth stats | `print_cache_stats` → hits/misses + `print_moe_miss_bandwidth` on misses only. |
| SSD load path | Shared `moe_expert_load_{packed,unpacked}.inl` + O_DIRECT `pread` (`cached/helper.cpp`). |

### `unified_llm_w4a16_predict`

| README claim | Code behavior |
|--------------|---------------|
| Expert cache + SSD on miss | Same slot LRU as cached; `ensure_expert_cached` on routing path. |
| **\(B\)** prefetch budget | `prefetch_experts_count_`, clamped to `max_cached_experts_`; `std::min(pred_size, prefetch_experts_count_)` before `load_predicted_experts`. |
| **\(J\)** routing only | `forced_top_n_` OR’d into `cache_mask`; not used to cap prefetch. |
| Predictor every **\(N\)** tokens | `decode_token_count_++ % lookahead_stride_ != 0` → return (`trigger_speculative_loading`). |
| Skip tick if prior prefetch busy | `speculative_load_future_.wait_for(0)` not ready → skip (no main-thread stall). |
| Up to **\(L \cdot B\)** loads per stride tick | One `speculative_load_future_` **per MoE layer**; each layer loads up to \(B\) experts when stride fires. Layers without a predictor model do not prefetch (`predict_layers`, empty `layer_predictor_path`). |
| Predictor inputs | `embedding`, prefill distribution, `prev_token_routing_mh_`, optional cross-layer `prev_layers_feat` → `predict_sync` (`predict/unified_llm_w4a16.cpp` ~697–761). |
| Horizon \(t+N\) eval | `pending_predictions_.push_back({source_decode_step + lookahead_stride_, top_experts})`; consumed at `current_decode_step` in `forward_generation`. |
| \(\rho\) / \(\pi\) in logs | `pred_hits_*` / `pred_total_*` (recall-style) and `pred_requested_*` (precision-style) in `print_cache_stats` (`predict/helper.cpp`). |
| Stall vs prefetch loads | `stall_loads_++` in `ensure_expert_cached` miss path; `prefetch_loads_++` in `load_predicted_experts`. |
| Wait on in-flight prefetch | Main thread `expert_slots_cv_.wait` until `expert_slot_ready_[s]` if expert reserved but still loading. |
| Prefill warmup | `warm_predictor_caches_after_prefill` → `run_predictor_prefill_warmup` (excluded from generation TPS in log message). |

### Gaps / mismatches to remember

1. **\(N\) is not auto-set from `fN` in the model path.** C++ default `lookahead_stride_ = 1` (predict every token). Python calls `set_predictor_lookahead` only when `predictor_lookahead > 1` (`qwen3_30B-A3B_w4a16_model.py`). For \(N>1\), pass `--predictor-lookahead N` (typically match `fN` in the predictor directory name).

2. **Prefetch is sequential per expert** within a layer (`for` over `predicted_expert_ids`), but **tensor reads inside one expert** use parallel `std::async` preads (`moe_expert_load_packed.inl`). The \(M_{\text{issue}} = LB\) count is still the right **expert-level** issue bound.

3. **Bitmask vs ready:** `expert_cache_bitmask_` is updated when a slot is *assigned*, before SSD I/O finishes—cache-conditional routing may bias toward experts that still block in `ensure_expert_cached`.

4. **`experts_loaded_this_step_`** increments on **any** `load_expert_weights` (stall or prefetch, any thread)—use `stall_loads_` / `prefetch_loads_` for the thesis split.

5. **Sections 4–8** (\(M_{\text{cap}}\), \(\bar{E}_{\text{hidden}}\), etc.) are **not** computed in C++; fit \(\rho,\pi,h\) from sweeps and apply the formulas offline (`moe_paper/model.py` when updated).

---

## 14. Next steps (when ready)

1. Lock thesis LaTeX to Sections 4–8 above.  
2. Rename `t_compute_ms` → `t_ceil_ms` in `params.py` and comments.  
3. Update `model.py` with \(N\)-stride averaging, \(M_{\text{issue}} = LB\), and incremental stall formula.  
4. Regenerate figures; compare anchors to `run_benchmarks.sh` logs.  
5. Optionally auto-call `set_predictor_lookahead(fN)` when `fN` is parsed from the predictor path (today manual CLI).
