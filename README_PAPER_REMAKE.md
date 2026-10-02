# Remake the submitted ExpertAhead paper

This file is the work order for replacing the WikiText-only numbers in `ASPDAC_27_ExpertAhead.zip` with the mixed-corpus predictor and the five-domain eval set. Follow it in order. Do not invent a new experimental design.

The submitted paper is the source of which artifacts must exist. Its measured claims are all WikiText-103, greedy decode, Qwen3-30B-A3B-AWQ, on the Ryzen AI 9 HX 370. Those claims are:

- Offline recall and precision of the predictor (two ablation tables, budget \(B=16\)).
- A \(C=48\) sweep of stride and budget, plotted as throughput against window recall and precision. The submitted peak is stride \(S=5\), \(B=24\).
- Perplexity change from cache-conditional routing while sweeping preserved experts \(J\) and cache size \(C\).
- One eviction-policy sentence at \(C=64\) (LFRU vs LRU vs most-recently-used).
- The end-to-end table and bar chart at \(C \in \{16,32,48,64\}\): SSD streaming, random eviction, oracle, cross-layer, cache-conditional, ExpertAhead, ExpertAhead-CC. Protocol: 10 prompts, 150 generated tokens, temperature 0.

Everything else in the paper (storage diagram, timelines, predictor diagram, hardware paragraph, latency equations) stays. Do not rerun it.

## Locked choices

| Item | Value |
| --- | --- |
| Predictor | `trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed` |
| Usable strides | 1, 4, 8, 16 (`transformer_eh4_h64_f1`, `f4`, `f8`, `f16`; 48 layers each) |
| Architecture | transformer, history 4, hidden 64, 2 layers, 4 heads, embedding + Markov + prefill |
| Model | Qwen3-30B-A3B-AWQ, W4A16, greedy (`temperature 0`) |
| Domains | `wikitext`, `fineweb`, `orca`, `gsm8k`, `cnn_dailymail` |
| Paper protocol | 10 held-out prompts per domain, 150 new tokens, `prompt-max-chars 4096` |
| Cache sizes | 16, 32, 48, 64 experts per layer |
| ExpertAhead-CC / cache-conditional | \(\lambda=1\), \(J=5\), same as the submitted table |
| Output root | `py/utils/final_results_runs/paper_remake/` |

`transformer_eh4_h64_f2` has only `layer_0` and `layer_24`. Do not use it.

The submitted operating points \(S \in \{2,5,6\}\) do not exist for this predictor. `predictor_path_for_lookahead` only resolves a directory whose name ends in `_f{S}`. A sweep with lookahead 2, 5, or 6 will not load these weights. Select a new \((S, B)\) per cache size from \(\{1,4,8,16\}\).

The old predictor `trainingData/qwen3_30b/transformer_final_pfill_markov_emb` and the old traces `trainingData/wikitext_test_traces` (10 WikiText traces, 512 prompt tokens, 150 generated tokens, different prompts) must not appear in any reported number.

Training shards for the mixed predictor were read from `/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_30b_mixed_sharded` (800 train / 200 val files per layer). That path is not on this machine. `training_metrics.json` under each layer has mixed-corpus validation `union_recall@K` and no precision and no per-domain split. It cannot fill the paper tables.

## Do not touch the running job

`sh_scripts/run_expanded_prompt_space.sh` is in tmux session 0. It is a pilot: 5 prompts per domain, 100 new tokens, cache 64, random baseline then ExpertAhead and ExpertAhead-CC. Its prompts are `py/utils/final_results_runs/expanded_prompt_space/by_domain/*.json`.

Leave that process alone. Do not start a second model load while it holds the GPU. Do not set `REFRESH_PROMPTS=1` on that directory. Do not copy its tokens-per-second into the paper table. The pilot and the paper protocol differ in prompt count and generation length.

`source utils/setup.sh` from the repo root before every sweep command below.

## 1. Freeze the eval manifest

Write a new manifest. Do not overwrite the pilot JSON.

```bash
python3 py/utils/cache_eval_prompts.py \
  --out py/utils/final_results_runs/paper_remake/examples_10each.json \
  --split-dir py/utils/final_results_runs/paper_remake/by_domain \
  --num-prompts 10 \
  --prompt-max-chars 4096 \
  --datasets wikitext fineweb orca gsm8k cnn_dailymail
```

Confirm each `by_domain/<domain>.json` has 10 objects with keys `domain`, `idx`, `text`, `gold`, `metric`, and `idx` equal to `0..9` inside that file.

Held-out split, from `py/utils/prompt_datasets.py`:

- WikiText-103, GSM8K, CNN/DailyMail: official `test` split.
- FineWeb-Edu (`sample-10BT`) and OpenOrca: hash holdout of the train stream (`eval_holdout_fraction=0.1`). Training skipped those examples.

Record a checksum of the five JSON files in `paper_remake/MANIFEST.sha256`. Every later run uses these files and `--num-prompts 0` so the sweep does not truncate them.

## 2. Capture oracle traces

Required before any oracle row. The pilot does not write traces. Capture only after the GPU is free.

One directory per domain. The sweep looks up batch index `i` as `oracle_trace_qwen3_30b_{i:05d}.txt` (also accepts `trace_{i:05d}.txt`) inside `--oracle-trace-dir`. Index `i` is the position in that domain's JSON, starting at 0.

For each domain, for each of the 10 prompts, run `py/utils/capture_oracle_trace.py` with:

- `--text` set to that example's `text` field, unchanged
- `--output trainingData/eval_traces_paper/<domain>/oracle_trace_qwen3_30b_{idx:05d}.txt`
- `--backend predict`
- `--max-cached-experts 128`
- `--max-new-tokens 150`
- `--temperature 0`

The script's own requirement: full cache and no cache-conditional bias, so the file is the router's expert sequence. Load the model once and loop the prompts if you wrap the script; do not boot the 30B model 50 times if you can avoid it.

Acceptance for every file, via `py/utils/oracle_trace.py`:

- `prompt_token_ids` is non-empty
- `len(generated_token_ids) == 150`
- `len(expert_trace) == 150`
- each step has 48 layers

Store traces only under `trainingData/eval_traces_paper/<domain>/`. A single shared directory is wrong: each domain sweep starts again at index 0.

## 3. Select stride and budget

This replaces the submitted \(C=48\) figure (`figures/tps_vs_recall_precision_C48.png`) and the \(S/B\) column of the end-to-end table. Run it on the frozen manifest, not on the pilot.

Per domain, one sweep at the four cache sizes is enough to pick the operating point. Use all 10 prompts and 150 tokens if the machine can take it; otherwise select on 5 prompts and 80 tokens (the submitted selection size), then freeze the chosen \((S,B)\) and rerun only those points at 10×150 in step 5. Do not mix the two lengths in one table.

```bash
PRED=trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed
# repeat per domain; C can be looped 16 32 48 64
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --examples-json py/utils/final_results_runs/paper_remake/by_domain/${DOMAIN}.json \
  --domain "$DOMAIN" \
  --num-prompts 0 \
  --prompt-max-chars 4096 \
  --max-new-tokens 150 \
  --temperature 0.0 \
  --cold-per-prompt \
  --cache-sizes 16 32 48 64 \
  --lookaheads 1 4 8 16 \
  --budget-fractions 0.25 0.5 0.75 \
  --non-baseline-cache-policy LFRU \
  --constraint-expert-reuse-csv py/expert_predictor/expert_reuse_qwen3_30b.csv \
  --cache-lookahead-slack 3 \
  --drop-page-cache-between-runs \
  --predictor-base-dir "$PRED" \
  --sweep-question custom_1_16_no_ppl \
  --skip-baselines \
  --prefetch-only \
  --enable-generation-perplexity \
  --out-dir py/utils/final_results_runs/paper_remake/${DOMAIN}/select \
  --csv-file py/utils/final_results_runs/paper_remake/${DOMAIN}/select/sweep.csv
```

Turn on `--enable-generation-perplexity` only for `wikitext`, `fineweb`, and `orca`.

From each domain's `sweep.csv`, record the best tokens/s at each \(C\). Columns already include `pred_window_recall_pct` and the precision fields parsed in `sweep_predict_cached_cache_metrics.py`. Write the winners to `paper_remake/selected_configs.json`:

```json
{ "wikitext": { "16": {"S": 1, "B": 4}, "32": {}, "48": {}, "64": {} }, "...": {} }
```

Also write, per domain, the \(C=48\) recall, precision, and tokens/s of every \((S,B)\) point. That is the replacement for `figures/tps_vs_recall_precision_C48.png`. Plot with `py/utils/section_5_2/plot_sec5_2_C48.py` only after pointing it at this CSV; the script's current input path is the old sweep.

Oracle scheduling in the submitted paper matches ExpertAhead's stride. After \(S\) is chosen for a cache size, the oracle row uses that same \(S\), not the old table's \(S\) (which was 2, 1, 1, 1).

## 4. Quality table

This replaces `tab:perplexity_change`. Cache-conditional routing does not use the predictor. It does use the new prompts, so the WikiText perplexity percentages cannot be copied.

For each domain and each \(C \in \{16,32,48,64\}\), sweep \(J \in \{3,4,5,6,7,8\}\) at \(\lambda=1\). \(J=8\) is unaltered top-8 routing and is the denominator.

```bash
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --examples-json py/utils/final_results_runs/paper_remake/by_domain/${DOMAIN}.json \
  --domain "$DOMAIN" \
  --num-prompts 0 \
  --prompt-max-chars 4096 \
  --max-new-tokens 150 \
  --temperature 0.0 \
  --cold-per-prompt \
  --cache-sizes 16 32 48 64 \
  --lookaheads 1 \
  --non-baseline-cache-policy LFRU \
  --drop-page-cache-between-runs \
  --sweep-question custom_1_16_no_ppl \
  --skip-baselines \
  --cache-cond-only \
  --lambdas 1 \
  --routing-bias-top-ns 3 4 5 6 7 8 \
  --out-dir py/utils/final_results_runs/paper_remake/${DOMAIN}/quality \
  --csv-file py/utils/final_results_runs/paper_remake/${DOMAIN}/quality/sweep.csv
```

Add `--enable-generation-perplexity` for wikitext, fineweb, and orca.

Then score:

```bash
python3 py/utils/eval_correctness.py \
  --examples py/utils/final_results_runs/paper_remake/by_domain/${DOMAIN}.json \
  --generations py/utils/final_results_runs/paper_remake/${DOMAIN}/quality/generations.md \
  --sweep-csv py/utils/final_results_runs/paper_remake/${DOMAIN}/quality/sweep.csv \
  --out py/utils/final_results_runs/paper_remake/${DOMAIN}/quality/correctness.json
```

Report the relative change versus \(J=8\), per domain:

- wikitext, fineweb, orca: generation perplexity
- gsm8k: exact match
- cnn_dailymail: ROUGE-L

The end-to-end table still uses \(J=5\). Quote the \(J=5\) quality delta at the same \(C\) as the throughput number. Do not quote the \(C=64\) perplexity next to a \(C=16\) speedup.

## 5. End-to-end table

This replaces `tab:end_to_end_results` and `figures/expert_ahead_evaluation_random_cropped.png`.

Run every method on the same manifest, 150 tokens, temperature 0, cold start per prompt. One policy for every non-random row: LFRU (`--non-baseline-cache-policy LFRU`). Random eviction stays the equal-memory baseline, matching the abstract.

Per domain, per cache size, using the \((S,B)\) from `selected_configs.json`. The script `sh_scripts/run_final_best_10prompt.sh` is the right shape (random, cross-layer, cache-conditional, ExpertAhead, ExpertAhead-CC) but it defaults to the old predictor, `--dataset wikitext`, and the old \(S/B\) table. Drive it with:

- `PRED=trainingData/qwen3_30b/multi_dataset/transformer_emb_markov_pfill_mixed`
- examples JSON for that domain instead of `--dataset wikitext`
- `NUM_PROMPTS` handled by `--num-prompts 0` plus `--examples-json`
- `MAX_NEW_TOKENS=150`
- `EA_S`, `EA_B`, `EACC_S`, `EACC_B` from `selected_configs.json`
- `EACC_J=5`, `CC_J=5`
- cross-layer budget retuned on this manifest, or a small grid (`--prefetch-budgets` at the same budgets as ExpertAhead) rather than the old `XL_B` values 8, 6, 2, 2

Oracle and SSD streaming are not in `run_final_best_10prompt.sh`. Add them.

SSD streaming, same prompts, one measurement reused across cache sizes (the submitted table does this; streaming TPS was 1.30 at every \(C\)):

```bash
FORCE_EXPERT_MISS=1 python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --examples-json py/utils/final_results_runs/paper_remake/by_domain/${DOMAIN}.json \
  --domain "$DOMAIN" \
  --num-prompts 0 \
  --max-new-tokens 150 \
  --temperature 0.0 \
  --cold-per-prompt \
  --cache-sizes 16 \
  --sweep-question custom_1_16_no_ppl \
  --random-baseline-only \
  --drop-page-cache-between-runs \
  --out-dir py/utils/final_results_runs/paper_remake/${DOMAIN}/streaming \
  --csv-file py/utils/final_results_runs/paper_remake/${DOMAIN}/streaming/sweep.csv
```

Oracle, one stride (the ExpertAhead \(S\) chosen for that \(C\)), full-union prefetch, no learned predictor:

```bash
python3 py/utils/sweep_predict_cached_cache_metrics.py \
  --model qwen \
  --examples-json py/utils/final_results_runs/paper_remake/by_domain/${DOMAIN}.json \
  --domain "$DOMAIN" \
  --oracle-trace-dir trainingData/eval_traces_paper/${DOMAIN} \
  --num-prompts 0 \
  --max-new-tokens 150 \
  --temperature 0.0 \
  --cold-per-prompt \
  --cache-sizes "$C" \
  --lookaheads "$S" \
  --non-baseline-cache-policy LFRU \
  --sweep-question oracle_baseline_sweep \
  --skip-baselines \
  --oracle-full-union-only \
  --no-actual-predictor \
  --drop-page-cache-between-runs \
  --out-dir py/utils/final_results_runs/paper_remake/${DOMAIN}/oracle_C${C} \
  --csv-file py/utils/final_results_runs/paper_remake/${DOMAIN}/oracle_C${C}/sweep.csv
```

`py/utils/build_end_to_end_table.py` emits the submitted LaTeX table from streaming, oracle, and method CSVs. It assumes one prompt set. Run it once per domain, then add a macro-average across the five domains with equal domain weight. The paper table's columns stay: for each \(C\), \(S/B/J\), tokens/s, speedup versus SSD streaming, speedup versus random eviction.

After ExpertAhead (prefetch only, \(\lambda=0\)), score generations against the random-eviction transcripts with `eval_correctness.py --normal-transcripts`. Token match should be exact. That is the "lossless" claim. ExpertAhead-CC is not lossless; publish its quality delta from step 4 at the same \(J\) and \(C\).

Oracle acceptance: window recall and precision at 100% on these traces. If they are not, the trace and the generated token sequence diverged (different text, different truncation, or a different `max-new-tokens`). Fix the capture. Do not report that row.

## 6. Eviction sentence

One cache size, \(C=64\), same 10×150 manifest, no predictor. Compare random, LRU, and LFRU on each domain. The submitted sentence also quotes most-recently-used at \(1.75\times\) LFRU; only include MRU if the sweep already has that policy. `--non-baseline-cache-policy` knows `LFRU`. LRU and random are first-class baselines in `oracle_baseline_sweep` and the random-only flag.

Replace the WikiText-only sentence with the per-domain result. A single averaged factor is fine only beside the per-domain numbers.

## 7. What to write into the paper

After the CSVs exist, update the tex. Do not update it before.

Replace:

- Abstract, introduction, and conclusion speedups. Name the cache size, the baseline (random eviction, equal memory), and whether the number is one domain or the macro-average. The submitted \(1.65\times\), \(2.05\times\), and \(3.11\times\) are WikiText at \(C=64\) (the oracle's best equal-memory point in that paper was actually \(3.17\times\) at \(C=48\)).
- Both predictor tables. Prefer per-domain window recall and precision of the deployed model at the selected \(B\), taken from the step-3 CSV. Skip a new MLP-vs-transformer search. The architecture is already chosen, and the mixed training shards are not on this machine.
- The \(C=48\) recall-precision figure.
- The perplexity table, now per domain and per metric.
- The end-to-end table, figure, and the oracle stride so it matches the new ExpertAhead stride.
- The training-data sentence: WikiText, FineWeb-Edu, OpenOrca, GSM8K, and CNN/DailyMail train partitions; eval is the frozen manifest above.

Delete rather than carry forward:

- The speculative-decoding comparison (2.6 vs 4.4 tokens/s). It is not in the submitted table.
- The FlashMoE 7% sentence. Different model, different machine.

Leave in place: the system diagram, the three timeline figures, the predictor diagram, the Ryzen AI 9 HX 370 paragraph, and the latency equations. The modeling section's prefetch equation is commented out and was not validated. Do not add a modeling figure unless you also fit \(T_{ceil}\) and \(T_{SSD,n}\) on this machine and show the fit. Otherwise leave that contribution as the qualitative account already in the text.

## Done when

- [ ] `paper_remake/by_domain/{wikitext,fineweb,orca,gsm8k,cnn_dailymail}.json` each have 10 prompts, checksummed
- [ ] `trainingData/eval_traces_paper/<domain>/` has 10 traces, 150 decode steps, 48 layers
- [ ] `selected_configs.json` has an \(S \in \{1,4,8,16\}\) and a \(B\) for every domain and every \(C\)
- [ ] ExpertAhead token match versus \(\lambda=0\) routing is exact on every domain
- [ ] Oracle recall and precision are 100% on the new traces
- [ ] Quality deltas exist at \(J=5\) for the \(C\) values quoted in the table
- [ ] Per-domain end-to-end CSVs and one macro-average table exist under `paper_remake/`
- [ ] No reported path contains `wikitext_test_traces` or `transformer_final_pfill_markov_emb`
- [ ] The pilot run under `expanded_prompt_space/` is not cited as the paper result
