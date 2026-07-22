#!/usr/bin/env bash
# Greedy generation perplexity for Cache-Cond vs ExpertAhead-CC (Hybrid) winner rows.
#
# Fills prompt_gen_perplexity on the existing final_10prompt sweep.csv rows — no TPS re-run.
# Uses the same prompts / max_new_tokens / cold-per-prompt settings as run_final_best_10prompt.sh.
#
# Usage:
#   ./run_final_gen_ppl_compare.sh
#   CACHE_SIZES="16 32" ./run_final_gen_ppl_compare.sh
#   DRY_RUN=1 ./run_final_gen_ppl_compare.sh
#
# Optional:
#   RUN_ROOT=py/utils/final_results_runs/final_10prompt
#   RUN_SUFFIX=20260709_001606   # canonical batch dir suffix (default: latest per C)
#   PPL_POLICIES="cache-cond,hybrid"   # default; add lru for a three-way baseline
#   PPL_OVERWRITE=1            # re-measure even when prompt_gen_perplexity is set
#   NUM_PROMPTS=10 MAX_NEW_TOKENS=150

set -euo pipefail
cd "$(dirname "$0")/.."
source utils/setup.sh

CACHE_SIZES="${CACHE_SIZES:-16 32 48 64}"
RUN_ROOT="${RUN_ROOT:-py/utils/final_results_runs/final_10prompt}"
PRED="${PRED:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-$(( 3600 + NUM_PROMPTS * MAX_NEW_TOKENS * 5 ))}"
PPL_POLICIES="${PPL_POLICIES:-cache-cond,hybrid}"

resolve_run_dir() {
  local C=$1
  if [[ -n "${RUN_SUFFIX:-}" ]]; then
    echo "$RUN_ROOT/C${C}_${RUN_SUFFIX}"
    return
  fi
  local latest
  latest=$(ls -d "$RUN_ROOT"/C${C}_* 2>/dev/null | sort | tail -1)
  if [[ -z "$latest" ]]; then
    echo "No run directory for C=$C under $RUN_ROOT" >&2
    return 1
  fi
  echo "$latest"
}

for C in $CACHE_SIZES; do
  OUT_DIR=$(resolve_run_dir "$C") || exit 1
  CSV="$OUT_DIR/sweep.csv"
  if [[ ! -f "$CSV" ]]; then
    echo "Missing $CSV" >&2
    exit 1
  fi

  cmd=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --model qwen
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --drop-page-cache-between-prompts
    --drop-page-cache-between-runs
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --predictor-base-dir "$PRED"
    --gen-ppl-winners-only
    --ppl-policies "$PPL_POLICIES"
    --out-dir "$OUT_DIR"
    --csv-file "$CSV"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    --log-file "$OUT_DIR/gen_ppl_compare.log"
  )
  if [[ "${PPL_OVERWRITE:-0}" == "1" ]]; then
    cmd+=(--ppl-overwrite)
  fi
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    cmd+=(--retry-failed)
  fi

  echo "[C=$C gen_ppl] $(date -Is) -> $CSV"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %q' "${cmd[@]}"
    echo
    continue
  fi
  "${cmd[@]}"
done

echo "Done. Check prompt_gen_perplexity in each sweep.csv under $RUN_ROOT"
