#!/usr/bin/env bash
# Unified Section 5.2 prefetch sweeps for precision-recall-TPS plots.
#
# Grid (prefetch-only + RANDOM/LRU baselines per cache size):
#   C=16: S=1,2,3   × B=4,8,12,16          (12 + 2 = 14 configs)
#   C=32: S=1,2,3,4 × B=4,8,12,16,20,24    (24 + 2 = 26 configs)
#   C=48: S=1,2,3,4,5 × B=8,16,20,24,32    (25 + 2 = 27 configs)
#
# 10 wikitext prompts per config. Outputs feed:
#   py/utils/section_5_2/sec_5_2_raw_C16.csv
#   py/utils/section_5_2/sec_5_2_raw_C32.csv
#   py/utils/section_5_2/sec_5_2_raw_C48.csv
#
# Also archives copies under:
#   py/utils/section_5_2/unified_10prompt_<timestamp>/
#
# Usage:
#   tmux new -s sec5_unified
#   cd ~/heteroPredict && ./run_sec5_2_unified_sweep.sh
#   # detach: Ctrl-b d
#
# Resume a failed cache size (re-runs only missing rows in that CSV):
#   RETRY_FAILED=1 START_CACHE=32 ./run_sec5_2_unified_sweep.sh
#
# Dry-run (print commands, no GPU):
#   DRY_RUN=1 ./run_sec5_2_unified_sweep.sh
#
# Plot after completion:
#   source utils/setup.sh
#   python3 py/utils/section_5_2/plot_sec5_2.py
#   python3 py/utils/section_5_2/plot_sec5_2_C48.py

set -euo pipefail
cd "$(dirname "$0")"
source utils/setup.sh

PRED="${PRED:-trainingData/qwen3_30b/transformer_final_pfill_markov_emb}"
REUSE_CSV="${REUSE_CSV:-py/expert_predictor/expert_reuse_qwen3_30b.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
PROMPT_MAX_CHARS="${PROMPT_MAX_CHARS:-4096}"
SUBPROCESS_TIMEOUT="${SUBPROCESS_TIMEOUT:-7200}"

SECTION_DIR="${SECTION_DIR:-py/utils/section_5_2}"
TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$SECTION_DIR/unified_10prompt_${TS}}"
START_CACHE="${START_CACHE:-16}"   # 16 | 32 | 48 — skip earlier cache sizes on resume

mkdir -p "$OUT_DIR"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  exec > >(tee -a "$OUT_DIR/collection.log") 2>&1
fi

backup_raw_csv() {
  local C=$1
  local dst="$SECTION_DIR/sec_5_2_raw_C${C}.csv"
  if [[ -f "$dst" ]]; then
    cp "$dst" "$OUT_DIR/sec_5_2_raw_C${C}.csv.bak"
    echo "[sec5-unified] Backed up existing $dst"
  fi
}

run_cache() {
  local C=$1
  shift
  local -a LOOKAHEADS=()
  local -a BUDGETS=()
  local mode=la
  for x in "$@"; do
    if [[ "$x" == "--" ]]; then mode=budget; continue; fi
    if [[ "$mode" == la ]]; then LOOKAHEADS+=("$x"); else BUDGETS+=("$x"); fi
  done

  local archive_csv="$OUT_DIR/sweep_C${C}.csv"
  local plot_csv="$SECTION_DIR/sec_5_2_raw_C${C}.csv"
  local log_file="$OUT_DIR/sweep_C${C}.log"
  local n_prefetch=$(( ${#LOOKAHEADS[@]} * ${#BUDGETS[@]} ))

  local -a EXTRA=()
  if [[ -f "$archive_csv" ]]; then
    EXTRA+=(--append)
  fi
  if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    EXTRA+=(--retry-failed)
  fi

  local -a CMD=(
    python3 py/utils/sweep_predict_cached_cache_metrics.py
    --sweep-question custom_1_16_no_ppl
    --model qwen
    --predictor-base-dir "$PRED"
    --dataset wikitext
    --num-prompts "$NUM_PROMPTS"
    --prompt-max-chars "$PROMPT_MAX_CHARS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature 0.0
    --cold-per-prompt
    --cache-sizes "$C"
    --lookaheads "${LOOKAHEADS[@]}"
    --custom-explicit-prefetch-budgets
    --prefetch-budgets "${BUDGETS[@]}"
    --prefetch-only
    --non-baseline-cache-policy LFRU
    --constraint-expert-reuse-csv "$REUSE_CSV"
    --drop-page-cache-between-runs
    --out-dir "$OUT_DIR"
    --csv-file "$archive_csv"
    --log-file "$log_file"
    --subprocess-timeout "$SUBPROCESS_TIMEOUT"
    "${EXTRA[@]}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] C=$C  S=${LOOKAHEADS[*]}  B=${BUDGETS[*]}  ($n_prefetch prefetch + 2 baselines)"
    printf '  %q' "${CMD[@]}"
    echo
    return
  fi

  backup_raw_csv "$C"
  echo ""
  echo "========================================"
  echo "[sec5-unified] C=$C  S=${LOOKAHEADS[*]}  B=${BUDGETS[*]}"
  echo "[sec5-unified] $n_prefetch prefetch configs + baselines"
  echo "[sec5-unified] archive: $archive_csv"
  echo "[sec5-unified] plot CSV: $plot_csv"
  echo "[sec5-unified] started: $(date -Is)"
  echo "========================================"

  "${CMD[@]}"

  cp "$archive_csv" "$plot_csv"
  echo "[sec5-unified] C=$C done → $plot_csv ($(date -Is))"
}

should_run() {
  local C=$1
  [[ "$C" -ge "$START_CACHE" ]]
}

{
  echo "run_sec5_2_unified_sweep.sh"
  echo "started: $(date -Is)"
  echo "OUT_DIR=$OUT_DIR"
  echo "NUM_PROMPTS=$NUM_PROMPTS"
  echo "PRED=$PRED"
  echo "START_CACHE=$START_CACHE"
  echo "RETRY_FAILED=${RETRY_FAILED:-0}"
} > "$OUT_DIR/command.txt"

echo "=== Section 5.2 unified prefetch sweep ==="
echo "OUT_DIR:      $OUT_DIR"
echo "Plot CSVs:    $SECTION_DIR/sec_5_2_raw_C{16,32,48}.csv"
echo "Prompts:      $NUM_PROMPTS"
echo "START_CACHE:  $START_CACHE"
echo ""
echo "Grid: 67 configs total (~5–10 h depending on machine)"
echo "  C=16: 12 prefetch (S=1,2,3 × B=4,8,12,16)"
echo "  C=32: 24 prefetch (S=1,2,3,4 × B=4,8,12,16,20,24)"
echo "  C=48: 25 prefetch (S=1,2,3,4,5 × B=8,16,20,24,32)"
echo ""
echo "Note: --prefetch-only still runs RANDOM+LRU at the end of each cache size."
echo "      You can kill after all prefetch rows appear in the CSV if LRU stalls."
echo ""

if should_run 16; then
  run_cache 16 1 2 3 -- 4 8 12 16
fi

if should_run 32; then
  run_cache 32 1 2 3 4 -- 4 8 12 16 20 24
fi

if should_run 48; then
  run_cache 48 1 2 3 4 5 -- 8 16 20 24 32
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1 — no jobs launched."
  exit 0
fi

echo ""
echo "=== All done ==="
echo "Plot CSVs:"
echo "  $SECTION_DIR/sec_5_2_raw_C16.csv"
echo "  $SECTION_DIR/sec_5_2_raw_C32.csv"
echo "  $SECTION_DIR/sec_5_2_raw_C48.csv"
echo "Archive: $OUT_DIR/"
echo ""
echo "Plot:"
echo "  python3 py/utils/section_5_2/plot_sec5_2.py"
echo "  python3 py/utils/section_5_2/plot_sec5_2_C48.py"
