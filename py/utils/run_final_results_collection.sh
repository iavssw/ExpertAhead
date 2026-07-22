#!/usr/bin/env bash
# Hands-off final thesis sweep: C=8..64, all valid lookaheads × budgets, 10 prompts,
# TPS for all four policies, then wikitext PPL on the best Cache-Cond + Hybrid per C.
#
# Phase 1 only:
#   sudo -E bash py/utils/run_final_results_collection.sh --skip-ppl-phase
#
# Phase 2 only (after phase 1 finished):
#   sudo -E bash py/utils/run_final_results_collection.sh --ppl-winners-only \
#     --csv-file py/utils/final_results_runs/final_results_collection/<timestamp>/sweep.csv
#
# Resume interrupted TPS sweep:
#   sudo -E bash py/utils/run_final_results_collection.sh --retry-failed \
#     --csv-file py/utils/final_results_runs/final_results_collection/<timestamp>/sweep.csv \
#     --run-dir py/utils/final_results_runs/final_results_collection/<timestamp>
#
# Dry-run:
#   bash py/utils/run_final_results_collection.sh --dry-run

set -eo pipefail
cd "$(dirname "$0")/../.."
source utils/setup.sh
set -u

PYTHON="${VIRTUAL_ENV}/bin/python3"
exec sudo -E "$PYTHON" py/utils/finals_experiment_runner.py \
  --experiment final_results_collection \
  "$@"
