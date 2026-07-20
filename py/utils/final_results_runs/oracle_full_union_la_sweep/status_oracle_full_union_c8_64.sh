#!/usr/bin/env bash
# Print progress for the active oracle C8-64 sweep (see oracle_full_union_c8_64_active.env).

set -euo pipefail
cd "$(dirname "$0")/../../../.."
source utils/setup.sh

OUT_DIR="py/utils/final_results_runs/oracle_full_union_la_sweep"
STATE_FILE="${OUT_DIR}/oracle_full_union_c8_64_active.env"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "No active run (missing $STATE_FILE)."
  echo "Start one with: bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "$STATE_FILE"

python3 <<EOF
import pandas as pd
from pathlib import Path

csv_path = Path("$CSV")
if not csv_path.exists():
    print(f"Active run CSV not found yet: {csv_path}")
    print("Resume with run_oracle_full_union_c8_64.sh")
    raise SystemExit(0)

cache_sizes = [8, 16, 24, 32, 40, 48, 56, 64]
lookaheads = [1, 2, 3, 4, 6]

expected = []
for c in cache_sizes:
    expected.append((c, 1, "Neither (LRU)"))
    expected.append((c, 1, "Neither (RANDOM)"))
    for la in lookaheads:
        expected.append((c, la, f"Oracle Noisy A=0.875 LA={la}"))
        expected.append((c, la, f"Oracle Full Union LA={la}"))

df = pd.read_csv(csv_path)
done = set(zip(df["cache_size"].astype(int), df["lookahead"].astype(int), df["label"].astype(str)))
missing = [e for e in expected if e not in done]

print(f"Active run: $CSV")
print(f"Log:        $LOG")
print(f"Completed:  {len(df)} / {len(expected)} configs ({100*len(df)/len(expected):.1f}%)")
print(f"Missing:    {len(missing)}")
print()
print("By cache size:")
for c in cache_sizes:
    exp = sum(1 for e in expected if e[0] == c)
    got = sum(1 for e in expected if e[0] == c and e in done)
    mark = "DONE" if got == exp else ""
    print(f"  C={c:2d}: {got}/{exp} {mark}")

if missing:
    print()
    print("Next up:")
    for m in missing[:5]:
        print(f"  C={m[0]} LA={m[1]} {m[2]}")
    if len(missing) > 5:
        print(f"  ... +{len(missing) - 5} more")
    print()
    print("Resume:")
    print("  bash py/utils/final_results_runs/oracle_full_union_la_sweep/run_oracle_full_union_c8_64.sh")
else:
    print()
    print("All configs complete. Regenerate plots:")
    print(f"  python py/utils/plot_oracle_cache_vs_lru.py --csv {csv_path!s}")
    print(f"  python py/utils/plot_oracle_cache_vs_lru.py --csv {csv_path!s} --tradeoff")
EOF
