# Correctness comparison — `orca`

Compare **Prefetch Only** (lossless routing) vs **Both** (hybrid λ/J) against **gold / correct** quality.

- Gold metric: `gen_ppl`
- `random` = Neither (RANDOM) quality reference (should match Prefetch Only)
- `prefetch_only` = ExpertAhead λ=0 (expect ≈ random / correct)
- `both` = ExpertAhead + forced routing (may diverge from gold)

## By mode

| Mode | Mean score | #configs |
|------|------------|----------|

## By label

| Mode | Label | Score | n |
|------|-------|-------|---|

