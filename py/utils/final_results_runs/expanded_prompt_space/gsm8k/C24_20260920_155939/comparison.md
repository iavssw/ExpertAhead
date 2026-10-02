# Correctness comparison — `gsm8k`

Compare **Prefetch Only** (lossless routing) vs **Both** (hybrid λ/J) against **gold / correct** quality.

- Gold metric: `exact_match`
- `random` = Neither (RANDOM) quality reference (should match Prefetch Only)
- `prefetch_only` = ExpertAhead λ=0 (expect ≈ random / correct)
- `both` = ExpertAhead + forced routing (may diverge from gold)

## By mode

| Mode | Mean score | #configs |
|------|------------|----------|
| random | 0.0% EM | 1 |
| prefetch_only | 0.0% EM | 1 |
| both | 12.0% EM | 1 |

## By label

| Mode | Label | Score | n |
|------|-------|-------|---|
| random | Neither (RANDOM) | 0.0% EM | 5 |
| prefetch_only | Prefetch Only B=12 | 0.0% EM | 25 |
| both | Both λ=1.0 B=12 | 12.0% EM | 25 |

