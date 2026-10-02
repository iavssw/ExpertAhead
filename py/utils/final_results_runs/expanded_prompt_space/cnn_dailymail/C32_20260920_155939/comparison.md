# Correctness comparison — `cnn_dailymail`

Compare **Prefetch Only** (lossless routing) vs **Both** (hybrid λ/J) against **gold / correct** quality.

- Gold metric: `rouge_l`
- `random` = Neither (RANDOM) quality reference (should match Prefetch Only)
- `prefetch_only` = ExpertAhead λ=0 (expect ≈ random / correct)
- `both` = ExpertAhead + forced routing (may diverge from gold)

## By mode

| Mode | Mean score | #configs |
|------|------------|----------|
| random | 0.1559 ROUGE-L | 1 |
| prefetch_only | 0.1559 ROUGE-L | 1 |
| both | 0.1742 ROUGE-L | 1 |

## By label

| Mode | Label | Score | n |
|------|-------|-------|---|
| random | Neither (RANDOM) | 0.1559 ROUGE-L | 5 |
| prefetch_only | Prefetch Only B=16 | 0.1559 ROUGE-L | 25 |
| both | Both λ=1.0 B=16 | 0.1742 ROUGE-L | 25 |

