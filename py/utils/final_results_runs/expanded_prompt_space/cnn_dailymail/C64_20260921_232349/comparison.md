# Quality vs normal operation — `cnn_dailymail`

**Normal** = RANDOM backend @ C=64 (fast, high-cache “full” operation).
Report degradation of Prefetch Only / Both relative to that baseline.

- Gold metric: `rouge_l`
- Normal gold score: 0.1559 ROUGE-L
- `match_to_normal` = fraction of responses identical to normal
- `degradation` = quality drop vs normal (0 = no worse; EM/ROUGE ↓ or gen-PPL ↑)

## Degradation by mode

| Mode | Score | vs normal | match_to_normal | degradation | #configs |
|------|-------|-----------|-----------------|-------------|----------|
| normal | 0.1559 ROUGE-L | +0.0000 | 100% | 0.0000 ROUGE-L | 1 |
| random | 0.1559 ROUGE-L | +0.0000 | 100% | 0.0000 ROUGE-L | 1 |
| prefetch_only | 0.1559 ROUGE-L | +0.0000 | 100% | 0.0000 ROUGE-L | 1 |
| both | 0.1620 ROUGE-L | +0.0062 | 20% | 0.0000 ROUGE-L | 1 |

## By label

| Mode | Label | Score | match | degradation | n |
|------|-------|-------|-------|-------------|---|
| normal | Normal (RANDOM C=64) | 0.1559 ROUGE-L | 100% | 0.0000 ROUGE-L | 5 |
| random | Neither (RANDOM) | 0.1559 ROUGE-L | 100% | 0.0000 ROUGE-L | 5 |
| prefetch_only | Prefetch Only B=32 | 0.1559 ROUGE-L | 100% | 0.0000 ROUGE-L | 20 |
| both | Both λ=1.0 B=32 | 0.1620 ROUGE-L | 20% | 0.0000 ROUGE-L | 20 |

