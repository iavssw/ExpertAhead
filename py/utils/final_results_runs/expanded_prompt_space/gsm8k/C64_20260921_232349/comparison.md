# Quality vs normal operation — `gsm8k`

**Normal** = RANDOM backend @ C=64 (fast, high-cache “full” operation).
Report degradation of Prefetch Only / Both relative to that baseline.

- Gold metric: `exact_match`
- Normal gold score: 0.0% EM
- `match_to_normal` = fraction of responses identical to normal
- `degradation` = quality drop vs normal (0 = no worse; EM/ROUGE ↓ or gen-PPL ↑)

## Degradation by mode

| Mode | Score | vs normal | match_to_normal | degradation | #configs |
|------|-------|-----------|-----------------|-------------|----------|
| normal | 0.0% EM | +0.0000 | 100% | 0.0% EM | 1 |
| random | 0.0% EM | +0.0000 | 100% | 0.0% EM | 1 |
| prefetch_only | 0.0% EM | +0.0000 | 100% | 0.0% EM | 1 |
| both | 15.0% EM | +0.1500 | 0% | 0.0% EM | 1 |

## By label

| Mode | Label | Score | match | degradation | n |
|------|-------|-------|-------|-------------|---|
| normal | Normal (RANDOM C=64) | 0.0% EM | 100% | 0.0% EM | 5 |
| random | Neither (RANDOM) | 0.0% EM | 100% | 0.0% EM | 5 |
| prefetch_only | Prefetch Only B=32 | 0.0% EM | 100% | 0.0% EM | 20 |
| both | Both λ=1.0 B=32 | 15.0% EM | 0% | 0.0% EM | 20 |

