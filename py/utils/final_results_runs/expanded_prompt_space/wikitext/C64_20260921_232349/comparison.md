# Quality vs normal operation — `wikitext`

**Normal** = RANDOM backend @ C=64 (fast, high-cache “full” operation).
Report degradation of Prefetch Only / Both relative to that baseline.

- Gold metric: `gen_ppl`
- Normal gold score: n/a
- `match_to_normal` = fraction of responses identical to normal
- `degradation` = quality drop vs normal (0 = no worse; EM/ROUGE ↓ or gen-PPL ↑)

## Degradation by mode

| Mode | Score | vs normal | match_to_normal | degradation | #configs |
|------|-------|-----------|-----------------|-------------|----------|
| normal | n/a | +0.0000 | 100% | 0.0000 (↓ better) | 1 |

## By label

| Mode | Label | Score | match | degradation | n |
|------|-------|-------|-------|-------------|---|
| normal | Normal (RANDOM C=64) | n/a | 100% | 0.0000 (↓ better) | 5 |

