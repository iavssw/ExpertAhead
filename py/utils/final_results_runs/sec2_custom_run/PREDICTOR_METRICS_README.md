Join LRU vs prefetch-only rows and summarize predictor recall/precision + load stats.

Post-processes a ``sweep.csv`` from the ``sec2_predictor_effectiveness`` experiment
(``finals_experiment_runner.py``), backing the section-3 predictor-vs-LRU story. Writes:

- ``predictor_speedup_attribution.csv`` — one row per prefetch config with baseline TPS /
  cache hit rate for the same (cache_size, lookahead, prompt_hash).

Metric glossary (serving decode, not training accuracy):

- **Recall (routed):** ``pred_hit_rate_routed_topk_pct`` if present, else ``pred_hit_rate_pct``.
  Fraction of *router-selected* experts that were already in the predictor’s prefetch set.

- **Precision (requested):** ``pred_requested_rate_topk_pct`` when the model prints it:
  fraction of *prefetched* experts that the router actually asked for on that step.

- **Stalls / prefetch loads:** from the ``Bandwidth: StallLoads=…`` line — fewer stalls with
  higher TPS usually means the predictor hid on-demand expert load latency.

Usage::

  python analyze_predictor_speedup_attribution.py --csv path/to/sweep.csv
