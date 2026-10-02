#!/usr/bin/env python3
"""Print paper Tables 1 and 2 from mixed-data ablation runs."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(
    "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/"
    "qwen3_30b_mixed_sharded/paper_tables_mixed"
)
LAYERS = [0, 23, 47]
BUDGET = 16
HORIZONS = [1, 4, 8, 16]

TABLE1 = {
    "MLP Emb (H=1)": "ablation_emb_only_hist1_h32_f{S}",
    "MLP Emb (H=4)": "ablation_emb_only_hist4_h32_f{S}",
    "Transformer (H=1)": "transformer_ablation_emb_only_hist1_h64_f{S}",
    "Transformer (H=4)": "transformer_ablation_emb_only_hist4_h64_f{S}",
}
TABLE2 = {
    "Base": "transformer_ablation_emb_only_hist4_h64_f{S}",
    "+Markov": "transformer_ablation_emb_markov_hist4_h64_f{S}",
    "+Markov & Prefill": "transformer_ablation_emb_markov_pfill_hist4_h64_f{S}",
}


def metrics_at(config_name: str):
    recs, precs = [], []
    d = ROOT / config_name
    for layer in LAYERS:
        p = d / f"layer_{layer}" / "training_metrics.json"
        if not p.exists():
            continue
        best = json.loads(p.read_text()).get("best_val_recalls") or {}
        r = best.get(f"topk_recall@{BUDGET}", best.get(f"union_recall@{BUDGET}"))
        pr = best.get(f"topk_precision@{BUDGET}", best.get(f"union_precision@{BUDGET}"))
        if r is not None:
            recs.append(r)
        if pr is not None:
            precs.append(pr)
    if not recs:
        return None, None, 0
    return float(np.mean(recs) * 100), float(np.mean(precs) * 100) if precs else None, len(recs)


def print_table(title, columns, horizons):
    print(f"\n{title}")
    header = "S"
    for col in columns:
        header += f" | {col} Rec | {col} Prec"
    print(header)
    print("-" * len(header))
    for s in horizons:
        cells = [str(s)]
        for tmpl in columns.values():
            rec, prec, n = metrics_at(tmpl.format(S=s))
            if rec is None:
                cells += ["n/a", "n/a"]
            else:
                cells.append(f"{rec:.2f}")
                cells.append(f"{prec:.2f}" if prec is not None else "n/a")
        print(" | ".join(cells))


def main():
    print(f"Reading {ROOT}")
    print(f"Layers {LAYERS}, budget B={BUDGET}")
    print_table("Table 1: Recall and Precision (%) at B=16. Embeddings only.", TABLE1, HORIZONS)
    print_table(
        "Table 2: Transformer(H=4) input feature ablation at B=16.",
        TABLE2,
        [1, 4],
    )


if __name__ == "__main__":
    main()
