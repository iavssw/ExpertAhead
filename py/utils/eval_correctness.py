#!/usr/bin/env python3
"""Dataset-appropriate correctness scorers for expanded_prompt_space.

Goal: compare **Prefetch Only** (lossless) vs **Both** (hybrid λ/J) vs **gold /
correct** quality (EM / ROUGE / gen-PPL), with RANDOM as the quality reference.

Metrics:
  - gsm8k: exact match on final numeric / #### answer
  - cnn_dailymail: ROUGE-L F1 vs highlights
  - mbpp: weak normalized code exact-match (pass@1 deferred)
  - wikitext / fineweb / orca: gen_ppl from sweep CSV (prompt_gen_perplexity)

Usage:
  python3 py/utils/eval_correctness.py \\
    --examples py/utils/final_results_runs/expanded_prompt_space/by_domain/gsm8k.json \\
    --generations .../gsm8k/C24_.../generations.md \\
    --sweep-csv .../gsm8k/C24_.../sweep.csv \\
    --out .../gsm8k/C24_.../correctness.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def classify_mode(label: str) -> str:
    """Map sweep label → mode bucket for comparison summaries."""
    l = label or ""
    if l.startswith("Normal (RANDOM") or l.startswith("Normal operation"):
        return "normal"
    if l.startswith("Neither (RANDOM)") or ("RANDOM" in l and "Prefetch" not in l):
        return "random"
    if l.startswith("Prefetch Only"):
        return "prefetch_only"
    if l.startswith("Both "):
        return "both"
    if l.startswith("Neither (LRU)") or l.startswith("LRU"):
        return "lru"
    if "Cache-Cond" in l:
        return "cache_cond"
    return "other"


def responses_by_config(generations: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Group responses by label, preserving prompt order within each config."""
    out: Dict[str, List[str]] = {}
    for label, gen in generations:
        out.setdefault(label or "", []).append(gen)
    return out


def load_normal_responses(path: Optional[Path]) -> List[str]:
    """Load normal-operation responses (one per prompt) from transcripts/generations."""
    if path is None or not path.exists():
        return []
    if path.suffix == ".jsonl" or path.name == "transcripts.jsonl":
        gens = parse_transcripts_jsonl(path)
    else:
        gens = parse_generations_md(path)
        jsonl = path.parent / "transcripts.jsonl"
        if jsonl.exists():
            gens = parse_transcripts_jsonl(jsonl)
    if not gens:
        return []
    # Prefer an explicit normal/random label group; else first config block.
    by_label = responses_by_config(gens)
    for label, resps in by_label.items():
        mode = classify_mode(label)
        if mode in ("normal", "random"):
            return list(resps)
    # Fallback: first label's responses
    first = next(iter(by_label.values()))
    return list(first)


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Pull the final answer (#### style or last number)."""
    if not text:
        return None
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        return _normalize_num(m[-1])
    # Fallback: last number in the string
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _normalize_num(s: str) -> str:
    s = s.strip().replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return str(v)
    except ValueError:
        return s


def score_gsm8k(pred: str, gold: Any) -> Dict[str, Any]:
    gold_ans = extract_gsm8k_answer(str(gold) if gold is not None else "")
    pred_ans = extract_gsm8k_answer(pred)
    ok = gold_ans is not None and pred_ans is not None and gold_ans == pred_ans
    return {
        "metric": "exact_match",
        "score": 1.0 if ok else 0.0,
        "pred_answer": pred_ans,
        "gold_answer": gold_ans,
    }


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # O(n*m) DP; prompts are short summaries
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, gold_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def score_cnn(pred: str, gold: Any) -> Dict[str, Any]:
    g = str(gold) if gold is not None else ""
    s = rouge_l_f1(pred, g)
    return {"metric": "rouge_l", "score": s}


def _normalize_code(code: str) -> str:
    lines = []
    for line in (code or "").splitlines():
        line = re.sub(r"#.*$", "", line)
        line = line.rstrip()
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def score_mbpp(pred: str, gold: Any) -> Dict[str, Any]:
    """Weak proxy: normalized code exact match against reference code."""
    gold_code = ""
    if isinstance(gold, dict):
        gold_code = gold.get("code") or ""
    elif gold is not None:
        gold_code = str(gold)
    # Prefer fenced python block if present
    m = re.search(r"```(?:python)?\n(.*?)```", pred, re.DOTALL | re.IGNORECASE)
    pred_code = m.group(1) if m else pred
    ok = _normalize_code(pred_code) == _normalize_code(gold_code) and bool(gold_code.strip())
    return {"metric": "code_em", "score": 1.0 if ok else 0.0}


def parse_generations_md(path: Path) -> List[Tuple[str, str]]:
    """Return list of (label, generated_text) in file order.

    Supports legacy (single ```text``` block) and new format with
    **Prompt:** / **Response:** fenced blocks.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"(?m)^### Prompt\s+\d+\s*$", text)
    out: List[Tuple[str, str]] = []
    for sec in sections[1:]:
        label_m = re.search(r"\*\*Label:\*\*\s*(.+)", sec)
        label = label_m.group(1).strip() if label_m else ""
        resp_m = re.search(r"\*\*Response:\*\*\s*\n```text\n(.*?)```", sec, re.DOTALL)
        if resp_m:
            gen = resp_m.group(1).strip()
        else:
            # Legacy: last/only fenced text block
            blocks = re.findall(r"```text\n(.*?)```", sec, re.DOTALL)
            gen = blocks[-1].strip() if blocks else ""
        out.append((label, gen))
    return out


def parse_transcripts_jsonl(path: Path) -> List[Tuple[str, str]]:
    """Prefer structured transcripts.jsonl when present: (label, response)."""
    if not path.exists():
        return []
    out: List[Tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((rec.get("label") or "", rec.get("response") or ""))
    return out


def score_examples(
    examples: List[Dict[str, Any]],
    generations: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Score each generation against the corresponding example (by ordinal).

    Generations.md writes one block per prompt per config; with cold-per-prompt
    and multiple configs, blocks are grouped by config (all prompts for config A,
    then all for B, …). We detect config boundaries via Label changes and map
    each block to example idx % n within that config group.
    """
    n = len(examples)
    if n == 0:
        return []

    results: List[Dict[str, Any]] = []
    prompt_ordinal = 0
    prev_label: Optional[str] = None

    for label, gen in generations:
        if prev_label is not None and label != prev_label:
            prompt_ordinal = 0
        prev_label = label
        ex = examples[prompt_ordinal % n]
        prompt_ordinal += 1

        metric = ex.get("metric") or "none"
        gold = ex.get("gold")
        domain = ex.get("domain")
        idx = ex.get("idx", prompt_ordinal - 1)

        if metric == "exact_match":
            scored = score_gsm8k(gen, gold)
        elif metric == "rouge_l":
            scored = score_cnn(gen, gold)
        elif metric == "code_em":
            scored = score_mbpp(gen, gold)
        elif metric == "gen_ppl":
            scored = {"metric": "gen_ppl", "score": None, "note": "see sweep CSV prompt_gen_perplexity"}
        else:
            scored = {"metric": metric, "score": None}

        results.append(
            {
                "domain": domain,
                "idx": idx,
                "label": label,
                "metric": scored.get("metric"),
                "score": scored.get("score"),
                "detail": {k: v for k, v in scored.items() if k not in ("metric", "score")},
            }
        )
    return results


def aggregate_by_label(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean score per config label (skip None scores)."""
    buckets: Dict[str, List[float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("score") is None:
            continue
        label = r.get("label") or ""
        buckets.setdefault(label, []).append(float(r["score"]))
        meta[label] = {"domain": r.get("domain"), "metric": r.get("metric")}
    out = []
    for label, scores in buckets.items():
        out.append(
            {
                "label": label,
                "mode": classify_mode(label),
                "domain": meta[label]["domain"],
                "metric": meta[label]["metric"],
                "correctness_mean": sum(scores) / len(scores),
                "n": len(scores),
            }
        )
    return out


def load_sweep_rows(sweep_csv: Path) -> List[Dict[str, str]]:
    if not sweep_csv.exists():
        return []
    with sweep_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def enrich_gen_ppl_from_sweep(
    aggregates: List[Dict[str, Any]],
    sweep_rows: List[Dict[str, str]],
    domain: Optional[str],
    metric: Optional[str],
) -> List[Dict[str, Any]]:
    """For gen_ppl domains, use prompt_gen_perplexity from sweep.csv as the score.

    Lower is better; stored as correctness_mean with metric=gen_ppl.
    """
    if metric != "gen_ppl" or not sweep_rows:
        return aggregates
    by_label = {a["label"]: a for a in aggregates}
    for row in sweep_rows:
        label = row.get("label") or ""
        ppl_s = row.get("prompt_gen_perplexity") or ""
        if not label or not ppl_s:
            continue
        try:
            ppl = float(ppl_s)
        except ValueError:
            continue
        if label in by_label:
            by_label[label]["correctness_mean"] = ppl
            by_label[label]["metric"] = "gen_ppl"
            by_label[label]["mode"] = classify_mode(label)
            by_label[label]["n"] = by_label[label].get("n") or 1
        else:
            by_label[label] = {
                "label": label,
                "mode": classify_mode(label),
                "domain": domain,
                "metric": "gen_ppl",
                "correctness_mean": ppl,
                "n": 1,
                "note": "from sweep.csv prompt_gen_perplexity",
            }
    return list(by_label.values())


def aggregate_by_mode(aggregates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean correctness / degradation across labels within each mode bucket."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for a in aggregates:
        if a.get("correctness_mean") is None and a.get("match_to_normal") is None:
            continue
        mode = a.get("mode") or classify_mode(a.get("label") or "")
        buckets.setdefault(mode, []).append(a)
    out = []
    for mode, rows in buckets.items():
        scores = [float(r["correctness_mean"]) for r in rows if r.get("correctness_mean") is not None]
        matches = [float(r["match_to_normal"]) for r in rows if r.get("match_to_normal") is not None]
        deltas = [float(r["delta_vs_normal"]) for r in rows if r.get("delta_vs_normal") is not None]
        degradations = [float(r["degradation_vs_normal"]) for r in rows if r.get("degradation_vs_normal") is not None]
        entry: Dict[str, Any] = {
            "mode": mode,
            "domain": rows[0].get("domain"),
            "metric": rows[0].get("metric"),
            "n_configs": len(rows),
        }
        if scores:
            entry["correctness_mean"] = sum(scores) / len(scores)
        if matches:
            entry["match_to_normal"] = sum(matches) / len(matches)
        if deltas:
            entry["delta_vs_normal"] = sum(deltas) / len(deltas)
        if degradations:
            entry["degradation_vs_normal"] = sum(degradations) / len(degradations)
        out.append(entry)
    return out


def _higher_is_better(metric: Optional[str]) -> bool:
    return metric != "gen_ppl"


def annotate_vs_normal(
    aggregates: List[Dict[str, Any]],
    per_prompt: List[Dict[str, Any]],
    generations: List[Tuple[str, str]],
    normal_responses: List[str],
    metric: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Attach match_to_normal / delta / degradation relative to normal operation.

    degradation_vs_normal:
      - EM/ROUGE: max(0, normal_score - mode_score)  (drop in quality)
      - gen_ppl:  max(0, mode_ppl - normal_ppl)       (rise in perplexity)
    match_to_normal: fraction of responses identical to normal (same prompt index).
    """
    if not normal_responses:
        return aggregates, None

    n = len(normal_responses)
    by_label_resps = responses_by_config(generations)

    # Gold score of the normal reference (mean over prompts that have scores).
    normal_gold_scores = [
        float(r["score"])
        for r in per_prompt
        if classify_mode(r.get("label") or "") in ("normal", "random") and r.get("score") is not None
    ]
    # If normal transcripts are external, score them via aggregates marked normal/random
    # or leave None and only report match rate.
    normal_score: Optional[float] = None
    for a in aggregates:
        if a.get("mode") in ("normal", "random") and a.get("correctness_mean") is not None:
            normal_score = float(a["correctness_mean"])
            break
    if normal_score is None and normal_gold_scores:
        normal_score = sum(normal_gold_scores) / len(normal_gold_scores)

    # Also allow normal_score from a dedicated aggregate injected by caller.
    for a in aggregates:
        label = a.get("label") or ""
        resps = by_label_resps.get(label, [])
        if not resps:
            continue
        matches = 0
        compared = 0
        for i, resp in enumerate(resps):
            if i >= n:
                break
            compared += 1
            if (resp or "").strip() == (normal_responses[i] or "").strip():
                matches += 1
        if compared:
            a["match_to_normal"] = matches / compared
            a["n_vs_normal"] = compared
        if normal_score is not None and a.get("correctness_mean") is not None:
            delta = float(a["correctness_mean"]) - normal_score
            a["delta_vs_normal"] = delta
            if _higher_is_better(metric):
                a["degradation_vs_normal"] = max(0.0, -delta)
            else:
                a["degradation_vs_normal"] = max(0.0, delta)
        a["normal_correctness"] = normal_score

    return aggregates, normal_score


def merge_into_sweep_csv(sweep_csv: Path, aggregates: List[Dict[str, Any]]) -> None:
    if not sweep_csv.exists() or not aggregates:
        return

    with sweep_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in ("domain", "mode", "correctness_metric", "correctness_mean",
                "match_to_normal", "degradation_vs_normal", "delta_vs_normal"):
        if col not in fieldnames:
            fieldnames.append(col)

    by_label = {a["label"]: a for a in aggregates}
    for row in rows:
        label = row.get("label", "")
        row["mode"] = classify_mode(label)
        agg = by_label.get(label)
        if not agg:
            if aggregates:
                row.setdefault("domain", aggregates[0].get("domain"))
                row.setdefault("correctness_metric", aggregates[0].get("metric"))
            continue
        row["domain"] = agg.get("domain")
        row["correctness_metric"] = agg.get("metric")
        if agg.get("correctness_mean") is not None:
            row["correctness_mean"] = f"{agg['correctness_mean']:.6f}"
        if agg.get("match_to_normal") is not None:
            row["match_to_normal"] = f"{agg['match_to_normal']:.6f}"
        if agg.get("degradation_vs_normal") is not None:
            row["degradation_vs_normal"] = f"{agg['degradation_vs_normal']:.6f}"
        if agg.get("delta_vs_normal") is not None:
            row["delta_vs_normal"] = f"{agg['delta_vs_normal']:.6f}"

    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_score(metric: Optional[str], value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if metric == "gen_ppl":
        return f"{value:.4f} (↓ better)"
    if metric in ("exact_match", "code_em"):
        return f"{value:.1%} EM"
    if metric == "rouge_l":
        return f"{value:.4f} ROUGE-L"
    return f"{value:.4f}"


def write_comparison_md(
    path: Path,
    *,
    domain: Optional[str],
    metric: Optional[str],
    by_mode: List[Dict[str, Any]],
    aggregates: List[Dict[str, Any]],
    normal_score: Optional[float] = None,
    normal_cache: Optional[int] = None,
) -> None:
    """Human-readable degradation vs normal (RANDOM C=64) + gold scores."""
    mode_order = ("normal", "random", "prefetch_only", "both", "lru", "cache_cond", "other")
    by_m = {m["mode"]: m for m in by_mode}
    nc = f"C={normal_cache}" if normal_cache else "C=64"
    lines = [
        f"# Quality vs normal operation — `{domain}`",
        "",
        f"**Normal** = RANDOM backend @ {nc} (fast, high-cache “full” operation).",
        "Report degradation of Prefetch Only / Both relative to that baseline.",
        "",
        f"- Gold metric: `{metric}`",
        f"- Normal gold score: {_fmt_score(metric, normal_score)}",
        "- `match_to_normal` = fraction of responses identical to normal",
        "- `degradation` = quality drop vs normal (0 = no worse; EM/ROUGE ↓ or gen-PPL ↑)",
        "",
        "## Degradation by mode",
        "",
        "| Mode | Score | vs normal | match_to_normal | degradation | #configs |",
        "|------|-------|-----------|-----------------|-------------|----------|",
    ]
    for mode in mode_order:
        m = by_m.get(mode)
        if not m:
            continue
        deg = m.get("degradation_vs_normal")
        match = m.get("match_to_normal")
        delta = m.get("delta_vs_normal")
        deg_s = "n/a" if deg is None else _fmt_score(metric, deg)
        match_s = "n/a" if match is None else f"{match:.0%}"
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"| {mode} | {_fmt_score(metric, m.get('correctness_mean'))} | {delta_s} | "
            f"{match_s} | {deg_s} | {m.get('n_configs')} |"
        )
    lines.extend(
        [
            "",
            "## By label",
            "",
            "| Mode | Label | Score | match | degradation | n |",
            "|------|-------|-------|-------|-------------|---|",
        ]
    )
    for a in sorted(
        aggregates,
        key=lambda x: (
            mode_order.index(x.get("mode") or "other")
            if (x.get("mode") or "other") in mode_order
            else 99,
            x.get("label") or "",
        ),
    ):
        match = a.get("match_to_normal")
        deg = a.get("degradation_vs_normal")
        match_s = "n/a" if match is None else f"{match:.0%}"
        deg_s = "n/a" if deg is None else _fmt_score(a.get("metric") or metric, deg)
        lines.append(
            f"| {a.get('mode')} | {a.get('label')} | "
            f"{_fmt_score(a.get('metric') or metric, a.get('correctness_mean'))} | "
            f"{match_s} | {deg_s} | {a.get('n')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_normal_reference(
    examples: List[Dict[str, Any]],
    normal_responses: List[str],
    domain: Optional[str],
    metric: Optional[str],
) -> Dict[str, Any]:
    """Score external normal responses against gold; return a synthetic aggregate row."""
    fake_gens = [("Normal (RANDOM C=64)", r) for r in normal_responses]
    rows = score_examples(examples, fake_gens)
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    agg: Dict[str, Any] = {
        "label": "Normal (RANDOM C=64)",
        "mode": "normal",
        "domain": domain,
        "metric": metric,
        "n": len(normal_responses),
        "match_to_normal": 1.0,
        "degradation_vs_normal": 0.0,
        "delta_vs_normal": 0.0,
    }
    if scores:
        agg["correctness_mean"] = sum(scores) / len(scores)
        agg["normal_correctness"] = agg["correctness_mean"]
    elif metric == "gen_ppl":
        agg["correctness_mean"] = None
    return agg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True, help="Domain examples JSON")
    p.add_argument("--generations", required=True, help="generations.md from sweep")
    p.add_argument("--sweep-csv", default=None, help="Optional sweep.csv to annotate")
    p.add_argument(
        "--normal-transcripts",
        default=None,
        help="transcripts.jsonl (or generations.md) from NORMAL=RANDOM@C64 baseline",
    )
    p.add_argument(
        "--normal-cache",
        type=int,
        default=64,
        help="Cache size used for the normal baseline (for reporting)",
    )
    p.add_argument("--out", required=True, help="Output correctness JSON")
    args = p.parse_args()

    examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))
    domain = examples[0]["domain"] if examples else None
    metric = examples[0].get("metric") if examples else None

    gen_path = Path(args.generations)
    jsonl_path = gen_path.parent / "transcripts.jsonl"
    if jsonl_path.exists():
        generations = parse_transcripts_jsonl(jsonl_path)
        print(f"[eval_correctness] using {jsonl_path} ({len(generations)} records)", flush=True)
    else:
        generations = parse_generations_md(gen_path)
    per_prompt = score_examples(examples, generations)
    for r in per_prompt:
        r["mode"] = classify_mode(r.get("label") or "")
    aggregates = aggregate_by_label(per_prompt)

    sweep_rows: List[Dict[str, str]] = []
    if args.sweep_csv:
        sweep_rows = load_sweep_rows(Path(args.sweep_csv))
        aggregates = enrich_gen_ppl_from_sweep(aggregates, sweep_rows, domain, metric)

    normal_path = Path(args.normal_transcripts) if args.normal_transcripts else None
    # Auto-detect sibling normal dir if not passed
    if normal_path is None:
        parent = gen_path.parent.parent  # domain/
        candidates = sorted(parent.glob("normal_C*/transcripts.jsonl"))
        if candidates:
            normal_path = candidates[-1]
            print(f"[eval_correctness] auto normal={normal_path}", flush=True)

    normal_responses = load_normal_responses(normal_path)
    normal_score: Optional[float] = None
    if normal_responses:
        normal_agg = score_normal_reference(examples, normal_responses, domain, metric)
        # Prefer gen_ppl from normal sweep.csv if available
        if metric == "gen_ppl" and normal_path is not None:
            normal_csv = normal_path.parent / "sweep.csv"
            if normal_csv.exists():
                n_rows = load_sweep_rows(normal_csv)
                enriched = enrich_gen_ppl_from_sweep([dict(normal_agg)], n_rows, domain, metric)
                if enriched and enriched[0].get("correctness_mean") is not None:
                    normal_agg = enriched[0]
                    normal_agg["mode"] = "normal"
                    normal_agg["label"] = "Normal (RANDOM C=64)"
                    normal_agg["match_to_normal"] = 1.0
                    normal_agg["degradation_vs_normal"] = 0.0
                    normal_agg["delta_vs_normal"] = 0.0
        if not any(a.get("mode") == "normal" for a in aggregates):
            aggregates.insert(0, normal_agg)
        normal_score = normal_agg.get("correctness_mean")
        aggregates, normal_score = annotate_vs_normal(
            aggregates, per_prompt, generations, normal_responses, metric
        )
        # Re-stamp normal row
        for a in aggregates:
            if a.get("mode") == "normal":
                a["match_to_normal"] = 1.0
                a["degradation_vs_normal"] = 0.0
                a["delta_vs_normal"] = 0.0
                if normal_score is not None:
                    a["correctness_mean"] = normal_score
                    a["normal_correctness"] = normal_score

    by_mode = aggregate_by_mode(aggregates)

    payload = {
        "domain": domain,
        "metric": metric,
        "n_examples": len(examples),
        "n_generation_blocks": len(generations),
        "goal": "degradation vs normal operation (RANDOM @ C=64)",
        "normal_cache": args.normal_cache,
        "normal_transcripts": str(normal_path) if normal_path else None,
        "normal_correctness": normal_score,
        "per_prompt": per_prompt,
        "by_label": aggregates,
        "by_mode": by_mode,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[eval_correctness] wrote {out} ({len(aggregates)} label aggregates)", flush=True)

    comparison_path = out.parent / "comparison.md"
    write_comparison_md(
        comparison_path,
        domain=domain,
        metric=metric,
        by_mode=by_mode,
        aggregates=aggregates,
        normal_score=normal_score,
        normal_cache=args.normal_cache,
    )
    print(f"[eval_correctness] wrote {comparison_path}", flush=True)

    if args.sweep_csv:
        merge_into_sweep_csv(Path(args.sweep_csv), aggregates)
        print(f"[eval_correctness] annotated {args.sweep_csv}", flush=True)

    print("[eval_correctness] degradation vs normal (RANDOM C=64):", flush=True)
    mode_order = ("normal", "prefetch_only", "both", "random")
    by_m = {m["mode"]: m for m in by_mode}
    for mode in mode_order:
        m = by_m.get(mode)
        if not m:
            continue
        deg = m.get("degradation_vs_normal")
        match = m.get("match_to_normal")
        print(
            f"  {mode}: score={_fmt_score(metric, m.get('correctness_mean'))}  "
            f"match={('n/a' if match is None else f'{match:.0%}')}  "
            f"degradation={('n/a' if deg is None else _fmt_score(metric, deg))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
