#!/usr/bin/env python3
"""Speed vs quality summary for run_quality_speed_eval.sh (all relative to RANDOM).

Per domain run dir (sweep.csv + transcripts.jsonl):
  - decode tokens/s and speedup vs RANDOM
  - reference perplexity (token-weighted) and relative change vs RANDOM
  - task score (gsm8k exact match, cnn_dailymail / orca ROUGE-L) and delta vs RANDOM
  - match_to_random: % of prompts whose response is identical to RANDOM's

Usage:
  python3 py/utils/eval_quality_speed.py --run-dir <out>/gsm8k/C64_<ts> [--examples by_domain/gsm8k.json]
  python3 py/utils/eval_quality_speed.py --summary-only --run-dirs <dir1> <dir2> ... --out-prefix <out>/summary_<ts>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_correctness import rouge_l_f1, score_gsm8k  # noqa: E402

# (key, display name) in report order.
METHODS: List[Tuple[str, str]] = [
    ("random", "RANDOM (baseline)"),
    ("prefetch_wikitext", "Prefetch only (WikiText predictor)"),
    ("prefetch_multi", "Prefetch only (multi-dataset predictor)"),
    ("cache_cond", "Cache-cond routing only"),
    ("gating", "Gating (cross-layer)"),
    ("hybrid_wikitext", "Hybrid (WikiText predictor)"),
    ("hybrid_multi", "Hybrid (multi-dataset predictor)"),
    ("oracle", "Oracle full union"),
]
METHOD_NAMES = dict(METHODS)

TASK_METRIC = {"gsm8k": "exact_match", "cnn_dailymail": "rouge_l", "orca": "rouge_l"}


def method_key(label: str, tag: Optional[str]) -> Optional[str]:
    tag = (tag or "").strip()
    if label.startswith("Neither (RANDOM)"):
        return "random"
    if label.startswith("Gating"):
        return "gating"
    if label.startswith("Cache-Cond"):
        return "cache_cond"
    if label.startswith("Oracle Full Union"):
        return "oracle"
    if label.startswith("Prefetch"):
        return f"prefetch_{tag}" if tag else "prefetch"
    if label.startswith("Both"):
        return f"hybrid_{tag}" if tag else "hybrid"
    return None


def _float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def load_sweep(path: Path) -> Dict[str, Dict[str, Any]]:
    """Last row with data per method."""
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = method_key(r.get("label", ""), r.get("predictor_tag"))
            if key and _float(r.get("tokens_per_second")) is not None:
                rows[key] = r
    return rows


def load_transcripts(path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """method -> prompt_ordinal -> record (last occurrence wins)."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = method_key(rec.get("label", ""), rec.get("predictor_tag"))
            if key:
                out.setdefault(key, {})[int(rec["prompt_ordinal"])] = rec
    return out


def task_score(domain: str, response: str, gold: Any) -> Optional[float]:
    metric = TASK_METRIC.get(domain)
    if metric is None or gold is None:
        return None
    if metric == "exact_match":
        return score_gsm8k(response, gold)["score"]
    return rouge_l_f1(response, str(gold))


def token_weighted_ppl(records: List[Dict[str, Any]]) -> Optional[float]:
    nll = toks = 0.0
    for r in records:
        ppl, n = _float(r.get("ref_perplexity")), _float(r.get("ref_tokens"))
        if ppl and n:
            nll += math.log(ppl) * n
            toks += n
    return math.exp(nll / toks) if toks else None


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def evaluate_run(run_dir: Path, examples: Optional[Path]) -> List[Dict[str, Any]]:
    sweep = load_sweep(run_dir / "sweep.csv")
    tx = load_transcripts(run_dir / "transcripts.jsonl")
    gold_by_ordinal: Dict[int, Any] = {}
    domain = None
    if examples and examples.exists():
        exs = json.loads(examples.read_text(encoding="utf-8"))
        gold_by_ordinal = {i + 1: e.get("gold") for i, e in enumerate(exs)}
        domain = exs[0].get("domain") if exs else None
    if domain is None:
        for recs in tx.values():
            for r in recs.values():
                domain = domain or r.get("domain")
    domain = domain or run_dir.parent.name

    random_resp = {o: r.get("response", "") for o, r in tx.get("random", {}).items()}

    rows: List[Dict[str, Any]] = []
    for key, name in METHODS:
        recs = tx.get(key, {})
        srow = sweep.get(key)
        if not recs and not srow:
            continue
        ordinals = sorted(recs)
        scores = [task_score(domain, recs[o].get("response", ""),
                             gold_by_ordinal.get(o, recs[o].get("gold"))) for o in ordinals]
        matches = [recs[o].get("response", "") == random_resp[o] for o in ordinals if o in random_resp]
        tps = _float(srow.get("tokens_per_second")) if srow else None
        if tps is None:
            tps = _mean([_float(recs[o].get("tokens_per_second")) for o in ordinals])
        ref_ppl = _float(srow.get("ref_perplexity")) if srow else None
        if ref_ppl is None:
            ref_ppl = token_weighted_ppl([recs[o] for o in ordinals])
        rows.append({
            "domain": domain,
            "method": key,
            "method_name": name,
            "label": srow.get("label") if srow else recs[ordinals[0]].get("label"),
            "n_prompts": len(ordinals),
            "tokens_per_second": tps,
            "hit_rate_pct": _float(srow.get("hit_rate_pct")) if srow else None,
            "ref_perplexity": ref_ppl,
            "task_metric": TASK_METRIC.get(domain),
            "task_score": _mean(scores),
            "match_to_random_pct": (100.0 * sum(matches) / len(matches)) if matches else None,
        })

    base = next((r for r in rows if r["method"] == "random"), None)
    for r in rows:
        r["speedup"] = (r["tokens_per_second"] / base["tokens_per_second"]
                        if base and base["tokens_per_second"] and r["tokens_per_second"] else None)
        r["ref_ppl_change_pct"] = (100.0 * (r["ref_perplexity"] / base["ref_perplexity"] - 1.0)
                                   if base and base["ref_perplexity"] and r["ref_perplexity"] else None)
        if base and base["task_score"] is not None and r["task_score"] is not None:
            r["task_delta"] = r["task_score"] - base["task_score"]
            r["task_change_pct"] = (100.0 * r["task_delta"] / base["task_score"]
                                    if base["task_score"] else None)
        else:
            r["task_delta"] = r["task_change_pct"] = None
    return rows


FIELDS = ["domain", "method", "method_name", "label", "n_prompts", "tokens_per_second", "speedup",
          "hit_rate_pct", "ref_perplexity", "ref_ppl_change_pct", "task_metric", "task_score",
          "task_delta", "task_change_pct", "match_to_random_pct"]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            if k not in ("domain", "method", "method_name", "label", "task_metric"):
                r[k] = _float(r.get(k))
    return rows


def _f(v: Optional[float], fmt: str, suffix: str = "") -> str:
    return "—" if v is None else f"{v:{fmt}}{suffix}"


def _signed(v: Optional[float], fmt: str, suffix: str = "") -> str:
    return "—" if v is None else f"{v:+{fmt}}{suffix}"


def domain_md(rows: List[Dict[str, Any]]) -> str:
    domain = rows[0]["domain"] if rows else "?"
    metric = rows[0].get("task_metric") if rows else None
    lines = [f"## {domain}", ""]
    head = "| Method | tok/s | Speedup | Ref PPL | ΔPPL vs RANDOM |"
    sep = "|---|---:|---:|---:|---:|"
    if metric:
        head += f" {metric} | Δ{metric} (abs) | Δ{metric} (rel) |"
        sep += "---:|---:|---:|"
    head += " Same output as RANDOM |"
    sep += "---:|"
    lines += [head, sep]
    for r in rows:
        line = (f"| {r['method_name']} | {_f(r['tokens_per_second'], '.2f')} | "
                f"{_f(r['speedup'], '.2f', '×')} | {_f(r['ref_perplexity'], '.3f')} | "
                f"{_signed(r['ref_ppl_change_pct'], '.2f', '%')} |")
        if metric:
            line += (f" {_f(r['task_score'], '.3f')} | {_signed(r['task_delta'], '.3f')} | "
                     f"{_signed(r['task_change_pct'], '.1f', '%')} |")
        line += f" {_f(r['match_to_random_pct'], '.0f', '%')} |"
        lines.append(line)
    n = max((int(r["n_prompts"] or 0) for r in rows), default=0)
    lines += ["", f"n={n} prompts. Reference PPL = teacher-forced NLL of held-out/gold text given the "
              "prompt, under each method's routing (↓ better).", ""]
    return "\n".join(lines)


def summary_md(all_rows: List[Dict[str, Any]]) -> str:
    domains = list(dict.fromkeys(r["domain"] for r in all_rows))
    by = {(r["domain"], r["method"]): r for r in all_rows}
    present = [(k, n) for k, n in METHODS if any((d, k) in by for d in domains)]

    lines = ["# Speed vs quality (relative to RANDOM)", "", "### Speedup (tok/s ÷ RANDOM tok/s)", ""]
    lines.append("| Method | " + " | ".join(domains) + " | Mean |")
    lines.append("|---|" + "---:|" * (len(domains) + 1))
    for k, n in present:
        vals = [by.get((d, k), {}).get("speedup") for d in domains]
        lines.append(f"| {n} | " + " | ".join(_f(v, ".2f", "×") for v in vals)
                     + f" | {_f(_mean(vals), '.2f', '×')} |")

    lines += ["", "### Reference perplexity change vs RANDOM (↓ better; 0% = no degradation)", ""]
    lines.append("| Method | " + " | ".join(domains) + " | Mean |")
    lines.append("|---|" + "---:|" * (len(domains) + 1))
    for k, n in present:
        vals = [by.get((d, k), {}).get("ref_ppl_change_pct") for d in domains]
        lines.append(f"| {n} | " + " | ".join(_signed(v, ".2f", "%") for v in vals)
                     + f" | {_signed(_mean(vals), '.2f', '%')} |")

    task_domains = [d for d in domains if TASK_METRIC.get(d)]
    if task_domains:
        lines += ["", "### Task score (absolute, Δ vs RANDOM in parentheses)", ""]
        lines.append("| Method | " + " | ".join(f"{d} ({TASK_METRIC[d]})" for d in task_domains) + " |")
        lines.append("|---|" + "---:|" * len(task_domains))
        for k, n in present:
            cells = []
            for d in task_domains:
                r = by.get((d, k), {})
                cells.append("—" if r.get("task_score") is None else
                             f"{r['task_score']:.3f} ({_signed(r.get('task_delta'), '.3f')})")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += ["", "### Same output as RANDOM (% of prompts)", ""]
    lines.append("| Method | " + " | ".join(domains) + " |")
    lines.append("|---|" + "---:|" * len(domains))
    for k, n in present:
        vals = [by.get((d, k), {}).get("match_to_random_pct") for d in domains]
        lines.append(f"| {n} | " + " | ".join(_f(v, ".0f", "%") for v in vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--examples", type=Path, default=None)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--run-dirs", type=Path, nargs="*", default=[])
    p.add_argument("--out-prefix", type=Path, default=None)
    args = p.parse_args()

    if not args.summary_only:
        if not args.run_dir:
            p.error("--run-dir is required unless --summary-only")
        rows = evaluate_run(args.run_dir, args.examples)
        if not rows:
            print(f"[eval] no results in {args.run_dir}")
            return 1
        write_csv(args.run_dir / "summary.csv", rows)
        md = domain_md(rows)
        (args.run_dir / "summary.md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    all_rows: List[Dict[str, Any]] = []
    for d in args.run_dirs:
        s = d / "summary.csv"
        if s.exists():
            all_rows.extend(read_csv(s))
        elif (d / "sweep.csv").exists():
            all_rows.extend(evaluate_run(d, None))
        else:
            print(f"[eval] skip {d}: no summary.csv / sweep.csv")
    if not all_rows:
        print("[eval] nothing to summarize")
        return 1
    md = summary_md(all_rows) + "\n" + "\n".join(
        domain_md([r for r in all_rows if r["domain"] == dom])
        for dom in dict.fromkeys(r["domain"] for r in all_rows)
    )
    prefix = args.out_prefix or Path("summary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".csv"), all_rows)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
