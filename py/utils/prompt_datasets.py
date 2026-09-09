#!/usr/bin/env python3
"""Shared prompt sources for expert-predictor training and evaluation.

Train and eval never share examples:

- WikiText, GSM8K, MBPP, CNN/DailyMail: official ``train`` vs ``test`` splits
  (disjoint by construction).
- FineWeb-Edu and OpenOrca have no test split. Those streams are hash-partitioned
  so each example is exclusively train or eval.

Sources cover distinct routing regimes for MoE expert prediction:

  wikitext       encyclopedia language modeling
  fineweb        web documents
  orca           instruction following
  gsm8k          math word problems
  mbpp           Python coding problems
  cnn_dailymail  news articles
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

ExtractFn = Callable[[dict], Optional[str]]


def _strip(value) -> str:
    return (value or "").strip() if isinstance(value, str) else ""


def _text_field(example: dict) -> Optional[str]:
    text = _strip(example.get("text"))
    return text or None


def _wikitext_text(example: dict) -> Optional[str]:
    text = _strip(example.get("text"))
    if not text:
        return None
    if _is_wikitext_header(text):
        return None
    return text


def _orca_train_text(example: dict) -> Optional[str]:
    text = "\n".join(
        part for part in (
            _strip(example.get("system_prompt")),
            _strip(example.get("question")),
            _strip(example.get("response")),
        ) if part
    )
    return text or None


def _orca_eval_text(example: dict) -> Optional[str]:
    text = "\n".join(
        part for part in (
            _strip(example.get("system_prompt")),
            _strip(example.get("question")),
        ) if part
    )
    return text or None


def _question_field(example: dict) -> Optional[str]:
    text = _strip(example.get("question"))
    return text or None


def _mbpp_text(example: dict) -> Optional[str]:
    text = _strip(example.get("text")) or _strip(example.get("prompt"))
    return text or None


def _article_field(example: dict) -> Optional[str]:
    text = _strip(example.get("article"))
    return text or None


def _is_wikitext_header(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(" = "):
        return True
    return stripped.startswith("=") and stripped.endswith("=")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    hf_path: str
    hf_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    streaming: bool = False
    eval_holdout_fraction: Optional[float] = None
    min_chars: int = 80
    min_tokens_override: Optional[int] = None
    eval_packing: str = "document"  # "document" | "wikitext_chunks"
    trust_remote_code: bool = False
    default_train_n: int = 0
    extract_train: ExtractFn = _text_field
    extract_eval: ExtractFn = _text_field

    def __post_init__(self) -> None:
        if self.train_split == self.eval_split and not self.eval_holdout_fraction:
            raise ValueError(
                f"{self.name}: train and eval share split {self.train_split!r}; "
                "set eval_holdout_fraction so an example cannot appear in both."
            )

    @property
    def cli_name(self) -> str:
        return self.name.replace("_", "-")

    @property
    def num_attr(self) -> str:
        return f"num_{self.name}"

    @property
    def skip_attr(self) -> str:
        return f"skip_{self.name}"


DATASETS: Dict[str, DatasetSpec] = {
    "wikitext": DatasetSpec(
        name="wikitext",
        description="WikiText-103 encyclopedia language modeling",
        hf_path="wikitext",
        hf_config="wikitext-103-raw-v1",
        train_split="train",
        eval_split="test",
        streaming=True,
        eval_packing="wikitext_chunks",
        default_train_n=1000,
        extract_train=_wikitext_text,
        extract_eval=_wikitext_text,
    ),
    "fineweb": DatasetSpec(
        name="fineweb",
        description="FineWeb-Edu web documents",
        hf_path="HuggingFaceFW/fineweb-edu",
        train_split="train",
        eval_split="train",
        streaming=True,
        eval_holdout_fraction=0.1,
        min_chars=100,
        extract_train=_text_field,
        extract_eval=_text_field,
    ),
    "orca": DatasetSpec(
        name="orca",
        description="OpenOrca instruction following",
        hf_path="Open-Orca/OpenOrca",
        train_split="train",
        eval_split="train",
        streaming=True,
        eval_holdout_fraction=0.1,
        min_chars=40,
        extract_train=_orca_train_text,
        extract_eval=_orca_eval_text,
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        description="GSM8K grade-school math word problems",
        hf_path="gsm8k",
        hf_config="main",
        train_split="train",
        eval_split="test",
        min_chars=20,
        min_tokens_override=8,
        extract_train=_question_field,
        extract_eval=_question_field,
    ),
    "mbpp": DatasetSpec(
        name="mbpp",
        description="MBPP Python programming problems",
        hf_path="mbpp",
        hf_config="sanitized",
        train_split="train",
        eval_split="test",
        min_chars=20,
        min_tokens_override=16,
        extract_train=_mbpp_text,
        extract_eval=_mbpp_text,
    ),
    "cnn_dailymail": DatasetSpec(
        name="cnn_dailymail",
        description="CNN/DailyMail news articles",
        hf_path="cnn_dailymail",
        hf_config="3.0.0",
        train_split="train",
        eval_split="test",
        streaming=True,
        min_chars=100,
        extract_train=_article_field,
        extract_eval=_article_field,
    ),
}

DATASET_NAMES: Tuple[str, ...] = tuple(DATASETS.keys())
HF_EVAL_CHOICES: Tuple[str, ...] = DATASET_NAMES + ("all",)


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Known sources: {', '.join(DATASET_NAMES)}"
        )
    return DATASETS[name]


def _load_hf(spec: DatasetSpec, split: str, streaming: Optional[bool] = None):
    from datasets import load_dataset

    kwargs = {}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    use_stream = spec.streaming if streaming is None else streaming
    if spec.hf_config:
        return load_dataset(
            spec.hf_path, spec.hf_config, split=split, streaming=use_stream, **kwargs
        )
    return load_dataset(spec.hf_path, split=split, streaming=use_stream, **kwargs)


def _iter_examples(spec: DatasetSpec, split: str) -> Iterator[dict]:
    ds = _load_hf(spec, split)
    for example in ds:
        yield example


def _example_key(example: dict, spec: DatasetSpec) -> str:
    for key in ("id", "id_string", "uuid"):
        value = example.get(key)
        if value is not None and str(value).strip():
            return f"{spec.name}:{value}"
    text = spec.extract_eval(example) or spec.extract_train(example) or ""
    return f"{spec.name}:{text[:2048]}"


def is_eval_holdout(example: dict, spec: DatasetSpec) -> bool:
    """True if this example is reserved for eval on a hash-partitioned corpus.

    Official train/test splits are disjoint upstream; this is only used when
    ``train_split == eval_split``.
    """
    if spec.eval_holdout_fraction is None:
        return False
    key = _example_key(example, spec)
    digest = hashlib.md5(key.encode("utf-8", errors="replace")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < spec.eval_holdout_fraction


def iter_train_texts(
    dataset_name: str,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    stream_skip: int = 0,
) -> Iterator[Tuple[str, object]]:
    """Yield ``(text, input_ids)`` pairs from the train partition only.

    ``stream_skip`` skips that many train-partition examples that already passed
    the token-length filter, so resume collection continues from the same
    stream position.
    """
    spec = get_spec(dataset_name)
    effective_min = spec.min_tokens_override if spec.min_tokens_override is not None else min_tokens
    skipped = 0
    for example in _iter_examples(spec, spec.train_split):
        if is_eval_holdout(example, spec):
            continue
        text = spec.extract_train(example)
        if not text:
            continue
        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_tokens,
        )
        ids = enc["input_ids"]
        if ids.shape[1] < effective_min:
            continue
        if skipped < stream_skip:
            skipped += 1
            continue
        yield text, ids


def _pack_wikitext_chunks(texts: Iterator[str], n: int, max_chars: int) -> List[str]:
    blob_parts: List[str] = []
    for text in texts:
        blob_parts.append(text)
    blob = "\n\n".join(blob_parts)
    paragraphs = [p.strip() for p in blob.split("\n\n") if p.strip()]

    prompts: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) >= max_chars:
            if current:
                prompts.append(current[:max_chars])
                if len(prompts) >= n:
                    break
            while len(para) >= max_chars:
                prompts.append(para[:max_chars])
                para = para[max_chars:]
                if len(prompts) >= n:
                    break
            current = para
        else:
            current = candidate
        if len(prompts) >= n:
            break
    if current and len(prompts) < n:
        prompts.append(current[:max_chars])
    return prompts[:n]


def load_eval_prompts(
    dataset_name: str,
    n: int,
    max_chars: int = 4096,
    stream_skip: Optional[int] = None,
) -> List[str]:
    """Load up to ``n`` evaluation prompts from the eval partition only.

    ``dataset_name='all'`` takes ``n`` prompts from each source and concatenates.
    ``stream_skip`` skips that many eval-partition examples (never train).
    """
    if dataset_name == "all":
        prompts: List[str] = []
        for name in DATASET_NAMES:
            try:
                part = load_eval_prompts(name, n, max_chars=max_chars, stream_skip=stream_skip)
            except Exception as e:
                print(f"[prompt_datasets] {name} load failed: {e}", flush=True)
                continue
            print(f"[prompt_datasets] {name}: {len(part)} prompt(s)", flush=True)
            prompts.extend(part)
        return prompts

    spec = get_spec(dataset_name)
    skip = stream_skip or 0
    extractor = spec.extract_eval

    def raw_texts() -> Iterator[str]:
        seen = 0
        for example in _iter_examples(spec, spec.eval_split):
            if spec.eval_holdout_fraction is not None and not is_eval_holdout(example, spec):
                continue
            text = extractor(example)
            if not text or len(text) < spec.min_chars:
                continue
            if seen < skip:
                seen += 1
                continue
            yield text

    if spec.eval_packing == "wikitext_chunks":
        prompts = _pack_wikitext_chunks(raw_texts(), n, max_chars)
    else:
        prompts = []
        for text in raw_texts():
            prompts.append(text[:max_chars])
            if len(prompts) >= n:
                break

    holdout = (
        f"hash_holdout={spec.eval_holdout_fraction}"
        if spec.eval_holdout_fraction is not None
        else "official_split"
    )
    print(
        f"[prompt_datasets] Loaded {len(prompts)} {dataset_name} prompt(s) "
        f"(split={spec.eval_split}, {holdout}, skip={skip}, max_chars={max_chars})",
        flush=True,
    )
    return prompts


def filter_pt_files(files: Sequence, include_datasets: Optional[Sequence[str]] = None):
    """Keep ``{dataset}_*.pt`` files matching ``include_datasets``, or all files."""
    if not include_datasets:
        return list(files)
    unknown = [d for d in include_datasets if d not in DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}. Known: {', '.join(DATASET_NAMES)}"
        )
    prefixes = tuple(f"{d}_" for d in include_datasets)
    return [f for f in files if getattr(f, "name", str(f)).startswith(prefixes)]


def add_collection_args(parser: argparse.ArgumentParser) -> None:
    for spec in DATASETS.values():
        parser.add_argument(
            f"--num-{spec.cli_name}",
            dest=spec.num_attr,
            type=int,
            default=spec.default_train_n,
            help=f"Number of {spec.description} samples (default: {spec.default_train_n}).",
        )
        parser.add_argument(
            f"--skip-{spec.cli_name}",
            dest=spec.skip_attr,
            action="store_true",
            help=f"Skip {spec.name} collection.",
        )


def requested_collection_counts(args: argparse.Namespace) -> Dict[str, int]:
    counts = {}
    for spec in DATASETS.values():
        if getattr(args, spec.skip_attr, False):
            counts[spec.name] = 0
        else:
            counts[spec.name] = int(getattr(args, spec.num_attr, 0) or 0)
    return counts


def _self_check_holdout() -> None:
    """Hash partitions must be disjoint and near the configured fraction."""
    for spec in DATASETS.values():
        if spec.eval_holdout_fraction is None:
            if spec.train_split == spec.eval_split:
                raise AssertionError(
                    f"{spec.name}: official split must use different train/eval names"
                )
            continue
        train_ids = set()
        eval_ids = set()
        n = 10_000
        for i in range(n):
            example = {"id": i, "question": f"q{i}", "system_prompt": "", "response": "a", "text": f"t{i}"}
            if is_eval_holdout(example, spec):
                eval_ids.add(i)
            else:
                train_ids.add(i)
        overlap = train_ids & eval_ids
        if overlap:
            raise AssertionError(f"{spec.name} train/eval holdout overlap: {len(overlap)} ids")
        frac = len(eval_ids) / n
        if abs(frac - spec.eval_holdout_fraction) > 0.02:
            raise AssertionError(
                f"{spec.name} eval holdout fraction {frac:.3f} != {spec.eval_holdout_fraction}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or preview shared prompt datasets used for training and eval."
    )
    parser.add_argument("--list", action="store_true", help="Print catalog and exit.")
    parser.add_argument(
        "--preview", type=str, default=None,
        help="Dataset name to preview, or 'all'.",
    )
    parser.add_argument("--n", type=int, default=2, help="Prompts to print for --preview.")
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--role", choices=["eval", "train"], default="eval")
    args = parser.parse_args(argv)

    if args.list or not args.preview:
        _self_check_holdout()
        print("name            train/eval          partition         description")
        for spec in DATASETS.values():
            if spec.eval_holdout_fraction is not None:
                partition = f"hash {spec.eval_holdout_fraction:.0%} eval"
            else:
                partition = "official splits"
            print(
                f"{spec.name:<15} {spec.train_split}/{spec.eval_split:<6}  "
                f"{partition:<16} {spec.description}"
            )
        if not args.preview:
            return 0

    names = DATASET_NAMES if args.preview == "all" else (args.preview,)
    for name in names:
        print(f"\n===== {name} ({args.role}) =====")
        try:
            if args.role == "eval":
                prompts = load_eval_prompts(name, args.n, max_chars=args.max_chars)
                for i, p in enumerate(prompts, 1):
                    shown = p.replace("\n", " ")[: args.max_chars]
                    print(f"[{i}] ({len(p)} chars) {shown}")
            else:
                print("Train preview requires a tokenizer; use collect_training_data_unified.py.")
        except Exception as e:
            print(f"FAILED: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
