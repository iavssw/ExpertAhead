"""
Set-based precision / recall for expert-cache prediction.

Two selection policies:
  topk    — fixed cache: top-K experts by logit score
  thresh  — variable cache: experts with score >= threshold (sigmoid or raw logit)
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch


def _scores(logits: torch.Tensor, score_mode: Literal["sigmoid", "logit"] = "sigmoid") -> torch.Tensor:
    return torch.sigmoid(logits) if score_mode == "sigmoid" else logits


def predict_topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    """[B, E] multi-hot with exactly k ones per row (or fewer if E < k)."""
    k = min(k, logits.size(-1))
    idx = logits.topk(k, dim=-1).indices
    pred = torch.zeros_like(logits)
    pred.scatter_(1, idx, 1.0)
    return pred


def predict_threshold_mask(
    logits: torch.Tensor,
    threshold: float,
    score_mode: Literal["sigmoid", "logit"] = "sigmoid",
    min_preds: int = 1,
    max_preds: int | None = None,
) -> torch.Tensor:
    """[B, E] multi-hot; at least min_preds (top-1 fallback), optionally cap at max_preds."""
    scores = _scores(logits, score_mode)
    pred = (scores >= threshold).float()
    empty = pred.sum(dim=1) == 0
    if empty.any():
        top1 = logits.topk(1, dim=-1).indices
        pred[empty] = 0.0
        pred[empty].scatter_(1, top1[empty], 1.0)
    if min_preds > 1:
        need = (pred.sum(dim=1) < min_preds)
        if need.any():
            k = min(min_preds, logits.size(-1))
            idx = logits.topk(k, dim=-1).indices
            extra = torch.zeros_like(pred)
            extra.scatter_(1, idx, 1.0)
            pred = torch.where(need.unsqueeze(1), torch.maximum(pred, extra), pred)
    if max_preds is not None and max_preds < logits.size(-1):
        for b in range(pred.size(0)):
            n = int(pred[b].sum().item())
            if n > max_preds:
                keep = logits[b].topk(max_preds).indices
                new = torch.zeros_like(pred[b])
                new[keep] = 1.0
                pred[b] = new
    return pred


def batch_set_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int | None = None,
    threshold: float | None = None,
    score_mode: Literal["sigmoid", "logit"] = "sigmoid",
    max_preds: int | None = None,
) -> Tuple[float, float, float, float, float]:
    """
    Returns (precision, recall, f1, avg_pred_size, avg_true_size) aggregated over batch.
    """
    if k is not None:
        pred = predict_topk_mask(logits, k)
    elif threshold is not None:
        pred = predict_threshold_mask(logits, threshold, score_mode=score_mode, max_preds=max_preds)
    else:
        raise ValueError("Specify k or threshold")

    tp = (pred * target).sum().item()
    pred_sz = pred.sum().clamp(min=1).item()
    true_sz = target.sum().clamp(min=1).item()
    prec = tp / pred_sz
    rec = tp / true_sz
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    B = target.size(0)
    avg_pred = pred.sum(dim=1).mean().item()
    avg_true = target.sum(dim=1).mean().item()
    return prec, rec, f1, avg_pred, avg_true


def _macro_mean_prf(
    precs: list[float], recs: list[float], f1s: list[float]
) -> Tuple[float, float, float]:
    n = len(precs)
    if n == 0:
        return 0.0, 0.0, 0.0
    return float(sum(precs) / n), float(sum(recs) / n), float(sum(f1s) / n)


def _per_prompt_prf(hits: float, pred_sz: int, true_sz: int) -> Tuple[float, float, float]:
    pred_sz = max(pred_sz, 1)
    true_sz = max(true_sz, 1)
    prec = hits / pred_sz
    rec = hits / true_sz
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def macro_topk_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> Tuple[float, float, float, int]:
    """Macro-average of per-prompt precision/recall/F1 at fixed top-K."""
    B, E = logits.shape
    if B == 0:
        return 0.0, 0.0, 0.0, 0
    k = min(k, E)
    precs, recs, f1s = [], [], []
    for i in range(B):
        idx = logits[i].topk(k, dim=-1).indices
        hits = target[i, idx].sum().item()
        true_sz = int(target[i].sum().item())
        p, r, f = _per_prompt_prf(hits, k, true_sz)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
    p, r, f = _macro_mean_prf(precs, recs, f1s)
    return p, r, f, B


def macro_threshold_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    score_mode: Literal["sigmoid", "logit"] = "sigmoid",
    max_preds: int | None = None,
) -> Tuple[float, float, float, int]:
    """Macro-average of per-prompt precision/recall/F1 for threshold-selected sets."""
    pred = predict_threshold_mask(
        logits, threshold, score_mode=score_mode, max_preds=max_preds
    )
    B = target.size(0)
    if B == 0:
        return 0.0, 0.0, 0.0, 0
    precs, recs, f1s = [], [], []
    for i in range(B):
        hits = (pred[i] * target[i]).sum().item()
        pred_sz = int(pred[i].sum().item())
        true_sz = int(target[i].sum().item())
        p, r, f = _per_prompt_prf(hits, pred_sz, true_sz)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
    p, r, f = _macro_mean_prf(precs, recs, f1s)
    return p, r, f, B


def _macro_prf_from_hits(
    hits: torch.Tensor, pred_sz: torch.Tensor, true_sz: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row precision, recall, F1; macro = mean over batch later."""
    pred_sz = pred_sz.clamp(min=1)
    true_sz = true_sz.clamp(min=1)
    prec = hits / pred_sz
    rec = hits / true_sz
    f1 = (2 * prec * rec) / (prec + rec).clamp(min=1e-9)
    return prec, rec, f1


def macro_eval_batch_sums(
    logits: torch.Tensor,
    target: torch.Tensor,
    eval_topks: list[int],
    thresholds: list[float] | None = None,
    *,
    thresh_score: Literal["sigmoid", "logit"] = "sigmoid",
    thresh_max_preds: int | None = None,
) -> dict:
    """
    One forward's logits → all metrics. Single top-k over max K; fixed-K and
    adaptive-K differ only in pred_sz / true_sz (denominators) and which prefix
    of the ranked list is used.
    """
    B, E = logits.shape
    if B == 0:
        return {"n": 0}

    true_sz = target.sum(dim=1)
    true_sz_denom = true_sz.clamp(min=1)
    max_k = min(
        E,
        max(
            [1] + [int(k) for k in eval_topks]
            + [int(true_sz.max().item()) if B else 1],
        ),
    )

    idx = logits.topk(max_k, dim=-1).indices
    hits_in_rank = torch.gather(target, 1, idx)
    hits_prefix = hits_in_rank.cumsum(dim=1)

    out: dict = {"n": B}

    # Adaptive: K_i = |union_i|, top-K_i hits / |union_i|
    ki = true_sz.long().clamp(min=1, max=max_k) - 1
    rows = torch.arange(B, device=logits.device)
    h_ad = hits_prefix[rows, ki]
    p, r, f = _macro_prf_from_hits(h_ad, true_sz_denom, true_sz_denom)
    out["adaptive"] = (p.sum().item(), r.sum().item(), f.sum().item(), true_sz.sum().item())

    for pk in eval_topks:
        k = min(int(pk), max_k)
        h = hits_prefix[:, k - 1]
        p, r, f = _macro_prf_from_hits(
            h, torch.full((B,), float(k), device=logits.device), true_sz_denom
        )
        out[f"topk@{pk}"] = (p.sum().item(), r.sum().item(), f.sum().item())

    if thresholds:
        scores = _scores(logits, thresh_score)
        for thr in thresholds:
            p_t, r_t, f_t = _macro_threshold_batch_sums(
                logits, target, scores, float(thr), thresh_max_preds
            )
            out[f"thresh@{thr:g}"] = (p_t, r_t, f_t)

    return out


def _macro_threshold_batch_sums(
    logits: torch.Tensor,
    target: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    max_preds: int | None,
) -> Tuple[float, float, float]:
    B, E = logits.shape
    pred = (scores >= threshold).float()
    empty = pred.sum(dim=1) == 0
    if empty.any():
        top1 = logits.topk(1, dim=-1).indices
        pred[empty] = 0.0
        pred[empty].scatter_(1, top1[empty], 1.0)
    if max_preds is not None and max_preds < E:
        over = pred.sum(dim=1) > max_preds
        if over.any():
            for i in torch.nonzero(over, as_tuple=False).flatten().tolist():
                keep = logits[i].topk(max_preds).indices
                new = torch.zeros(E, device=logits.device)
                new[keep] = 1.0
                pred[i] = new
    hits = (pred * target).sum(dim=1)
    pred_sz = pred.sum(dim=1)
    true_sz = target.sum(dim=1)
    p, r, f = _macro_prf_from_hits(hits, pred_sz, true_sz)
    return p.sum().item(), r.sum().item(), f.sum().item()


def adaptive_union_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, float, float, float, int]:
    """
    Per prompt: K_i = |true union_i|; take top-K_i predictions.
    Macro-average per-prompt precision, recall, and F1 (not micro, not mean-K).
    """
    B, E = logits.shape
    if B == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    precs, recs, f1s = [], [], []
    union_sizes = []
    for i in range(B):
        k_i = int(target[i].sum().item())
        k_i = max(1, min(k_i, E))
        idx = logits[i].topk(k_i, dim=-1).indices
        hits = target[i, idx].sum().item()
        p, r, f = _per_prompt_prf(hits, k_i, k_i)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        union_sizes.append(float(k_i))

    p, r, f = _macro_mean_prf(precs, recs, f1s)
    return p, r, f, float(sum(union_sizes) / B), B


def accumulate_set_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int | None = None,
    threshold: float | None = None,
    score_mode: Literal["sigmoid", "logit"] = "sigmoid",
    max_preds: int | None = None,
) -> Tuple[float, float, float, float, float, int]:
    """Micro-averaged sums for epoch aggregation. Returns tp, pred_sz, true_sz, n, sum_pred, sum_true."""
    if k is not None:
        pred = predict_topk_mask(logits, k)
    else:
        pred = predict_threshold_mask(
            logits, threshold, score_mode=score_mode, max_preds=max_preds
        )
    tp = (pred * target).sum().item()
    pred_sz = pred.sum().item()
    true_sz = target.sum().item()
    n = target.size(0)
    return tp, pred_sz, true_sz, n, pred.sum(dim=1).sum().item(), target.sum(dim=1).sum().item()
