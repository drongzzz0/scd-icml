from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


def token_f1(pred: str, ref: str) -> float:
    p = pred.split()
    r = ref.split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def mean_f1(preds: Iterable[str], refs: Iterable[str]) -> float:
    vals = [token_f1(p, r) for p, r in zip(preds, refs)]
    return float(np.mean(vals)) if vals else 0.0


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q), axis=-1).mean())


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum(axis=-1).mean())


def perplexity_from_nll(nll: float) -> float:
    return float(math.exp(nll))


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)
