from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def ece(probs: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = 15) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(sel):
            continue
        acc = np.mean((pred[sel] == y[sel]).astype(np.float64))
        avg_conf = np.mean(conf[sel])
        ece_val += (np.sum(sel) / N) * abs(acc - avg_conf)
    return float(ece_val)


def brier_score(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    K = p.shape[1]
    y = np.zeros_like(p)
    y[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def nll(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

