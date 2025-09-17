from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Thresholds:
    global_threshold: float
    per_domain: Dict[str, float]

    def for_domain(self, domain: Optional[str]) -> float:
        if domain and domain in self.per_domain:
            return self.per_domain[domain]
        return self.global_threshold


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    from collections import Counter

    y_true = list(y_true)
    y_pred = list(y_pred)
    f1s: List[float] = []
    for c in range(n_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def tune_threshold(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    domains: Optional[Sequence[str]] = None,
    nei_index: int = 2,
    metric: str = "f1",
    grid: Optional[Sequence[float]] = None,
) -> Thresholds:
    """Tune threshold for abstaining to NEI when confidence is low.

    Rules:
      - If argmax == NEI -> keep NEI
      - Else if max_prob < thr(domain) -> force NEI
      - Else -> keep argmax
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    dom = list(domains) if domains is not None else [None] * len(y)
    if grid is None:
        grid = np.linspace(0.3, 0.9, 25)

    def eval_for_thr(thr: float, mask: np.ndarray) -> float:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return -1.0
        pm = p[idx]
        ym = y[idx]
        pred = pm.argmax(axis=1)
        conf = pm.max(axis=1)
        pred = np.where((pred != nei_index) & (conf < thr), nei_index, pred)
        if metric == "f1":
            return macro_f1(ym, pred, p.shape[1])
        else:
            # accuracy
            return float(np.mean((ym == pred).astype(np.float64)))

    # Global threshold
    best_g, best_g_val = 0.5, -1.0
    mask_all = np.ones(len(y), dtype=bool)
    for t in grid:
        v = eval_for_thr(float(t), mask_all)
        if v > best_g_val:
            best_g_val = v
            best_g = float(t)

    # Per-domain
    per_dom: Dict[str, float] = {}
    if domains is not None:
        uniq = sorted(set(dom))
        for d in uniq:
            m = np.array([dd == d for dd in dom], dtype=bool)
            best_d, best_d_val = best_g, -1.0
            for t in grid:
                v = eval_for_thr(float(t), m)
                if v > best_d_val:
                    best_d_val = v
                    best_d = float(t)
            if d is not None:
                per_dom[d] = best_d

    return Thresholds(global_threshold=best_g, per_domain=per_dom)

