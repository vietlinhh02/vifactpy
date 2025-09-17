from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import json
import numpy as np

from .calibration import TemperatureScaler
from .threshold import Thresholds


def load_calibration(path: Optional[str]) -> Optional[TemperatureScaler]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        T = float(obj.get("temperature", 1.0))
        return TemperatureScaler(T=T)
    except Exception:
        return None


def load_thresholds(path: Optional[str]) -> Optional[Thresholds]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Thresholds(global_threshold=float(obj.get("global_threshold", 0.5)), per_domain=obj.get("per_domain", {}))


def probs_from_logits_or_probs(
    logits: Optional[Sequence[Sequence[float]]], probs: Optional[Sequence[Sequence[float]]], calibrator: Optional[TemperatureScaler]
) -> np.ndarray:
    if logits is not None and len(logits) > 0:
        arr = np.asarray(logits, dtype=np.float64)
        if calibrator is not None:
            return calibrator.calibrate_batch(arr)
        # softmax
        m = np.max(arr, axis=1, keepdims=True)
        exps = np.exp(arr - m)
        s = np.sum(exps, axis=1, keepdims=True) + 1e-12
        return exps / s
    elif probs is not None and len(probs) > 0:
        parr = np.asarray(probs, dtype=np.float64)
        return parr
    else:
        return np.zeros((0, 0), dtype=np.float64)


def apply_thresholds(
    probs: np.ndarray, domains: Sequence[Optional[str]], thresholds: Optional[Thresholds], nei_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: (pred_raw, conf_raw, pred_final)
    """
    pred_raw = probs.argmax(axis=1)
    conf_raw = probs.max(axis=1)
    if thresholds is None:
        return pred_raw, conf_raw, pred_raw
    pred_final = pred_raw.copy()
    for i, (y, c) in enumerate(zip(pred_raw, conf_raw)):
        if y != nei_index:
            thr = thresholds.for_domain(domains[i] if i < len(domains) else None)
            if c < thr:
                pred_final[i] = nei_index
    return pred_raw, conf_raw, pred_final

