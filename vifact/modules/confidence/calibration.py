from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - m)
    s = np.sum(exps, axis=1, keepdims=True) + 1e-12
    return exps / s


def _to_numpy(arr: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    # Negative log likelihood
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(p + 1e-12)))


@dataclass
class TemperatureScaler:
    T: float = 1.0

    def calibrate(self, logits: List[float]) -> List[float]:
        arr = np.asarray([logits], dtype=np.float64)
        scaled = arr / max(self.T, 1e-6)
        return _softmax_np(scaled)[0].tolist()

    def calibrate_batch(self, logits: Sequence[Sequence[float]]) -> np.ndarray:
        arr = _to_numpy(logits) / max(self.T, 1e-6)
        return _softmax_np(arr)

    def fit(self, logits: Sequence[Sequence[float]], labels: Sequence[int], *, grid: Sequence[float] | None = None) -> float:
        """Fit temperature by minimizing NLL over a simple grid search.

        Args:
            logits: shape (N, C)
            labels: shape (N,)
            grid: optional temperature grid to search
        Returns:
            best temperature
        """
        x = _to_numpy(logits)
        y = np.asarray(labels, dtype=np.int64)
        if grid is None:
            grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        best_T, best_nll = self.T, float("inf")
        for T in grid:
            probs = _softmax_np(x / max(T, 1e-6))
            val = _nll(probs, y)
            if val < best_nll:
                best_nll = val
                best_T = T
        self.T = float(best_T)
        return self.T

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"T": self.T}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(T=float(obj.get("T", 1.0)))
