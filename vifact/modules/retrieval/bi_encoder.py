from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Sequence, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n


def _try_load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name)
    except Exception:
        return None


@dataclass
class DenseConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64


class DenseIndex:
    def __init__(self, cfg: DenseConfig | None = None):
        self.cfg = cfg or DenseConfig()
        self.model = None
        self.doc_ids: List[str] = []
        self.emb: np.ndarray | None = None

    def _ensure_model(self):
        if self.model is None:
            self.model = _try_load_model(self.cfg.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        if self.model is None:
            # Fallback cheap embedding: bag of char-code means
            arr = np.array([[float(sum(map(ord, t))) / max(len(t), 1)] for t in texts], dtype=np.float32)
            return _l2_normalize(np.pad(arr, ((0, 0), (0, 383)), mode="constant") if arr.shape[1] < 384 else arr)
        vecs = self.model.encode(list(texts), batch_size=self.cfg.batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def fit(self, docs: Iterable[Tuple[str, str]]):
        ids: List[str] = []
        texts: List[str] = []
        for did, txt in docs:
            ids.append(did)
            texts.append(txt)
        self.doc_ids = ids
        self.emb = self.encode(texts)

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, emb=self.emb, ids=np.array(self.doc_ids, dtype=object))

    @classmethod
    def load(cls, path: str | Path, cfg: DenseConfig | None = None) -> "DenseIndex":
        data = np.load(path, allow_pickle=True)
        obj = cls(cfg)
        obj.emb = data["emb"]
        obj.doc_ids = list(data["ids"].tolist())
        return obj

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.emb is None or not self.doc_ids:
            return []
        q = self.encode([query])
        sims = (q @ self.emb.T)[0]
        idx = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]

