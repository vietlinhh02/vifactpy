from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional


def _load_cross_encoder(model_name: str, device: Optional[str] = None):
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        kw = {}
        if device is not None:
            kw["device"] = device
        return CrossEncoder(model_name, **kw)
    except Exception:
        return None


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 64, device: Optional[str] = None, max_length: Optional[int] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = _load_cross_encoder(model_name, device=device)
        if self.model is not None and max_length is not None:
            # sentence-transformers CrossEncoder exposes max_length
            try:
                self.model.max_length = int(max_length)
            except Exception:
                pass

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        if self.model is None:
            # Fallback heuristic: token overlap length
            qset = set(query.lower().split())
            return [float(len(qset & set(d.lower().split()))) for d in docs]
        pairs = [(query, d) for d in docs]
        return list(map(float, self.model.predict(pairs, batch_size=self.batch_size)))
