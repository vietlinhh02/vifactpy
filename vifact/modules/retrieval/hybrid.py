from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .bm25 import BM25Index
from .bi_encoder import DenseIndex
from .cross_encoder import CrossEncoderReranker


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def hybrid_search(
    query: str,
    bm25: Optional[BM25Index],
    dense: Optional[DenseIndex],
    *,
    k_sparse: int = 50,
    k_dense: int = 50,
    k_final: int = 10,
    rerank_pool: Optional[int] = None,
    w_sparse: float = 0.5,
    w_dense: float = 0.5,
    cross: Optional[CrossEncoderReranker] = None,
    alpha_ce: Optional[float] = None,
    normalize: bool = True,
    max_chars: Optional[int] = None,
) -> List[Dict]:
    pool: Dict[str, Dict] = {}

    if bm25 is not None:
        sres = bm25.search(query, top_k=k_sparse)
        if sres:
            scores = _minmax([s for _, s in sres])
            for (doc_id, _), ns in zip(sres, scores):
                pool.setdefault(doc_id, {"doc_id": doc_id, "text": bm25.get_text(doc_id), "score_sparse": 0.0, "score_dense": 0.0})
                pool[doc_id]["score_sparse"] = ns

    if dense is not None:
        dres = dense.search(query, top_k=k_dense)
        if dres:
            scores = _minmax([s for _, s in dres])
            for (doc_id, _), ns in zip(dres, scores):
                # Try to fill text from BM25 index if available
                text_val = bm25.get_text(doc_id) if bm25 is not None else ""
                pool.setdefault(doc_id, {"doc_id": doc_id, "text": text_val, "score_sparse": 0.0, "score_dense": 0.0})
                pool[doc_id]["score_dense"] = ns

    # Weighted fusion
    fused: List[Dict] = []
    for doc in pool.values():
        score = w_sparse * doc.get("score_sparse", 0.0) + w_dense * doc.get("score_dense", 0.0)
        fused.append({**doc, "score": score})
    fused.sort(key=lambda x: x["score"], reverse=True)
    pool_size = rerank_pool or k_final
    fused = fused[: max(pool_size, 0)]

    if cross is not None and fused:
        texts = [d.get("text", "") for d in fused]
        if max_chars:
            texts = [t[:max_chars] for t in texts]
        ce_scores = cross.score(query, texts)
        # Attach raw ce score
        for d, s in zip(fused, ce_scores):
            d["score_ce"] = float(s)

        if alpha_ce is None:
            # Default: pure CE ranking
            fused.sort(key=lambda x: x.get("score_ce", 0.0), reverse=True)
        else:
            # Blend fused score and CE score
            fused_vals = [d.get("score", 0.0) for d in fused]
            ce_vals = [d.get("score_ce", 0.0) for d in fused]
            if normalize:
                fused_norm = _minmax(fused_vals)
                ce_norm = _minmax(ce_vals)
            else:
                fused_norm = fused_vals
                ce_norm = ce_vals
            for d, fn, cn in zip(fused, fused_norm, ce_norm):
                d["score_final"] = (1.0 - alpha_ce) * float(fn) + alpha_ce * float(cn)
            fused.sort(key=lambda x: x.get("score_final", x.get("score_ce", 0.0)), reverse=True)

    return fused[:k_final]
