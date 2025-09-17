#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List

# Allow running from scripts/ by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.retrieval import BM25Index
from vifact.modules.retrieval.bi_encoder import DenseIndex
from vifact.modules.retrieval.hybrid import hybrid_search


def base_id(doc_id: str) -> str:
    return str(doc_id).split("#", 1)[0]


def recall_at_k(ranked: List[str], target: str, k: int) -> float:
    tgt = base_id(target)
    ranked_base = [base_id(d) for d in ranked[:k]]
    return 1.0 if tgt in ranked_base else 0.0


def mrr(ranked: List[str], target: str) -> float:
    tgt = base_id(target)
    for i, did in enumerate(ranked, 1):
        if base_id(did) == tgt:
            return 1.0 / i
    return 0.0


def iter_examples(shards_dir: Path):
    for shard in sorted(shards_dir.glob("*.jsonl")):
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval over processed shards against retrieval indexes")
    p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--index", help="Path to BM25 index .pkl.gz (bm25/hybrid) OR dense index .npz (dense, for convenience)")
    p.add_argument("--dense-index", help="Path to dense index .npz (dense/hybrid)")
    p.add_argument("--data", required=True, help="Processed shards directory")
    p.add_argument("--k", type=int, default=10)
    # Hybrid tuning
    p.add_argument("--k-sparse", type=int, default=100)
    p.add_argument("--k-dense", type=int, default=100)
    p.add_argument("--w-sparse", type=float, default=0.6)
    p.add_argument("--w-dense", type=float, default=0.4)
    p.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking on a larger fused pool")
    p.add_argument("--rerank-pool", type=int, default=50)
    p.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--ce-batch-size", type=int, default=64)
    p.add_argument("--ce-device", default=None, help="Force CE device: cuda|cpu (auto if None)")
    p.add_argument("--ce-max-length", type=int, default=256, help="Max seq length for CE (reduce for speed)")
    p.add_argument("--max-ctx-chars", type=int, default=800, help="Truncate doc text before CE scoring for speed")
    p.add_argument("--alpha-ce", type=float, default=0.5, help="Blend weight for CE score (0..1). If omitted, pure CE sort when --rerank")
    p.add_argument("--no-norm", action="store_true", help="Disable min-max normalization before blending scores")
    p.add_argument("--limit", type=int, default=0, help="Limit examples for quick eval (0 = all)")
    args = p.parse_args()

    # Argument validation and convenience fallback
    if args.mode in ("bm25", "hybrid") and not args.index:
        p.error("--index is required for bm25/hybrid modes")
    if args.mode == "dense":
        # Allow using --index for dense for convenience
        if not args.dense_index and args.index:
            args.dense_index = args.index
        if not args.dense_index:
            p.error("--dense-index (or --index pointing to .npz) is required for dense mode")

    bm25 = BM25Index.load(args.index) if args.mode in ("bm25", "hybrid") and args.index else None
    dense = DenseIndex.load(args.dense_index) if args.mode in ("dense", "hybrid") and args.dense_index else None

    n = 0
    r_sum = 0.0
    mrr_sum = 0.0
    from vifact.modules.retrieval.cross_encoder import CrossEncoderReranker
    from vifact.modules.retrieval.hybrid import hybrid_search

    # Create CE reranker if requested
    if args.rerank:
        dev = args.ce_device
        # Auto-detect cuda
        if dev is None:
            try:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = None
        cross = CrossEncoderReranker(args.cross_model, batch_size=args.ce_batch_size, device=dev, max_length=args.ce_max_length)
    else:
        cross = None

    for ex in iter_examples(Path(args.data)):
        n += 1
        claim = ex.get("claim", "")
        target = str(ex.get("id", ex.get("example_id", "")))
        if args.mode == "bm25":
            results = bm25.search(claim, top_k=args.k) if bm25 else []
            ranked_ids = [doc_id for doc_id, score in results]
        elif args.mode == "dense":
            results = dense.search(claim, top_k=args.k) if dense else []
            ranked_ids = [doc_id for doc_id, score in results]
        else:
            fused = hybrid_search(
                claim,
                bm25,
                dense,
                k_sparse=args.k_sparse,
                k_dense=args.k_dense,
                k_final=args.k,
                rerank_pool=args.rerank_pool if args.rerank else None,
                w_sparse=args.w_sparse,
                w_dense=args.w_dense,
                cross=cross,
                alpha_ce=(args.alpha_ce if args.rerank else None),
                normalize=(not args.no_norm),
                max_chars=args.max_ctx_chars if args.rerank else None,
            )
            ranked_ids = [d["doc_id"] for d in fused]
        r_sum += recall_at_k(ranked_ids, target, args.k)
        mrr_sum += mrr(ranked_ids, target)
        if args.limit and n >= args.limit:
            break

    if n == 0:
        print("No examples found.")
        return
    print({
        "n": n,
        f"recall@{args.k}": r_sum / n,
        "mrr": mrr_sum / n,
    })


if __name__ == "__main__":
    main()
