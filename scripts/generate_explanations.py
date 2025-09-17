#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.retrieval import BM25Index
from vifact.modules.retrieval.bi_encoder import DenseIndex
from vifact.modules.retrieval.hybrid import hybrid_search
from vifact.modules.explainer.generator import generate_explanation
from vifact.modules.rationale.extractor import extract_rationale


def load_rationales(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = str(obj.get("id", obj.get("example_id", "")))
            rat = obj.get("rationale")
            if rid and isinstance(rat, str):
                out[rid] = rat
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate explanations using predicted rationales and hybrid retrieval sources")
    ap.add_argument("--data", required=True, help="Processed shards directory (jsonl)")
    ap.add_argument("--out", required=True, help="Output JSONL of explanations")
    ap.add_argument("--bm25-index", required=True, help="Path to BM25 index .pkl.gz")
    ap.add_argument("--dense-index", help="Path to Dense index .npz (optional)")
    ap.add_argument("--rationales", help="Predicted rationales JSONL {id,rationale} (optional)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k-sparse", type=int, default=150)
    ap.add_argument("--k-dense", type=int, default=150)
    ap.add_argument("--w-sparse", type=float, default=0.7)
    ap.add_argument("--w-dense", type=float, default=0.3)
    args = ap.parse_args()

    bm25 = BM25Index.load(args.bm25_index)
    dense = DenseIndex.load(args.dense_index) if args.dense_index else None
    rat_map = load_rationales(args.rationales)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f_out:
        for shard in sorted(Path(args.data).glob("*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    ex_id = str(ex.get("id", ex.get("example_id", "")))
                    claim = ex.get("claim") or ""
                    verdict = ex.get("verdict", "NEI")
                    # Retrieval
                    fused = hybrid_search(
                        claim,
                        bm25,
                        dense,
                        k_sparse=args.k_sparse,
                        k_dense=args.k_dense,
                        k_final=args.k,
                        w_sparse=args.w_sparse,
                        w_dense=args.w_dense,
                        cross=None,
                    )
                    sources = [
                        {"doc_id": d.get("doc_id"), "text": d.get("text", ""), "score": float(d.get("score", 0.0))}
                        for d in fused
                    ]
                    # Rationale
                    rationale = rat_map.get(ex_id)
                    if not rationale:
                        rationale = extract_rationale(ex, max_sentences=2)
                    # Explanation
                    explanation = generate_explanation(claim, verdict, rationale, sources)
                    out = {
                        "id": ex_id,
                        "claim": claim,
                        "domain": ex.get("domain"),
                        "verdict": verdict,
                        "rationale": rationale,
                        "sources": sources,
                        "explanation": explanation,
                    }
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote explanations to {args.out}")


if __name__ == "__main__":
    main()

