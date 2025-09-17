#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path

# Allow running from scripts/ by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.retrieval.bi_encoder import DenseConfig, DenseIndex


def main():
    p = argparse.ArgumentParser(description="Build dense (bi-encoder) index from corpus JSONL")
    p.add_argument("--corpus", required=True, help="Path to corpus JSONL (doc_id,text)")
    p.add_argument("--output", required=True, help="Path to save dense index (e.g., models/dense_index.npz)")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cfg = DenseConfig(model_name=args.model, batch_size=args.batch_size)
    idx = DenseIndex(cfg)

    def gen():
        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield obj["doc_id"], obj["text"]

    idx.fit(gen())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.output)
    print(f"Dense index built: N={len(idx.doc_ids)} -> {args.output}")


if __name__ == "__main__":
    main()
