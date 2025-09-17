#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# Allow running from scripts/ by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.retrieval import BM25Index


def main():
    p = argparse.ArgumentParser(description="Build BM25 index from a corpus JSONL")
    p.add_argument("--corpus", required=True, help="Path to corpus JSONL (doc_id,text)")
    p.add_argument("--output", required=True, help="Path to save BM25 index (e.g., models/bm25.pkl.gz)")
    args = p.parse_args()

    idx = BM25Index.from_corpus_jsonl(args.corpus)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.output)
    print(f"Index built: N={idx.N}, avgdl={idx.avgdl:.2f} -> {args.output}")


if __name__ == "__main__":
    main()
