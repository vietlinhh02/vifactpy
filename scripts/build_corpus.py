#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Set

# Allow running from scripts/ by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.data.loader import iter_vifact_json
from vifact.data.preprocess import normalize_text


SENT_SPLIT = re.compile(r"(?<=[.!?\t\n\r])\s+")


def make_passages(text: str, sentences: int, overlap: int) -> List[str]:
    if sentences <= 0:
        return [text]
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return [text]
    win = sentences
    step = max(1, sentences - overlap)
    out: List[str] = []
    for i in range(0, len(sents), step):
        chunk = " ".join(sents[i : i + win]).strip()
        if chunk:
            out.append(chunk)
        if i + win >= len(sents):
            break
    return out


def main():
    p = argparse.ArgumentParser(description="Build corpus from processed/raw data")
    p.add_argument("--input", required=True, help="Directory with processed shards (*.jsonl) or a raw JSON file")
    p.add_argument("--output", required=True, help="Output JSONL path for corpus docs")
    p.add_argument("--min-len", type=int, default=20, help="Minimum characters for a document")
    p.add_argument("--passage-sentences", type=int, default=0, help="If >0, split context/evidence into passages of N sentences")
    p.add_argument("--overlap", type=int, default=1, help="Sentences overlap between passages (when splitting)")
    p.add_argument("--include-claims", action="store_true", help="Also include claim text as separate docs (doc_id '#claim') when context/evidence missing")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    doc_count = 0
    stats = {"ctx": 0, "ev": 0, "claim": 0}

    def write_doc(fh, doc_id: str, text: str):
        nonlocal doc_count
        text_n = normalize_text(text)
        if len(text_n) < args.min_len:
            return
        h = hashlib.md5(text_n.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        fh.write(json.dumps({"doc_id": doc_id, "text": text_n}, ensure_ascii=False) + "\n")
        doc_count += 1

    with open(out, "w", encoding="utf-8") as f_out:
        if inp.is_dir():
            for shard in sorted(inp.glob("*.jsonl")):
                with open(shard, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        base_id = str(obj.get("id", obj.get("example_id", "")))
                        wrote_any = False
                        ctx = obj.get("context")
                        if ctx:
                            if args.passage_sentences > 0:
                                passages = make_passages(ctx, args.passage_sentences, args.overlap)
                                for pi, pv in enumerate(passages):
                                    write_doc(f_out, f"{base_id}#p{pi}", pv)
                                    stats["ctx"] += 1
                            else:
                                write_doc(f_out, base_id, ctx)
                                stats["ctx"] += 1
                            wrote_any = True
                        ev = obj.get("evidence")
                        if isinstance(ev, str):
                            if args.passage_sentences > 0:
                                passages = make_passages(ev, args.passage_sentences, args.overlap)
                                for pi, pv in enumerate(passages):
                                    write_doc(f_out, f"{base_id}#ev{pi}", pv)
                                    stats["ev"] += 1
                            else:
                                write_doc(f_out, f"{base_id}#ev", ev)
                                stats["ev"] += 1
                            wrote_any = True
                        if not wrote_any and args.include_claims:
                            claim = obj.get("claim")
                            if isinstance(claim, str) and claim.strip():
                                write_doc(f_out, f"{base_id}#claim", claim)
                                stats["claim"] += 1
        else:
            # raw JSON/JSONL
            for ex_id, ex in iter_vifact_json(inp):
                base_id = str(ex_id)
                wrote_any = False
                if ex.get("context"):
                    if args.passage_sentences > 0:
                        for pi, pv in enumerate(make_passages(ex["context"], args.passage_sentences, args.overlap)):
                            write_doc(f_out, f"{base_id}#p{pi}", pv)
                            stats["ctx"] += 1
                    else:
                        write_doc(f_out, base_id, ex["context"])
                        stats["ctx"] += 1
                    wrote_any = True
                if isinstance(ex.get("evidence"), str):
                    if args.passage_sentences > 0:
                        for pi, pv in enumerate(make_passages(ex["evidence"], args.passage_sentences, args.overlap)):
                            write_doc(f_out, f"{base_id}#ev{pi}", pv)
                            stats["ev"] += 1
                    else:
                        write_doc(f_out, f"{base_id}#ev", ex["evidence"])
                        stats["ev"] += 1
                    wrote_any = True
                if not wrote_any and args.include_claims:
                    claim = ex.get("claim")
                    if isinstance(claim, str) and claim.strip():
                        write_doc(f_out, f"{base_id}#claim", claim)
                        stats["claim"] += 1

    print(f"Wrote {doc_count} documents to {out}")
    print(f"Breakdown: ctx={stats['ctx']} ev={stats['ev']} claim={stats['claim']}")


if __name__ == "__main__":
    main()
