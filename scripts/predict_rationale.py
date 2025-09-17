#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.rationale.dataset import split_sentences


def main():
    ap = argparse.ArgumentParser(description="Predict top-k rationale sentences for each example")
    ap.add_argument("--data", required=True, help="Processed shards directory (jsonl)")
    ap.add_argument("--model", required=True, help="Path or name of trained rationale model")
    ap.add_argument("--out", required=True, help="Output JSONL path with id and rationale")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring sentences")
    ap.add_argument("--threshold", type=float, default=None, help="Probability threshold for selecting sentences (class 1)")
    ap.add_argument("--include-meta", action="store_true", help="Include sentences, selected indices and scores in output JSONL")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Determine positive class index robustly
    pos_idx = 1
    try:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == model.config.num_labels:
            # Prefer label containing '1' or starting with POS/TRUE
            for i, name in id2label.items():
                n = str(name).upper()
                if n in {"LABEL_1", "1"} or n.startswith("POS") or n.startswith("TRUE"):
                    pos_idx = int(i)
                    break
    except Exception:
        pass

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for shard in sorted(Path(args.data).glob("*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    ex_id = str(obj.get("id", obj.get("example_id", "")))
                    claim = obj.get("claim") or ""
                    context = obj.get("context") or ""
                    sents = split_sentences(context)
                    if not sents or not claim:
                        skipped += 1
                        continue
                    # Batch score sentences
                    scores: List[float] = []
                    with torch.no_grad():
                        bs = max(1, int(args.batch_size))
                        for start in range(0, len(sents), bs):
                            batch_sents = sents[start : start + bs]
                            enc = tok([claim] * len(batch_sents), batch_sents, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
                            out = model(**enc)
                            logits = out.logits  # [B, num_labels]
                            if args.threshold is None:
                                # Use positive logit for ranking
                                batch_scores = logits[:, pos_idx].detach().cpu().tolist()
                            else:
                                # Convert to probability for thresholding
                                probs = F.softmax(logits, dim=-1)[:, pos_idx]
                                batch_scores = probs.detach().cpu().tolist()
                            scores.extend(batch_scores)

                    order = np.argsort(scores)[::-1]
                    if args.threshold is None:
                        sel_idx = order[: args.topk]
                    else:
                        # Select all meeting threshold, then cap at topk
                        passing = [i for i in order if scores[i] >= args.threshold]
                        sel_idx = passing[: args.topk]
                    top_sents = [sents[i] for i in sel_idx]
                    rationale = " ".join(top_sents)

                    out_obj = {"id": ex_id, "rationale": rationale}
                    if args.include_meta:
                        out_obj.update({
                            "sentences": sents,
                            "selected_indices": list(map(int, sel_idx)),
                            "scores": scores,
                        })
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    processed += 1
    print(f"Wrote rationales to {args.out} (processed={processed}, skipped={skipped})")


if __name__ == "__main__":
    main()
