#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.rationale.dataset import PairExample, generate_pairs
from vifact.modules.rationale.model import PairDataset, RationaleClassifier, RationaleConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_pairs_from_shards(data_dir: str, *, max_pos: int, max_neg: int, pos_threshold: float, limit: int | None = None) -> List[PairExample]:
    pairs: List[PairExample] = []
    n = 0
    for shard in sorted(Path(data_dir).glob("*.jsonl")):
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ps = generate_pairs(obj, max_pos=max_pos, max_neg=max_neg, pos_threshold=pos_threshold)
                if ps:
                    pairs.extend(ps)
                n += 1
                if limit and n >= limit:
                    break
        if limit and n >= limit:
            break
    random.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Train rationale (sentence-level) classifier as cross-encoder on GPU/CPU")
    ap.add_argument("--data", required=True, help="Processed shards directory (jsonl)")
    ap.add_argument("--output", required=True, help="Output dir to save model")
    ap.add_argument("--model-name", default="xlm-roberta-base")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--max-neg", type=int, default=4)
    ap.add_argument("--pos-threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of examples to sample pairs from (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--early-stopping", type=int, default=0, help="Patience for early stopping (0=disabled)")
    ap.add_argument("--balance", action="store_true", help="Oversample positives to balance classes in train set")
    ap.add_argument("--no-fp16", action="store_true", help="Force disable fp16 mixed precision")
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    limit = args.limit if args.limit and args.limit > 0 else None
    pairs = load_pairs_from_shards(args.data, max_pos=args.max_pos, max_neg=args.max_neg, pos_threshold=args.pos_threshold, limit=limit)
    n = len(pairs)
    if n == 0:
        print("No training pairs found. Ensure 'context' and 'evidence' exist in data.")
        return

    # Stratified split by label for stability
    pos_pairs = [p for p in pairs if int(getattr(p, "label", 0)) == 1]
    neg_pairs = [p for p in pairs if int(getattr(p, "label", 0)) == 0]
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    pos_split = int(0.9 * len(pos_pairs))
    neg_split = int(0.9 * len(neg_pairs))
    train_pairs = pos_pairs[:pos_split] + neg_pairs[:neg_split]
    eval_pairs = pos_pairs[pos_split:] + neg_pairs[neg_split:]
    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)

    # Optional oversampling to balance train set
    if args.balance:
        n_pos = sum(1 for p in train_pairs if p.label == 1)
        n_neg = sum(1 for p in train_pairs if p.label == 0)
        if n_pos and n_neg:
            if n_pos < n_neg:
                gap = n_neg - n_pos
                pos_only = [p for p in train_pairs if p.label == 1]
                if pos_only:
                    train_pairs.extend(random.choices(pos_only, k=gap))
            elif n_neg < n_pos:
                gap = n_pos - n_neg
                neg_only = [p for p in train_pairs if p.label == 0]
                if neg_only:
                    train_pairs.extend(random.choices(neg_only, k=gap))
        random.shuffle(train_pairs)

    cfg = RationaleConfig(model_name=args.model_name, max_length=args.max_length)
    clf = RationaleClassifier(cfg)
    tok = clf.tokenizer
    train_ds = PairDataset(tok, train_pairs, cfg.max_length)
    eval_ds = PairDataset(tok, eval_pairs, cfg.max_length)

    # Metrics for evaluation
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    use_fp16 = torch.cuda.is_available() and (not args.no_fp16)
    pad_to_multiple_of = 8 if use_fp16 else None
    early_patience = args.early_stopping if args.early_stopping and args.early_stopping > 0 else None

    trainer = clf.get_trainer(
        train_ds,
        eval_ds,
        output_dir=args.output,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wd=args.wd,
        fp16=use_fp16,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        compute_metrics=compute_metrics,
        early_stopping_patience=early_patience,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"Saved trained rationale model to {args.output}")


if __name__ == "__main__":
    main()
