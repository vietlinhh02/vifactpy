#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from vifact.modules.confidence.calibration import TemperatureScaler
from vifact.modules.confidence.metrics import ece, brier_score, nll


def read_preds(path: str, label_order: List[str]):
    probs = []
    logits = []
    labels = []
    domains = []
    lab2idx = {l: i for i, l in enumerate(label_order)}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            y = obj.get("label")
            if y is None:
                continue
            labels.append(lab2idx.get(y, -1))
            domains.append(obj.get("domain"))
            if "logits" in obj:
                logits.append(obj["logits"])
            elif "probs" in obj:
                probs.append(obj["probs"])
            else:
                continue
    labels = [i for i in labels if i >= 0]
    if logits:
        logits = np.asarray(logits, dtype=np.float64)
        m = np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits - m) / (np.sum(np.exp(logits - m), axis=1, keepdims=True) + 1e-12)
    else:
        probs = np.asarray(probs, dtype=np.float64)
        logits = np.log(probs + 1e-12)
    return logits, probs, np.asarray(labels, dtype=np.int64), domains


def main():
    ap = argparse.ArgumentParser(description="Calibrate logits/probs with temperature scaling and report calibration metrics")
    ap.add_argument("--preds", required=True, help="JSONL with fields: id,label,domain,logits|probs")
    ap.add_argument("--out", required=True, help="Output JSON path for calibration params and metrics")
    ap.add_argument("--labels", nargs="+", default=["SUPPORTED", "REFUTED", "NEI"], help="Label order mapping to indices")
    args = ap.parse_args()

    logits, probs, labels, domains = read_preds(args.preds, args.labels)

    pre = {
        "ece": ece(probs, labels),
        "brier": brier_score(probs, labels),
        "nll": nll(probs, labels),
    }

    ts = TemperatureScaler()
    T = ts.fit(logits, labels)
    cal_probs = ts.calibrate_batch(logits)
    post = {
        "ece": ece(cal_probs, labels),
        "brier": brier_score(cal_probs, labels),
        "nll": nll(cal_probs, labels),
    }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"temperature": T, "metrics_pre": pre, "metrics_post": post}, f, ensure_ascii=False, indent=2)
    print({"temperature": T, **{f"pre_{k}": v for k, v in pre.items()}, **{f"post_{k}": v for k, v in post.items()}})


if __name__ == "__main__":
    main()

