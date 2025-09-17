#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.confidence.calibration import TemperatureScaler
from vifact.modules.confidence.threshold import Thresholds, tune_threshold


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
            if y is None or y not in lab2idx:
                continue
            labels.append(lab2idx[y])
            domains.append(obj.get("domain"))
            if "logits" in obj:
                logits.append(obj["logits"])
            elif "probs" in obj:
                probs.append(obj["probs"])
    if logits:
        logits = np.asarray(logits, dtype=np.float64)
        m = np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits - m) / (np.sum(np.exp(logits - m), axis=1, keepdims=True) + 1e-12)
    else:
        probs = np.asarray(probs, dtype=np.float64)
    return probs, np.asarray(labels, dtype=np.int64), domains


def main():
    ap = argparse.ArgumentParser(description="Tune adaptive confidence thresholds for abstaining to NEI")
    ap.add_argument("--preds", required=True, help="JSONL with fields: id,label,domain,logits|probs")
    ap.add_argument("--out", required=True, help="Output JSON path for thresholds")
    ap.add_argument("--labels", nargs="+", default=["SUPPORTED", "REFUTED", "NEI"], help="Label order mapping to indices")
    ap.add_argument("--nei-label", default="NEI")
    ap.add_argument("--metric", choices=["f1", "accuracy"], default="f1")
    ap.add_argument("--calibration-json", help="Optional temperature file from confidence_calibrate.py to apply before tuning")
    args = ap.parse_args()

    probs, labels, domains = read_preds(args.preds, args.labels)

    # Optional temperature calibration
    if args.calibration_json and os.path.exists(args.calibration_json):
        with open(args.calibration_json, "r", encoding="utf-8") as f:
            T = float(json.load(f).get("temperature", 1.0))
        # Convert probs to logits then re-calibrate for better consistency
        logits = np.log(probs + 1e-12)
        from vifact.modules.confidence.calibration import TemperatureScaler

        ts = TemperatureScaler(T=T)
        probs = ts.calibrate_batch(logits)

    nei_index = args.labels.index(args.nei_label)
    thrs = tune_threshold(probs, labels, domains=domains, nei_index=nei_index, metric=args.metric)

    obj = {
        "global_threshold": thrs.global_threshold,
        "per_domain": thrs.per_domain,
        "labels": args.labels,
        "nei_label": args.nei_label,
        "metric": args.metric,
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print({"global_threshold": thrs.global_threshold, "n_domains": len(thrs.per_domain)})


if __name__ == "__main__":
    main()

