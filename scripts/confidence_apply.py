#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vifact.modules.confidence.apply import apply_thresholds, load_calibration, load_thresholds, probs_from_logits_or_probs


def main():
    ap = argparse.ArgumentParser(description="Apply calibration + adaptive thresholds to predictions JSONL and emit final labels")
    ap.add_argument("--preds", required=True, help="Input JSONL with id, optional domain, and logits|probs")
    ap.add_argument("--out", required=True, help="Output JSONL path with final predictions")
    ap.add_argument("--labels", nargs="+", default=["SUPPORTED", "REFUTED", "NEI"], help="Label order mapping to indices")
    ap.add_argument("--nei-label", default="NEI")
    ap.add_argument("--calibration-json", help="Calibration JSON (temperature) from confidence_calibrate.py")
    ap.add_argument("--thresholds-json", help="Thresholds JSON from confidence_tune.py")
    args = ap.parse_args()

    calibrator = load_calibration(args.calibration_json)
    thresholds = load_thresholds(args.thresholds_json)
    nei_index = args.labels.index(args.nei_label)

    ids: List[str] = []
    domains: List[Optional[str]] = []
    logits_list: List[List[float]] = []
    probs_list: List[List[float]] = []
    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(str(obj.get("id", obj.get("example_id", ""))))
            domains.append(obj.get("domain"))
            if "logits" in obj:
                logits_list.append(obj["logits"])
            elif "probs" in obj:
                probs_list.append(obj["probs"])
            else:
                # skip entries without model outputs
                logits_list.append([])

    logits_opt = logits_list if any(len(v) > 0 for v in logits_list) else None
    probs_opt = probs_list if logits_opt is None else None
    probs = probs_from_logits_or_probs(logits_opt, probs_opt, calibrator)
    pred_raw, conf_raw, pred_final = apply_thresholds(probs, domains, thresholds, nei_index)

    lab = args.labels
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f_out:
        for i, _id in enumerate(ids):
            obj = {
                "id": _id,
                "domain": domains[i],
                "pred_raw": lab[int(pred_raw[i])] if probs.shape[1] else None,
                "pred_final": lab[int(pred_final[i])] if probs.shape[1] else None,
                "confidence": float(conf_raw[i]) if probs.shape[1] else None,
                "abstained": bool(pred_final[i] == nei_index and pred_raw[i] != nei_index) if probs.shape[1] else None,
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote results: {args.out}")


if __name__ == "__main__":
    main()

