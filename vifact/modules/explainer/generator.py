from __future__ import annotations

from typing import Dict, List


def generate_explanation(claim: str, verdict: str, rationale: str, sources: List[Dict] | None = None) -> str:
    srcs = sources or []
    cite = "; ".join(f"[{i+1}] {s.get('doc_id','')}" for i, s in enumerate(srcs[:3]))
    bits = [
        f"Phán quyết: {verdict}.",
        f"Tóm tắt luận cứ: {rationale}" if rationale else "",
        f"Nguồn: {cite}" if cite else "",
    ]
    return " ".join([b for b in bits if b])

