from __future__ import annotations

from typing import Dict, List


def simple_fuse(evidences: List[Dict], max_items: int = 3) -> List[Dict]:
    """Baseline fusion: return top-k evidences as-is.
    Each evidence dict may include {doc_id, text, score}.
    """
    sorted_evs = sorted(evidences, key=lambda e: e.get("score", 0.0), reverse=True)
    return sorted_evs[:max_items]

