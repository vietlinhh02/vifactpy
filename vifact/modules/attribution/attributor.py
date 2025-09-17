from __future__ import annotations

from typing import Dict, List


def attach_sources(ranked: List[Dict]) -> List[Dict]:
    """Attach minimal citation info to evidences.
    Currently passes through doc_id and text; URLs can be added when available in corpus.
    """
    out = []
    for ev in ranked:
        out.append({
            "doc_id": ev.get("doc_id", ""),
            "text": ev.get("text", ""),
            "score": ev.get("score", 0.0),
            "url": ev.get("url"),
        })
    return out

