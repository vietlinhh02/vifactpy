from __future__ import annotations

import re
from typing import Dict


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_rationale(ex: Dict, max_sentences: int = 2) -> str:
    """Heuristic rationale: select up to N sentences with most word overlap with claim.
    Expects ex to have keys: claim, context.
    """
    claim = (ex.get("claim") or "").lower()
    ctx = ex.get("context") or ""
    if not claim or not ctx:
        return ""
    claim_tokens = set(claim.split())
    sents = SENT_SPLIT.split(ctx.strip()) if ctx.strip() else []
    scored = []
    for s in sents:
        toks = set(s.lower().split())
        score = len(claim_tokens & toks)
        if score:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored[:max_sentences])

