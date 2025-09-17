from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np


SENT_SPLIT = re.compile(r"(?<=[.!?\n\r])\s+")


def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    return sents if sents else [text]


def jaccard(a: str, b: str) -> float:
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def label_sentences(context: str, evidence: str | None, pos_threshold: float = 0.3) -> List[Tuple[str, int]]:
    sents = split_sentences(context or "")
    if not evidence:
        return [(s, 0) for s in sents]
    ev = evidence.strip()
    labeled: List[Tuple[str, int]] = []
    for s in sents:
        lbl = 1 if (ev in s) or (jaccard(s, ev) >= pos_threshold) else 0
        labeled.append((s, lbl))
    return labeled


@dataclass
class PairExample:
    text_a: str  # claim
    text_b: str  # sentence
    label: int   # 0/1


def generate_pairs(ex: Dict, *, max_pos: int = 2, max_neg: int = 4, pos_threshold: float = 0.3) -> List[PairExample]:
    claim = ex.get("claim") or ""
    labeled = label_sentences(ex.get("context") or "", ex.get("evidence"), pos_threshold)
    pos = [s for s, l in labeled if l == 1]
    neg = [s for s, l in labeled if l == 0]
    random.shuffle(pos)
    random.shuffle(neg)
    pairs: List[PairExample] = []
    for s in pos[:max_pos]:
        pairs.append(PairExample(text_a=claim, text_b=s, label=1))
    for s in neg[:max_neg]:
        pairs.append(PairExample(text_a=claim, text_b=s, label=0))
    return pairs

