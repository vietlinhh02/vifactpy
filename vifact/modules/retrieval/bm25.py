from __future__ import annotations

import gzip
import json
import math
import pickle
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def tokenize(text: str) -> List[str]:
    t = _normalize(text)
    return [tok for tok in t.split() if tok]


@dataclass
class BM25Config:
    k1: float = 1.2
    b: float = 0.75


class BM25Index:
    def __init__(self, cfg: BM25Config | None = None):
        self.cfg = cfg or BM25Config()
        self.N = 0
        self.avgdl = 0.0
        self.doc_len: Dict[int, int] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_text: Dict[int, str] = {}
        self.doc_ids: List[str] = []

    def fit(self, docs: Iterable[Tuple[str, str]]) -> None:
        total_len = 0
        for internal_id, (doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            total_len += len(tokens)
            self.doc_len[internal_id] = len(tokens)
            self.doc_text[internal_id] = text
            self.doc_ids.append(doc_id)
            tf = Counter(tokens)
            for term, c in tf.items():
                self.postings[term][internal_id] = c
            for term in tf.keys():
                self.df[term] += 1
        self.N = len(self.doc_ids)
        self.avgdl = (total_len / max(self.N, 1)) if self.N else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1e-12)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.N:
            return []
        k1, b = self.cfg.k1, self.cfg.b
        q_terms = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        for term in q_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self._idf(term)
            for doc_id, tf in plist.items():
                dl = self.doc_len[doc_id]
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1e-6))
                scores[doc_id] += idf * (tf * (k1 + 1)) / denom
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[d], s) for d, s in items]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wb") as f:
            pickle.dump(
                {
                    "cfg": self.cfg,
                    "N": self.N,
                    "avgdl": self.avgdl,
                    "doc_len": self.doc_len,
                    "df": dict(self.df),
                    "postings": {t: dict(m) for t, m in self.postings.items()},
                    "doc_text": self.doc_text,
                    "doc_ids": self.doc_ids,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
        idx = cls(obj.get("cfg", BM25Config()))
        idx.N = obj["N"]
        idx.avgdl = obj["avgdl"]
        idx.doc_len = obj["doc_len"]
        idx.df = defaultdict(int, obj["df"])
        idx.postings = defaultdict(dict, {t: dict(m) for t, m in obj["postings"].items()})
        idx.doc_text = obj["doc_text"]
        idx.doc_ids = obj["doc_ids"]
        return idx

    @classmethod
    def from_corpus_jsonl(
        cls, path: str | Path, text_key: str = "text", id_key: str = "doc_id"
    ) -> "BM25Index":
        def gen():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    yield obj[id_key], obj[text_key]

        idx = cls()
        idx.fit(gen())
        return idx

    def get_text(self, ext_doc_id: str) -> str:
        try:
            pos = self.doc_ids.index(ext_doc_id)
        except ValueError:
            return ""
        return self.doc_text.get(pos, "")
