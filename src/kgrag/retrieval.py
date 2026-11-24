"""Retrieval baselines for multi-hop QA, graded with an oracle reader.

A question is **answerable** from a retrieved set iff every hop's supporting fact
appears in at least one retrieved document. That is an upper bound on any reader
over those documents, which isolates retrieval from reading.

* ``single_shot``  — retrieve top-k once with the question (classic RAG).
* ``iterative``    — one retrieval per hop (IRCoT / self-ask style). The next query is
  ``<bridge entity name> + <relation keywords>``, where the bridge comes from the
  previous hop's retrieved documents (an oracle reader hands it over, and only if
  the supporting document was actually retrieved).
* ``entity_index`` — GraphRAG-style hop: candidates are the documents in which the
  entity linker found the bridge **entity** (not its name), ranked by BM25 against
  the relation keywords. It inherits the linker's errors, so it shows how EL quality
  carries into retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .world import Question, World

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in is was which where who what by that city company".split())
REL_WORDS = {
    "born_in": "born birthplace",
    "works_at": "works at joined employs",
    "hq_in": "headquartered based in",
    "founded_by": "founded by founder",
    "acquired_by": "acquired",
}


def tokenize(t: str) -> list[str]:
    return [w for w in _TOK.findall(t.lower()) if w not in STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], float)
        self.avg, self.k1, self.b = self.len.mean(), k1, b
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.inv: dict[str, list[int]] = {}
        for i, d in enumerate(self.docs):
            for w in d:
                self.inv.setdefault(w, []).append(i)

    def scores(self, query: str, subset: list[int] | None = None) -> dict[int, float]:
        s: dict[int, float] = {}
        allowed = set(subset) if subset is not None else None
        for w in set(tokenize(query)):
            for i in self.inv.get(w, []):
                if allowed is not None and i not in allowed:
                    continue
                f = self.docs[i][w]
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] = s.get(i, 0.0) + self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int, subset: list[int] | None = None) -> list[int]:
        s = self.scores(query, subset)
        ranked = sorted(s, key=lambda i: (-s[i], i))[:k]
        if subset is not None and len(ranked) < k:  # pad with unscored subset docs
            ranked += [i for i in sorted(subset) if i not in s][: k - len(ranked)]
        return ranked


class Dense:
    """LSA embeddings (TF-IDF word+char → SVD); an offline stand-in for a neural embedder."""

    def __init__(self, texts: list[str], dim: int = 128, seed: int = 0):
        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        x = hstack([self.w.fit_transform(texts), self.c.fit_transform(texts)])
        self.svd = TruncatedSVD(dim, random_state=seed).fit(x)
        self.v = self._enc(x)

    def _enc(self, x) -> np.ndarray:
        v = self.svd.transform(x)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def search(self, query: str, k: int, subset=None) -> list[int]:
        q = self._enc(hstack([self.w.transform([query]), self.c.transform([query])]))[0]
        return [int(i) for i in np.argsort(-(self.v @ q), kind="stable")[:k]]


def _supported(world: World, fact, docs: list[int]) -> bool:
    return any(fact in world.docs[i].facts for i in docs)


def single_shot(world: World, retriever, q: Question, k: int) -> tuple[bool, list[int]]:
    docs = retriever.search(q.text, k)
    return all(_supported(world, f, docs) for f in q.support), docs


def iterative(world: World, retriever, q: Question, k: int) -> bool:
    query = q.text
    for hop, fact in enumerate(q.support):
        docs = retriever.search(query, k)
        if not _supported(world, fact, docs):
            return False
        if hop + 1 < len(q.support):
            bridge = world.entities[q.chain[hop]].name
            query = f"{bridge} {REL_WORDS[q.path[hop + 1]]}"
    return True


def entity_index(world: World, bm25: BM25, entity_docs, start_node, q: Question, k: int) -> bool:
    """Hop over linked-entity postings; ``start_node`` is the linker's question entity."""
    node = start_node
    for hop, fact in enumerate(q.support):
        cand = sorted(entity_docs.get(node, ()))
        docs = bm25.search(REL_WORDS[q.path[hop]], k, subset=cand) if cand else []
        if not _supported(world, fact, docs):
            return False
        node = q.chain[hop]  # oracle reader hands over the bridge entity
    return True


def ndcg_at_k(world: World, q: Question, ranked: list[int], k: int = 10) -> float:
    rel = [1.0 if any(f in world.docs[i].facts for f in q.support) else 0.0 for i in ranked[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_rel = sum(any(f in d.facts for f in q.support) for d in world.docs)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, n_rel)))
    return dcg / idcg if idcg else 0.0
