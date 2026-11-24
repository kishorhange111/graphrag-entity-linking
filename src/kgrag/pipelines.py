"""End-to-end multi-hop QA with one shared extractive reader (no oracle anywhere).

The reader is the KG extractor applied to *retrieved documents only*: it links
mentions, extracts relation triples and follows the question's relation path. So
every system below differs only in **which documents the reader gets to see**:

* ``rag_single``   — top-k by the question, once;
* ``rag_iterative``— per hop: query = reader's current bridge-entity name + relation
  words, top-k each, reading only that hop's documents;
* ``graph_hop``    — per hop: documents where the linker found the current entity,
  BM25-ranked by relation words, top-k (GraphRAG-style);
* ``kg_global``    — traverse the graph extracted from the whole corpus offline.
"""

from __future__ import annotations

from .kg import KnowledgeGraph
from .retrieval import BM25, REL_WORDS
from .world import Question


def rag_single(kg: KnowledgeGraph, retriever, q: Question, k: int) -> tuple[str | None, int]:
    docs = retriever.search(q.text, k)
    return kg.answer(q, docs), len(docs)


def _hop(kg: KnowledgeGraph, docs: list[int], node: str, rel: str) -> str | None:
    c = kg.local_edges(docs).get((node, rel))
    return c.most_common(1)[0][0] if c else None


def rag_iterative(kg: KnowledgeGraph, retriever, q: Question, k: int) -> tuple[str | None, int]:
    node, path = kg.parse(q)
    read = 0
    if node is None or path is None:
        return None, read
    for i, rel in enumerate(path):
        query = q.text if i == 0 else f"{kg.node_name(node)} {REL_WORDS[rel]}"
        docs = retriever.search(query, k)
        read += len(docs)
        node = _hop(kg, docs, node, rel)
        if node is None:
            return None, read
    return kg.to_answer(node), read


def graph_hop(kg: KnowledgeGraph, bm25: BM25, q: Question, k: int) -> tuple[str | None, int]:
    node, path = kg.parse(q)
    read = 0
    if node is None or path is None:
        return None, read
    for rel in path:
        cand = sorted(kg.entity_docs.get(node, ()))
        docs = bm25.search(REL_WORDS[rel], k, subset=cand) if cand else []
        read += len(docs)
        node = _hop(kg, docs, node, rel)
        if node is None:
            return None, read
    return kg.to_answer(node), read


def kg_global(kg: KnowledgeGraph, q: Question) -> tuple[str | None, int]:
    return kg.answer(q), 0
