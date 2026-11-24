"""Knowledge-graph pipeline: mention detection → entity linking → relation extraction → QA.

Entity-linking strategies (the ablation axis):

* ``surface``  — no linking: graph nodes are surface strings ("Weber", "NL").
* ``naive``    — link each mention to its first KB candidate (a "most common sense" baseline).
* ``context``  — rank candidates by TF-IDF similarity between the document and the
  candidate's KB description (research field / sector).
* ``context+doc`` — ``context`` plus in-document coherence: a short mention ("Weber")
  reuses the entity already linked for a longer mention in the same document
  whose aliases include it (a standard EL heuristic).
* ``oracle``   — gold links; the upper bound for the rest of the pipeline.

Relation extraction uses patterns over linked mentions and resolves "It was founded
by …" to the document's first organisation. The document grammar is known to the
extractor, so extraction itself is near-exact; **linking is the variable under test**.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .world import Doc, Question, World

RELATION_SIGNATURE = {
    "born_in": ("person", "city"),
    "works_at": ("person", "org"),
    "hq_in": ("org", "city"),
    "founded_by": ("org", "person"),
    "acquired_by": ("org", "org"),
}

# (regex over marked text, relation, which group is subject, which is object)
_M = r"⟦(\d+)⟧"
PATTERNS = [
    (rf"{_M} was born in {_M}", "born_in", 1, 2),
    (rf"{_M} is the birthplace of {_M}", "born_in", 2, 1),
    (rf"{_M} works at {_M}", "works_at", 1, 2),
    (rf"{_M} joined {_M}", "works_at", 1, 2),
    (rf"{_M} employs {_M}", "works_at", 2, 1),
    (rf"{_M} is (?:headquartered|based) in {_M}", "hq_in", 1, 2),
    (rf"{_M}, an? \w+ researcher, founded {_M}", "founded_by", 2, 1),
    (rf"{_M} founded {_M}", "founded_by", 2, 1),
    (rf"{_M} acquired (?:the \w+ firm )?{_M}", "acquired_by", 2, 1),
]
IT_FOUNDED = re.compile(rf"It was founded by {_M}")

QUESTION_PATHS = [
    (
        r"^Where was the founder of the company that employs (.+) born\?$",
        ["works_at", "founded_by", "born_in"],
    ),
    (
        r"^In which city is the company that acquired the employer of (.+) headquartered\?$",
        ["works_at", "acquired_by", "hq_in"],
    ),
    (r"^In which city is the employer of (.+) headquartered\?$", ["works_at", "hq_in"]),
    (r"^Where was the founder of (.+) born\?$", ["founded_by", "born_in"]),
    (r"^In which city is (.+) headquartered\?$", ["hq_in"]),
    (r"^Where was (.+) born\?$", ["born_in"]),
]


@dataclass
class Span:
    start: int
    end: int
    surface: str
    candidates: list[str]


class MentionDetector:
    """Longest-match dictionary tagger over all KB aliases."""

    def __init__(self, world: World):
        self.alias: dict[str, list[str]] = defaultdict(list)
        for e in world.entities.values():
            for a in e.aliases:
                if e.id not in self.alias[a]:
                    self.alias[a].append(e.id)
        alts = sorted(self.alias, key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, alts)) + r")(?!\w)")

    def __call__(self, text: str) -> list[Span]:
        return [
            Span(m.start(), m.end(), m.group(1), self.alias[m.group(1)])
            for m in self.rx.finditer(text)
        ]


class Linker:
    STRATEGIES = ("surface", "naive", "context", "context+doc", "oracle")

    def __init__(self, world: World, strategy: str):
        assert strategy in self.STRATEGIES, strategy
        self.world, self.strategy = world, strategy
        ents = list(world.entities.values())
        self.vec = TfidfVectorizer(sublinear_tf=True).fit(
            [e.description or e.name for e in ents] + [d.text for d in world.docs]
        )
        self.desc_ids = [e.id for e in ents]
        self.desc = self.vec.transform([e.description or e.name for e in ents])
        self.row = {eid: i for i, eid in enumerate(self.desc_ids)}

    def _context_rank(self, text: str, cands: list[str]) -> str:
        q = self.vec.transform([text])
        sims = (self.desc[[self.row[c] for c in cands]] @ q.T).toarray().ravel()
        return cands[int(np.argmax(sims))]

    def link(self, text: str, spans: list[Span], gold: dict[tuple[int, int], str] | None = None):
        """Returns one node id per span (entity id, or surface string for ``surface``)."""
        if self.strategy == "surface":
            return [s.surface for s in spans]
        if self.strategy == "oracle":
            assert gold is not None
            return [gold.get((s.start, s.end), s.candidates[0]) for s in spans]
        out: list[str | None] = [None] * len(spans)
        order = sorted(range(len(spans)), key=lambda i: -len(spans[i].surface))
        for i in order:  # longest (most specific) mentions first
            s = spans[i]
            if len(s.candidates) == 1:
                out[i] = s.candidates[0]
                continue
            if self.strategy == "naive":
                out[i] = s.candidates[0]
                continue
            if self.strategy == "context+doc":
                prior = [
                    out[j]
                    for j in range(len(spans))
                    if out[j] in s.candidates
                    and len(spans[j].surface) > len(s.surface)
                    and s.surface in self.world.entities[out[j]].aliases
                ]
                if prior:
                    out[i] = Counter(prior).most_common(1)[0][0]
                    continue
            out[i] = self._context_rank(text, s.candidates)
        return out


def _kind(world: World, node: str, strategy: str) -> str | None:
    if strategy == "surface":
        ids = MentionDetectorCache.alias.get(node, [])
        kinds = {world.entities[i].kind for i in ids}
        return kinds.pop() if len(kinds) == 1 else None
    return world.entities[node].kind


class MentionDetectorCache:
    alias: dict[str, list[str]] = {}


@dataclass
class KGStats:
    el_accuracy: float
    el_by_ambiguity: dict[str, float]
    triple_precision: float
    triple_recall: float
    n_triples: int


class KnowledgeGraph:
    def __init__(self, world: World, strategy: str):
        self.world, self.strategy = world, strategy
        self.detect = MentionDetector(world)
        MentionDetectorCache.alias = self.detect.alias
        self.linker = Linker(world, strategy)
        self.edges: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.entity_docs: dict[str, set[int]] = defaultdict(set)
        self.doc_triples: dict[int, list[tuple[str, str, str]]] = {}
        el_hits: dict[str, list[bool]] = defaultdict(list)
        for d in world.docs:
            spans, nodes = self._process(d, el_hits)
            for n in nodes:
                self.entity_docs[n].add(d.id)
            self.doc_triples[d.id] = self._extract(d, spans, nodes)
            for s, r, o in self.doc_triples[d.id]:
                self.edges[(s, r)][o] += 1
        self.el_hits = el_hits

    def _process(self, d: Doc, el_hits):
        spans = self.detect(d.text)
        gold = {(m.start, m.end): m.entity for m in d.mentions}
        nodes = self.linker.link(d.text, spans, gold)
        if self.strategy != "surface":
            for s, n in zip(spans, nodes, strict=True):
                g = gold.get((s.start, s.end))
                if g is None:
                    continue
                k = len(s.candidates)
                bucket = "unambiguous" if k == 1 else ("2-3 candidates" if k <= 3 else "4+")
                el_hits[bucket].append(n == g)
        return spans, nodes

    def _extract(self, d: Doc, spans: list[Span], nodes: list[str]):
        marked, last = [], 0
        for i, s in enumerate(spans):
            marked.append(d.text[last : s.start])
            marked.append(f"⟦{i}⟧")
            last = s.end
        marked.append(d.text[last:])
        mt = "".join(marked)
        out = []
        for rx, rel, si, oi in PATTERNS:
            for m in re.finditer(rx, mt):
                out.append((nodes[int(m.group(si))], rel, nodes[int(m.group(oi))]))
        m = IT_FOUNDED.search(mt)
        if m:
            orgs = [n for n in nodes if _kind(self.world, n, self.strategy) == "org"]
            if orgs:
                out.append((orgs[0], "founded_by", nodes[int(m.group(1))]))
        # type-check against the relation signature
        good = []
        for s, r, o in out:
            ks, ko = RELATION_SIGNATURE[r]
            if self.strategy == "surface" or (
                _kind(self.world, s, self.strategy) == ks
                and _kind(self.world, o, self.strategy) == ko
            ):
                good.append((s, r, o))
        return good

    def stats(self) -> KGStats:
        triples = {(s, r, o) for (s, r), c in self.edges.items() for o in c}
        if self.strategy == "surface":
            p = rcl = float("nan")
        else:
            tp = len(triples & self.world.triples)
            p, rcl = tp / max(1, len(triples)), tp / len(self.world.triples)
        allh = [x for v in self.el_hits.values() for x in v]
        return KGStats(
            el_accuracy=float(np.mean(allh)) if allh else float("nan"),
            el_by_ambiguity={k: float(np.mean(v)) for k, v in sorted(self.el_hits.items())},
            triple_precision=p,
            triple_recall=rcl,
            n_triples=len(triples),
        )

    # ------------------------------------------------------------------ QA

    def parse(self, q: Question) -> tuple[str | None, list[str] | None]:
        for rx, path in QUESTION_PATHS:
            m = re.match(rx, q.text)
            if m:
                spans = self.detect(q.text)
                spans = [s for s in spans if s.start >= m.start(1)] or spans
                if not spans:
                    return None, path
                s = max(spans, key=lambda s: len(s.surface))
                if self.strategy == "oracle":
                    return q.start, path
                node = self.linker.link(q.text, [s])[0]
                return node, path
        return None, None

    def node_name(self, node: str) -> str:
        return node if self.strategy == "surface" else self.world.entities[node].name

    def to_answer(self, node: str | None) -> str | None:
        if node is None or self.strategy != "surface":
            return node
        ids = self.detect.alias.get(node, [])
        return ids[0] if len(ids) == 1 else None

    def answer(self, q: Question, docs: list[int] | None = None) -> str | None:
        """Traverse the graph built from ``docs`` (default: the whole corpus)."""
        node, path = self.parse(q)
        if node is None or path is None:
            return None
        edges = self.edges if docs is None else self.local_edges(docs)
        for r in path:
            c = edges.get((node, r))
            if not c:
                return None
            node = c.most_common(1)[0][0]
        if self.strategy == "surface":  # final hop is a city: map the name back to its id
            ids = self.detect.alias.get(node, [])
            return ids[0] if len(ids) == 1 else None
        return node

    def local_edges(self, docs: list[int]) -> dict[tuple[str, str], Counter]:
        e: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for i in docs:
            for s, r, o in self.doc_triples[i]:
                e[(s, r)][o] += 1
        return e
