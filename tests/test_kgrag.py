import pytest

from kgrag.kg import KnowledgeGraph, MentionDetector
from kgrag.pipelines import graph_hop, kg_global, rag_iterative, rag_single
from kgrag.retrieval import BM25, ndcg_at_k, single_shot
from kgrag.world import make_world


@pytest.fixture(scope="module")
def world():
    return make_world(n_people=120, n_orgs=40, n_distractors=60, seed=3)


def test_world_is_consistent(world):
    for d in world.docs:
        for m in d.mentions:
            assert d.text[m.start : m.end] in world.entities[m.entity].aliases
        for f in d.facts:
            assert f in world.triples
    for q in world.questions:
        assert len(q.chain) == q.hops == len(q.support)
        assert q.support[0][0] == q.start and q.support[-1][2] == q.answer


def test_generation_deterministic():
    a, b = make_world(50, 20, 10, seed=1), make_world(50, 20, 10, seed=1)
    assert [d.text for d in a.docs] == [d.text for d in b.docs]


def test_mention_detector_recovers_gold_spans(world):
    det = MentionDetector(world)
    found = gold = 0
    for d in world.docs:
        spans = {(s.start, s.end) for s in det(d.text)}
        g = {(m.start, m.end) for m in d.mentions}
        found += len(spans & g)
        gold += len(g)
    assert found / gold > 0.99


def test_oracle_linking_gives_exact_kg(world):
    kg = KnowledgeGraph(world, "oracle")
    s = kg.stats()
    assert s.triple_precision == 1.0 and s.triple_recall == 1.0
    assert all(kg.answer(q) == q.answer for q in world.questions)


def test_context_linking_beats_naive(world):
    naive, ctx = KnowledgeGraph(world, "naive").stats(), KnowledgeGraph(world, "context").stats()
    assert ctx.el_accuracy > naive.el_accuracy + 0.1
    assert ctx.triple_precision > naive.triple_precision


def test_single_shot_rag_fails_multi_hop_but_graph_hop_does_not(world):
    bm = BM25([d.text for d in world.docs])
    kg = KnowledgeGraph(world, "oracle")
    multi = [q for q in world.questions if q.hops >= 2]
    single = sum(rag_single(kg, bm, q, 10)[0] == q.answer for q in multi) / len(multi)
    hop = sum(graph_hop(kg, bm, q, 5)[0] == q.answer for q in multi) / len(multi)
    it = sum(rag_iterative(kg, bm, q, 5)[0] == q.answer for q in multi) / len(multi)
    assert single < 0.2 and hop > 0.95 and it > single
    assert all(kg_global(kg, q)[0] == q.answer for q in multi)


def test_ndcg_bounds(world):
    bm = BM25([d.text for d in world.docs])
    q = world.questions[0]
    _, docs = single_shot(world, bm, q, 10)
    assert 0.0 <= ndcg_at_k(world, q, docs) <= 1.0
