# graphrag-entity-linking

**Multi-hop QA, measured: why single-shot RAG fails on bridge questions, and why entity linking, not retrieval, decides whether GraphRAG works.**

The test bed is a synthetic knowledge world where the hard parts of real corpora are explicit and graded:
- **homonyms:** about 1 in 7 people share a full name;
- **colliding acronyms:** "NL" can be Northvale Labs or Nimbus Labs;
- **ambiguous short mentions:** "Dr. Weber", "Weber";
- **coreference:** "It was founded by…";
- **distractor news:** "X gave a talk in Lyon" states no relation.

Questions follow 1-, 2- and 3-hop relation paths, for example *"In which city is the company that acquired the employer of Liam Yilmaz headquartered?"*

```
pip install -e ".[dev]"
kgrag --seeds 0 1 2 --out results      # ~1 min on CPU
```

## Systems

The knowledge-graph pipeline (`kg.py`):
1. dictionary mention detection over KB aliases (longest match);
2. **entity linking**, which is the ablation axis;
3. relation extraction with patterns, coreference for "It" and a type check against the relation signature;
4. graph traversal.

The KB given to the linker holds names, aliases and *descriptions* (a person's research field, a company's sector), like a Wikidata-style catalogue. It does **not** hold the relations the questions ask about.

| linker | how it picks among candidates |
|---|---|
| `surface` | no linking; nodes are strings ("Weber") |
| `naive` | first KB candidate |
| `context` | TF-IDF similarity of the document to each candidate's description |
| `context+doc` | plus in-document coherence (a short mention reuses a longer linked mention's entity) |
| `oracle` | gold links (upper bound) |

**End-to-end systems** (`pipelines.py`) all use **the same extractive reader**: the KG extractor run on the documents each system retrieves. They differ only in which documents the reader sees:

| system | documents |
|---|---|
| RAG single-shot | top-10 by the question |
| RAG iterative | per hop, top-5 for *bridge-entity name + relation words* (IRCoT-style) |
| graph hop | per hop, top-5 of the documents where the linker found the bridge **entity** (GraphRAG-style) |
| KG traversal | the whole graph, extracted offline |

## Results

Mean over 3 world seeds, 480 questions each. Full tables, including per-hop retrieval and nDCG, are in [`results/results.md`](results/results.md).

### End-to-end accuracy

| linker | system | 1-hop | 2-hop | 3-hop | overall | docs read |
|---|---|---|---|---|---|---|
| context | RAG single-shot (BM25) | 98.5 | **1.0** | **0.2** | 33.3 | 10.0 |
| context | RAG iterative (BM25) | 95.8 | 71.2 | 48.8 | 71.9 ± 4.1 | 9.4 |
| context | RAG iterative (dense) | 87.3 | 65.6 | 41.5 | 64.8 ± 3.6 | 9.0 |
| context | **graph hop** | 97.7 | 75.4 | 60.8 | **78.0 ± 5.3** | **6.5** |
| context | KG traversal | 97.9 | 75.2 | 60.4 | 77.8 ± 5.1 | 0 |
| oracle | RAG iterative (BM25) | 96.7 | 92.7 | 74.8 | 88.1 | 9.4 |
| oracle | **graph hop** | 100 | 100 | 100 | **100** | 6.8 |

### Entity linking drives everything downstream

| linker | EL accuracy | 2–3 candidates | 4+ candidates | triple precision | graph-hop QA |
|---|---|---|---|---|---|
| surface | — | — | — | — | 10.4 |
| naive | 74.0 | 47.4 | 12.6 | 44.6 | 29.0 |
| context | 94.2 | 77.7 | 86.1 | 88.5 | 78.0 |
| context+doc | 94.4 | 77.7 | 86.7 | 88.8 | 78.3 |
| oracle | 100 | 100 | 100 | 100 | 100 |

## Findings

1. **Single-shot RAG cannot answer bridge questions.** With every supporting fact required, BM25 top-20 covers 3.5% of 2-hop questions and LSA dense top-20 covers 9.8%. 1-hop coverage is 100% for both. The bridge entity is not in the question, so no amount of `k` helps much.

2. **Hop-wise retrieval fixes coverage, but retrieving by *name* has a ceiling.** Even with perfect linking, iterative RAG reaches 88%, against 100% for graph hops reading fewer documents. The difference is that a name query ("Anna Weber works at") also pulls the homonym's documents, and the reader then follows the wrong person. Indexing by **entity ID** rather than by string removes that failure mode.

3. **With real linking, GraphRAG's advantage depends on the linker.**
   - **Going down the linkers** (oracle → context EL at 94.2% → naive EL at 74%), graph-hop accuracy falls from 100% to 78% to 29%. With no linking at all it is 10%.
   - **A few EL points cost many QA points.** Errors compound per hop: 3-hop accuracy is 61% at 94% EL. A 6-point EL gap turns into a 39-point 3-hop gap.
   - **Where the KG still wins:** at realistic EL it beats iterative RAG by about 6 points (78 vs 72) and reads 30% fewer documents.

4. **The hardest mentions are short org aliases, and document coherence doesn't rescue them.** Measured on seed 0 with the `context` linker, over ambiguous mentions only:

   | mention type | count | EL accuracy |
   |---|---|---|
   | person, full name (homonyms) | 58 | 98.3% |
   | person, surname / "Dr." / initial | 757 | 93% |
   | **org, acronym or short name** | 183 | **60%** |

   - A person's bio states their field, which matches the KB description.
   - A short org mention ("P. Okafor works at NL", "NL acquired …") usually has no sector cue nearby, and the org's full name never appears in that document. So in-document coherence adds only +0.2 points.
   - Fixing this needs **collective / cross-document** linking: for example, prefer the candidate whose other extracted relations are consistent, or use an LLM linker with the KB descriptions.

5. **BM25 beats LSA for this kind of retrieval** (nDCG@10 0.63 vs 0.53; iterative 3-hop 48.8 vs 41.5). Entity-heavy, name-matching queries reward exact lexical match.

## Limitations

- The documents are template-generated, so relation extraction is near-exact by construction (100% precision and recall with oracle links). This isolates **linking**, but real extraction errors would stack on top.
- The LSA embedder is a CPU stand-in for neural embedders.
- The reader is extractive and rule-based. An LLM reader could recover some linking errors from context, and could also add its own.

## Layout

| file | what |
|---|---|
| `src/kgrag/world.py` | KB, documents with gold mention links and facts, multi-hop questions |
| `src/kgrag/kg.py` | mention detection, five linkers, relation extraction, graph build, QA, EL/KG metrics |
| `src/kgrag/retrieval.py` | BM25, LSA dense, single-shot/iterative support recall, nDCG |
| `src/kgrag/pipelines.py` | end-to-end systems with a shared reader |
| `src/kgrag/cli.py` | the experiment grid; writes `results/` |
