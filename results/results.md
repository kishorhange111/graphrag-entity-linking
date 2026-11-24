Mean over world seeds [0, 1, 2] (480 questions per seed: 160 each of 1/2/3 hops).

## End-to-end QA accuracy (shared extractive reader, no oracle)

| entity linker | system | 1-hop | 2-hop | 3-hop | overall (± sd) | docs read |
|---|---|---|---|---|---|---|
| surface | RAG single-shot (BM25, k=10) | 26.7 | 0.0 | 0.0 | 8.9 ± 0.8 | 10.0 |
| surface | RAG iterative (BM25, k=5/hop) | 26.7 | 4.8 | 0.0 | 10.5 ± 0.8 | 5.6 |
| surface | RAG iterative (dense, k=5/hop) | 25.8 | 3.5 | 0.0 | 9.8 ± 0.3 | 5.6 |
| surface | graph hop (entity index, k=5/hop) | 26.7 | 4.6 | 0.0 | 10.4 ± 0.7 | 2.4 |
| surface | KG traversal (global) | 26.7 | 4.2 | 0.0 | 10.3 ± 1.0 | 0.0 |
| naive | RAG single-shot (BM25, k=10) | 60.2 | 0.2 | 0.2 | 20.2 ± 0.5 | 10.0 |
| naive | RAG iterative (BM25, k=5/hop) | 60.2 | 30.4 | 11.0 | 33.9 ± 1.2 | 7.2 |
| naive | RAG iterative (dense, k=5/hop) | 52.7 | 27.1 | 8.5 | 29.4 ± 1.3 | 7.2 |
| naive | graph hop (entity index, k=5/hop) | 52.3 | 23.8 | 10.8 | 29.0 ± 0.5 | 5.0 |
| naive | KG traversal (global) | 53.8 | 23.8 | 11.9 | 29.8 ± 1.1 | 0.0 |
| context | RAG single-shot (BM25, k=10) | 98.5 | 1.0 | 0.2 | 33.3 ± 0.4 | 10.0 |
| context | RAG iterative (BM25, k=5/hop) | 95.8 | 71.2 | 48.8 | 71.9 ± 4.1 | 9.4 |
| context | RAG iterative (dense, k=5/hop) | 87.3 | 65.6 | 41.5 | 64.8 ± 3.6 | 9.0 |
| context | graph hop (entity index, k=5/hop) | 97.7 | 75.4 | 60.8 | 78.0 ± 5.3 | 6.5 |
| context | KG traversal (global) | 97.9 | 75.2 | 60.4 | 77.8 ± 5.1 | 0.0 |
| context+doc | RAG single-shot (BM25, k=10) | 98.5 | 1.0 | 0.2 | 33.3 ± 0.4 | 10.0 |
| context+doc | RAG iterative (BM25, k=5/hop) | 95.8 | 71.9 | 49.0 | 72.2 ± 3.9 | 9.4 |
| context+doc | RAG iterative (dense, k=5/hop) | 87.3 | 66.0 | 41.7 | 65.0 ± 3.4 | 9.0 |
| context+doc | graph hop (entity index, k=5/hop) | 97.7 | 76.0 | 61.0 | 78.3 ± 5.1 | 6.5 |
| context+doc | KG traversal (global) | 97.9 | 76.2 | 61.2 | 78.5 ± 4.7 | 0.0 |
| oracle | RAG single-shot (BM25, k=10) | 99.8 | 1.5 | 0.2 | 33.8 ± 0.3 | 10.0 |
| oracle | RAG iterative (BM25, k=5/hop) | 96.7 | 92.7 | 74.8 | 88.1 ± 1.1 | 9.4 |
| oracle | RAG iterative (dense, k=5/hop) | 89.0 | 86.0 | 65.8 | 80.3 ± 1.1 | 9.1 |
| oracle | graph hop (entity index, k=5/hop) | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 | 6.8 |
| oracle | KG traversal (global) | 100.0 | 100.0 | 100.0 | 100.0 ± 0.0 | 0.0 |

## Entity linking and KG quality

| linker | EL accuracy | unambiguous | 2-3 candidates | 4+ candidates | triple precision | triple recall |
|---|---|---|---|---|---|---|
| surface | — | — | — | — | — | — |
| naive | 74.0 | 100.0 | 47.4 | 12.6 | 44.6 | 43.3 |
| context | 94.2 | 100.0 | 77.7 | 86.1 | 88.5 | 88.3 |
| context+doc | 94.4 | 100.0 | 77.7 | 86.7 | 88.8 | 88.7 |
| oracle | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Retrieval: all supporting facts retrieved (oracle reader)

| retriever | mode | k (per hop) | 1-hop | 2-hop | 3-hop | nDCG@k |
|---|---|---|---|---|---|---|
| bm25 | single-shot | 5 | 96.7 | 0.6 | 0.0 | 0.624 |
| bm25 | iterative | 5 | 96.7 | 92.7 | 74.8 | — |
| bm25 | single-shot | 10 | 99.8 | 1.5 | 0.2 | 0.633 |
| bm25 | iterative | 10 | 99.8 | 98.8 | 83.1 | — |
| bm25 | single-shot | 20 | 100.0 | 3.5 | 1.7 | 0.639 |
| bm25 | iterative | 20 | 100.0 | 100.0 | 86.5 | — |
| dense | single-shot | 5 | 89.0 | 0.2 | 0.0 | 0.507 |
| dense | iterative | 5 | 89.0 | 86.0 | 65.8 | — |
| dense | single-shot | 10 | 98.5 | 2.5 | 1.3 | 0.533 |
| dense | iterative | 10 | 98.5 | 99.0 | 82.7 | — |
| dense | single-shot | 20 | 100.0 | 9.8 | 2.9 | 0.548 |
| dense | iterative | 20 | 100.0 | 100.0 | 92.3 | — |
