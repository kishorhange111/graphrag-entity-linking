"""Run all experiments over several world seeds and write results.

kgrag --seeds 0 1 2 --out results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .kg import KnowledgeGraph
from .pipelines import graph_hop, kg_global, rag_iterative, rag_single
from .retrieval import BM25, Dense, iterative, ndcg_at_k, single_shot
from .world import make_world

EL = ("surface", "naive", "context", "context+doc", "oracle")


def run_seed(seed: int, k_single: int, k_hop: int) -> dict:
    w = make_world(seed=seed)
    texts = [d.text for d in w.docs]
    bm, de = BM25(texts), Dense(texts, seed=seed)
    out: dict = {"seed": seed, "el": {}, "e2e": [], "retrieval": []}

    for name, r in (("bm25", bm), ("dense", de)):
        for k in (5, 10, 20):
            ss, it, nd = defaultdict(list), defaultdict(list), []
            for q in w.questions:
                ok, docs = single_shot(w, r, q, k)
                ss[q.hops].append(ok)
                it[q.hops].append(iterative(w, r, q, k))
                nd.append(ndcg_at_k(w, q, docs, k))
            for mode, d in (("single-shot", ss), ("iterative", it)):
                out["retrieval"].append(
                    {
                        "retriever": name,
                        "mode": mode,
                        "k": k,
                        **{f"hop{h}": float(np.mean(v)) for h, v in sorted(d.items())},
                        "ndcg": float(np.mean(nd)) if mode == "single-shot" else None,
                    }
                )

    for st in EL:
        kg = KnowledgeGraph(w, st)
        s = kg.stats()
        out["el"][st] = s.__dict__
        systems = {
            "RAG single-shot (BM25, k=10)": lambda q, kg=kg: rag_single(kg, bm, q, k_single),
            "RAG iterative (BM25, k=5/hop)": lambda q, kg=kg: rag_iterative(kg, bm, q, k_hop),
            "RAG iterative (dense, k=5/hop)": lambda q, kg=kg: rag_iterative(kg, de, q, k_hop),
            "graph hop (entity index, k=5/hop)": lambda q, kg=kg: graph_hop(kg, bm, q, k_hop),
            "KG traversal (global)": lambda q, kg=kg: kg_global(kg, q),
        }
        for name, fn in systems.items():
            acc, read = defaultdict(list), []
            for q in w.questions:
                ans, n = fn(q)
                acc[q.hops].append(ans == q.answer)
                read.append(n)
            out["e2e"].append(
                {
                    "linker": st,
                    "system": name,
                    **{f"hop{h}": float(np.mean(v)) for h, v in sorted(acc.items())},
                    "overall": float(np.mean([x for v in acc.values() for x in v])),
                    "docs_read": float(np.mean(read)),
                }
            )
        print(f"seed {seed}: {st} done (EL acc {s.el_accuracy:.3f})", flush=True)
    return out


def _mean(rows: list[dict], keys: tuple[str, ...], vals: tuple[str, ...]) -> list[dict]:
    g: dict[tuple, list[dict]] = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in g.items():
        d = dict(zip(keys, key, strict=True))
        for v in vals:
            xs = [r[v] for r in rs if r.get(v) is not None]
            d[v] = float(np.mean(xs)) if xs else None
            d[v + "_sd"] = float(np.std(xs)) if xs else None
        out.append(d)
    return out


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kgrag")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-single", type=int, default=10)
    ap.add_argument("--k-hop", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(s, a.k_single, a.k_hop) for s in a.seeds]
    (a.out / "raw.json").write_text(json.dumps(runs, indent=1))

    e2e = _mean(
        [r for run in runs for r in run["e2e"]],
        ("linker", "system"),
        ("hop1", "hop2", "hop3", "overall", "docs_read"),
    )
    ret = _mean(
        [r for run in runs for r in run["retrieval"]],
        ("retriever", "mode", "k"),
        ("hop1", "hop2", "hop3", "ndcg"),
    )
    el_rows = []
    for st in EL:
        xs = [run["el"][st] for run in runs]
        el_rows.append(
            f"| {st} | {pct(np.mean([x['el_accuracy'] for x in xs]))} | "
            + " | ".join(
                pct(np.mean([x["el_by_ambiguity"].get(b, np.nan) for x in xs]))
                for b in ("unambiguous", "2-3 candidates", "4+")
            )
            + f" | {pct(np.mean([x['triple_precision'] for x in xs]))} | "
            f"{pct(np.mean([x['triple_recall'] for x in xs]))} |"
        )
    md = [
        f"Mean over world seeds {a.seeds} (480 questions per seed: 160 each of 1/2/3 hops).",
        "",
        "## End-to-end QA accuracy (shared extractive reader, no oracle)",
        "",
        "| entity linker | system | 1-hop | 2-hop | 3-hop | overall (± sd) | docs read |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['linker']} | {r['system']} | {pct(r['hop1'])} | {pct(r['hop2'])} | "
            f"{pct(r['hop3'])} | {pct(r['overall'])} ± {pct(r['overall_sd'])} | "
            f"{r['docs_read']:.1f} |"
            for r in e2e
        ],
        "",
        "## Entity linking and KG quality",
        "",
        "| linker | EL accuracy | unambiguous | 2-3 candidates | 4+ candidates | "
        "triple precision | triple recall |",
        "|---|---|---|---|---|---|---|",
        *el_rows,
        "",
        "## Retrieval: all supporting facts retrieved (oracle reader)",
        "",
        "| retriever | mode | k (per hop) | 1-hop | 2-hop | 3-hop | nDCG@k |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['retriever']} | {r['mode']} | {r['k']} | {pct(r['hop1'])} | {pct(r['hop2'])} "
            f"| {pct(r['hop3'])} | {f3(r['ndcg'])} |"
            for r in ret
        ],
    ]
    (a.out / "results.md").write_text("\n".join(md) + "\n")
    (a.out / "summary.json").write_text(json.dumps({"e2e": e2e, "retrieval": ret}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
