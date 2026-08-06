"""Federate an upstream HippoRAG benchmark (hotpotqa / musique) across 3 clients.

Unlike 2wikimultihop (see build_2wiki_upstream_fed_bench.py), our local FL
shards don't cover these corpora (musique: 42% doc coverage, 0/1000 question
ids), so the benchmark itself is partitioned: the upstream corpus is shuffled
(seed) and split into `num_clients` disjoint shards, and the 1000 upstream
questions are assigned round-robin (bench_idx % num_clients == client_id),
mirroring the ownership rule of the FL pipeline. Gold docs land on the
"right" shard only by chance, preserving the federated-partition ceiling
(oracle recall ~1/3 + co-location luck) that the 2wiki benches show.

Chunks are written as "<i>:<title>: <text>" (LinearRAG convention); the
HippoRAG adapter reconstructs "<title>\\n<text>" from them, which matches the
upstream corpus format exactly — so precomputed OpenIE files keyed by passage
text still hit (titles containing ':' are the exception; counted and printed).

Questions are normalised to {id, question, answer, evidence=[[title,[sents]]]}
so eval_bench_recall.py and the adapter's `_gold_docs` evidence path work for
every dataset.

Usage:
  python scripts/build_upstream_fed_bench.py --dataset hotpotqa
  python scripts/build_upstream_fed_bench.py --dataset musique
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.graph_rag.hipporag_local import _chunk_to_doc  # noqa: E402
from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402

UP_DIR = _ROOT / "third_party" / "HippoRAG" / "reproduce" / "dataset"


def _gold_evidence(sample: dict) -> list:
    """[[title, [sentences]], ...] for the supporting passages of a question."""
    if "supporting_facts" in sample:  # hotpotqa
        ctx = {t: sents for t, sents in sample.get("context", [])}
        titles = list(dict.fromkeys(t for t, _ in sample["supporting_facts"]))
        return [[t, ctx.get(t, [])] for t in titles]
    if "paragraphs" in sample:  # musique
        out = []
        for p in sample["paragraphs"]:
            if p.get("is_supporting"):
                out.append([p.get("title", ""), [p.get("paragraph_text", "")]])
        return out
    raise ValueError(f"Unknown gold format: {list(sample.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--bench-name", default=None)
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench_name = args.bench_name or f"{args.dataset}_upfed"
    dst = _ROOT / "processed" / bench_name
    corpus = json.loads((UP_DIR / f"{args.dataset}_corpus.json").read_text())
    up_questions = json.loads((UP_DIR / f"{args.dataset}.json").read_text())

    rng = random.Random(args.seed)
    chunk_strings = [f"{i}:{d['title']}: {d['text']}" for i, d in enumerate(corpus)]
    colon_mismatch = sum(
        1 for i, d in enumerate(corpus)
        if _chunk_to_doc(chunk_strings[i])[0] != f"{d['title']}\n{d['text']}"
    )
    order = list(range(len(chunk_strings)))
    rng.shuffle(order)
    shards = [sorted(order[cid::args.num_clients]) for cid in range(args.num_clients)]

    m = len(up_questions) // args.num_clients
    bench_questions = []
    for j in range(m):
        for cid in range(args.num_clients):
            uq = up_questions[j * args.num_clients + cid]
            bench_questions.append({
                "id": str(uq.get("_id", uq.get("id", len(bench_questions)))),
                "source": args.dataset,
                "question": uq["question"],
                "answer": uq.get("answer", ""),
                "evidence": _gold_evidence(uq),
            })

    meta = {"seed": args.seed, "source": f"upstream {args.dataset}", "clients": []}
    shard_title_sets = []
    for cid in range(args.num_clients):
        sub = [chunk_strings[i] for i in shards[cid]]
        out_dir = dst / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps(sub, ensure_ascii=False))
        shard_title_sets.append({norm_passage_title(c) for c in sub})

    for cid in range(args.num_clients):
        qs = [q for j, q in enumerate(bench_questions) if j % args.num_clients == cid]
        oracle = []
        for q in qs:
            gt = {norm_passage_title(t) for t, _ in q["evidence"] if t}
            if gt:
                oracle.append(len(gt & shard_title_sets[cid]) / len(gt))
        oracle_recall = sum(oracle) / max(len(oracle), 1)
        print(f"client_{cid}: {len(qs)} questions, {len(shards[cid])} docs, "
              f"oracle recall {100 * oracle_recall:.2f}%")
        meta["clients"].append({
            "client_id": cid,
            "num_questions": len(qs),
            "num_docs": len(shards[cid]),
            "oracle_recall": round(oracle_recall, 4),
        })

    (dst / "questions.json").write_text(json.dumps(bench_questions, ensure_ascii=False))
    (dst / "bench_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(bench_questions)} questions -> {dst} "
          f"({colon_mismatch} docs with ':' in title won't hit OpenIE caches)")


if __name__ == "__main__":
    main()
