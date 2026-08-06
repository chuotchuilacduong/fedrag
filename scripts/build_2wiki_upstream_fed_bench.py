"""Federated version of the upstream HippoRAG/PropRAG 2wiki benchmark.

Takes the exact upstream evaluation set (reproduce/dataset/2wikimultihopqa.json,
1000 questions + 6119-passage corpus) and distributes it across our 3 client
shards:

- Every upstream question exists in our questions.json (matched by id); it is
  assigned to the client that owns its global index (idx % num_clients), the
  same ownership rule as the FL pipeline. Client question lists are truncated
  to equal length so the interleaved bench keeps bench_idx % 3 == client_id.
- Per-client corpus = our shard chunks whose "Title\\nbody" form appears in the
  upstream corpus (the shards are disjoint, so upstream docs partition cleanly;
  ~6.07k of 6119 exist in our corpus).

Because the corpus is the upstream one, the PropRAG-precomputed OpenIE file
(dataset/ner/2wikimultihopqa/openie_results_ner_meta-llama_llama-3.3-70b-instruct.json,
Llama-3.3-70B extractions) covers ~100%% of docs — HippoRAG indexes with
published-grade OpenIE and needs no local LLM there.

Usage:
  python scripts/build_2wiki_upstream_fed_bench.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.graph_rag.hipporag_local import _chunk_to_doc  # noqa: E402
from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", default="2wikimultihop")
    ap.add_argument("--bench-name", default="2wikimultihop_upfed")
    ap.add_argument("--num-clients", type=int, default=3)
    args = ap.parse_args()

    src = _ROOT / "processed" / args.source_dataset
    dst = _ROOT / "processed" / args.bench_name
    up_dir = _ROOT / "third_party" / "HippoRAG" / "reproduce" / "dataset"

    ours = json.loads((src / "questions.json").read_text())
    id2idx = {q["id"]: i for i, q in enumerate(ours)}
    up_questions = json.loads((up_dir / "2wikimultihopqa.json").read_text())
    up_corpus = json.loads((up_dir / "2wikimultihopqa_corpus.json").read_text())
    up_docs = {f"{d['title']}\n{d['text']}" for d in up_corpus}

    per_client: list[list[dict]] = [[] for _ in range(args.num_clients)]
    missing = 0
    for uq in up_questions:
        gi = id2idx.get(uq["_id"])
        if gi is None:
            missing += 1
            continue
        q = dict(ours[gi])
        q["_global_idx"] = gi
        per_client[gi % args.num_clients].append(q)
    for qs in per_client:
        qs.sort(key=lambda q: q["_global_idx"])
    m = min(len(qs) for qs in per_client)
    print(f"matched {1000 - missing}/1000 upstream questions; "
          f"per-client {[len(qs) for qs in per_client]} -> truncated to {m} each")

    meta = {"source": "upstream HippoRAG reproduce/dataset 2wikimultihopqa", "clients": []}
    for cid in range(args.num_clients):
        chunks = json.loads((src / f"client_{cid}" / "chunks.json").read_text())
        sub_corpus = [c for c in chunks if _chunk_to_doc(str(c))[0] in up_docs]
        shard_titles = {norm_passage_title(c) for c in sub_corpus}

        qs = per_client[cid][:m]
        oracle = []
        for q in qs:
            gt = {norm_passage_title(t) for t, _ in q.get("evidence", [])}
            if gt:
                oracle.append(len(gt & shard_titles) / len(gt))
        oracle_recall = sum(oracle) / max(len(oracle), 1)

        out_dir = dst / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps(sub_corpus, ensure_ascii=False))
        print(f"client_{cid}: {len(qs)} questions, {len(sub_corpus)} docs, "
              f"oracle recall {100 * oracle_recall:.2f}%")
        meta["clients"].append({
            "client_id": cid,
            "num_questions": len(qs),
            "num_docs": len(sub_corpus),
            "oracle_recall": round(oracle_recall, 4),
        })

    bench_questions = []
    for j in range(m):
        for cid in range(args.num_clients):
            bench_questions.append(per_client[cid][j])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "questions.json").write_text(json.dumps(bench_questions, ensure_ascii=False))
    (dst / "bench_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(bench_questions)} questions -> {dst}")


if __name__ == "__main__":
    main()
