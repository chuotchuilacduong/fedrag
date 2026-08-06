"""Build a scaled-down, internally-fair retrieval benchmark from 2wiki shards.

Design (mirrors how upstream HippoRAG built its 2wiki subset: sampled
questions + their gold passages + distractors):

- Sample N test-split questions per client, using the same ownership rule as
  the FL pipeline (global_idx % num_clients == client_id).
- Per-client sub-corpus = every shard chunk whose title is gold for one of
  that client's sampled questions, plus random distractor chunks from the
  same shard up to --docs-per-client.

Because distractors and gold docs both come from the client's own shard, the
federated partition effect is preserved: a question whose gold passages never
existed on its shard still cannot be answered (oracle recall ceiling ~41%).
Both HippoRAG (adapter) and LinearRAG (eval_retrieval_recall.py) can then run
on the same processed/<name>/ layout for an apples-to-apples R@k.

Output questions.json is interleaved c0,c1,c2,... so that
bench_idx % num_clients == client_id keeps assignments consistent. Each
question dict gains "_global_idx" (index into the full questions.json).

Usage:
  python scripts/build_2wiki_sub_benchmark.py \
    --questions-per-client 500 --docs-per-client 6000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", default="2wikimultihop")
    ap.add_argument("--bench-name", default="2wikimultihop_bench")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--questions-per-client", type=int, default=500)
    ap.add_argument("--docs-per-client", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = _ROOT / "processed" / args.source_dataset
    dst = _ROOT / "processed" / args.bench_name
    questions = json.loads((src / "questions.json").read_text())
    n = len(questions)

    split_dir = _ROOT / "dataset" / "fedcond_qa" / "split"
    test_path = split_dir / "test_indices.txt"
    if test_path.exists():
        test_indices = [int(x) for x in test_path.read_text().split() if x.strip()]
        test_indices = [i for i in test_indices if 0 <= i < n]
    else:
        test_indices = list(range(int(0.9 * n), n))
    print(f"{n} questions total, {len(test_indices)} in test split")

    rng = random.Random(args.seed)
    per_client_questions: list[list[dict]] = []
    meta = {"seed": args.seed, "source": args.source_dataset, "clients": []}

    for cid in range(args.num_clients):
        mine = [i for i in test_indices if i % args.num_clients == cid]
        picked = sorted(rng.sample(mine, args.questions_per_client))
        qs = []
        for gi in picked:
            q = dict(questions[gi])
            q["_global_idx"] = gi
            qs.append(q)
        per_client_questions.append(qs)

        gold_titles: set[str] = set()
        for q in qs:
            gold_titles |= {norm_passage_title(t) for t, _ in q.get("evidence", [])}

        chunks = json.loads((src / f"client_{cid}" / "chunks.json").read_text())
        gold_chunks, other_chunks = [], []
        n_gold_titles_on_shard: set[str] = set()
        for c in chunks:
            t = norm_passage_title(c)
            if t in gold_titles:
                gold_chunks.append(c)
                n_gold_titles_on_shard.add(t)
            else:
                other_chunks.append(c)

        n_distractors = max(0, args.docs_per_client - len(gold_chunks))
        distractors = rng.sample(other_chunks, min(n_distractors, len(other_chunks)))
        sub_corpus = gold_chunks + distractors
        rng.shuffle(sub_corpus)

        # Oracle ceiling on this bench = avg fraction of each question's gold
        # titles present on its shard (unchanged from the full-corpus setting).
        oracle = []
        for q in qs:
            gt = {norm_passage_title(t) for t, _ in q.get("evidence", [])}
            if gt:
                oracle.append(len(gt & n_gold_titles_on_shard) / len(gt))
        oracle_recall = sum(oracle) / max(len(oracle), 1)

        out_dir = dst / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps(sub_corpus, ensure_ascii=False))
        print(
            f"client_{cid}: {len(qs)} questions, {len(gold_chunks)} gold chunks "
            f"({len(n_gold_titles_on_shard)}/{len(gold_titles)} gold titles on shard), "
            f"{len(sub_corpus)} total docs, oracle recall {100 * oracle_recall:.2f}%"
        )
        meta["clients"].append({
            "client_id": cid,
            "num_questions": len(qs),
            "num_docs": len(sub_corpus),
            "num_gold_chunks": len(gold_chunks),
            "oracle_recall": round(oracle_recall, 4),
        })

    # Interleave so bench_idx % num_clients == client_id.
    bench_questions = []
    for j in range(args.questions_per_client):
        for cid in range(args.num_clients):
            bench_questions.append(per_client_questions[cid][j])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "questions.json").write_text(json.dumps(bench_questions, ensure_ascii=False))
    (dst / "bench_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(bench_questions)} questions -> {dst}")


if __name__ == "__main__":
    main()
