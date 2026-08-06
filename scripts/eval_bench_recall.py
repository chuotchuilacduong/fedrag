"""Aggregate R@k on the sub-benchmark for HippoRAG and/or LinearRAG runs.

Gold titles come from each bench question's `evidence` field (identical to
records.jsonl retrieved_passages titles), normalised with norm_passage_title —
the same metric used for the full-corpus LinearRAG numbers (R@5 ~24%).

Sources:
- HippoRAG: output/baselines/hipporag_local/<bench>/<split>/client_<id>/predictions.jsonl
  ("docs" = rank-ordered "Title\\nbody" strings, "idx" = bench index)
- LinearRAG: processed/<bench>/client_<id>/retrieval_top<K>_titles.jsonl
  (written by scripts/eval_retrieval_recall.py; "titles" already normalised)

Usage:
  python scripts/eval_bench_recall.py --bench 2wikimultihop_bench \
      --hipporag-split test --linearrag-top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title, retrieval_recall_at_k  # noqa: E402

KS = (1, 2, 5, 10)


def _gold_sets(bench_dir: Path) -> list[set[str]]:
    questions = json.loads((bench_dir / "questions.json").read_text())
    return [
        {norm_passage_title(t) for t, _ in q.get("evidence", []) if t}
        for q in questions
    ]


def _report(name: str, per_client: dict[int, list[dict]], gold_sets: list[set[str]]) -> None:
    print(f"\n=== {name} ===")
    tot = {k: 0.0 for k in KS}
    n_tot = 0
    for cid in sorted(per_client):
        rows = per_client[cid]
        c_tot = {k: 0.0 for k in KS}
        c_n = 0
        for row in rows:
            rec = retrieval_recall_at_k(gold_sets[row["idx"]], row["titles"], KS)
            if not rec:
                continue
            c_n += 1
            for k in KS:
                c_tot[k] += rec[k]
        n_tot += c_n
        for k in KS:
            tot[k] += c_tot[k]
        print(f"  client_{cid}: n={c_n}  "
              + "  ".join(f"R@{k}={100 * c_tot[k] / max(c_n, 1):.2f}%" for k in KS))
    print(f"  OVERALL (n={n_tot}): "
          + "  ".join(f"R@{k}={100 * tot[k] / max(n_tot, 1):.2f}%" for k in KS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="2wikimultihop_bench")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--hipporag-split", default=None,
                    help="Split dir under output/baselines/hipporag_local/<bench>/ to score.")
    ap.add_argument("--linearrag-top-k", type=int, default=None,
                    help="Score processed/<bench>/client_*/retrieval_top<K>_titles.jsonl.")
    args = ap.parse_args()

    bench_dir = _ROOT / "processed" / args.bench
    gold_sets = _gold_sets(bench_dir)

    # Oracle ceiling: fraction of gold titles present on the assigned shard.
    per_client_titles: dict[int, set[str]] = {}
    oracle = {k: [] for k in ("all",)}
    for cid in range(args.num_clients):
        chunks = json.loads((bench_dir / f"client_{cid}" / "chunks.json").read_text())
        per_client_titles[cid] = {norm_passage_title(c) for c in chunks}
    hits = []
    for i, gold in enumerate(gold_sets):
        if not gold:
            continue
        shard = per_client_titles[i % args.num_clients]
        hits.append(len(gold & shard) / len(gold))
    print(f"Oracle recall ceiling on bench: {100 * sum(hits) / max(len(hits), 1):.2f}% "
          f"(n={len(hits)})")

    if args.hipporag_split:
        per_client: dict[int, list[dict]] = {}
        base = _ROOT / "output" / "baselines" / "hipporag_local" / args.bench / args.hipporag_split
        for cid in range(args.num_clients):
            path = base / f"client_{cid}" / "predictions.jsonl"
            if not path.exists():
                print(f"WARNING: {path} missing")
                continue
            rows = []
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    titles = [norm_passage_title(d.split("\n", 1)[0]) for d in r.get("docs", [])]
                    rows.append({"idx": r["idx"], "titles": titles})
            per_client[cid] = rows
        _report(f"HippoRAG ({args.hipporag_split})", per_client, gold_sets)

    if args.linearrag_top_k:
        per_client = {}
        for cid in range(args.num_clients):
            path = bench_dir / f"client_{cid}" / f"retrieval_top{args.linearrag_top_k}_titles.jsonl"
            if not path.exists():
                print(f"WARNING: {path} missing")
                continue
            rows = []
            seen: set[int] = set()
            with path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r["idx"] in seen:
                        continue
                    seen.add(r["idx"])
                    rows.append({"idx": r["idx"], "titles": r["titles"]})
            per_client[cid] = rows
        _report(f"LinearRAG (top{args.linearrag_top_k})", per_client, gold_sets)


if __name__ == "__main__":
    main()
