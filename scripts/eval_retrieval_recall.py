"""Standalone retrieval R@k evaluation for FedCondGraphRAG.

Re-runs LinearRAG retrieval (entity activation → BFS → PPR → passage ranking)
per client with a configurable top-k — independent of ppr_node_map.pt, which
only stores top-5 — and measures passage recall R@k against the gold evidence
passages in dataset/fedcond_qa/records.jsonl.

Does NOT touch ppr_node_map.pt or any other training artifact. Retrieved
titles are checkpointed per question, so a killed run resumes for free.

Usage (fedcond env, from project root):
  # per-client worker (resumable):
  python scripts/eval_retrieval_recall.py --dataset 2wikimultihop --client-id 0 --top-k 10
  # final aggregation once all clients are done:
  python scripts/eval_retrieval_recall.py --dataset 2wikimultihop --top-k 10 --aggregate
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title, retrieval_recall_at_k

PROCESSED_ROOT = _ROOT / "processed"
QA_RECORDS = _ROOT / "dataset" / "fedcond_qa" / "records.jsonl"
ENCODER_MODEL = "all-MiniLM-L6-v2"
BATCH = 500


def _client_dirs(dataset: str) -> list[Path]:
    return sorted(
        p for p in (PROCESSED_ROOT / dataset).glob("client_*")
        if (p / "chunks.json").exists()
    )


def _titles_path(dataset: str, client_id: int, top_k: int) -> Path:
    return PROCESSED_ROOT / dataset / f"client_{client_id}" / f"retrieval_top{top_k}_titles.jsonl"


def run_client(dataset: str, client_id: int, n_clients: int, top_k: int,
               limit: int | None = None) -> None:
    from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
    from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceLinearRAG

    client_dir = PROCESSED_ROOT / dataset / f"client_{client_id}"
    questions = json.loads((PROCESSED_ROOT / dataset / "questions.json").read_text())
    mine = [(i, questions[i]) for i in range(len(questions)) if i % n_clients == client_id]

    out_path = _titles_path(dataset, client_id, top_k)
    done: set[int] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue  # torn tail line from a killed run — will be redone
    todo = [(i, q) for i, q in mine if i not in done]
    if limit:
        todo = todo[:limit]
    print(f"client_{client_id}: {len(mine)} questions, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    print(f"  Indexing chunks with LinearRAG (top_k={top_k})...", flush=True)
    t0 = time.time()
    chunks = json.loads((client_dir / "chunks.json").read_text())
    retriever = EvidenceLinearRAG(
        working_dir=client_dir / "linearrag_cache",
        dataset_name=dataset,
        encoder=load_encoder(ENCODER_MODEL),
        retrieval_top_k=top_k,
    )
    retriever.index(chunks)
    del chunks
    gc.collect()
    print(f"  Indexed in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    skipped = 0
    with out_path.open("a") as f:
        for bs in range(0, len(todo), BATCH):
            b = todo[bs : bs + BATCH]
            try:
                results = retriever.retrieve_with_evidence(
                    [{"question": q["question"], "answer": q.get("answer", "")} for _, q in b]
                )
            except Exception as exc:
                skipped += len(b)
                print(f"  batch at {bs} failed: {exc}", flush=True)
                continue
            for (gi, _), r in zip(b, results):
                titles = [norm_passage_title(p) for p in r.top_k_passages[:top_k]]
                f.write(json.dumps({"idx": gi, "titles": titles}) + "\n")
            f.flush()
            n_done = bs + len(b)
            el = time.time() - t0
            eta_h = (len(todo) - n_done) * el / n_done / 3600
            print(f"  {n_done}/{len(todo)} — {el:.0f}s ({el/n_done*1000:.0f}ms/q), "
                  f"ETA {eta_h:.1f}h, {skipped} skipped", flush=True)
    print(f"  Saved {out_path}", flush=True)


def aggregate(dataset: str, n_clients: int, top_k: int, ks: tuple[int, ...]) -> None:
    gold_sets: list[set[str]] = []
    with QA_RECORDS.open() as f:
        for line in f:
            r = json.loads(line)
            gold_sets.append(
                {norm_passage_title(p) for p in r.get("retrieved_passages", []) if p}
            )

    tot = {k: 0.0 for k in ks}
    n_scored = 0
    missing = 0
    for cid in range(n_clients):
        path = _titles_path(dataset, cid, top_k)
        if not path.exists():
            print(f"WARNING: {path} missing — client_{cid} not evaluated yet")
            continue
        c_tot = {k: 0.0 for k in ks}
        c_n = 0
        seen: set[int] = set()
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = row["idx"]
                if idx in seen:
                    continue
                seen.add(idx)
                rec = retrieval_recall_at_k(gold_sets[idx], row["titles"], ks)
                if not rec:
                    continue
                c_n += 1
                for k in ks:
                    c_tot[k] += rec[k]
        expected = len(gold_sets) // n_clients
        missing += max(0, expected - c_n)
        n_scored += c_n
        for k in ks:
            tot[k] += c_tot[k]
        print(f"client_{cid}: n={c_n}  "
              + "  ".join(f"R@{k}={100*c_tot[k]/max(c_n,1):.2f}%" for k in ks))

    print(f"\n=== {dataset} retrieval recall ({n_scored}/{len(gold_sets)} scored"
          + (f", ~{missing} missing" if missing else "") + ") ===")
    print("  ".join(f"R@{k}={100*tot[k]/max(n_scored,1):.2f}%" for k in ks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2wikimultihop")
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,2,5,10",
                        help="Comma-separated k values for R@k (aggregate step)")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pending questions (smoke test)")
    args = parser.parse_args()

    n_clients = len(_client_dirs(args.dataset))
    if not n_clients:
        print(f"ERROR: no client dirs under {PROCESSED_ROOT / args.dataset}")
        sys.exit(1)

    if args.aggregate:
        ks = tuple(int(k) for k in args.ks.split(","))
        aggregate(args.dataset, n_clients, args.top_k, ks)
        return
    if args.client_id is None:
        print("ERROR: pass --client-id N (worker) or --aggregate")
        sys.exit(1)
    run_client(args.dataset, args.client_id, n_clients, args.top_k, limit=args.limit)


if __name__ == "__main__":
    main()
