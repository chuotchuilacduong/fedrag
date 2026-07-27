"""Standalone smoke test for LinearRAG retrieval quality.

Two modes:
  1. Raw corpus (default) — indexes the FULL corpus from dataset/linearrag/
     as a single index. Direct test of the LinearRAG engine, decoupled from
     Stage A/trigraph/FL.
  2. Federated (--num-clients N) — indexes each client's own shard from
     processed/{dataset}/client_{i}/chunks.json (built by
     scripts/preprocess_data.py) separately, and queries the SAME shared
     question set (processed/{dataset}/questions.json) against each client's
     local index — since a client only ever sees its own passages.

Reports title-match hit@k / recall@k against gold evidence per client, plus
(federated mode) a "union hit" — whether ANY client's shard surfaces a gold
title, i.e. whether the answer is reachable at all in this partition.

Usage (fedrag env, from project root):
    python scripts/test_linearrag.py --dataset hotpotqa
    python scripts/test_linearrag.py --dataset hotpotqa --max-questions 100
    python scripts/test_linearrag.py --dataset hotpotqa --num-clients 3 --max-questions 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceLinearRAG

LINEARRAG_ROOT = _ROOT / "dataset" / "linearrag"
PROCESSED_ROOT = _ROOT / "processed"
ENCODER_MODEL = "all-MiniLM-L6-v2"
_PREFIX_RE = re.compile(r"^\d+:")


def _norm_title(text: str) -> str:
    text = _PREFIX_RE.sub("", str(text), count=1).strip()
    head, _, _ = text.partition(":")
    return head.strip().lower()


def _gold_titles(question: dict) -> set[str]:
    """Extract gold supporting-fact titles from the evidence field.

    Handles the [title, [sentence, ...]] pair format (hotpotqa/musique-style).
    """
    titles = set()
    for item in question.get("evidence") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
            titles.add(item[0].strip().lower())
    return titles


def _index_and_retrieve(chunks, questions, working_dir, dataset_name, encoder, top_k, tag):
    retriever = EvidenceLinearRAG(
        working_dir=working_dir,
        dataset_name=dataset_name,
        encoder=encoder,
        retrieval_top_k=top_k,
    )
    print(f"[{tag}] Indexing {len(chunks)} passages with LinearRAG...")
    t0 = time.time()
    retriever.index(chunks)
    print(f"[{tag}]   Indexed in {time.time()-t0:.0f}s")

    print(f"[{tag}] Running retrieval for {len(questions)} questions (top_k={top_k})...")
    t0 = time.time()
    results = retriever.retrieve_with_evidence(
        [{"question": q["question"], "answer": q["answer"]} for q in questions]
    )
    elapsed = time.time() - t0
    print(f"[{tag}]   Retrieved in {elapsed:.0f}s ({elapsed/max(len(questions),1)*1000:.0f}ms/question)")
    return results


def _score(questions, results, top_k):
    """Per-question retrieved title sets + hit/recall summary."""
    retrieved_title_sets = []
    hit_at_k = 0
    recall_sum = 0.0
    no_gold = 0
    for q, r in zip(questions, results):
        gold = _gold_titles(q)
        retrieved_titles = {_norm_title(p) for p in r.top_k_passages}
        retrieved_title_sets.append(retrieved_titles)
        if not gold:
            no_gold += 1
            continue
        found = gold & retrieved_titles
        if found:
            hit_at_k += 1
        recall_sum += len(found) / len(gold)

    n_scored = len(questions) - no_gold
    summary = {
        "n_scored": n_scored,
        "hit_at_k_pct": 100 * hit_at_k / n_scored if n_scored else None,
        "recall_at_k_pct": 100 * recall_sum / n_scored if n_scored else None,
    }
    return retrieved_title_sets, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=0,
                         help="If >0, test federated mode: index each client's own "
                              "shard from processed/{dataset}/client_i/chunks.json "
                              "and query the shared question set against each.")
    args = parser.parse_args()

    encoder = load_encoder(ENCODER_MODEL)

    if args.num_clients > 0:
        proc_dir = PROCESSED_ROOT / args.dataset
        questions = json.loads((proc_dir / "questions.json").read_text())
        if args.max_questions:
            questions = questions[: args.max_questions]
        print(f"Dataset: {args.dataset} (federated, {args.num_clients} clients) — "
              f"{len(questions)} shared questions")

        per_client_title_sets = []
        for cid in range(args.num_clients):
            client_dir = proc_dir / f"client_{cid}"
            chunks = json.loads((client_dir / "chunks.json").read_text())
            results = _index_and_retrieve(
                chunks, questions,
                working_dir=_ROOT / "processed" / "_linearrag_smoke" / args.dataset / f"client_{cid}",
                dataset_name=args.dataset,
                encoder=encoder, top_k=args.top_k, tag=f"client_{cid}",
            )
            titles, summary = _score(questions, results, args.top_k)
            per_client_title_sets.append(titles)
            print(f"[client_{cid}] Hit@{args.top_k}: {summary['hit_at_k_pct']:.1f}% | "
                  f"Recall@{args.top_k}: {summary['recall_at_k_pct']:.1f}%\n")

        # Union across clients — is the gold evidence reachable at all in this partition?
        union_hits = 0
        union_recall_sum = 0.0
        no_gold = 0
        for i, q in enumerate(questions):
            gold = _gold_titles(q)
            if not gold:
                no_gold += 1
                continue
            union_retrieved = set()
            for titles in per_client_title_sets:
                union_retrieved |= titles[i]
            found = gold & union_retrieved
            if found:
                union_hits += 1
            union_recall_sum += len(found) / len(gold)
        n_scored = len(questions) - no_gold
        print("=== Federated summary (union across all clients) ===")
        print(f"  Questions with gold evidence: {n_scored}/{len(questions)}")
        if n_scored:
            print(f"  Union Hit@{args.top_k} (>=1 gold title found by ANY client): "
                  f"{100*union_hits/n_scored:.1f}%")
            print(f"  Union Recall@{args.top_k}: {100*union_recall_sum/n_scored:.1f}%")
        return

    # --- single global-corpus mode ---
    ds_dir = LINEARRAG_ROOT / args.dataset
    chunks = json.loads((ds_dir / "chunks.json").read_text())
    questions = json.loads((ds_dir / "questions.json").read_text())
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"Dataset: {args.dataset} — {len(chunks)} passages, testing on {len(questions)} questions")
    results = _index_and_retrieve(
        chunks, questions,
        working_dir=_ROOT / "processed" / "_linearrag_smoke" / args.dataset,
        dataset_name=args.dataset, encoder=encoder, top_k=args.top_k, tag="global",
    )
    _, summary = _score(questions, results, args.top_k)

    print("\n=== Retrieval quality ===")
    print(f"  Questions with gold evidence: {summary['n_scored']}/{len(questions)}")
    if summary["n_scored"]:
        print(f"  Hit@{args.top_k} (>=1 gold title retrieved): {summary['hit_at_k_pct']:.1f}%")
        print(f"  Mean recall@{args.top_k} (fraction of gold titles retrieved): {summary['recall_at_k_pct']:.1f}%")

    print("\n=== Sample (first 3) ===")
    for q, r in zip(questions[:3], results[:3]):
        print(f"- Q: {q['question']}")
        print(f"  Gold answer: {q['answer']}")
        print(f"  Gold titles: {sorted(_gold_titles(q))}")
        print(f"  Retrieved (top-{args.top_k}): {[_norm_title(p) for p in r.top_k_passages]}")
        print()


if __name__ == "__main__":
    main()
