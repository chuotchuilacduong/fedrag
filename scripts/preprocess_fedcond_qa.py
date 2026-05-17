"""Preprocess a LinearRAG-format QA dataset into FedCondQA cache format.

Reads from dataset/linearrag/{dataset}/ (same format as LinearRAG's repo):
  chunks.json    — passage strings
  questions.json — QA pairs

Outputs to dataset/fedcond_qa/ (or --out_dir):
  records.jsonl                       — question/answer metadata per sample
  cached_graphs/{id}.pt               — evidence subgraph per question
  cached_condensed_graphs/{id}.pt     — server synthetic graph per question

Prerequisites:
  1. Run build_client_pipeline.py first to build trigraph.pt + synthetic_graph.pt
     for each client under processed/{dataset}/client_{m}/.
  2. This script indexes each client's chunks with EvidenceLinearRAG, then
     routes each question to the client that retrieves the most passages.

Usage (fedcond env, from project root):
    python scripts/preprocess_fedcond_qa.py --dataset hotpotqa
    python scripts/preprocess_fedcond_qa.py --dataset hotpotqa --max_questions 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_a_trigraph import load_trigraph
from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceLinearRAG
from fedcond_grag.client.stage_d_retrieve.evidence_graph_builder import build_evidence_graph

LINEARRAG_ROOT = _ROOT / "dataset" / "linearrag"
PROCESSED_ROOT = _ROOT / "processed"
ENCODER_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_synthetic_graph(client_dir: Path) -> Data:
    path = client_dir / "synthetic_graph.pt"
    if not path.exists():
        path = client_dir / "condensed_graph.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return Data(
        x=payload["x"].float(),
        edge_index=payload["edge_index"].long(),
        edge_weight=(payload.get("edge_weight") or torch.ones(payload["edge_index"].shape[1])).float(),
        node_type=payload["node_type"].long(),
    )


def index_client(client_dir: Path, dataset: str) -> EvidenceLinearRAG:
    """Build and index an EvidenceLinearRAG for one client."""
    chunks_path = client_dir / "chunks.json"
    with chunks_path.open() as f:
        chunks = json.load(f)
    encoder = load_encoder(ENCODER_MODEL)
    retriever = EvidenceLinearRAG(
        working_dir=client_dir / "linearrag_cache",
        dataset_name=dataset,
        encoder=encoder,
    )
    retriever.index(chunks)
    return retriever


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "2wikimultihop", "musique", "medical"])
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: dataset/fedcond_qa)")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--top_k_passages", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "dataset" / "fedcond_qa"
    graphs_dir = out_dir / "cached_graphs"
    condensed_dir = out_dir / "cached_condensed_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    condensed_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    qs_path = LINEARRAG_ROOT / args.dataset / "questions.json"
    with qs_path.open() as f:
        questions = json.load(f)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Questions: {len(questions)}")

    # Discover available clients
    dataset_dir = PROCESSED_ROOT / args.dataset
    client_dirs = sorted(
        p for p in dataset_dir.glob("client_*")
        if (p / "trigraph.pt").exists() and (p / "chunks.json").exists()
    )
    if not client_dirs:
        print(f"ERROR: No built clients found under {dataset_dir}. "
              "Run build_client_pipeline.py first.")
        sys.exit(1)
    print(f"Using {len(client_dirs)} client(s): {[p.name for p in client_dirs]}")

    # Index all available clients
    print("\nIndexing clients (LinearRAG.index)...")
    retrievers: list[EvidenceLinearRAG] = []
    trigraphs: list[Data] = []
    synthetic_graphs: list[Data] = []
    for cdir in client_dirs:
        print(f"  {cdir.name}...", end=" ", flush=True)
        t = time.time()
        retrievers.append(index_client(cdir, args.dataset))
        trigraphs.append(load_trigraph(cdir / "trigraph.pt"))
        synthetic_graphs.append(load_synthetic_graph(cdir))
        print(f"OK ({time.time()-t:.0f}s)")

    # Process questions
    records = []
    skipped = 0
    print(f"\nProcessing {len(questions)} questions...")
    t0 = time.time()

    for i, q in enumerate(questions):
        qid = str(q.get("id", q.get("_id", i)))
        question_text = q["question"]
        answer = q["answer"]

        # Route: try each client, pick the one that activates the most entities
        best_result = None
        best_client_idx = 0
        best_score = -1

        for ci, retriever in enumerate(retrievers):
            try:
                results = retriever.retrieve_with_evidence([{"question": question_text, "answer": answer}])
                r = results[0]
                score = len(r.actived_entities)
                if score > best_score:
                    best_score = score
                    best_result = r
                    best_client_idx = ci
            except Exception:
                continue

        if best_result is None or len(best_result.actived_entities) == 0:
            # Fallback: use client 0, accept DPR-only result
            try:
                results = retrievers[0].retrieve_with_evidence([{"question": question_text, "answer": answer}])
                best_result = results[0]
                best_client_idx = 0
            except Exception:
                skipped += 1
                continue

        # Build evidence graph
        try:
            ev = build_evidence_graph(
                trigraphs[best_client_idx],
                best_result,
                top_k=args.top_k_passages,
            )
        except Exception:
            skipped += 1
            continue

        # Save evidence graph (strip node_text for upload cleanliness — keep for local training)
        torch.save(ev.data, graphs_dir / f"{qid}.pt")

        # Save condensed graph (synthetic graph from that client's server condensation)
        torch.save(synthetic_graphs[best_client_idx], condensed_dir / f"{qid}.pt")

        records.append({
            "id": qid,
            "question": question_text,
            "answer": answer,
            "retrieved_passages": best_result.top_k_passages,
            "client_id": best_client_idx,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(questions)} — {elapsed:.0f}s elapsed, {skipped} skipped")

    # Write records.jsonl
    records_path = out_dir / "records.jsonl"
    with records_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  {len(records)} records written to {records_path}")
    print(f"  {skipped} questions skipped (no retrieval result)")
    print(f"  Evidence graphs: {graphs_dir}")
    print(f"  Condensed graphs: {condensed_dir}")
    print(f"\nRun training with:")
    print(f"  python main.py train --dataset fedcond_qa --model_name dual_graph_llm \\")
    print(f"      --gnn_in_dim 384 --gnn_hidden_dim 384 \\")
    print(f"      --gnn_in_dim_c 384 --gnn_hidden_dim_c 384")


if __name__ == "__main__":
    main()
