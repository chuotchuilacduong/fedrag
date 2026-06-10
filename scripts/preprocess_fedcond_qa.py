"""Build per-client PPR passage node maps for FedCondGraphRAG.

For each client, runs LinearRAG (entity activation → BFS → PPR) on every
question using that client's own chunks and trigraph. Saves the resulting
trigraph node IDs as:

    processed/{dataset}/client_{c}/ppr_node_map.pt   [Q, top_k]  int64
        Value: local trigraph node_id of the PPR-selected passage, or -1.

This is purely per-client: client c only uses its own chunks and trigraph.
No cross-client data sharing occurs.

Prerequisites:
    Run build_client_pipeline.py first to build trigraph.pt for each client.

Usage (fedcond env, from project root):
    python scripts/preprocess_fedcond_qa.py --dataset musique
    python scripts/preprocess_fedcond_qa.py --dataset musique --max_questions 500
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

import torch

from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceLinearRAG

LINEARRAG_ROOT = _ROOT / "dataset" / "linearrag"
PROCESSED_ROOT = _ROOT / "processed"
ENCODER_MODEL = "all-MiniLM-L6-v2"
_PREFIX_RE = re.compile(r"^\d+:")


def _norm_title(text: str) -> str:
    """Normalise a passage text to its title for node matching."""
    text = _PREFIX_RE.sub("", str(text), count=1).strip()
    head, _, _ = text.partition(":")
    return head.strip().lower()


def _build_title_to_node(trigraph_path: Path) -> dict[str, int]:
    """passage title → trigraph node_id for all passage nodes (type==2)."""
    g = torch.load(trigraph_path, map_location="cpu", weights_only=False)
    node_texts: list[str] = g["node_text"]
    node_types: list[int] = g["node_type"].tolist()
    title_to_node: dict[str, int] = {}
    for i, nt in enumerate(node_types):
        if nt == 2:
            title = _norm_title(node_texts[i])
            if title and title not in title_to_node:
                title_to_node[title] = i
    return title_to_node


def _index_client(client_dir: Path, dataset: str) -> EvidenceLinearRAG:
    chunks_path = client_dir / "chunks.json"
    chunks = json.loads(chunks_path.read_text())
    encoder = load_encoder(ENCODER_MODEL)
    retriever = EvidenceLinearRAG(
        working_dir=client_dir / "linearrag_cache",
        dataset_name=dataset,
        encoder=encoder,
    )
    retriever.index(chunks)
    return retriever


def process_client(
    client_dir: Path,
    dataset: str,
    questions: list[dict],
    top_k: int,
) -> None:
    """Run per-query PPR on one client and save ppr_node_map.pt."""
    cid = client_dir.name
    print(f"\n=== {cid} ===")

    print(f"  Building title→node map from trigraph...", flush=True)
    title_to_node = _build_title_to_node(client_dir / "trigraph.pt")
    print(f"  {len(title_to_node)} passage nodes indexed", flush=True)

    print(f"  Indexing chunks with LinearRAG...", flush=True)
    t0 = time.time()
    retriever = _index_client(client_dir, dataset)
    print(f"  Indexed in {time.time()-t0:.0f}s", flush=True)

    Q = len(questions)
    node_map = torch.full((Q, top_k), -1, dtype=torch.long)
    n_hits = 0
    skipped = 0

    print(f"  Running PPR for {Q} questions...", flush=True)
    t0 = time.time()
    BATCH = 500
    for batch_start in range(0, Q, BATCH):
        batch_qs = questions[batch_start: batch_start + BATCH]
        try:
            results = retriever.retrieve_with_evidence(
                [{"question": q["question"], "answer": q["answer"]} for q in batch_qs]
            )
        except Exception as e:
            skipped += len(batch_qs)
            print(f"  batch {batch_start} failed: {e}", flush=True)
            continue

        for j, r in enumerate(results):
            i = batch_start + j
            slot = 0
            for passage_text in r.top_k_passages:
                if slot >= top_k:
                    break
                title = _norm_title(passage_text)
                node_id = title_to_node.get(title, -1)
                node_map[i, slot] = node_id
                if node_id >= 0:
                    n_hits += 1
                slot += 1

        elapsed = time.time() - t0
        done = min(batch_start + BATCH, Q)
        print(f"  {done}/{Q} — {elapsed:.0f}s elapsed ({elapsed/max(done,1)*1000:.0f}ms/q), {skipped} skipped",
              flush=True)

    out_path = client_dir / "ppr_node_map.pt"
    torch.save(node_map, out_path)
    total = Q * top_k
    print(f"  Saved {out_path}  shape={tuple(node_map.shape)}", flush=True)
    print(f"  Hit rate: {n_hits}/{total} = {100*n_hits/max(total,1):.1f}%", flush=True)
    print(f"  Skipped: {skipped}/{Q} questions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "2wikimultihop", "musique", "medical"])
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--top_k_passages", type=int, default=5,
                        help="Top-k PPR passages to map per client per question")
    parser.add_argument("--client-id", type=int, default=None,
                        help="Process only this client ID (0-indexed). If omitted, all clients.")
    args = parser.parse_args()

    qs_path = PROCESSED_ROOT / args.dataset / "questions.json"
    questions: list[dict] = json.loads(qs_path.read_text())
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Questions: {len(questions)}")

    dataset_dir = PROCESSED_ROOT / args.dataset
    client_dirs = sorted(
        p for p in dataset_dir.glob("client_*")
        if (p / "trigraph.pt").exists() and (p / "chunks.json").exists()
    )
    if not client_dirs:
        print(f"ERROR: No built clients under {dataset_dir}. "
              "Run build_client_pipeline.py first.")
        sys.exit(1)
    if args.client_id is not None:
        client_dirs = [p for p in client_dirs if p.name == f"client_{args.client_id}"]
        if not client_dirs:
            print(f"ERROR: client_{args.client_id} not found.")
            sys.exit(1)
    print(f"Found {len(client_dirs)} client(s): {[p.name for p in client_dirs]}")

    t_total = time.time()
    for cdir in client_dirs:
        process_client(cdir, args.dataset, questions, args.top_k_passages)

    print(f"\nAll done in {time.time()-t_total:.0f}s")
    print("Each client's ppr_node_map.pt is saved under its own processed/ directory.")


if __name__ == "__main__":
    main()
