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
import gc
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


def _index_client(client_dir: Path, dataset: str, top_k: int = 5) -> EvidenceLinearRAG:
    chunks_path = client_dir / "chunks.json"
    chunks = json.loads(chunks_path.read_text())
    encoder = load_encoder(ENCODER_MODEL)
    retriever = EvidenceLinearRAG(
        working_dir=client_dir / "linearrag_cache",
        dataset_name=dataset,
        encoder=encoder,
        retrieval_top_k=top_k,
    )
    retriever.index(chunks)
    return retriever


def process_client(
    client_dir: Path,
    dataset: str,
    questions: list[dict],
    top_k: int,
    client_id: int = 0,
    n_clients: int = 1,
    all_questions: bool = False,
    out_suffix: str | None = None,
) -> None:
    """Run per-query PPR on one client and save ppr_node_map.pt.

    Each client only processes questions where global_idx % n_clients == client_id,
    matching the FL training split in trainer.py (cid_indices = [i for i in train_idx
    if i % n == cid]).  The saved tensor keeps global shape (Q_total, top_k) so the
    client can index by dataset integer index directly; rows belonging to other clients
    remain -1 (gracefully handled as "no PPR anchors").

    Checkpoints after every batch so a killed run can resume.
    progress.json next_batch tracks LOCAL question index (0..len(my_questions)-1).
    """
    cid = client_dir.name
    suffix = out_suffix if out_suffix is not None else ("_all" if all_questions else "")
    out_path = client_dir / f"ppr_node_map{suffix}.pt"
    progress_path = client_dir / f"ppr_progress{suffix}.json"

    Q_total = len(questions)
    # Default: only questions that belong to this client in the FL split.
    # --all-questions: every question (broadcast eval), separate output file.
    if all_questions:
        my_indices: list[int] = list(range(Q_total))
    else:
        my_indices = [i for i in range(Q_total) if i % n_clients == client_id]
    my_questions: list[dict] = [questions[i] for i in my_indices]
    Q_local = len(my_questions)

    print(f"\n=== {cid} === ({Q_local}/{Q_total} questions, "
          f"{'ALL (broadcast)' if all_questions else f'idx%{n_clients}=={client_id}'})",
          flush=True)

    # Full-size tensor (global indices) so client can do ppr_node_map[dataset_idx]
    node_map = torch.full((Q_total, top_k), -1, dtype=torch.long)
    start_batch = 0
    n_hits = 0
    skipped = 0

    # Resume from checkpoint if available
    if progress_path.exists() and out_path.exists():
        try:
            partial = torch.load(out_path, weights_only=True)
            prog = json.loads(progress_path.read_text())
            if partial.shape == node_map.shape and prog.get("n_clients") == n_clients:
                node_map = partial
                start_batch = prog.get("next_batch", 0)
                n_hits = prog.get("n_hits", 0)
                skipped = prog.get("skipped", 0)
                print(f"  Resuming from local question {start_batch}/{Q_local} "
                      f"({100*start_batch/Q_local:.1f}% done)", flush=True)
            else:
                print(f"  Checkpoint incompatible (shape or n_clients mismatch), "
                      f"starting fresh", flush=True)
        except Exception as e:
            print(f"  Could not load checkpoint ({e}), starting fresh", flush=True)

    print(f"  Building title→node map from trigraph...", flush=True)
    title_to_node = _build_title_to_node(client_dir / "trigraph.pt")
    print(f"  {len(title_to_node)} passage nodes indexed", flush=True)

    print(f"  Indexing chunks with LinearRAG...", flush=True)
    t0 = time.time()
    retriever = _index_client(client_dir, dataset, top_k=top_k)
    gc.collect()
    print(f"  Indexed in {time.time()-t0:.0f}s", flush=True)

    print(f"  Running PPR for {Q_local} local questions "
          f"(starting at local {start_batch})...", flush=True)
    t0 = time.time()
    BATCH = 500
    for batch_start in range(start_batch, Q_local, BATCH):
        local_batch = my_questions[batch_start: batch_start + BATCH]
        global_batch_indices = my_indices[batch_start: batch_start + BATCH]
        try:
            results = retriever.retrieve_with_evidence(
                [{"question": q["question"], "answer": q["answer"]} for q in local_batch]
            )
        except Exception as e:
            skipped += len(local_batch)
            print(f"  batch {batch_start} failed: {e}", flush=True)
            continue

        for j, r in enumerate(results):
            global_i = global_batch_indices[j]  # write at global question index
            slot = 0
            for passage_text in r.top_k_passages:
                if slot >= top_k:
                    break
                title = _norm_title(passage_text)
                node_id = title_to_node.get(title, -1)
                node_map[global_i, slot] = node_id
                if node_id >= 0:
                    n_hits += 1
                slot += 1

        elapsed = time.time() - t0
        done_local = min(batch_start + BATCH, Q_local)
        print(f"  {done_local}/{Q_local} local — {elapsed:.0f}s elapsed "
              f"({elapsed/max(done_local,1)*1000:.0f}ms/q), {skipped} skipped",
              flush=True)

        # Checkpoint uses local index for resume; n_clients validates compatibility
        torch.save(node_map, out_path)
        progress_path.write_text(json.dumps({
            "next_batch": done_local,
            "n_hits": n_hits,
            "skipped": skipped,
            "n_clients": n_clients,
        }))

    # Mark complete
    if progress_path.exists():
        progress_path.unlink()

    total = Q_local * top_k
    print(f"  Saved {out_path}  shape={tuple(node_map.shape)}", flush=True)
    print(f"  Hit rate: {n_hits}/{total} = {100*n_hits/max(total,1):.1f}%", flush=True)
    print(f"  Skipped: {skipped}/{Q_local} local questions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa",
                        )
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--top_k_passages", type=int, default=5,
                        help="Top-k PPR passages to map per client per question")
    parser.add_argument("--client-id", type=int, default=None,
                        help="Process only this client ID (0-indexed). If omitted, all clients.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if ppr_node_map.pt already exists.")
    parser.add_argument("--all-questions", action="store_true",
                        help="Compute PPR anchors for EVERY question on each client "
                             "(broadcast eval); writes ppr_node_map_all.pt alongside "
                             "the ownership map.")
    parser.add_argument("--out-suffix", default=None,
                        help="Explicit output suffix: writes ppr_node_map<suffix>.pt "
                             "(e.g. '_top10'); overrides the --all-questions default suffix.")
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
    # n_clients is the total number of clients (used for FL question split)
    n_clients_total = len(client_dirs)
    if args.client_id is not None:
        client_dirs = [p for p in client_dirs if p.name == f"client_{args.client_id}"]
        if not client_dirs:
            print(f"ERROR: client_{args.client_id} not found.")
            sys.exit(1)
    print(f"Found {n_clients_total} total client(s), processing: {[p.name for p in client_dirs]}")
    print(f"Each client handles questions where global_idx % {n_clients_total} == client_id")

    t_total = time.time()
    suffix = args.out_suffix if args.out_suffix is not None else ("_all" if args.all_questions else "")
    for cdir in client_dirs:
        cid_int = int(cdir.name.split("_")[1])
        ppr_path = cdir / f"ppr_node_map{suffix}.pt"
        progress_path = cdir / f"ppr_progress{suffix}.json"
        if not args.force and not args.max_questions and ppr_path.exists() and not progress_path.exists():
            try:
                existing = torch.load(ppr_path, weights_only=True)
                if existing.shape[0] >= len(questions):
                    print(f"  {cdir.name}: ppr_node_map already complete {tuple(existing.shape)}, skipping")
                    continue
            except Exception:
                pass
        process_client(cdir, args.dataset, questions, args.top_k_passages,
                       client_id=cid_int, n_clients=n_clients_total,
                       all_questions=args.all_questions,
                       out_suffix=args.out_suffix)

    print(f"\nAll done in {time.time()-t_total:.0f}s")
    print("Each client's ppr_node_map.pt is saved under its own processed/ directory.")


if __name__ == "__main__":
    main()
