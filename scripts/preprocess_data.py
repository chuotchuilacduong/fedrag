"""Preprocess LinearRAG-format datasets into per-client chunk files.

Usage (from project root, fedcond conda env):
    python scripts/preprocess_data.py --dataset hotpotqa --num_clients 5
    python scripts/preprocess_data.py --dataset all --num_clients 5

Output layout:
    processed/{dataset}/
        questions.json          # all questions (unchanged)
        client_{m}/
            chunks.json         # LinearRAG-format chunk strings for client m

The chunks.json files can be passed directly to LinearRAG.index() or
to trigraph_builder.build_trigraph_for_client().

Checkpoint (plan 09_INT_HOST_REPO.md §35 Step 5 equivalent):
    Each client gets >0 chunks; total == original; no chunk appears twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path when run directly
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.dataloader.data_preprocess import (
    chunk_partition_stats,
    partition_linearrag_chunks,
    load_linearrag_dataset,
    save_chunk_list,
    save_question_list,
)

DATASET_ROOT = _ROOT / "dataset" / "linearrag"
PROCESSED_ROOT = _ROOT / "processed"

ALL_DATASETS = ["hotpotqa", "2wikimultihop", "musique", "medical"]


def preprocess_one(dataset_name: str, num_clients: int, verbose: bool = True) -> dict:
    src = DATASET_ROOT / dataset_name
    if not src.exists():
        print(f"  [SKIP] {dataset_name}: source not found at {src}")
        return {}

    print(f"\n=== {dataset_name} ===")
    dataset = load_linearrag_dataset(DATASET_ROOT, dataset_name)
    print(f"  Loaded {len(dataset.chunks)} chunks, {len(dataset.questions)} questions")

    # Partition chunks
    clients = partition_linearrag_chunks(dataset.chunks, num_clients=num_clients)
    stats = chunk_partition_stats(clients)
    print(f"  Partition stats: {stats}")

    # Validate checkpoint: no overlap, full coverage
    assert stats["no_overlap"], "BUG: duplicate chunk index across clients"
    assert stats["total_chunks"] == len(dataset.chunks), "BUG: chunks lost in partition"
    assert stats["min"] > 0, "WARNING: some client has 0 chunks"

    # Write per-client files
    out_root = PROCESSED_ROOT / dataset_name
    for client in clients:
        out_dir = out_root / f"client_{client.client_id}"
        save_chunk_list(client.chunks, out_dir / "chunks.json")
        if verbose:
            print(f"  client_{client.client_id}: {len(client.chunks)} chunks → {out_dir}/chunks.json")

    # Write questions once (shared across all clients)
    save_question_list(dataset.questions, out_root / "questions.json")
    print(f"  questions → {out_root}/questions.json")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess LinearRAG datasets into per-client chunks.")
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=ALL_DATASETS + ["all"],
                        help="Dataset to process (or 'all')")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of federated clients (default: 5)")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    print(f"Preprocessing {datasets} → {PROCESSED_ROOT}")
    print(f"num_clients = {args.num_clients}")

    for ds in datasets:
        preprocess_one(ds, num_clients=args.num_clients)

    print("\nDone. Verify with:")
    print(f"  ls {PROCESSED_ROOT}/{datasets[0]}/")


if __name__ == "__main__":
    main()
