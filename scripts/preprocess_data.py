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
# Separate 1000-question training pools (scripts/download_train_split.py),
# distinct from the eval benchmark above -- excluded from "--dataset all".
TRAIN_VARIANTS = ["hotpotqa_train", "2wikimultihop_train", "musique_train"]
# <ds>_merged: the train and eval corpora concatenated into one passage set
# (scripts/build_merged_corpus.py). The two are 100% disjoint, so training on
# <ds>_train alone never touches a single passage the eval corpus contains.
# Also excluded from "--dataset all".
MERGED_VARIANTS = ["hotpotqa_merged", "2wikimultihop_merged", "musique_merged"]


def preprocess_one(
    dataset_name: str,
    num_clients: int,
    verbose: bool = True,
    *,
    partition_mode: str = "index",
    alpha: float | None = None,
    num_topic_clusters: int = 20,
    partition_seed: int = 42,
) -> dict:
    src = DATASET_ROOT / dataset_name
    if not src.exists():
        print(f"  [SKIP] {dataset_name}: source not found at {src}")
        return {}

    print(f"\n=== {dataset_name} ===")
    dataset = load_linearrag_dataset(DATASET_ROOT, dataset_name)
    print(f"  Loaded {len(dataset.chunks)} chunks, {len(dataset.questions)} questions")

    # Partition chunks
    if partition_mode == "index":
        clients = partition_linearrag_chunks(dataset.chunks, num_clients=num_clients)
        stats = chunk_partition_stats(clients)
        print(f"  Partition stats: {stats}")
        # Validate checkpoint: no overlap, full coverage
        assert stats["no_overlap"], "BUG: duplicate chunk index across clients"
        assert stats["total_chunks"] == len(dataset.chunks), "BUG: chunks lost in partition"
        assert stats["min"] > 0, "WARNING: some client has 0 chunks"
    else:
        from fedcond_grag.dataloader.topic_partition import partition_chunks_by_topic

        cache_path = DATASET_ROOT / dataset_name / "topic_clusters.json"
        clients, stats = partition_chunks_by_topic(
            dataset.chunks,
            num_clients=num_clients,
            mode=partition_mode,
            alpha=alpha,
            num_topic_clusters=num_topic_clusters,
            seed=partition_seed,
            cluster_cache_path=cache_path,
        )
        print(f"  Partition ({partition_mode}, alpha={alpha}) stats: "
              f"articles={stats['num_articles']} passages={stats['num_passages_after_dedup']} "
              f"(deduped from {stats['num_passages_before_dedup']}) "
              f"client_sizes_passages={stats['client_sizes_passages']}")
        if partition_mode == "dirichlet":
            print(f"  mean pairwise JS divergence across clients: {stats['mean_pairwise_js_divergence']:.4f}")
        assert sum(stats["client_sizes_passages"]) == stats["num_passages_after_dedup"]
        (PROCESSED_ROOT / dataset_name).mkdir(parents=True, exist_ok=True)
        diag_path = PROCESSED_ROOT / dataset_name / f"partition_diagnostics_{partition_mode}_{alpha}.json"
        diag_path.write_text(json.dumps(stats, indent=2, default=str))
        print(f"  diagnostics -> {diag_path}")

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
                        help="Dataset to process (one of "
                             f"{ALL_DATASETS + TRAIN_VARIANTS + MERGED_VARIANTS}, 'all', or any "
                             "other name with an existing dataset/linearrag/<name>/ -- e.g. a "
                             "topic-skew partition variant)")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of federated clients (default: 5)")
    parser.add_argument("--partition-mode", dest="partition_mode", default="index",
                        choices=["index", "random", "dirichlet"],
                        help="'index' (default) = current chunk.index %% num_clients behavior. "
                             "'random' = article-grouped, topic-agnostic shuffle+split. "
                             "'dirichlet' = article-grouped topic-skew partition: cluster the "
                             "corpus into --num-topic-clusters semantic proxies via K-means, "
                             "then allocate each topic across clients via Dirichlet(--alpha).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet concentration for --partition-mode dirichlet "
                             "(lower = more topic-concentrated per client, e.g. 0.1; "
                             "higher = more uniform, e.g. 1.0). Required for that mode.")
    parser.add_argument("--num-topic-clusters", dest="num_topic_clusters", type=int, default=20,
                        help="K-means cluster count for --partition-mode dirichlet (default 20 -- "
                             "a proposed experimental parameter, not an established topic count).")
    parser.add_argument("--partition-seed", dest="partition_seed", type=int, default=42,
                        help="Seed for K-means, article shuffling, and Dirichlet sampling.")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    print(f"Preprocessing {datasets} → {PROCESSED_ROOT}")
    print(f"num_clients = {args.num_clients}, partition_mode = {args.partition_mode}")

    for ds in datasets:
        preprocess_one(
            ds, num_clients=args.num_clients,
            partition_mode=args.partition_mode, alpha=args.alpha,
            num_topic_clusters=args.num_topic_clusters, partition_seed=args.partition_seed,
        )

    print("\nDone. Verify with:")
    print(f"  ls {PROCESSED_ROOT}/{datasets[0]}/")


if __name__ == "__main__":
    main()
