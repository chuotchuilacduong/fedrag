"""Run the per-client local GRAG baseline (see
fedcond_grag/baselines/grag/client_runner.py for the full picture).

Builds each client's own knowledge graph from HippoRAG's cached OpenIE
triples if that baseline has already been run for this dataset/client
(reused, not recomputed), else falls back to this project's own Tri-Graph
with generic edge labels.

Prerequisites (per dataset):
    python main.py preprocess --dataset <dataset> --num-clients <N>
    python scripts/build_fedcond_qa_dataset.py --dataset <dataset>
    python scripts/preprocess_fedcond_qa.py --dataset <dataset>

Usage:
    python scripts/run_grag_baseline.py --dataset hotpotqa --num_clients 3
    python scripts/run_grag_baseline.py --dataset hotpotqa --client 0
    python scripts/run_grag_baseline.py --dataset hotpotqa --num_clients 3 \
        --llm_model_name qwen2.5-1.5b --use_cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

from fedcond_grag.baselines.grag.client_runner import run_all_clients, run_client_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop", "medical"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client", type=int, default=None,
                         help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--llm_model_name", default="qwen2.5-1.5b")
    parser.add_argument("--llm_model_path", default="")
    parser.add_argument("--llm_frozen", default="True", choices=["True", "False"])
    parser.add_argument("--gnn_model_name", default="gat", choices=["gat", "trans", "gcn"])
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--local_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_train_samples", type=int, default=0,
                         help="Cap this client's train shard (0 = use full shard)")
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument("--retrieval_k", type=int, default=2)
    parser.add_argument("--retrieval_topk_entity", type=int, default=5)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    common_kwargs = dict(
        device=device,
        llm_model_name=args.llm_model_name,
        llm_model_path=args.llm_model_path,
        llm_frozen=args.llm_frozen,
        gnn_model_name=args.gnn_model_name,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_samples=args.max_eval_samples,
        max_train_samples=args.max_train_samples,
        retrieval_topk=args.retrieval_topk,
        retrieval_k=args.retrieval_k,
        retrieval_topk_entity=args.retrieval_topk_entity,
    )

    if args.client is not None:
        result = run_client_baseline(args.dataset, args.client, args.num_clients, **common_kwargs)
        print(json.dumps(result, indent=2))
    else:
        summary = run_all_clients(args.dataset, args.num_clients, **common_kwargs)
        print(f"\n=== {args.dataset}: mean over {args.num_clients} clients ===")
        print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
