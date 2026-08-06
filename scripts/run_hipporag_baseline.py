"""Run the per-client local HippoRAG baseline (see
fedcond_grag/baselines/hipporag/client_runner.py for the full picture).

Requires an OpenAI-compatible LLM endpoint already running (e.g. `ollama
serve`, default http://localhost:11434/v1) with the requested model pulled.

Usage:
    python scripts/run_hipporag_baseline.py --dataset hotpotqa --num_clients 5
    python scripts/run_hipporag_baseline.py --dataset musique --client 0
    python scripts/run_hipporag_baseline.py --dataset all --num_clients 5 \
        --llm_name qwen2.5:7b-instruct --llm_base_url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.hipporag.client_runner import (
    DEFAULT_SAVE_ROOT,
    HIPPO_DATASET_NAMES,
    run_all_clients,
    run_client_baseline,
)
from fedcond_grag.baselines.wandb_logging import log_baseline_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(HIPPO_DATASET_NAMES) + ["all"], default="all")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client", type=int, default=None,
                         help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--llm_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--llm_name", default="qwen2.5:7b-instruct")
    parser.add_argument("--embedding_name", default="Transformers/sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--force_index_from_scratch", action="store_true")
    parser.add_argument("--force_openie_from_scratch", action="store_true")
    args = parser.parse_args()

    datasets = list(HIPPO_DATASET_NAMES) if args.dataset == "all" else [args.dataset]

    common_kwargs = dict(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        embedding_batch_size=args.embedding_batch_size,
        save_root=args.save_root,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
    )

    for dataset in datasets:
        if args.client is not None:
            result = run_client_baseline(dataset, args.client, args.num_clients, **common_kwargs)
            print(json.dumps(result, indent=2))
            log_baseline_result("hipporag", dataset, result)
        else:
            summary = run_all_clients(dataset, args.num_clients, **common_kwargs)
            print(f"\n=== {dataset}: mean over {args.num_clients} clients ===")
            print(json.dumps(summary["mean"], indent=2))
            log_baseline_result("hipporag", dataset, summary)


if __name__ == "__main__":
    main()
