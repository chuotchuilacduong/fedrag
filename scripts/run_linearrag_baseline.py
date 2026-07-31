"""Run the per-client local LinearRAG baseline (see
fedcond_grag/baselines/linearrag/client_runner.py for the full picture).

Training-free, same as baselines/hipporag/comorag/flare: each client
indexes+retrieves from its own passage shard with LinearRAG's own
(unmodified) entity-seeded PPR retrieval, then answers with LinearRAG's own
(unmodified) LLM reading step -- evaluated against the *global* question set.

Requires an OpenAI-compatible LLM endpoint already running (e.g. `ollama
serve`, default http://localhost:11434/v1) with the requested model pulled.

Usage:
    python scripts/run_linearrag_baseline.py --dataset hotpotqa --num_clients 3
    python scripts/run_linearrag_baseline.py --dataset hotpotqa --client 0 --num_clients 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.linearrag.client_runner import DEFAULT_SAVE_ROOT, run_all_clients, run_client_baseline
from fedcond_grag.baselines.wandb_logging import log_baseline_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client", type=int, default=None,
                         help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--llm_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--llm_name", default="qwen2.5:7b-instruct")
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT))
    args = parser.parse_args()

    common_kwargs = dict(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        retrieval_top_k=args.retrieval_top_k,
        max_eval_samples=args.max_eval_samples,
        save_root=args.save_root,
    )

    if args.client is not None:
        result = run_client_baseline(args.dataset, args.client, args.num_clients, **common_kwargs)
        print(json.dumps(result, indent=2))
        log_baseline_result("linearrag", args.dataset, result)
    else:
        summary = run_all_clients(args.dataset, args.num_clients, **common_kwargs)
        print(f"\n=== {args.dataset}: mean over {args.num_clients} clients ===")
        print(json.dumps(summary["mean"], indent=2))
        log_baseline_result("linearrag", args.dataset, summary)


if __name__ == "__main__":
    main()
