"""Run the per-client local FLARE baseline (see
fedcond_grag/baselines/flare/client_runner.py for the full picture).

Training-free: retrieves from each client's own passage shard and answers
with iterative look-ahead generation + confidence-based retrieval, no local
training step (unlike gretriever/grag).

Requires an OpenAI-compatible LLM endpoint that returns chat-completion
logprobs (default: local Ollama at http://localhost:11434/v1 -- verified to
return real per-token logprobs via `/v1/chat/completions`; its legacy
`/v1/completions` endpoint does not, which is why this baseline uses chat
completions).

Prerequisites:
    python scripts/build_fedcond_qa_dataset.py --dataset <dataset>

Usage:
    python scripts/run_flare_baseline.py --dataset hotpotqa --num_clients 5
    python scripts/run_flare_baseline.py --dataset hotpotqa --client 0 --num_clients 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.flare.client_runner import DEFAULT_SAVE_ROOT, run_all_clients, run_client_baseline
from fedcond_grag.baselines.wandb_logging import log_baseline_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client", type=int, default=None,
                         help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--llm_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--llm_name", default="qwen2.5:1.5b-instruct")
    parser.add_argument("--look_ahead_steps", type=int, default=64)
    parser.add_argument("--mask_prob", type=float, default=0.4)
    parser.add_argument("--retrieval_topk", type=int, default=2)
    parser.add_argument("--max_sentences", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT))
    args = parser.parse_args()

    common_kwargs = dict(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        look_ahead_steps=args.look_ahead_steps,
        mask_prob=args.mask_prob,
        retrieval_topk=args.retrieval_topk,
        max_sentences=args.max_sentences,
        max_eval_samples=args.max_eval_samples,
        save_root=args.save_root,
    )

    if args.client is not None:
        result = run_client_baseline(args.dataset, args.client, args.num_clients, **common_kwargs)
        print(json.dumps(result, indent=2))
        log_baseline_result("flare", args.dataset, result)
    else:
        summary = run_all_clients(args.dataset, args.num_clients, **common_kwargs)
        print(f"\n=== {args.dataset}: mean over {args.num_clients} clients ===")
        print(json.dumps(summary["mean"], indent=2))
        log_baseline_result("flare", args.dataset, summary)


if __name__ == "__main__":
    main()
