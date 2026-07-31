"""Run the per-client local G-Retriever baseline (see
fedcond_grag/baselines/gretriever/client_runner.py for the full picture).

Prerequisites (per dataset):
    python main.py preprocess --dataset <dataset> --num-clients <N>
    python scripts/build_fedcond_qa_dataset.py --dataset <dataset>
    python scripts/preprocess_fedcond_qa.py --dataset <dataset>

Usage:
    python scripts/run_gretriever_baseline.py --dataset hotpotqa --num_clients 3
    python scripts/run_gretriever_baseline.py --dataset hotpotqa --client 0
    python scripts/run_gretriever_baseline.py --dataset hotpotqa --num_clients 3 \
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

from fedcond_grag.baselines.gretriever.client_runner import run_all_clients, run_client_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop", "medical"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client", type=int, default=None,
                         help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--llm_model_name", default="qwen2.5-1.5b")
    parser.add_argument("--llm_model_path", default="")
    parser.add_argument("--llm_frozen", default="True", choices=["True", "False"])
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--local_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_train_samples", type=int, default=0,
                         help="Cap this client's train shard (0 = use full shard)")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--llm_load_in_4bit", action="store_true",
                         help="Quantize the local LLM to 4-bit (bitsandbytes) -- needed for qwen2.5-7b on <16GB VRAM")
    parser.add_argument("--qa_data_root", default="dataset/fedcond_qa",
                         help="QA cache (+ trigraph, via --dataset) used for val/test (and, without "
                              "--qa_train_root, train too). Build via 'main.py preprocess --dataset "
                              "<dataset> --qa-out-root <root>'.")
    parser.add_argument("--qa_train_root", default="",
                         help="If set, train on a SEPARATE '<dataset>_train' pseudo-dataset's own "
                              "trigraph/PPR-map/qa-cache instead of --qa_data_root's -- build via "
                              "scripts/download_train_split.py + 'main.py preprocess --dataset "
                              "<dataset>_train --qa-out-root <this>'. Pair with --qa_data_root built "
                              "via 'main.py preprocess --qa-test-only' so none of val/test's questions "
                              "were ever seen during training.")
    args = parser.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    common_kwargs = dict(
        device=device,
        llm_model_name=args.llm_model_name,
        llm_model_path=args.llm_model_path,
        llm_frozen=args.llm_frozen,
        llm_load_in_4bit=args.llm_load_in_4bit,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_samples=args.max_eval_samples,
        max_train_samples=args.max_train_samples,
        qa_data_root=args.qa_data_root,
        qa_train_root=args.qa_train_root,
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
