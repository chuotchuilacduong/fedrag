"""Run the per-client FD-RAG baseline (see
fedcond_grag/baselines/fdrag/client_runner.py for the full picture and
implementfd_rag.md for the algorithmic spec extracted from fd-rag.pdf).

FD-RAG has two reporting modes (§5.3):
  --mode local      each client's local memory bank only (FD-RAG-Local)
  --mode federated  Alg 4 sanitized cross-client fusion (full FD-RAG)

The LLM endpoint is expected to be an OpenAI-compatible server (e.g. ollama
serve at http://localhost:11434/v1). Set --no_llm to skip LLM calls entirely
(useful for smoke tests -- degrades to a deterministic cloze fallback for
QA-memory synthesis and returns the top memory verbatim on the slow path).

Usage:
    python scripts/run_fdrag_baseline.py --dataset hotpotqa --num_clients 3
    python scripts/run_fdrag_baseline.py --dataset hotpotqa --num_clients 5 --mode federated
    python scripts/run_fdrag_baseline.py --dataset hotpotqa --client 0 --num_clients 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.fdrag.client_runner import (
    DEFAULT_SAVE_ROOT,
    run_all_clients,
    run_client_baseline,
)
from fedcond_grag.baselines.wandb_logging import log_baseline_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--client", type=int, default=None,
                        help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--mode", choices=["local", "federated"], default="local",
                        help="local = FD-RAG-Local (§5.3), federated = full FD-RAG with Alg 4 fusion.")
    parser.add_argument("--llm_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--llm_name", default="qwen2.5:7b-instruct")
    parser.add_argument("--embedding_model", default=None,
                        help="SentenceTransformer name. Defaults to fedcond_grag's shared encoder.")
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_docs_per_client", type=int, default=None)
    parser.add_argument("--memories_per_edge", type=int, default=1)
    parser.add_argument("--opt_steps", type=int, default=300)
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM calls (smoke test only -- accuracy will drop).")
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--override", action="append", default=[],
                        help="cfg override: KEY=VALUE (can be repeated). "
                             "E.g. --override delta=0.75 --override alpha=0.6")
    args = parser.parse_args()

    cfg_overrides: dict = {}
    for kv in args.override:
        if "=" not in kv:
            parser.error(f"--override expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            v_cast = json.loads(v)
        except json.JSONDecodeError:
            v_cast = v
        cfg_overrides[k.strip()] = v_cast

    common = dict(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        max_eval_samples=args.max_eval_samples,
        max_docs_per_client=args.max_docs_per_client,
        memories_per_edge=args.memories_per_edge,
        opt_steps=args.opt_steps,
        use_llm=not args.no_llm,
        save_root=args.save_root,
        cfg_overrides=cfg_overrides,
    )
    if args.embedding_model:
        common["embedding_model_name"] = args.embedding_model

    if args.client is not None:
        # Per-client, always "local" (federated needs all clients present).
        if args.mode == "federated":
            parser.error("--mode federated requires all clients; drop --client or use run_all_clients")
        result = run_client_baseline(args.dataset, args.client, args.num_clients, **common)
        print(json.dumps(result, indent=2))
        log_baseline_result(f"fdrag-{args.mode}", args.dataset, result)
    else:
        summary = run_all_clients(
            args.dataset, args.num_clients,
            federated=(args.mode == "federated"),
            **common,
        )
        print(f"\n=== {args.dataset} [{args.mode}]: mean over {args.num_clients} clients ===")
        print(json.dumps(summary["mean"], indent=2))
        log_baseline_result(f"fdrag-{args.mode}", args.dataset, summary)


if __name__ == "__main__":
    main()
