"""Run the DGRAG baseline (see fedcond_grag/baselines/dgrag/ and
implement_dg_rag.md for the full algorithmic spec).

Two modes (--mode):
  local      per-client shard + local gate only (FD-RAG-Local equivalent)
  federated  full §9 no-cloud adaptation: gate + cross-edge escalation to peers

Ablations (--ablation, spec §7):
  none       full DGRAG
  w/o_CD     skip Confidence Detection; similarity-only gate
  w/o_SE     skip Similarity Evaluation; confidence-only gate
  w/o_BQ     B=1 single inference; gate falls back to confidence only
  w/o_KG     retrieval from chunk VDB only, no graph expansion

Requires an OpenAI-compatible LLM endpoint already running (e.g.
`ollama serve`, default http://localhost:11434/v1).

Use --no_llm to skip LLM calls entirely (smoke test; accuracy degrades to
a word-overlap heuristic for QA and the gate always routes locally).

Usage:
    python scripts/run_dgrag_baseline.py --dataset hotpotqa --num_clients 4
    python scripts/run_dgrag_baseline.py --dataset hotpotqa --num_clients 4 --mode federated
    python scripts/run_dgrag_baseline.py --dataset hotpotqa --client 0 --num_clients 4
    python scripts/run_dgrag_baseline.py --dataset hotpotqa --num_clients 4 \\
        --ablation w/o_KG --mode federated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.dgrag.client_runner import (
    DEFAULT_SAVE_ROOT,
    run_all_clients,
    run_client_baseline,
)
from fedcond_grag.baselines.wandb_logging import log_baseline_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihop"])
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--client", type=int, default=None,
                        help="Run only this client id (0-indexed). Default: run all clients.")
    parser.add_argument("--mode", choices=["local", "federated"], default="local",
                        help="local = per-client only; federated = §9 cross-edge escalation.")
    parser.add_argument("--ablation",
                        choices=["none", "w/o_CD", "w/o_SE", "w/o_BQ", "w/o_KG"],
                        default="none")
    parser.add_argument("--llm_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--llm_name", default="qwen2.5:7b-instruct")
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_docs_per_client", type=int, default=None)
    parser.add_argument("--batch_b", type=int, default=3,
                        help="Number of candidate answers per query (spec B=3).")
    parser.add_argument("--gate_threshold", type=float, default=0.75,
                        help="Gate similarity threshold (spec [FREE] default 0.75).")
    parser.add_argument("--no_llm", action="store_true",
                        help="Smoke test: skip all LLM calls.")
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--override", action="append", default=[],
                        help="Config override KEY=VALUE (repeatable). E.g. --override thr_summary=0.3")
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
        batch_b=args.batch_b,
        gate_threshold=args.gate_threshold,
        ablation=args.ablation,
        use_llm=not args.no_llm,
        save_root=args.save_root,
        cfg_overrides=cfg_overrides,
    )
    if args.embedding_model:
        common["embedding_model_name"] = args.embedding_model

    method_tag = f"dgrag-{args.mode}"
    if args.ablation != "none":
        method_tag += f"-{args.ablation}"

    if args.client is not None:
        if args.mode == "federated":
            parser.error("--mode federated requires all clients; drop --client.")
        result = run_client_baseline(args.dataset, args.client, args.num_clients, **common)
        print(json.dumps(result, indent=2))
        log_baseline_result(method_tag, args.dataset, result)
    else:
        summary = run_all_clients(
            args.dataset, args.num_clients,
            federated=(args.mode == "federated"),
            **common,
        )
        print(f"\n=== {args.dataset} [{args.mode}|{args.ablation}]: mean over {args.num_clients} clients ===")
        print(json.dumps(summary["mean"], indent=2))
        log_baseline_result(method_tag, args.dataset, summary)


if __name__ == "__main__":
    main()
