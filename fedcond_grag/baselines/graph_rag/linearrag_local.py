"""Local-only LinearRAG baseline over FedCondGraphRAG client shards.

Runs the vendored LinearRAG (https://github.com/DEEP-PolyU/LinearRAG) as each
federated client's model: one LinearRAG index per client shard, retrieval via
entity activation -> BFS -> PPR, then QA with upstream LinearRAG's own
reading-comprehension prompt. Training-free, so a single pass ("1 round") is
the whole baseline.

Question ownership follows the FL rule (bench_idx % num_clients == client_id),
the same rule used to build the *_upfed benches and the HippoRAG baseline.

Outputs (per bench):
  output/baselines/linearrag_local/<dataset>/client_<c>/predictions.jsonl
  output/baselines/linearrag_local/<dataset>/client_<c>/metrics.json
  output/baselines/linearrag_local/<dataset>/summary.json
  processed/<dataset>/client_<c>/retrieval_top<K>_titles.jsonl (eval_bench_recall.py format)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = ROOT / "processed"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "baselines" / "linearrag_local"
ENCODER_MODEL = "all-MiniLM-L6-v2"
RETRIEVE_BATCH = 500

# Upstream LinearRAG QA prompt (LinearRAG.qa), verbatim.
QA_SYSTEM_PROMPT = (
    'As an advanced reading comprehension assistant, your task is to analyze '
    'text passages and corresponding questions meticulously. Your response start '
    'after "Thought: ", where you will methodically break down the reasoning '
    'process, illustrating how you arrive at conclusions. Conclude with '
    '"Answer: " to present a concise, definitive response, devoid of additional '
    'elaborations.'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedcond_grag linearrag-baseline",
        description="Run one local-only LinearRAG instance per federated client shard.",
    )
    parser.add_argument("--dataset", default="2wikimultihop_upfed")
    parser.add_argument("--num-clients", dest="num_clients", type=int, default=3)
    parser.add_argument("--client-id", dest="client_id", type=int, default=None,
                        help="Run only one client. Default runs all clients.")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-questions-per-client", type=int, default=0,
                        help="Cap queries per client for smoke tests. 0 means no cap.")
    parser.add_argument("--all-questions", action="store_true",
                        help="Every client answers ALL questions on its local shard "
                             "(broadcast eval) instead of only those it owns via idx %% C.")
    parser.add_argument("--split-file", default=None,
                        help="Restrict to the question indices listed in this file "
                             "(e.g. dataset/fedcond_qa/<ds>/split/test_indices.txt).")
    parser.add_argument("--retrieval-top-k", type=int, default=10,
                        help="Passages retrieved per question (titles file uses this k).")
    parser.add_argument("--qa-top-k", type=int, default=5,
                        help="Passages placed in the QA prompt (LinearRAG default 5).")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip QA generation; only write retrieval titles + R@k.")
    parser.add_argument("--llm-name", default="gpt-4o-mini",
                        help="Reader model served through the TimelyGPT gateway.")
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--local-llm", action="store_true",
                        help="Use a local 4-bit HF model as the QA reader instead of the "
                             "TimelyGPT gateway; overrides --llm-name.")
    parser.add_argument("--local-llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-llm-batch-size", type=int, default=2)
    parser.add_argument("--local-llm-gpu-mem", default=None,
                        help="Cap GPU memory for the local reader (e.g. '5GiB'); overflow "
                             "layers are CPU-offloaded. Use when the GPU is shared.")
    parser.add_argument("--max-new-tokens", type=int, default=320,
                        help="Generation budget for the local reader (Thought + Answer).")
    parser.add_argument("--embedding-name", default=ENCODER_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from fedcond_grag.utils.evaluate import (
        exact_match,
        norm_passage_title,
        normalize,
        retrieval_recall_at_k,
        token_f1,
    )

    processed_root = Path(args.processed_root) / args.dataset
    questions_path = processed_root / "questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    output_root = Path(args.output_dir) / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)

    llm = None
    if not args.retrieval_only:
        if args.local_llm:
            from .local_llm import QwenLocalLLM
            args.llm_name = args.local_llm_model
            llm = QwenLocalLLM(
                model_name=args.local_llm_model,
                cache_dir=output_root / "llm_cache_shared",
                batch_size=args.local_llm_batch_size,
                max_gpu_mem=args.local_llm_gpu_mem,
            )
        else:
            from .timely_llm import TimelyLLM
            llm = TimelyLLM(
                model_name=args.llm_name,
                cache_dir=output_root / "llm_cache_shared",
                max_concurrency=args.llm_concurrency,
            )

    client_ids = [args.client_id] if args.client_id is not None else list(range(args.num_clients))
    summaries: list[dict[str, Any]] = []
    for client_id in client_ids:
        summaries.append(_run_client(
            args=args,
            client_id=client_id,
            questions=questions,
            processed_root=processed_root,
            output_root=output_root,
            llm=llm,
            normalize=normalize,
            exact_match=exact_match,
            token_f1=token_f1,
            norm_passage_title=norm_passage_title,
            retrieval_recall_at_k=retrieval_recall_at_k,
        ))

    summary = {
        "baseline": "linearrag_local",
        "upstream": {
            "repo": "https://github.com/DEEP-PolyU/LinearRAG",
            "vendored_path": "fedcond_grag/linearrag",
        },
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "retrieval_top_k": args.retrieval_top_k,
        "qa_top_k": args.qa_top_k,
        "llm_name": None if args.retrieval_only else args.llm_name,
        "clients": summaries,
        "weighted": _weighted_summary(summaries),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[linearrag-local] summary -> {summary_path}", flush=True)
    print(json.dumps(summary["weighted"], indent=2), flush=True)
    return 0


def _run_client(
    *,
    args,
    client_id: int,
    questions: list[dict],
    processed_root: Path,
    output_root: Path,
    llm,
    normalize,
    exact_match,
    token_f1,
    norm_passage_title,
    retrieval_recall_at_k,
) -> dict[str, Any]:
    from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
    from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceLinearRAG

    client_dir = processed_root / f"client_{client_id}"
    chunks_path = client_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Client chunks not found: {chunks_path}")

    keep = None
    if args.split_file:
        keep = {int(l) for l in Path(args.split_file).read_text().splitlines() if l.strip()}
    mine = [(i, questions[i]) for i in range(len(questions))
            if (args.all_questions or i % args.num_clients == client_id)
            and (keep is None or i in keep)]
    if args.max_questions_per_client > 0:
        mine = mine[: args.max_questions_per_client]

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[linearrag-local] client_{client_id}: {len(chunks)} docs, "
          f"{len(mine)} queries", flush=True)

    t0 = time.time()
    retriever = EvidenceLinearRAG(
        working_dir=client_dir / "linearrag_cache",
        dataset_name=args.dataset,
        encoder=load_encoder(args.embedding_name),
        retrieval_top_k=args.retrieval_top_k,
    )
    retriever.index(chunks)
    del chunks
    gc.collect()
    print(f"  indexed in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # Retrieval (batched)
    # ------------------------------------------------------------------
    t0 = time.time()
    all_results = []
    for bs in range(0, len(mine), RETRIEVE_BATCH):
        batch = mine[bs : bs + RETRIEVE_BATCH]
        all_results.extend(retriever.retrieve_with_evidence(
            [{"question": q["question"], "answer": str(q.get("answer", ""))} for _, q in batch]
        ))
    print(f"  retrieved {len(all_results)} in {time.time() - t0:.0f}s", flush=True)

    # Titles file in the eval_bench_recall.py / eval_retrieval_recall.py format.
    # Skipped on capped smoke runs — and on broadcast / split-restricted runs —
    # so an existing full-ownership-run file is not clobbered.
    if args.max_questions_per_client <= 0 and not args.all_questions and not args.split_file:
        titles_path = client_dir / f"retrieval_top{args.retrieval_top_k}_titles.jsonl"
        with titles_path.open("w", encoding="utf-8") as f:
            for (gi, _), r in zip(mine, all_results):
                titles = [norm_passage_title(p) for p in r.top_k_passages[: args.retrieval_top_k]]
                f.write(json.dumps({"idx": gi, "titles": titles}) + "\n")

    # R@k against gold evidence titles.
    ks = (1, 2, 5, 10)
    rec_tot = {k: 0.0 for k in ks}
    rec_n = 0
    for (gi, q), r in zip(mine, all_results):
        gold = {norm_passage_title(t) for t, _ in q.get("evidence", []) if t}
        rec = retrieval_recall_at_k(
            gold, [norm_passage_title(p) for p in r.top_k_passages], ks
        )
        if not rec:
            continue
        rec_n += 1
        for k in ks:
            rec_tot[k] += rec[k]
    retrieval_metrics = {
        f"r@{k}": round(100.0 * rec_tot[k] / max(rec_n, 1), 2) for k in ks
    }
    retrieval_metrics["n_scored"] = rec_n
    print(f"  retrieval: {retrieval_metrics}", flush=True)

    # ------------------------------------------------------------------
    # QA with upstream LinearRAG prompt
    # ------------------------------------------------------------------
    qa_metrics: dict[str, float] = {}
    preds: list[str] = []
    raw_outputs: list[str] = []
    if llm is not None and mine:
        all_messages = []
        for r in all_results:
            prompt_user = ""
            for passage in r.top_k_passages[: args.qa_top_k]:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {r.question}\n Thought: "
            all_messages.append([
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ])

        t0 = time.time()
        if hasattr(llm, "batch_infer"):
            # Local model: threads would serialize on the generate lock anyway.
            raw = [text for text, _ in llm.batch_infer(
                all_messages,
                max_new_tokens=args.max_new_tokens,
                desc=f"QA client_{client_id}",
            )]
        else:
            with ThreadPoolExecutor(max_workers=args.llm_concurrency) as ex:
                raw = list(ex.map(lambda m: llm.infer(m)[0], all_messages))
        print(f"  QA generated {len(raw)} in {time.time() - t0:.0f}s", flush=True)

        hits = em_total = 0.0
        f1_total = 0.0
        for out, (gi, q) in zip(raw, mine):
            raw_outputs.append(out)
            pred = out.split("Answer:")[1].strip() if "Answer:" in out else out.strip()
            preds.append(pred)
            gold = str(q.get("answer", ""))
            if normalize(gold) in normalize(pred):
                hits += 1
            if exact_match(pred, gold):
                em_total += 1
            f1_total += token_f1(pred, gold)
        n = max(len(mine), 1)
        qa_metrics = {
            "hit": round(100.0 * hits / n, 2),
            "em": round(100.0 * em_total / n, 2),
            "f1": round(100.0 * f1_total / n, 2),
        }
        print(f"  qa: {qa_metrics}", flush=True)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    save_dir = output_root / f"client_{client_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, ((gi, q), r) in enumerate(zip(mine, all_results)):
            row = {
                "client_id": client_id,
                "idx": gi,
                "id": str(q.get("id", gi)),
                "question": r.question,
                "gold_answer": str(q.get("answer", "")),
                "retrieved_titles": [norm_passage_title(p) for p in r.top_k_passages],
                "retrieved_passages": list(r.top_k_passages[: args.qa_top_k]),
            }
            if preds:
                row["pred_answer"] = preds[i]
                row["raw_output"] = raw_outputs[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "client_id": client_id,
        "num_queries": len(mine),
        "retrieval": retrieval_metrics,
        "qa": qa_metrics,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def _weighted_summary(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    total = sum(int(item.get("num_queries", 0)) for item in summaries)
    if total <= 0:
        return {"retrieval": {}, "qa": {}}
    out: dict[str, dict[str, float]] = {"retrieval": {}, "qa": {}}
    for key in ("retrieval", "qa"):
        metric_names = sorted({
            metric
            for item in summaries
            for metric in (item.get(key) or {}).keys()
            if metric != "n_scored"
        })
        for metric in metric_names:
            weighted = 0.0
            for item in summaries:
                n = int(item.get("num_queries", 0))
                weighted += n * float((item.get(key) or {}).get(metric, 0.0))
            out[key][metric] = round(weighted / total, 2)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
