"""Local-only HippoRAG baseline over FedCondGraphRAG client shards.

This module is intentionally a thin adapter around the vendored upstream
HippoRAG implementation under ``third_party/HippoRAG``. It prepares our
per-client LinearRAG shard files in the same ``title\\ntext`` style used by
HippoRAG's own experiment runner, then calls upstream HippoRAG for indexing,
retrieval, QA, and metric computation.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = ROOT / "third_party" / "HippoRAG"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
DEFAULT_PROCESSED_ROOT = ROOT / "processed"
DEFAULT_QA_ROOT = ROOT / "dataset" / "fedcond_qa"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "baselines" / "hipporag_local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedcond_grag hipporag-baseline",
        description="Run one local-only HippoRAG instance per federated client shard.",
    )
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--num-clients", dest="num_clients", type=int, default=3)
    parser.add_argument("--client-id", dest="client_id", type=int, default=None,
                        help="Run only one client. Default runs all clients.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--all-questions", action="store_true",
                        help="Every client answers ALL split questions on its local shard "
                             "(broadcast eval) instead of only those it owns via idx %% C.")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--qa-data-root", default=str(DEFAULT_QA_ROOT),
                        help="Optional source of split files; falls back to contiguous 80/10/10.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-docs-per-client", type=int, default=0,
                        help="Cap documents per client for smoke tests. 0 means no cap.")
    parser.add_argument("--max-questions-per-client", type=int, default=0,
                        help="Cap queries per client for smoke tests. 0 means no cap.")
    parser.add_argument("--precomputed-openie-path", default=None,
                        help="Copy a pre-extracted HippoRAG/PropRAG OpenIE JSON into each client save dir.")

    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-name", default="gpt-4o-mini")
    parser.add_argument("--local-llm", action="store_true",
                        help="Use a local 4-bit HF model (no API) for OpenIE + fact rerank; "
                             "overrides --llm-name/--llm-base-url.")
    parser.add_argument("--local-llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-llm-batch-size", type=int, default=4)
    parser.add_argument("--timely-llm", action="store_true",
                        help="Use the TimelyGPT gateway (TIMELY_GPT_KEY in env/.env) for "
                             "OpenIE + fact rerank; model comes from --llm-name.")
    parser.add_argument("--timely-concurrency", type=int, default=8)
    parser.add_argument("--rerank-local-llm", default=None,
                        help="With --timely-llm: keep the API for OpenIE (data processing) but "
                             "run retrieval-time fact rerank on this local HF model instead.")
    parser.add_argument("--local-llm-max-gpu-mem", default=None,
                        help="Cap GPU memory for the local model (e.g. '5.5GiB'); overflow "
                             "layers are offloaded to CPU. Needed for 7B on a shared GPU.")
    parser.add_argument("--embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--openie-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--force-index-from-scratch", action="store_true")
    parser.add_argument("--force-openie-from-scratch", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Run upstream HippoRAG retrieval + Recall@k, but skip QA generation.")
    parser.add_argument("--index-only", action="store_true",
                        help="Preprocess/index only (OpenIE + graph + embeddings); skip retrieval.")

    parser.add_argument("--retrieval-top-k", type=int, default=200)
    parser.add_argument("--linking-top-k", type=int, default=5)
    parser.add_argument("--max-qa-steps", type=int, default=3)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--vector-store-type", choices=["parquet", "qdrant", "chroma"], default="parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_upstream_on_path()

    if args.local_llm and args.timely_llm:
        raise SystemExit("--local-llm and --timely-llm are mutually exclusive")
    if args.local_llm or args.timely_llm:
        # Keep CacheOpenAI construction (unused after the swap below) from
        # failing on a missing key; no API call is ever made through it.
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-local-unused")
        if args.local_llm:
            args.llm_name = args.local_llm_model

    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    logging.basicConfig(level=logging.INFO)

    processed_root = Path(args.processed_root) / args.dataset
    questions_path = processed_root / "questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    questions = _read_json(questions_path)
    split_indices = _load_split_indices(
        n_questions=len(questions),
        split=args.split,
        qa_root=Path(args.qa_data_root),
    )

    client_ids = [args.client_id] if args.client_id is not None else list(range(args.num_clients))
    output_root = Path(args.output_dir) / args.dataset / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for client_id in client_ids:
        client_dir = processed_root / f"client_{client_id}"
        chunks_path = client_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Client chunks not found: {chunks_path}")

        docs, title_to_doc = _load_client_docs(chunks_path, max_docs=args.max_docs_per_client)
        if args.all_questions:
            sample_indices = list(split_indices)
        else:
            sample_indices = [idx for idx in split_indices if idx % args.num_clients == client_id]
        if args.max_questions_per_client > 0:
            sample_indices = sample_indices[: args.max_questions_per_client]
        samples = [questions[idx] for idx in sample_indices]

        print(
            f"[hipporag-local] client_{client_id}: "
            f"{len(docs)} docs, {len(samples)} {args.split} queries",
            flush=True,
        )
        if not samples:
            summaries.append({
                "client_id": client_id,
                "num_docs": len(docs),
                "num_queries": 0,
                "retrieval": {},
                "qa": {},
            })
            continue

        queries = [str(sample.get("question", "")) for sample in samples]
        gold_answers = [_gold_answers(sample) for sample in samples]
        gold_docs = [_gold_docs(sample, title_to_doc) for sample in samples]

        save_dir = output_root / f"client_{client_id}"
        if args.precomputed_openie_path:
            _install_precomputed_openie(
                source=Path(args.precomputed_openie_path),
                save_dir=save_dir,
                llm_name=args.llm_name,
                docs=docs,
            )
        config = BaseConfig(
            save_dir=str(save_dir),
            llm_base_url=args.llm_base_url,
            llm_name=args.llm_name,
            embedding_base_url=args.embedding_base_url,
            embedding_model_name=args.embedding_name,
            dataset=args.dataset,
            force_index_from_scratch=args.force_index_from_scratch,
            force_openie_from_scratch=args.force_openie_from_scratch,
            rerank_dspy_file_path=str(
                UPSTREAM_SRC / "hipporag" / "prompts" / "dspy_prompts"
                / "filter_llama3.3-70B-Instruct.json"
            ),
            retrieval_top_k=args.retrieval_top_k,
            linking_top_k=args.linking_top_k,
            max_qa_steps=args.max_qa_steps,
            qa_top_k=args.qa_top_k,
            graph_type="facts_and_sim_passage_node_unidirectional",
            embedding_batch_size=args.embedding_batch_size,
            max_new_tokens=args.max_new_tokens,
            corpus_len=len(docs),
            openie_mode=args.openie_mode,
            vector_store_type=args.vector_store_type,
        )

        hipporag = HippoRAG(global_config=config)
        if args.local_llm:
            from .local_llm import BatchedLocalOpenIE
            hipporag.llm_model = _get_local_llm(args, output_root)
            hipporag.openie = BatchedLocalOpenIE(llm_model=hipporag.llm_model)
            # rerank_filter captured the old llm_model.infer at construction
            from hipporag.rerank import DSPyFilter
            hipporag.rerank_filter = DSPyFilter(hipporag)
        elif args.timely_llm:
            from hipporag.information_extraction.openie_openai import OpenIE
            from hipporag.rerank import DSPyFilter
            from .timely_llm import TimelyLLM
            hipporag.llm_model = TimelyLLM(
                model_name=args.llm_name,
                cache_dir=output_root / "llm_cache_shared",
                max_concurrency=args.timely_concurrency,
            )
            hipporag.openie = OpenIE(llm_model=hipporag.llm_model)
            if args.rerank_local_llm:
                # API handles OpenIE (data processing) via the reference held
                # by hipporag.openie; rerank AND QA inference run on the local
                # model, so llm_model stays swapped after DSPyFilter captures
                # its infer at construction.
                hipporag.llm_model = _get_local_llm(args, output_root,
                                                    model_name=args.rerank_local_llm)
                hipporag.rerank_filter = DSPyFilter(hipporag)
            else:
                hipporag.rerank_filter = DSPyFilter(hipporag)
        hipporag.index(docs)

        if args.index_only:
            summaries.append({
                "client_id": client_id,
                "num_docs": len(docs),
                "num_queries": 0,
                "retrieval": {},
                "qa": {},
            })
            continue

        if args.retrieval_only:
            query_solutions, retrieval_metrics = hipporag.retrieve(
                queries=queries,
                gold_docs=gold_docs,
            )
            qa_metrics: dict[str, float] = {}
            metadata: list[dict[str, Any]] = []
        else:
            query_solutions, _responses, metadata, retrieval_metrics, qa_metrics = hipporag.rag_qa(
                queries=queries,
                gold_docs=gold_docs,
                gold_answers=gold_answers,
            )

        _write_client_outputs(
            save_dir=save_dir,
            client_id=client_id,
            sample_indices=sample_indices,
            samples=samples,
            query_solutions=query_solutions,
            metadata=metadata,
            retrieval_metrics=retrieval_metrics or {},
            qa_metrics=qa_metrics or {},
        )
        summaries.append({
            "client_id": client_id,
            "num_docs": len(docs),
            "num_queries": len(samples),
            "retrieval": retrieval_metrics or {},
            "qa": qa_metrics or {},
        })

    summary = {
        "baseline": "hipporag_local",
        "upstream": {
            "repo": "https://github.com/OSU-NLP-Group/HippoRAG",
            "vendored_path": str(UPSTREAM_ROOT.relative_to(ROOT)),
        },
        "dataset": args.dataset,
        "split": args.split,
        "num_clients": args.num_clients,
        "clients": summaries,
        "weighted": _weighted_summary(summaries),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hipporag-local] summary -> {summary_path}", flush=True)
    return 0


_LOCAL_LLM_SINGLETON = None


def _get_local_llm(args, output_root: Path, model_name: str | None = None):
    """One 4-bit model instance shared across all clients in this process."""
    global _LOCAL_LLM_SINGLETON
    if _LOCAL_LLM_SINGLETON is None:
        from .local_llm import QwenLocalLLM
        _LOCAL_LLM_SINGLETON = QwenLocalLLM(
            model_name=model_name or args.local_llm_model,
            cache_dir=output_root / "llm_cache_shared",
            batch_size=args.local_llm_batch_size,
            max_gpu_mem=args.local_llm_max_gpu_mem,
        )
    return _LOCAL_LLM_SINGLETON


def _ensure_upstream_on_path() -> None:
    if not UPSTREAM_SRC.exists():
        raise FileNotFoundError(
            f"Vendored HippoRAG source not found at {UPSTREAM_SRC}. "
            "Expected upstream code under third_party/HippoRAG/src."
        )
    src = str(UPSTREAM_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_split_indices(n_questions: int, split: str, qa_root: Path) -> list[int]:
    if split == "all":
        return list(range(n_questions))

    split_path = qa_root / "split" / f"{split}_indices.txt"
    if split_path.exists():
        indices = [
            int(line.strip())
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [idx for idx in indices if 0 <= idx < n_questions]

    train_end = int(0.8 * n_questions)
    val_end = int(0.9 * n_questions)
    if split == "train":
        return list(range(0, train_end))
    if split == "val":
        return list(range(train_end, val_end))
    return list(range(val_end, n_questions))


def _load_client_docs(chunks_path: Path, max_docs: int = 0) -> tuple[list[str], dict[str, str]]:
    chunks = _read_json(chunks_path)
    if max_docs and max_docs > 0:
        chunks = chunks[:max_docs]
    docs: list[str] = []
    title_to_doc: dict[str, str] = {}
    for chunk in chunks:
        doc, title = _chunk_to_doc(str(chunk))
        if not doc:
            continue
        docs.append(doc)
        if title:
            title_to_doc.setdefault(_title_key(title), doc)
    return docs, title_to_doc


def _chunk_to_doc(chunk: str) -> tuple[str, str]:
    # LinearRAG chunks are usually "global_idx:Title: passage text".
    _, sep, without_index = chunk.partition(":")
    text = without_index.strip() if sep and chunk[: chunk.find(":")].isdigit() else chunk.strip()
    title, sep, body = text.partition(":")
    if sep:
        title = title.strip()
        body = body.strip()
        return (f"{title}\n{body}" if body else title), title
    return text, ""


def _title_key(title: str) -> str:
    return " ".join(str(title).strip().lower().split())


def _gold_answers(sample: dict[str, Any]) -> list[str]:
    answer = sample.get("answer", sample.get("gold_ans", ""))
    if isinstance(answer, list):
        answers = [str(item) for item in answer]
    else:
        answers = [str(answer)]
    aliases = sample.get("answer_aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    answers.extend(str(item) for item in aliases)
    return sorted({item for item in answers if item})


def _gold_docs(sample: dict[str, Any], title_to_doc: dict[str, str]) -> list[str]:
    if "supporting_facts" in sample and "context" in sample:
        titles = {str(item[0]) for item in sample.get("supporting_facts", [])}
        docs = []
        for item in sample.get("context", []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = str(item[0])
            if title not in titles:
                continue
            local_doc = title_to_doc.get(_title_key(title))
            if local_doc:
                docs.append(local_doc)
            else:
                docs.append(f"{title}\n{' '.join(str(s) for s in item[1])}")
        return _dedupe(docs)

    if "contexts" in sample:
        docs = []
        for item in sample.get("contexts", []):
            if item.get("is_supporting"):
                title = str(item.get("title", "")).strip()
                local_doc = title_to_doc.get(_title_key(title)) if title else None
                docs.append(local_doc or f"{title}\n{item.get('text', '')}".strip())
        return _dedupe(docs)

    if "paragraphs" in sample:
        docs = []
        for item in sample.get("paragraphs", []):
            if item.get("is_supporting") is False:
                continue
            title = str(item.get("title", "")).strip()
            local_doc = title_to_doc.get(_title_key(title)) if title else None
            text = item.get("text", item.get("paragraph_text", ""))
            docs.append(local_doc or f"{title}\n{text}".strip())
        return _dedupe(docs)

    evidence = sample.get("evidence") or sample.get("evidence_relations") or []
    if not isinstance(evidence, list):
        evidence = [evidence]

    docs: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        doc = _evidence_item_to_doc(item, title_to_doc)
        if doc and doc not in seen:
            docs.append(doc)
            seen.add(doc)
    return docs


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _evidence_item_to_doc(item: Any, title_to_doc: dict[str, str]) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, (list, tuple)) or not item:
        return str(item).strip()

    title = str(item[0]).strip()
    if title:
        local_doc = title_to_doc.get(_title_key(title))
        if local_doc is not None:
            return local_doc

    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
        body = " ".join(str(sentence).strip() for sentence in item[1] if str(sentence).strip())
        return f"{title}\n{body}" if title and body else title or body
    if len(item) >= 3:
        return f"{item[0]}\n{item[1]} {item[2]}".strip()
    if len(item) == 2:
        return f"{item[0]}\n{item[1]}".strip()
    return title


def _install_precomputed_openie(source: Path, save_dir: Path, llm_name: str, docs: list[str]) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Precomputed OpenIE file not found: {source}")
    save_dir.mkdir(parents=True, exist_ok=True)
    normalized_llm_name = llm_name[len("Transformers/"):] if llm_name.startswith("Transformers/") else llm_name
    target = save_dir / f"openie_results_ner_{normalized_llm_name.replace('/', '_')}.json"

    payload = json.loads(source.read_text(encoding="utf-8"))
    source_docs = payload.get("docs", payload) if isinstance(payload, dict) else payload
    if not isinstance(source_docs, list):
        raise ValueError(f"Unsupported OpenIE payload in {source}; expected list or dict with docs")

    wanted = set(docs)
    filtered_docs = [item for item in source_docs if item.get("passage") in wanted]
    if len(filtered_docs) != len(wanted):
        print(
            f"[hipporag-local] OpenIE cache matched {len(filtered_docs)}/{len(wanted)} indexed docs "
            f"from {source}",
            flush=True,
        )

    out_payload = dict(payload) if isinstance(payload, dict) else {"docs": filtered_docs}
    out_payload["docs"] = filtered_docs
    target.write_text(json.dumps(out_payload, ensure_ascii=False), encoding="utf-8")


def _write_client_outputs(
    *,
    save_dir: Path,
    client_id: int,
    sample_indices: list[int],
    samples: list[dict[str, Any]],
    query_solutions: list[Any],
    metadata: list[dict[str, Any]],
    retrieval_metrics: dict[str, float],
    qa_metrics: dict[str, float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = save_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for i, query_solution in enumerate(query_solutions):
            row = query_solution.to_dict()
            # Upstream to_dict() truncates docs[:5]; keep the full ranked list
            # so R@k can be computed for k > 5.
            row["docs"] = list(query_solution.docs)
            if query_solution.doc_scores is not None:
                row["doc_scores"] = [round(float(v), 4) for v in list(query_solution.doc_scores)]
            row.update({
                "client_id": client_id,
                "idx": sample_indices[i],
                "id": str(samples[i].get("id", sample_indices[i])),
                "metadata": metadata[i] if i < len(metadata) else {},
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "client_id": client_id,
        "num_queries": len(samples),
        "retrieval": retrieval_metrics,
        "qa": qa_metrics,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
        })
        for metric in metric_names:
            weighted = 0.0
            for item in summaries:
                n = int(item.get("num_queries", 0))
                weighted += n * float((item.get(key) or {}).get(metric, 0.0))
            out[key][metric] = round(weighted / total, 4)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
