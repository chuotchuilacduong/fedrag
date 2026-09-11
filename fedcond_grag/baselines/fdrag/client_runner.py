"""Run FD-RAG per-client, either "local" (each client keeps its own
memories only) or "federated" (Alg 4: sanitized cross-client fusion).

Same shape as baselines/linearrag/hipporag/comorag: reuse the shared
`idx % num_clients` corpus sharding, index per client, answer the *global*
question set, average metrics across clients.

FD-RAG's own two-mode reporting (§5.3 "Local" vs "Federated" rows) maps to
this file's ``run_all_clients(..., federated=True/False)`` -- both flow
through the same ``FDRAG`` per-client pipeline, only the presence of the
inter-client fusion step at the end differs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.federation import build_shared_vocab, fuse_and_broadcast
from fedcond_grag.baselines.fdrag.pipeline import FDRAG
from fedcond_grag.baselines.linearrag.utils import LLM_Model
from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1

_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "dataset" / "raw"
DEFAULT_SAVE_ROOT = _ROOT / "output" / "baselines" / "fdrag"

_log = logging.getLogger("fdrag.runner")

# project dataset name -> raw file stem (same mapping as baselines/hipporag)
DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihop": "2wikimultihopqa",
}


def _base_dataset_name(dataset: str) -> str:
    """Strip a partition-variant suffix (e.g. 'hotpotqa__dirichlet_0.1' ->
    'hotpotqa') so raw-corpus/question lookups keyed by DATASET_NAMES still
    resolve -- the questions/gold answers are the same regardless of which
    client-corpus partition is in use."""
    return dataset.split("__", 1)[0]


def _parse_chunk_body(body: str) -> dict:
    """`processed/<ds>/client_X/chunks.json` entries are "{title}: {text}"
    strings (see scripts/setup_datasets.py's _build_chunks and
    fedcond_grag/dataloader/topic_partition.py's _chunk_title_and_text) --
    recover a {title, text} dict since FDRAG.index() expects the same shape
    as the raw *_corpus.json."""
    title, sep, text = body.partition(": ")
    if not sep:
        return {"title": body.strip(), "text": ""}
    return {"title": title.strip(), "text": text.strip()}


def _load_client_corpus(dataset: str, hippo_name: str, client_id: int, num_clients: int, max_docs: Optional[int] = None) -> list[dict]:
    # A prebuilt partition (e.g. a Dirichlet client-skew variant) takes
    # precedence over the legacy iid `index % num_clients` split below --
    # see `main.py preprocess --partition-mode` / topic_partition.py.
    partition_path = _ROOT / "processed" / dataset / f"client_{client_id}" / "chunks.json"
    if partition_path.exists():
        bodies = json.loads(partition_path.read_text(encoding="utf-8"))
        shard = [_parse_chunk_body(b) for b in bodies]
    else:
        corpus = json.loads((RAW_DIR / f"{hippo_name}_corpus.json").read_text(encoding="utf-8"))
        shard = [item for i, item in enumerate(corpus) if i % num_clients == client_id]
    if max_docs is not None:
        shard = shard[:max_docs]
    return shard


def _load_global_samples(hippo_name: str) -> list[dict]:
    return json.loads((RAW_DIR / f"{hippo_name}.json").read_text(encoding="utf-8"))


def _load_gold_answers(samples: list[dict]) -> list[set[str]]:
    # Same logic as baselines/hipporag/client_runner.py
    gold_answers = []
    for sample in samples:
        gold_answer = sample.get("answer", sample.get("gold_ans"))
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers = set(answers)
        answers.update(sample.get("answer_aliases", []))
        gold_answers.append(answers)
    return gold_answers


def _score(pred: str, gold: set[str]) -> tuple[float, float, float]:
    hit = 1.0 if any(normalize(a) in normalize(pred) for a in gold) else 0.0
    em = 1.0 if any(exact_match(pred, a) for a in gold) else 0.0
    f1 = max((token_f1(pred, a) for a in gold), default=0.0)
    return hit, em, f1


def _make_llm(llm_base_url: str, llm_name: str, use_llm: bool):
    if not use_llm:
        return None
    os.environ["OPENAI_BASE_URL"] = llm_base_url
    os.environ.setdefault("OPENAI_API_KEY", "sk-")
    model = LLM_Model(llm_name)
    def _infer(prompt: str) -> str:
        return model.infer([{"role": "user", "content": prompt}]) or ""
    return _infer


# ---------------------------------------------------------------------------
# Non-federated: one client at a time, no cross-client memory
# ---------------------------------------------------------------------------
def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    *,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_name: str = "qwen2.5:7b-instruct",
    embedding_model_name: str = DEFAULT_MODEL,
    max_eval_samples: int = 200,
    max_docs_per_client: Optional[int] = None,
    memories_per_edge: int = 1,
    opt_steps: int = 300,
    use_llm: bool = True,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    cfg_overrides: Optional[dict[str, Any]] = None,
    dump_predictions_path: str | None = None,
) -> dict[str, Any]:
    hippo_name = DATASET_NAMES[_base_dataset_name(dataset)]
    corpus = _load_client_corpus(dataset, hippo_name, client_id, num_clients, max_docs=max_docs_per_client)
    samples = _load_global_samples(hippo_name)
    if max_eval_samples:
        samples = samples[:max_eval_samples]
    gold = _load_gold_answers(samples)

    cfg = FDRAGConfig(
        num_clients=num_clients,
        memories_per_edge=memories_per_edge,
        opt_steps=opt_steps,
    )
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    llm_infer = _make_llm(llm_base_url, llm_name, use_llm=use_llm)
    encoder = load_encoder(embedding_model_name)
    rag = FDRAG(cfg, encoder=encoder, llm_infer=llm_infer, client_id=client_id)
    stats = rag.index(corpus)

    hits = em_total = f1_total = 0.0
    fast_ct = slow_ct = llm_calls = 0
    total_latency = 0.0
    dump_f = open(dump_predictions_path, "a") if dump_predictions_path else None
    try:
        for sample, answers in zip(samples, gold):
            r = rag.answer(sample["question"])
            h, e, f = _score(r.answer, answers)
            hits += h; em_total += e; f1_total += f
            llm_calls += r.llm_calls
            total_latency += r.latency_sec
            if r.path == "fast":
                fast_ct += 1
            elif r.path.startswith("slow"):
                slow_ct += 1
            if dump_f is not None:
                dump_f.write(json.dumps({
                    "id": str(sample.get("_id", sample.get("id"))), "client_id": client_id,
                    "pred": r.answer, "label": "|".join(answers),
                }) + "\n")
    finally:
        if dump_f is not None:
            dump_f.close()

    n = max(len(samples), 1)
    result = {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "federated": False,
        "num_docs": stats.n_docs,
        "num_hyperedges": stats.n_hyperedges,
        "num_memories": stats.n_memories,
        "num_questions": len(samples),
        "hit": 100.0 * hits / n,
        "em": 100.0 * em_total / n,
        "f1": 100.0 * f1_total / n,
        "fast_frac": 100.0 * fast_ct / n,
        "avg_llm_calls": llm_calls / n,
        "avg_latency_sec": total_latency / n,
        "build_seconds": stats.build_seconds,
    }
    save_dir = Path(save_root) / dataset / f"client_{client_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Federated: build every client, exchange sanitized memories, then eval
# ---------------------------------------------------------------------------
def run_all_clients(
    dataset: str,
    num_clients: int,
    *,
    federated: bool = False,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_name: str = "qwen2.5:7b-instruct",
    embedding_model_name: str = DEFAULT_MODEL,
    max_eval_samples: int = 200,
    max_docs_per_client: Optional[int] = None,
    memories_per_edge: int = 1,
    opt_steps: int = 300,
    use_llm: bool = True,
    cfg_overrides: Optional[dict[str, Any]] = None,
    dump_predictions_path: str | None = None,
) -> dict[str, Any]:
    if dump_predictions_path:
        Path(dump_predictions_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dump_predictions_path).write_text("")   # reset on new run
    hippo_name = DATASET_NAMES[_base_dataset_name(dataset)]
    samples = _load_global_samples(hippo_name)
    if max_eval_samples:
        samples = samples[:max_eval_samples]
    gold = _load_gold_answers(samples)

    llm_infer = _make_llm(llm_base_url, llm_name, use_llm=use_llm)
    encoder = load_encoder(embedding_model_name)

    # ---- build each client's local index ----
    clients: list[FDRAG] = []
    build_stats = []
    for cid in range(num_clients):
        cfg = FDRAGConfig(
            num_clients=num_clients,
            memories_per_edge=memories_per_edge,
            opt_steps=opt_steps,
        )
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)
        corpus = _load_client_corpus(dataset, hippo_name, cid, num_clients, max_docs=max_docs_per_client)
        rag = FDRAG(cfg, encoder=encoder, llm_infer=llm_infer, client_id=cid)
        stats = rag.index(corpus)
        clients.append(rag)
        build_stats.append(stats)

    # ---- Stage 3 fusion (federated only) ----
    if federated and clients:
        vocab = build_shared_vocab(
            [c.hyperedges for c in clients],
            sensitive_types=clients[0].cfg.sensitive_types,
            encoder=encoder,
        )
        bundles = [c.export_bundle(vocab) for c in clients]
        gk = fuse_and_broadcast(bundles, clients[0].cfg)
        for c in clients:
            c.ingest_global(gk)

    # ---- eval every client on the global question set ----
    per_client = []
    for cid, rag in enumerate(clients):
        hits = em_total = f1_total = 0.0
        fast_ct = 0
        llm_calls = 0
        total_latency = 0.0
        dump_f = open(dump_predictions_path, "a") if dump_predictions_path else None
        try:
            for sample, answers in zip(samples, gold):
                r = rag.answer(sample["question"])
                h, e, f = _score(r.answer, answers)
                hits += h; em_total += e; f1_total += f
                llm_calls += r.llm_calls
                total_latency += r.latency_sec
                if r.path == "fast":
                    fast_ct += 1
                if dump_f is not None:
                    dump_f.write(json.dumps({
                        "id": str(sample.get("_id", sample.get("id"))), "client_id": cid,
                        "pred": r.answer, "label": "|".join(answers),
                    }) + "\n")
        finally:
            if dump_f is not None:
                dump_f.close()
        n = max(len(samples), 1)
        per_client.append({
            "dataset": dataset,
            "client_id": cid,
            "num_clients": num_clients,
            "federated": federated,
            "num_docs": build_stats[cid].n_docs,
            "num_hyperedges": build_stats[cid].n_hyperedges,
            "num_memories_local": build_stats[cid].n_memories,
            "num_memories_effective": len(rag.memories),
            "num_questions": len(samples),
            "hit": 100.0 * hits / n,
            "em": 100.0 * em_total / n,
            "f1": 100.0 * f1_total / n,
            "fast_frac": 100.0 * fast_ct / n,
            "avg_llm_calls": llm_calls / n,
            "avg_latency_sec": total_latency / n,
            "build_seconds": build_stats[cid].build_seconds,
        })

    metric_keys = sorted({k for r in per_client for k, v in r.items() if isinstance(v, (int, float))})
    mean_metrics = {
        k: sum(r.get(k, 0.0) or 0.0 for r in per_client) / max(len(per_client), 1)
        for k in metric_keys
    }
    summary = {
        "dataset": dataset,
        "num_clients": num_clients,
        "federated": federated,
        "per_client": per_client,
        "mean": mean_metrics,
    }

    save_dir = Path(save_root) / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "federated" if federated else "local"
    (save_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
