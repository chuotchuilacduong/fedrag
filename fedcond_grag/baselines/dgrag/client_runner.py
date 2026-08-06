"""Run DGRAG per-client in local or federated mode.

Same shape as baselines/fdrag/client_runner.py (and linearrag/hipporag):
- Corpus sharding: `idx % num_clients` on `dataset/raw/<name>_corpus.json`
- Evaluation: full global question set, ACC/F1/hit/latency/LLM-calls
- Two modes:
    local     = each client's own KG only; gate routes locally or falls back
    federated = full §9 no-cloud adaptation with cross-edge retrieval

LLM-call accounting follows spec §9.5:
  1 keyword + B batch + 1 confidence + 1 semantic + 1 select (local path)
  + on escalation: N_peer keyword calls + 1 cloud generation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fedcond_grag.baselines.dgrag.config import AblationFlag, DGRAGConfig
from fedcond_grag.baselines.dgrag.federation import (
    build_coordinator,
    build_peer_retrieve_fns,
    distribute_summary_vdb,
)
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.pipeline import DGRAG
from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1

_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "dataset" / "raw"
DEFAULT_SAVE_ROOT = _ROOT / "output" / "baselines" / "dgrag"

_log = logging.getLogger("dgrag.runner")

DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihop": "2wikimultihopqa",
}


def _load_client_corpus(hippo_name: str, client_id: int, num_clients: int, max_docs: Optional[int]) -> list[dict]:
    corpus = json.loads((RAW_DIR / f"{hippo_name}_corpus.json").read_text(encoding="utf-8"))
    shard = [item for i, item in enumerate(corpus) if i % num_clients == client_id]
    if max_docs is not None:
        shard = shard[:max_docs]
    # Inject idx for stable chunk_ids
    for i, item in enumerate(shard):
        if "idx" not in item:
            item["idx"] = f"{client_id}-{i}"
    return shard


def _load_global_samples(hippo_name: str) -> list[dict]:
    return json.loads((RAW_DIR / f"{hippo_name}.json").read_text(encoding="utf-8"))


def _load_gold_answers(samples: list[dict]) -> list[set[str]]:
    gold = []
    for sample in samples:
        ans = sample.get("answer", sample.get("gold_ans"))
        answers = [ans] if isinstance(ans, str) else list(ans or [])
        answers = set(answers)
        answers.update(sample.get("answer_aliases", []))
        gold.append(answers)
    return gold


def _score(pred: str, gold: set[str]) -> tuple[float, float, float]:
    hit = 1.0 if any(normalize(a) in normalize(pred) for a in gold) else 0.0
    em = 1.0 if any(exact_match(pred, a) for a in gold) else 0.0
    f1 = max((token_f1(pred, a) for a in gold), default=0.0)
    return hit, em, f1


def _make_model(cfg: DGRAGConfig) -> Optional[DGRAGModel]:
    os.environ["OPENAI_BASE_URL"] = cfg.llm_base_url
    os.environ.setdefault("OPENAI_API_KEY", "sk-")
    return DGRAGModel(model_name=cfg.slm_name, base_url=cfg.llm_base_url)


def _build_cfg(
    num_clients: int,
    ablation: AblationFlag = "none",
    cfg_overrides: Optional[dict] = None,
    **kwargs,
) -> DGRAGConfig:
    cfg = DGRAGConfig(num_clients=num_clients, ablation=ablation, **kwargs)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Single client (local mode only)
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
    batch_b: int = 3,
    gate_threshold: float = 0.75,
    ablation: AblationFlag = "none",
    use_llm: bool = True,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    hippo_name = DATASET_NAMES[dataset]
    corpus = _load_client_corpus(hippo_name, client_id, num_clients, max_docs_per_client)
    samples = _load_global_samples(hippo_name)
    if max_eval_samples:
        samples = samples[:max_eval_samples]
    gold = _load_gold_answers(samples)

    cfg = _build_cfg(
        num_clients, ablation=ablation, cfg_overrides=cfg_overrides,
        llm_base_url=llm_base_url, slm_name=llm_name, llm_name=llm_name,
        batch_b=batch_b, gate_threshold=gate_threshold,
    )
    encoder = load_encoder(embedding_model_name)
    slm = _make_model(cfg) if use_llm else None
    edge_id = f"edge-{client_id}"
    rag = DGRAG(cfg, encoder=encoder, edge_id=edge_id, slm=slm)
    stats = rag.index(corpus)

    hits = em_t = f1_t = 0.0
    local_ct = global_ct = llm_total = 0
    total_lat = 0.0

    for sample, answers in zip(samples, gold):
        r = rag.answer(sample["question"])
        h, e, f = _score(r.answer, answers)
        hits += h; em_t += e; f1_t += f
        llm_total += r.llm_calls
        total_lat += r.latency_sec
        if r.route == "local":
            local_ct += 1
        elif r.route == "global":
            global_ct += 1

    n = max(len(samples), 1)
    result = {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "federated": False,
        "ablation": ablation,
        "num_docs": stats.n_docs,
        "num_entities": stats.n_entities,
        "num_relations": stats.n_relations,
        "num_subgraphs": stats.n_subgraphs,
        "num_subgraph_sizes_p50": stats.n_subgraph_sizes_p50,
        "num_questions": len(samples),
        "hit": 100.0 * hits / n,
        "em": 100.0 * em_t / n,
        "f1": 100.0 * f1_t / n,
        "local_rate": 100.0 * local_ct / n,
        "global_rate": 100.0 * global_ct / n,
        "avg_llm_calls": llm_total / n,
        "avg_latency_sec": total_lat / n,
        "build_seconds": stats.build_seconds,
    }
    out_dir = Path(save_root) / dataset / f"client_{client_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# All clients (local or federated)
# ---------------------------------------------------------------------------

def run_all_clients(
    dataset: str,
    num_clients: int,
    *,
    federated: bool = False,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_name: str = "qwen2.5:7b-instruct",
    embedding_model_name: str = DEFAULT_MODEL,
    max_eval_samples: int = 200,
    max_docs_per_client: Optional[int] = None,
    batch_b: int = 3,
    gate_threshold: float = 0.75,
    ablation: AblationFlag = "none",
    use_llm: bool = True,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    hippo_name = DATASET_NAMES[dataset]
    samples = _load_global_samples(hippo_name)
    if max_eval_samples:
        samples = samples[:max_eval_samples]
    gold = _load_gold_answers(samples)

    encoder = load_encoder(embedding_model_name)
    clients: list[DGRAG] = []
    build_stats = []

    for cid in range(num_clients):
        cfg = _build_cfg(
            num_clients, ablation=ablation, cfg_overrides=cfg_overrides,
            llm_base_url=llm_base_url, slm_name=llm_name, llm_name=llm_name,
            batch_b=batch_b, gate_threshold=gate_threshold,
        )
        corpus = _load_client_corpus(hippo_name, cid, num_clients, max_docs_per_client)
        slm = _make_model(cfg) if use_llm else None
        rag = DGRAG(cfg, encoder=encoder, edge_id=f"edge-{cid}", slm=slm)
        stat = rag.index(corpus)
        clients.append(rag)
        build_stats.append(stat)

    # ── Federation: build coordinator summary VDB and distribute ────────
    if federated:
        summary_vdb = build_coordinator(clients)
        distribute_summary_vdb(clients, summary_vdb)

    # ── Evaluate every client on global question set ─────────────────────
    per_client = []
    for cid, rag in enumerate(clients):
        peer_fns = build_peer_retrieve_fns(clients, rag.edge_id) if federated else None
        hits = em_t = f1_t = 0.0
        local_ct = global_ct = llm_total = 0
        total_lat = 0.0

        for sample, answers in zip(samples, gold):
            r = rag.answer(sample["question"], peer_retrieve_fns=peer_fns)
            h, e, f = _score(r.answer, answers)
            hits += h; em_t += e; f1_t += f
            llm_total += r.llm_calls
            total_lat += r.latency_sec
            if r.route == "local":
                local_ct += 1
            elif r.route == "global":
                global_ct += 1

        n = max(len(samples), 1)
        per_client.append({
            "dataset": dataset,
            "client_id": cid,
            "num_clients": num_clients,
            "federated": federated,
            "ablation": ablation,
            "num_docs": build_stats[cid].n_docs,
            "num_entities": build_stats[cid].n_entities,
            "num_relations": build_stats[cid].n_relations,
            "num_subgraphs": build_stats[cid].n_subgraphs,
            "num_subgraph_sizes_p50": build_stats[cid].n_subgraph_sizes_p50,
            "num_questions": len(samples),
            "hit": 100.0 * hits / n,
            "em": 100.0 * em_t / n,
            "f1": 100.0 * f1_t / n,
            "local_rate": 100.0 * local_ct / n,
            "global_rate": 100.0 * global_ct / n,
            "avg_llm_calls": llm_total / n,
            "avg_latency_sec": total_lat / n,
            "build_seconds": build_stats[cid].build_seconds,
        })

    metric_keys = sorted({k for r in per_client for k, v in r.items() if isinstance(v, (int, float))})
    mean_metrics = {
        k: sum(r.get(k, 0.0) or 0.0 for r in per_client) / max(len(per_client), 1)
        for k in metric_keys
    }
    tag = "federated" if federated else "local"
    summary = {
        "dataset": dataset, "num_clients": num_clients,
        "federated": federated, "ablation": ablation,
        "per_client": per_client, "mean": mean_metrics,
    }
    out_dir = Path(save_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
