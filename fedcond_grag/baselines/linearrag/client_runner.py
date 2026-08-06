"""Run the (unmodified) LinearRAG engine independently on each federated
client's own passage shard, and evaluate every client against the *same*
global benchmark question set used by fedcond_grag's own Stage D eval.

`fedcond_grag/baselines/linearrag/LinearRAG.py` is this project's own
LinearRAG implementation -- also the evidence-retrieval backbone behind
fedrag's own Stage D (`fedcond_grag/client/stage_d_retrieve/evidence_linearrag.py`
wraps the same class for retrieval only, no LLM). This baseline drives the
class directly via its own unmodified `index()` + `qa()` (entity-seeded PPR
retrieval, then LLM reading over the retrieved passages) -- nothing about
the retrieval or generation logic is touched.

Same "traditional single-node RAG under corpus fragmentation" baseline as
baselines/hipporag, baselines/comorag, baselines/flare: each client only
ever indexes+retrieves from its own passage shard (`idx % num_clients`,
same rule and same `dataset/raw/<name>_corpus.json` ordering those three
use), then answers the full global question set. `LinearRAG.qa()` needs its
own ordinally-prefixed chunk format ("0:Title: text", used for its adjacent-
passage edges) -- reuses `dataset/linearrag/<dataset>/chunks.json`, which is
already in that format and index-aligned with the raw corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

from fedcond_grag.baselines.linearrag import LinearRAG, LinearRAGConfig
from fedcond_grag.baselines.linearrag.utils import LLM_Model
from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1

RAW_DIR = _ROOT / "dataset" / "raw"
LINEARRAG_DIR = _ROOT / "dataset" / "linearrag"
DEFAULT_SAVE_ROOT = _ROOT / "output" / "baselines" / "linearrag"

# project dataset name -> raw sample file stem (see scripts/setup_datasets.py)
DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihop": "2wikimultihopqa",
}


def _load_client_chunks(dataset: str, client_id: int, num_clients: int) -> list[str]:
    chunks = json.loads((LINEARRAG_DIR / dataset / "chunks.json").read_text(encoding="utf-8"))
    return [c for i, c in enumerate(chunks) if i % num_clients == client_id]


def _load_global_samples(dataset: str) -> list[dict]:
    raw_name = DATASET_NAMES[dataset]
    return json.loads((RAW_DIR / f"{raw_name}.json").read_text(encoding="utf-8"))


def _load_gold_answers(samples: list[dict]) -> list[set[str]]:
    # Same logic as baselines/hipporag/client_runner.py::_load_gold_answers
    gold_answers = []
    for sample in samples:
        gold_answer = sample.get("answer", sample.get("gold_ans"))
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers = set(answers)
        answers.update(sample.get("answer_aliases", []))
        gold_answers.append(answers)
    return gold_answers


def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_name: str = "qwen2.5:7b-instruct",
    embedding_model_name: str = DEFAULT_MODEL,
    retrieval_top_k: int = 5,
    max_eval_samples: int = 200,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
) -> dict[str, Any]:
    """Index client `client_id`'s local chunk shard with LinearRAG and
    evaluate it against the full global question set for `dataset`."""
    # LLM_Model (baselines/linearrag/utils.py) reads these env vars -- same
    # OpenAI-compatible-endpoint convention as the other baselines.
    os.environ["OPENAI_BASE_URL"] = llm_base_url
    os.environ.setdefault("OPENAI_API_KEY", "sk-")

    chunks = _load_client_chunks(dataset, client_id, num_clients)
    samples = _load_global_samples(dataset)
    if max_eval_samples:
        samples = samples[:max_eval_samples]
    gold_answers = _load_gold_answers(samples)

    working_dir = Path(save_root) / dataset / f"client_{client_id}" / "index"
    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=load_encoder(embedding_model_name),
        llm_model=LLM_Model(llm_name),
        working_dir=str(working_dir),
        retrieval_top_k=retrieval_top_k,
    )
    rag = LinearRAG(config)
    rag.index(chunks)

    questions = [{"question": s["question"], "answer": next(iter(a), "")}
                 for s, a in zip(samples, gold_answers)]
    qa_results = rag.qa(questions)

    hits = em_total = f1_total = 0.0
    for result, answers in zip(qa_results, gold_answers):
        pred = result.get("pred_answer", "")
        if any(normalize(a) in normalize(pred) for a in answers):
            hits += 1
        if any(exact_match(pred, a) for a in answers):
            em_total += 1.0
        f1_total += max((token_f1(pred, a) for a in answers), default=0.0)

    n = max(len(samples), 1)
    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "num_chunks": len(chunks),
        "num_questions": len(samples),
        "hit": 100.0 * hits / n,
        "em": 100.0 * em_total / n,
        "f1": 100.0 * f1_total / n,
    }


def run_all_clients(dataset: str, num_clients: int, save_root: Path | str = DEFAULT_SAVE_ROOT, **kwargs: Any) -> dict[str, Any]:
    """Run every client's local LinearRAG baseline independently and write a
    summary JSON (per-client metrics + mean across clients)."""
    per_client = [
        run_client_baseline(dataset, client_id, num_clients, save_root=save_root, **kwargs)
        for client_id in range(num_clients)
    ]

    metric_keys = sorted({k for r in per_client for k, v in r.items() if isinstance(v, (int, float))})
    mean_metrics = {
        k: sum(r.get(k, 0.0) or 0.0 for r in per_client) / len(per_client)
        for k in metric_keys
    }

    summary = {"dataset": dataset, "num_clients": num_clients, "per_client": per_client, "mean": mean_metrics}

    out_path = Path(save_root) / dataset / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
