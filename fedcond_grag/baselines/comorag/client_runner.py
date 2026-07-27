"""Run the (vendored) ComoRAG package independently on each federated
client's own passage shard, and evaluate every client against the *same*
global benchmark question set used by fedcond_grag's own Stage D eval.

ComoRAG (https://github.com/EternityJune25/ComoRAG, MIT licensed) is itself
a fork of HippoRAG that adds a "veridical / semantic / episodic" memory-pool
retrieval loop on top -- see `_vendor/VENDORED.md` for the lineage and the
handful of packaging fixes needed to run it here (same categories as
`baselines/hipporag`: eager vllm import, Windows path colon sanitization,
a genuine upstream bug in its embedding-model factory).

This is the "traditional single-node RAG under corpus fragmentation"
baseline, same methodology as `baselines/hipporag`/`baselines/grag`/
`baselines/flare`: each client only ever sees its own slice of the corpus,
answering the full benchmark test set, so multi-hop questions whose
evidence spans more than one client are expected to fail for most clients.

Corpus sharding reuses the exact same rule as
`fedcond_grag.dataloader.data_preprocess.partition_linearrag_chunks`
(`idx % num_clients`), applied directly to the raw corpus files under
`dataset/raw/` -- same source `baselines/hipporag` uses.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))


def _install_vllm_stub() -> None:
    """ComoRAG.py unconditionally imports its vllm-offline OpenIE backend at
    module load time, even though this baseline only ever uses the
    OpenAI-compatible endpoint backend (llm_base_url). Real `vllm` pins an
    exact torch build that conflicts with this project's own torch/cuda pin
    (requirements.txt) -- installing it would break the main FL pipeline in
    this same env. Stub the module in sys.modules so the import succeeds;
    it's never actually called since we never select the vllm backend.
    Same approach as baselines/hipporag/client_runner.py.
    """
    if "vllm" in sys.modules:
        return
    stub = types.ModuleType("vllm")

    class _Unavailable:
        def __init__(self, *_a, **_kw):
            raise RuntimeError(
                "vllm backend not installed in this env -- use llm_base_url "
                "(OpenAI-compatible endpoint) instead"
            )

    stub.LLM = _Unavailable
    stub.SamplingParams = _Unavailable
    sys.modules["vllm"] = stub


_install_vllm_stub()

from src.comorag.ComoRAG import ComoRAG
from src.comorag.utils.config_utils import BaseConfig

RAW_DIR = _ROOT / "dataset" / "raw"

# Must match baselines/hipporag/client_runner.py::DEFAULT_SAVE_ROOT (not
# under _ROOT -- ComoRAG's OpenAI response cache uses filelock.FileLock,
# which fails on Windows over a UNC/network path; see that file's comment).
DEFAULT_SAVE_ROOT = Path(os.environ.get("LOCALAPPDATA", "C:/")) / "fedrag_baselines" / "comorag"

# project dataset name -> raw corpus file stem (see scripts/setup_datasets.py)
DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihop": "2wikimultihopqa",
}


def _load_client_docs(hippo_name: str, client_id: int, num_clients: int) -> list[str]:
    corpus = json.loads((RAW_DIR / f"{hippo_name}_corpus.json").read_text(encoding="utf-8"))
    return [
        f"{item['title']}\n{item['text']}"
        for i, item in enumerate(corpus)
        if i % num_clients == client_id
    ]


def _load_global_samples(hippo_name: str) -> list[dict]:
    return json.loads((RAW_DIR / f"{hippo_name}.json").read_text(encoding="utf-8"))


def _load_gold_answers(samples: list[dict]) -> list[list[str]]:
    # Same shape as baselines/hipporag/client_runner.py::_load_gold_answers
    gold_answers = []
    for sample in samples:
        gold_answer = sample.get("answer", sample.get("gold_ans"))
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers = list(dict.fromkeys(answers + list(sample.get("answer_aliases", []))))
        gold_answers.append(answers)
    return gold_answers


def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_name: str = "qwen2.5:1.5b-instruct",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    need_cluster: bool = True,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
) -> dict[str, Any]:
    """Index client `client_id`'s local shard with ComoRAG and evaluate it
    against the full global question set for `dataset`."""
    from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1

    hippo_name = DATASET_NAMES[dataset]
    docs = _load_client_docs(hippo_name, client_id, num_clients)
    samples = _load_global_samples(hippo_name)
    queries = [s["question"] for s in samples]
    gold_answers = _load_gold_answers(samples)

    save_dir = Path(save_root) / dataset / f"client_{client_id}"
    config = BaseConfig(
        save_dir=str(save_dir),
        dataset=hippo_name if hippo_name in ("hotpotqa", "musique", "2wikimultihopqa") else None,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        need_cluster=need_cluster,
        corpus_len=len(docs),
    )

    comorag = ComoRAG(global_config=config)
    comorag.index(docs)
    solutions = comorag.try_answer(queries)

    hits = em_total = f1_total = 0.0
    for solution, answers in zip(solutions, gold_answers):
        pred = solution.answer or ""
        if any(normalize(a) in normalize(pred) for a in answers):
            hits += 1
        if any(exact_match(pred, a) for a in answers):
            em_total += 1.0
        f1_total += max((token_f1(pred, a) for a in answers), default=0.0)

    n = max(len(queries), 1)
    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "num_docs": len(docs),
        "num_questions": len(queries),
        "hit": 100.0 * hits / n,
        "em": 100.0 * em_total / n,
        "f1": 100.0 * f1_total / n,
    }


def run_all_clients(dataset: str, num_clients: int, save_root: Path | str = DEFAULT_SAVE_ROOT, **kwargs: Any) -> dict[str, Any]:
    """Run every client's local ComoRAG baseline and write a summary JSON
    (per-client metrics + mean across clients)."""
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
