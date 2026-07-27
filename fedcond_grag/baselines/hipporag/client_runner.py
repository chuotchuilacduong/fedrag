"""Run the (unmodified) HippoRAG package independently on each federated
client's own passage shard, and evaluate every client against the *same*
global benchmark question set used by fedcond_grag's own Stage D eval.

HippoRAG is a real pip dependency now (see requirements.txt — installed
straight from https://github.com/OSU-NLP-Group/HippoRAG, MIT licensed, pinned
to a commit SHA), not a copy checked into this repo. Nothing in its own
source is patched; `_install_vllm_stub()` below works around one packaging
wrinkle without touching it (see that function's docstring).

This is the "traditional single-node RAG under corpus fragmentation"
baseline: each client only ever sees its own slice of the corpus (no
cross-client fusion), so multi-hop questions whose evidence spans more than
one client are expected to fail for most clients -- that gap is the point of
the comparison against FedCondGraphRAG's federated retrieval.

Corpus sharding reuses the exact same rule as
`fedcond_grag.dataloader.data_preprocess.partition_linearrag_chunks`
(`idx % num_clients`), applied directly to the raw HippoRAG corpus files
under `dataset/raw/` so client boundaries line up 1:1 with the shards used
by the rest of the pipeline (`processed/<dataset>/client_<m>/chunks.json`).

`_load_gold_docs` / `_load_gold_answers` below are adapted from HippoRAG's
own `main.py` (https://github.com/OSU-NLP-Group/HippoRAG, MIT licensed) --
same gold-label extraction logic, not reimplemented from scratch.
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


def _install_vllm_stub() -> None:
    """hipporag/HippoRAG.py unconditionally imports its vllm-offline OpenIE
    backend at module load time, even though we only ever use the
    OpenAI-compatible endpoint backend (llm_base_url). Real `vllm` pins an
    exact torch build that conflicts with this project's own torch/cuda pin
    (requirements.txt) -- installing it would break the main FL pipeline in
    this same env. Stub the module in sys.modules so the import succeeds;
    it's never actually called since we never select the vllm backend.
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

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

RAW_DIR = _ROOT / "dataset" / "raw"

# NOTE: deliberately NOT under _ROOT. HippoRAG's OpenAI response cache uses
# `filelock.FileLock`, and on Windows that's backed by `msvcrt.locking`,
# which does not support UNC/network paths -- this repo lives at
# \\wsl.localhost\... (confirmed: FileLock raises OSError: [Errno 22]
# Invalid argument there, works fine on a local drive). Save HippoRAG's
# index/cache/lock files to a local path instead; only the JSON inputs
# under dataset/raw/ are read from the UNC path (no locking involved there).
DEFAULT_SAVE_ROOT = Path(os.environ.get("LOCALAPPDATA", "C:/")) / "fedrag_baselines" / "hipporag"

# project dataset name -> HippoRAG raw file stem (see scripts/setup_datasets.py)
HIPPO_DATASET_NAMES = {
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


def _load_gold_docs(samples: list[dict], hippo_name: str) -> list[list[str]]:
    # Adapted from HippoRAG's main.py::get_gold_docs
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            separator = "" if hippo_name.startswith("hotpotqa") else " "
            gold_doc = [item[0] + "\n" + separator.join(item[1]) for item in supporting_contexts]
        else:
            assert "paragraphs" in sample, "`paragraphs` should be in sample, or disable retrieval evaluation"
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"]) for item in paragraphs]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def _load_gold_answers(samples: list[dict]) -> list[set[str]]:
    # Adapted from HippoRAG's main.py::get_gold_answers
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
    embedding_model_name: str = "Transformers/sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    force_index_from_scratch: bool = False,
    force_openie_from_scratch: bool = False,
) -> dict[str, Any]:
    """Index client `client_id`'s local shard with HippoRAG and evaluate it
    against the full global question set for `dataset`."""
    hippo_name = HIPPO_DATASET_NAMES[dataset]
    docs = _load_client_docs(hippo_name, client_id, num_clients)
    samples = _load_global_samples(hippo_name)
    queries = [s["question"] for s in samples]
    gold_answers = _load_gold_answers(samples)
    gold_docs = _load_gold_docs(samples, hippo_name)

    save_dir = Path(save_root) / dataset / f"client_{client_id}"
    config = BaseConfig(
        save_dir=str(save_dir),
        dataset=hippo_name,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
    )

    rag = HippoRAG(global_config=config)
    rag.index(docs)
    _, _, _, overall_retrieval_result, overall_qa_results = rag.rag_qa(
        queries=queries, gold_docs=gold_docs, gold_answers=gold_answers
    )

    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "num_docs": len(docs),
        "num_questions": len(queries),
        **(overall_retrieval_result or {}),
        **(overall_qa_results or {}),
    }


def run_all_clients(dataset: str, num_clients: int, save_root: Path | str = DEFAULT_SAVE_ROOT, **kwargs) -> dict[str, Any]:
    """Run every client's local HippoRAG baseline and write a summary JSON
    (per-client metrics + mean across clients) next to the per-client
    HippoRAG save dirs."""
    per_client = [
        run_client_baseline(dataset, client_id, num_clients, save_root=save_root, **kwargs)
        for client_id in range(num_clients)
    ]

    metric_keys = sorted({k for r in per_client for k, v in r.items() if isinstance(v, (int, float))})
    mean_metrics = {
        k: sum(r.get(k, 0.0) for r in per_client) / len(per_client)
        for k in metric_keys
    }

    summary = {"dataset": dataset, "num_clients": num_clients, "per_client": per_client, "mean": mean_metrics}

    out_path = Path(save_root) / dataset / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
