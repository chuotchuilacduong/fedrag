"""Per-client local FLARE baseline (https://github.com/jzbjyb/FLARE, MIT
licensed -- vendored `ApiReturn` under `_vendor/`, see `_vendor/VENDORED.md`
for what's vendored vs. reimplemented here and why).

Like `baselines/hipporag` and `baselines/grag`: single-client, non-federated
-- each client only ever retrieves from its own passage shard, evaluated
against the *global* question set, to measure what a traditional
single-node RAG method loses when its corpus is fragmented.

FLARE's method (from the paper): iteratively generate a temporary
"look-ahead" continuation of the answer, decide whether to retrieve by
checking whether it contains low-confidence tokens (masking them via the
vendored `ApiReturn.use_as_query`), retrieve with the masked span as the
query if it's non-empty, and regenerate the continuation conditioned on
what was retrieved -- repeated sentence-by-sentence. Unlike the other three
baselines, this one is training-free (no GNN, no LoRA): pure retrieval +
prompting, so there's no local-training loop here, just retrieve-generate-
evaluate per question.

Corpus + sharding: same `dataset/raw/<name>_corpus.json` + `idx %
num_clients` rule as `baselines/hipporag` / `baselines/grag`. Retrieval is
plain embedding-similarity search over that shard (all-MiniLM-L6-v2,
matching the rest of this project) -- FLARE's paper doesn't mandate a
specific retriever, it calls out to "an off-the-shelf retriever" via its own
pluggable interface.
"""

from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from api_return import ApiReturn  # noqa: E402  (vendored FLARE package)

from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder  # noqa: E402
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1  # noqa: E402

RAW_DIR = _ROOT / "dataset" / "raw"
DEFAULT_SAVE_ROOT = _ROOT / "output" / "baselines" / "flare"

# project dataset name -> raw corpus file stem (see scripts/setup_datasets.py)
DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihop": "2wikimultihopqa",
}

DEFAULT_ARGS: dict[str, Any] = dict(
    llm_base_url="http://localhost:11434/v1",
    llm_name="qwen2.5:1.5b-instruct",
    look_ahead_steps=64,       # matches FLARE's own 2wikihop_flare_config.json
    mask_prob=0.4,             # "
    retrieval_topk=2,          # "
    max_sentences=4,           # answer length cap for short-answer QA (vs. FLARE's long-form default)
    max_eval_samples=200,
)


class _CorpusRetriever:
    """Plain embedding-similarity retriever over one client's passage
    shard -- FLARE's own interface is just `.retrieve(query, topk)`, ES/BM25
    in upstream's code; this is the per-client-corpus equivalent used by
    `baselines/hipporag`/`baselines/grag`."""

    def __init__(self, passages: list[str], encoder):
        self.passages = passages
        self.encoder = encoder
        self.embeddings = encoder.encode(
            passages, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        ).float().cpu() if passages else None

    def retrieve(self, query: str, topk: int) -> list[str]:
        if not query.strip() or self.embeddings is None:
            return []
        import torch
        q_emb = self.encoder.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        ).float().cpu()[0]
        sims = self.embeddings @ q_emb
        k = min(topk, len(self.passages))
        top_idx = torch.topk(sims, k).indices.tolist()
        return [self.passages[i] for i in top_idx]


def _load_client_passages(dataset: str, client_id: int, num_clients: int) -> list[str]:
    hippo_name = DATASET_NAMES[dataset]
    corpus = json.loads((RAW_DIR / f"{hippo_name}_corpus.json").read_text(encoding="utf-8"))
    return [
        f"{item['title']}\n{item['text']}"
        for i, item in enumerate(corpus)
        if i % num_clients == client_id
    ]


def _generate_with_logprobs(client, model: str, prompt: str, max_tokens: int) -> ApiReturn:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        logprobs=True,
        top_logprobs=1,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    lp = choice.logprobs.content if choice.logprobs else []
    tokens = [t.token for t in lp]
    probs = [math.exp(t.logprob) for t in lp]
    offsets, cum = [], len(prompt)
    for t in tokens:
        offsets.append(cum)
        cum += len(t)
    return ApiReturn(
        prompt=prompt, text=text, tokens=tokens or None, probs=probs or None, offsets=offsets or None,
        finish_reason=choice.finish_reason, model=model,
    )


def _build_prompt(question: str, answer_so_far: str, context: str) -> str:
    parts = []
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"Question: {question}\nAnswer: {answer_so_far}")
    return "\n".join(parts)


def _flare_answer(client, model: str, retriever: _CorpusRetriever, question: str, args) -> str:
    answer = ""
    retrieved_context: list[str] = retriever.retrieve(question, args.retrieval_topk)  # initial retrieval (FLARE's own "pre-retrieval for look ahead")

    for _ in range(args.max_sentences):
        context = "\n".join(retrieved_context)
        prompt = _build_prompt(question, answer, context)

        # 1. generate a temporary look-ahead continuation
        lookahead = _generate_with_logprobs(client, model, prompt, args.look_ahead_steps)
        lookahead.truncate_at_boundary("sentence")
        if lookahead.is_empty:
            break

        # 2. decide whether to retrieve: mask low-confidence tokens to form the query
        query = lookahead.use_as_query(mask_prob=args.mask_prob, mask_method="simple")
        if query:
            new_passages = retriever.retrieve(query, args.retrieval_topk)
            for p in new_passages:
                if p not in retrieved_context:
                    retrieved_context.append(p)
            prompt = _build_prompt(question, answer, "\n".join(retrieved_context))

        # 3. regenerate the real continuation conditioned on (possibly updated) context
        real = _generate_with_logprobs(client, model, prompt, args.look_ahead_steps)
        real.truncate_at_boundary("sentence")
        if real.is_empty:
            break
        sep = " " if answer and not answer.endswith((" ", "\n")) else ""
        answer += sep + real.text.lstrip()
        if real.finish_reason == "stop":
            break

    return answer.strip()


def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    save_root: Path | str = DEFAULT_SAVE_ROOT,
    **arg_overrides: Any,
) -> dict[str, Any]:
    """Answer the full global question set for `dataset` using FLARE,
    retrieving only from client `client_id`'s own passage shard."""
    from openai import OpenAI
    from fedcond_grag.dataloader import FedCondQADataset

    args = Namespace(**{**DEFAULT_ARGS, **arg_overrides})

    encoder = load_encoder(DEFAULT_MODEL)
    passages = _load_client_passages(dataset, client_id, num_clients)
    retriever = _CorpusRetriever(passages, encoder)

    client = OpenAI(base_url=args.llm_base_url, api_key="sk-")

    qa_dataset = FedCondQADataset(root="dataset/fedcond_qa")
    idx_split = qa_dataset.get_idx_split()
    max_eval = int(args.max_eval_samples)
    eval_idx = (idx_split["val"] + idx_split["test"])[:max_eval]

    hits = em_total = f1_total = 0.0
    for i in eval_idx:
        row = qa_dataset[i]
        raw_question = row["question"].split("Question: ", 1)[-1].split("\nAnswer:")[0]
        pred = _flare_answer(client, args.llm_name, retriever, raw_question, args)
        label = row["label"]
        if normalize(label) in normalize(pred):
            hits += 1
        if exact_match(pred, label):
            em_total += 1.0
        f1_total += token_f1(pred, label)

    n = max(len(eval_idx), 1)
    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "num_docs": len(passages),
        "num_questions": len(eval_idx),
        "hit": 100.0 * hits / n,
        "em": 100.0 * em_total / n,
        "f1": 100.0 * f1_total / n,
    }


def run_all_clients(dataset: str, num_clients: int, save_root: Path | str = DEFAULT_SAVE_ROOT, **kwargs: Any) -> dict[str, Any]:
    """Run every client's local FLARE baseline independently and write a
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
