"""Build a separate 1000-question TRAINING set per dataset, sourced from the
official full train split (NOT the 1000-question HippoRAG eval benchmark
under dataset/raw/). Keeps Stage D training from ever touching the questions
used to report final test metrics.

Sources (verified on HuggingFace):
  hotpotqa       -> hotpot_qa (distractor config), split=train
  musique        -> voidful/MuSiQue, split=train
  2wikimultihop  -> framolfese/2WikiMultihopQA, split=train
    (repackaged by the uploader to HotpotQA's own field layout: context is
    {title: [...], sentences: [[...]]}, same as hotpot_qa)

Output (same LinearRAG format as dataset/linearrag/<name>/, so the existing
preprocess pipeline can treat "<name>_train" as its own pseudo-dataset):
    dataset/linearrag/<name>_train/questions.json
    dataset/linearrag/<name>_train/chunks.json

Any train-split question whose id OR normalized question text matches one of
the 1000 eval-benchmark questions (dataset/raw/<name>.json) is excluded
before sampling, so the two 1000-question sets never overlap.

Usage (fedcond env, from project root):
    python scripts/download_train_split.py --dataset musique
    python scripts/download_train_split.py --dataset all
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

RAW_DIR = _ROOT / "dataset" / "raw"
OUT_ROOT = _ROOT / "dataset" / "linearrag"
SAMPLE_SIZE = 1000
SEED = 42

RAW_NAMES = {"hotpotqa": "hotpotqa", "musique": "musique", "2wikimultihop": "2wikimultihopqa"}

HF_SOURCE = {
    "hotpotqa": {"path": "hotpot_qa", "name": "distractor", "split": "train"},
    "musique": {"path": "voidful/MuSiQue", "split": "train", "streaming": True},
    "2wikimultihop": {"path": "framolfese/2WikiMultihopQA", "split": "train"},
}
# Both hotpotqa and the 2wiki repackaging share the same row schema.
HOTPOT_SCHEMA_DATASETS = {"hotpotqa", "2wikimultihop"}


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _existing_eval_keys(dataset: str) -> tuple[set[str], set[str]]:
    """(ids, normalized question texts) already used by the 1000-question
    eval benchmark -- excluded from the new training pool so the two sets
    never overlap, even if the HF mirror renumbers ids."""
    raw_path = RAW_DIR / f"{RAW_NAMES[dataset]}.json"
    if not raw_path.exists():
        print(f"  WARNING: {raw_path} not found -- cannot exclude eval overlap, "
              "proceeding without dedup against it.")
        return set(), set()
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    ids = {str(r.get("id", r.get("_id", ""))) for r in rows if r.get("id") or r.get("_id")}
    qs = {_norm_question(r["question"]) for r in rows if r.get("question")}
    return ids, qs


def _parse_musique_paragraphs(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []
    return []


def _iter_hotpot_schema_rows(dataset: str):
    from datasets import load_dataset

    src = HF_SOURCE[dataset]
    ds = load_dataset(src["path"], src.get("name"), split=src["split"])
    print(f"  {len(ds)} rows in {src['path']} ({src.get('name', '')}) split={src['split']}", flush=True)
    for row in ds:
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        evidence = [[title, sents] for title, sents in zip(titles, sentences_list)]
        yield {
            "id": str(row["id"]),
            "source": dataset,
            "question": row["question"],
            "answer": row["answer"],
            "question_type": row.get("type", ""),
            "evidence": evidence,
        }


def _iter_musique_rows():
    from datasets import load_dataset

    src = HF_SOURCE["musique"]
    ds = load_dataset(src["path"], split=src["split"], streaming=bool(src.get("streaming")))
    for row in ds:
        answerable = row.get("answerable", "True")
        if str(answerable).lower() == "false":
            continue
        paragraphs = _parse_musique_paragraphs(row.get("paragraphs", []))
        evidence = []
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            sentences = [s.strip() for s in text.split(". ") if s.strip()]
            evidence.append([title, sentences])
        answer_aliases = row.get("answer_aliases", [])
        if isinstance(answer_aliases, str):
            try:
                answer_aliases = ast.literal_eval(answer_aliases)
            except Exception:
                answer_aliases = []
        yield {
            "id": str(row["id"]),
            "source": "musique",
            "question": row["question"],
            "answer": row["answer"],
            "answer_aliases": answer_aliases,
            "question_type": "multihop",
            "evidence": evidence,
        }


def build_one(dataset: str) -> None:
    print(f"\n=== {dataset} ===", flush=True)
    exclude_ids, exclude_qs = _existing_eval_keys(dataset)
    print(f"  excluding against {len(exclude_ids)} eval ids / {len(exclude_qs)} eval question texts", flush=True)

    print(f"  downloading full train split...", flush=True)
    rows_iter = (
        _iter_hotpot_schema_rows(dataset)
        if dataset in HOTPOT_SCHEMA_DATASETS
        else _iter_musique_rows()
    )

    pool: list[dict] = []
    seen_q: set[str] = set()
    total = 0
    for row in rows_iter:
        total += 1
        if row["id"] in exclude_ids:
            continue
        nq = _norm_question(row["question"])
        if nq in exclude_qs:
            continue
        # Also dedup within the train pool itself (HF splits occasionally
        # repeat rows across shards).
        if nq in seen_q:
            continue
        seen_q.add(nq)
        pool.append(row)

    print(f"  {total} rows scanned, {len(pool)} eligible after excluding eval overlap + in-pool dupes", flush=True)
    if len(pool) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(pool)} eligible training rows for {dataset}, need {SAMPLE_SIZE}. "
            "Lower SAMPLE_SIZE or check the exclusion logic isn't over-matching."
        )

    rng = random.Random(SEED)
    sampled = rng.sample(pool, SAMPLE_SIZE)
    print(f"  sampled {len(sampled)} questions (seed={SEED})", flush=True)

    # Build deduplicated chunk list from only the sampled questions' own evidence.
    chunk_key_to_idx: dict[str, int] = {}
    chunks: list[str] = []
    questions: list[dict] = []
    for row in sampled:
        evidence_out = []
        for title, sents in row["evidence"]:
            key = title + "|||" + " ".join(sents)
            if key not in chunk_key_to_idx:
                idx = len(chunks)
                chunk_key_to_idx[key] = idx
                text = f"{title}: {' '.join(sents)}" if title else " ".join(sents)
                chunks.append(f"{idx}:{text}")
            evidence_out.append([title, sents])
        q = dict(row)
        q["evidence"] = evidence_out
        questions.append(q)

    out_dir = OUT_ROOT / f"{dataset}_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"  {len(questions)} questions, {len(chunks)} unique chunks -> {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(HF_SOURCE) + ["all"])
    args = parser.parse_args()
    datasets = list(HF_SOURCE) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        build_one(ds)
    print("\nDone. Next: python main.py preprocess --dataset <name>_train --num-clients 3")


if __name__ == "__main__":
    main()
