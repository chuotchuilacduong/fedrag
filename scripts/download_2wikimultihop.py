"""Download full 2WikiMultihopQA and convert to LinearRAG format.

Output:
    dataset/linearrag/2wikimultihop/questions.json
    dataset/linearrag/2wikimultihop/chunks.json

Source: xanhho/2WikiMultihopQA on Hugging Face.

By default this installs the full labeled data: train + dev. The hosted test
split has empty answers/supporting facts, so it is skipped unless explicitly
requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "dataset" / "linearrag" / "2wikimultihop"
HF_DATASET = "xanhho/2WikiMultihopQA"


def _passage_text(title: str, sentences: Iterable[str]) -> str:
    body = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    return f"{title}: {body}" if title else body


def _relation_triples(evidences: dict) -> list[list[str]]:
    facts = evidences.get("fact", []) or []
    rels = evidences.get("relation", []) or []
    entities = evidences.get("entity", []) or []
    return [
        [str(fact), str(rel), str(entity)]
        for fact, rel, entity in zip(facts, rels, entities)
    ]


def _supporting_evidence(row: dict, title_to_sentences: dict[str, list[str]]) -> list[list]:
    supporting = row.get("supporting_facts", {}) or {}
    titles = supporting.get("title", []) or []
    sent_ids = supporting.get("sent_id", []) or []
    evidence: list[list] = []
    for title, sent_id in zip(titles, sent_ids):
        title = str(title)
        sentences = title_to_sentences.get(title, [])
        if isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
            evidence.append([title, [sentences[sent_id]]])
        elif sentences:
            evidence.append([title, sentences[:3]])
    return evidence


def build_linearrag_format(splits: list[str]) -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict] = []
    chunk_key_to_idx: dict[str, int] = {}
    chunks: list[str] = []
    skipped_unlabeled = 0

    for split in splits:
        print(f"Loading {HF_DATASET} {split}...", flush=True)
        ds = load_dataset(HF_DATASET, split=split)
        print(f"  {len(ds)} rows", flush=True)

        for row in ds:
            answer = str(row.get("answer", "") or "").strip()
            if not answer:
                skipped_unlabeled += 1
                continue

            context = row.get("context", {}) or {}
            titles = context.get("title", []) or []
            contents = context.get("content", []) or []
            title_to_sentences: dict[str, list[str]] = {}

            for title, sentences in zip(titles, contents):
                title = str(title)
                clean_sentences = [
                    str(sentence).strip()
                    for sentence in sentences
                    if str(sentence).strip()
                ]
                title_to_sentences[title] = clean_sentences
                key = title + "|||" + " ".join(clean_sentences)
                if key not in chunk_key_to_idx:
                    idx = len(chunks)
                    chunk_key_to_idx[key] = idx
                    chunks.append(f"{idx}:{_passage_text(title, clean_sentences)}")

            evidence = _supporting_evidence(row, title_to_sentences)
            evidence_relations = _relation_triples(row.get("evidences", {}) or {})
            if not evidence and evidence_relations:
                evidence = evidence_relations

            all_questions.append({
                "id": str(row["_id"]),
                "source": "2wikimultihopqa",
                "question": str(row["question"]),
                "answer": answer,
                "question_type": str(row.get("type", "")),
                "evidence": evidence,
                "evidence_relations": evidence_relations,
            })

    print(f"\nTotal labeled questions : {len(all_questions)}", flush=True)
    print(f"Unique chunks           : {len(chunks)}", flush=True)
    if skipped_unlabeled:
        print(f"Skipped unlabeled rows  : {skipped_unlabeled}", flush=True)

    q_path = OUT_DIR / "questions.json"
    c_path = OUT_DIR / "chunks.json"
    q_path.write_text(
        json.dumps(all_questions, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    c_path.write_text(
        json.dumps(chunks, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Saved -> {q_path}", flush=True)
    print(f"Saved -> {c_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "test"],
        help="Splits to install. Defaults to the full labeled train+dev data.",
    )
    args = parser.parse_args()
    build_linearrag_format(args.splits)


if __name__ == "__main__":
    main()
