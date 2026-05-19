"""Download full HotpotQA dataset and convert to LinearRAG format.

Output:
    dataset/linearrag/hotpotqa/questions.json   — all questions (train + validation)
    dataset/linearrag/hotpotqa/chunks.json      — deduplicated passage chunks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "dataset" / "linearrag" / "hotpotqa"


def _passage_text(title: str, sentences: list[str]) -> str:
    body = " ".join(sentences)
    return f"{title}: {body}"


def build_linearrag_format(splits: list[str] = ("train", "validation")) -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict] = []
    # chunk_key → global index (for dedup)
    chunk_key_to_idx: dict[str, int] = {}
    chunks: list[str] = []

    for split in splits:
        print(f"Loading hotpot_qa distractor {split}...", flush=True)
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        print(f"  {len(ds)} questions", flush=True)

        for row in ds:
            titles = row["context"]["title"]
            sentences_list = row["context"]["sentences"]

            evidence: list[list] = []
            for title, sents in zip(titles, sentences_list):
                # Register chunk (dedup by title+content)
                key = title + "|||" + " ".join(sents)
                if key not in chunk_key_to_idx:
                    idx = len(chunks)
                    chunk_key_to_idx[key] = idx
                    chunks.append(f"{idx}:{_passage_text(title, sents)}")
                evidence.append([title, sents])

            all_questions.append({
                "id": row["id"],
                "source": "hotpotqa",
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("type", ""),
                "evidence": evidence,
            })

    print(f"\nTotal questions : {len(all_questions)}", flush=True)
    print(f"Unique chunks   : {len(chunks)}", flush=True)

    q_path = OUT_DIR / "questions.json"
    c_path = OUT_DIR / "chunks.json"
    q_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2))
    c_path.write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"Saved → {q_path}", flush=True)
    print(f"Saved → {c_path}", flush=True)


if __name__ == "__main__":
    build_linearrag_format()
