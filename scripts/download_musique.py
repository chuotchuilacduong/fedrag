"""Download full MuSiQue dataset from HuggingFace and convert to LinearRAG format.

Output:
    dataset/linearrag/musique/questions.json   — all questions (train + validation)
    dataset/linearrag/musique/chunks.json      — deduplicated passage chunks

Source: voidful/MuSiQue on HuggingFace
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "dataset" / "linearrag" / "musique"


def _parse_paragraphs(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []
    return []


def build_linearrag_format(splits: list[str] = ("train",)) -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict] = []
    chunk_key_to_idx: dict[str, int] = {}
    chunks: list[str] = []

    for split in splits:
        print(f"Loading voidful/MuSiQue {split}...", flush=True)
        ds = load_dataset("voidful/MuSiQue", split=split, streaming=True)
        skipped = 0
        count = 0
        for row in ds:
            count += 1
            # Only keep answerable questions
            answerable = row.get("answerable", "True")
            if str(answerable).lower() == "false":
                skipped += 1
                continue

            paragraphs = _parse_paragraphs(row.get("paragraphs", []))

            evidence: list[list] = []
            for para in paragraphs:
                title = para.get("title", "")
                text = para.get("paragraph_text", "")
                sentences = [s.strip() for s in text.split(". ") if s.strip()]
                # Register chunk (dedup by title+text)
                key = title + "|||" + text
                if key not in chunk_key_to_idx:
                    idx = len(chunks)
                    chunk_key_to_idx[key] = idx
                    chunk_text = f"{title}: {text}" if title else text
                    chunks.append(f"{idx}:{chunk_text}")
                evidence.append([title, sentences])

            answer_aliases = row.get("answer_aliases", [])
            if isinstance(answer_aliases, str):
                try:
                    answer_aliases = ast.literal_eval(answer_aliases)
                except Exception:
                    answer_aliases = []

            all_questions.append({
                "id": row["id"],
                "source": "musique",
                "question": row["question"],
                "answer": row["answer"],
                "answer_aliases": answer_aliases,
                "question_type": "multihop",
                "evidence": evidence,
            })

        print(f"  {count} rows total, kept {len(all_questions)} answerable, skipped {skipped}", flush=True)

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
