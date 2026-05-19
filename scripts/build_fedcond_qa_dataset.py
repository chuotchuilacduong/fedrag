"""Build the fedcond_qa dataset: encode questions, write records + splits.

Evidence graphs are retrieved on-the-fly from each client's trigraph
during training, so NO per-question subgraph files are needed.

Saves:
  dataset/fedcond_qa/q_embs.pt          [Q, 384] float32 (question embeddings)
  dataset/fedcond_qa/records.jsonl      {id, question, answer, desc, retrieved_passages}
  dataset/fedcond_qa/split/             train / val / test indices  (80 / 10 / 10)

Run from project root after `preprocess` has completed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

ENCODER_MODEL = "all-MiniLM-L6-v2"
DATASET = "hotpotqa"
PROCESSED_ROOT = _ROOT / "processed" / DATASET
OUT_ROOT = _ROOT / "dataset" / "fedcond_qa"


def _build_description(question: dict) -> tuple[str, list[str]]:
    evidence = question.get("evidence", [])
    passages = []
    for title, sentences in evidence:
        text = f"{title}: {' '.join(str(s).strip() for s in sentences[:3])}"
        passages.append(text)
    desc = "\n\n".join(passages[:4]) if passages else question.get("question", "")
    return desc, passages


def main() -> None:
    from sentence_transformers import SentenceTransformer

    questions_path = PROCESSED_ROOT / "questions.json"
    if not questions_path.exists():
        print(f"ERROR: questions not found at {questions_path}")
        sys.exit(1)

    questions = json.load(questions_path.open())
    print(f"Loaded {len(questions)} questions from {questions_path}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "split").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Encode question texts → q_embs.pt
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(questions)} questions with {ENCODER_MODEL} on {device} ...")
    encoder = SentenceTransformer(ENCODER_MODEL)
    texts = [q.get("question", "") for q in questions]

    ENCODE_BATCH = 512
    emb_list = []
    for i in range(0, len(texts), ENCODE_BATCH):
        embs = encoder.encode(
            texts[i : i + ENCODE_BATCH],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )
        emb_list.append(embs.float().cpu())
        if i % 20000 == 0 and i > 0:
            print(f"  encoded {i}/{len(texts)}")
    del encoder

    q_embs = torch.cat(emb_list, dim=0)          # [Q, 384]
    torch.save(q_embs, OUT_ROOT / "q_embs.pt")
    print(f"  q_embs.pt saved: {q_embs.shape}")

    # ------------------------------------------------------------------
    # 2. Build records.jsonl
    # ------------------------------------------------------------------
    print("Building records.jsonl ...")
    records = []
    for q in questions:
        desc, passages = _build_description(q)
        records.append({
            "id": str(q["id"]),
            "question": q.get("question", ""),
            "answer": str(q.get("answer", "")),
            "desc": desc,
            "retrieved_passages": passages,
        })
    with (OUT_ROOT / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  records.jsonl: {len(records)} records")

    # ------------------------------------------------------------------
    # 3. Train / val / test splits  (80 / 10 / 10)
    # ------------------------------------------------------------------
    n = len(records)
    splits = {
        "train": list(range(0, int(0.8 * n))),
        "val":   list(range(int(0.8 * n), int(0.9 * n))),
        "test":  list(range(int(0.9 * n), n)),
    }
    for name, idx in splits.items():
        (OUT_ROOT / "split" / f"{name}_indices.txt").write_text("\n".join(map(str, idx)))
    print(f"  splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"\nDone. Dataset at {OUT_ROOT}")


if __name__ == "__main__":
    main()
