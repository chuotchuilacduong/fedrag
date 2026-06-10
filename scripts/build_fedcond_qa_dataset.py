"""Build the fedcond_qa dataset: encode questions, write records + splits.

Evidence graphs are retrieved on-the-fly from each client's trigraph
during training, so NO per-question subgraph files are needed.

Saves:
  dataset/fedcond_qa/q_embs.pt          [Q, 384] float32 (question embeddings)
  dataset/fedcond_qa/records.jsonl      {id, question, answer, desc, retrieved_passages}
  dataset/fedcond_qa/split/             train / val / test indices  (80 / 10 / 10)

desc is built by cosine-ranking each question's evidence passages against the
question embedding (all-MiniLM-L6-v2), then taking top TOP_K_DESC passages.
This replaces the old "first 4 unranked paragraphs" approach.

Run from project root after `preprocess` has completed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

ENCODER_MODEL = "all-MiniLM-L6-v2"
TOP_K_DESC = 5          # passages to include in LLM desc
ENCODE_BATCH = 512


def _collect_passages(questions: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Return per-question passage lists and deduplicated unique passage list."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    per_q: list[list[str]] = []
    for q in questions:
        plist: list[str] = []
        for title, sentences in q.get("evidence", []):
            text = f"{title}: {' '.join(str(s).strip() for s in sentences[:3])}"
            plist.append(text)
            if text not in seen:
                seen[text] = len(unique)
                unique.append(text)
        per_q.append(plist)
    return per_q, unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "musique", "2wikimultihop", "medical"])
    parser.add_argument("--top-k-desc", type=int, default=TOP_K_DESC,
                        help="Number of cosine-ranked passages to use as LLM desc")
    args = parser.parse_args()

    processed_root = _ROOT / "processed" / args.dataset
    out_root = _ROOT / "dataset" / "fedcond_qa"

    from sentence_transformers import SentenceTransformer

    questions_path = processed_root / "questions.json"
    if not questions_path.exists():
        print(f"ERROR: questions not found at {questions_path}")
        sys.exit(1)

    questions = json.load(questions_path.open())
    print(f"Loaded {len(questions)} questions from {questions_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "split").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(ENCODER_MODEL)

    # ------------------------------------------------------------------
    # 1. Encode question texts → q_embs.pt
    # ------------------------------------------------------------------
    print(f"Encoding {len(questions)} questions with {ENCODER_MODEL} on {device} ...")
    texts = [q.get("question", "") for q in questions]
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
            print(f"  encoded {i}/{len(texts)} questions")

    q_embs = torch.cat(emb_list, dim=0)   # [Q, 384]
    torch.save(q_embs, out_root / "q_embs.pt")
    print(f"  q_embs.pt saved: {q_embs.shape}")

    # ------------------------------------------------------------------
    # 2. Encode all unique evidence passages (deduplicated)
    # ------------------------------------------------------------------
    per_q_passages, unique_passages = _collect_passages(questions)
    print(f"Encoding {len(unique_passages)} unique evidence passages ...")
    p_emb_list = []
    for i in range(0, len(unique_passages), ENCODE_BATCH):
        embs = encoder.encode(
            unique_passages[i : i + ENCODE_BATCH],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )
        p_emb_list.append(embs.float().cpu())
        if i % 20000 == 0 and i > 0:
            print(f"  encoded {i}/{len(unique_passages)} passages")

    del encoder
    all_p_embs = torch.cat(p_emb_list, dim=0)   # [U, 384]
    text_to_idx = {t: i for i, t in enumerate(unique_passages)}
    print(f"  passage embeddings: {all_p_embs.shape}")

    # ------------------------------------------------------------------
    # 3. Build records.jsonl with cosine-ranked desc
    # ------------------------------------------------------------------
    top_k = args.top_k_desc
    print(f"Building records.jsonl (cosine-ranked desc, top-{top_k} passages) ...")
    records = []
    for qi, q in enumerate(questions):
        passages = per_q_passages[qi]
        if not passages:
            desc = q.get("question", "")
        elif len(passages) <= top_k:
            desc = "\n\n".join(passages)
        else:
            p_idxs = [text_to_idx[p] for p in passages]
            p_embs = all_p_embs[p_idxs]          # [P, 384]
            scores = p_embs @ q_embs[qi]          # [P]
            topk_local = scores.topk(top_k).indices.tolist()
            ranked = sorted(topk_local, key=lambda x: -scores[x].item())
            desc = "\n\n".join(passages[i] for i in ranked)

        records.append({
            "id": str(q["id"]),
            "question": q.get("question", ""),
            "answer": str(q.get("answer", "")),
            "desc": desc,
            "retrieved_passages": passages,
        })

    with (out_root / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  records.jsonl: {len(records)} records")

    # ------------------------------------------------------------------
    # 4. Train / val / test splits  (80 / 10 / 10)
    # ------------------------------------------------------------------
    n = len(records)
    splits = {
        "train": list(range(0, int(0.8 * n))),
        "val":   list(range(int(0.8 * n), int(0.9 * n))),
        "test":  list(range(int(0.9 * n), n)),
    }
    for name, idx in splits.items():
        (out_root / "split" / f"{name}_indices.txt").write_text("\n".join(map(str, idx)))
    print(f"  splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"\nDone. Dataset at {out_root}")


if __name__ == "__main__":
    main()
