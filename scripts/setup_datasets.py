"""Set up the hotpotqa / musique / 2wikimultihop benchmark datasets and
convert them to this project's LinearRAG format.

These are our primary evaluation benchmarks. To keep comparisons against
other multi-hop RAG systems meaningful, we use the standard 1000-question
dev split + full retrieval corpus for each dataset, sourced from the copies
checked into OSU-NLP-Group/HippoRAG's `reproduce/dataset/`
(https://github.com/OSU-NLP-Group/HippoRAG) rather than re-deriving our own
sample from a fresh HuggingFace download.

Source files (no auth needed):
    reproduce/dataset/{name}.json         -- 1000 dev questions
    reproduce/dataset/{name}_corpus.json  -- full retrieval corpus (pooled,
                                              deduplicated passages)

Output:
    dataset/linearrag/hotpotqa/{chunks.json,questions.json}
    dataset/linearrag/musique/{chunks.json,questions.json}
    dataset/linearrag/2wikimultihop/{chunks.json,questions.json}

Raw downloads are cached under dataset/raw/ so re-runs don't re-fetch
(pass --force to bypass the cache).

Usage:
    python scripts/setup_datasets.py
    python scripts/setup_datasets.py --dataset musique --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

RAW_DIR = _ROOT / "dataset" / "raw"
OUT_ROOT = _ROOT / "dataset" / "linearrag"

BASE_URL = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"

# source file stem -> output dir name used across this repo's pipeline
DATASETS = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihopqa": "2wikimultihop",
}


def _download(name: str, force: bool) -> tuple[list, list]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    q_path = RAW_DIR / f"{name}.json"
    c_path = RAW_DIR / f"{name}_corpus.json"
    for path, fname in ((q_path, f"{name}.json"), (c_path, f"{name}_corpus.json")):
        if force or not path.exists():
            url = f"{BASE_URL}/{fname}"
            print(f"  downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, path)
    questions = json.loads(q_path.read_text(encoding="utf-8"))
    corpus = json.loads(c_path.read_text(encoding="utf-8"))
    return questions, corpus


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in text.split(". ") if s.strip()]
    return parts or [text.strip()]


def _build_chunks(corpus: list[dict]) -> list[str]:
    # idx is the array position, not corpus["idx"] (musique/2wiki corpora don't
    # carry one) -- kept consistent so partition_linearrag_chunks() (index % N)
    # still gives a full, non-overlapping split.
    return [f"{i}:{item['title']}: {item['text']}" for i, item in enumerate(corpus)]


def _convert_hotpotqa(questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        gold_titles = sorted({t for t, _ in q["supporting_facts"]})
        out.append({
            "id": q["_id"],
            "source": "hotpotqa",
            "question": q["question"],
            "answer": q["answer"],
            "question_type": q.get("type", ""),
            "evidence": q["context"],  # [title, [sentences]] x10 (2 gold + 8 distractor)
            "gold_titles": gold_titles,
        })
    return out


def _convert_musique(questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        aliases, seen = [], set()
        for a in [q["answer"], *[a for a in (q.get("answer_aliases") or [])]]:
            if a and a not in seen:
                seen.add(a)
                aliases.append(a)
        evidence = [[p["title"], _split_sentences(p["paragraph_text"])] for p in q["paragraphs"]]
        gold_titles = sorted({p["title"] for p in q["paragraphs"] if p.get("is_supporting")})
        out.append({
            "id": q["id"],
            "source": "musique",
            "question": q["question"],
            "answer": "|".join(aliases),
            "question_type": "multihop",
            "evidence": evidence,
            "gold_titles": gold_titles,
        })
    return out


def _convert_2wikimultihopqa(questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        gold_titles = sorted({t for t, _ in q.get("supporting_facts", [])})
        out.append({
            "id": q["_id"],
            "source": "2wikimultihop",
            "question": q["question"],
            "answer": q["answer"],
            "question_type": q.get("type", ""),
            "evidence": q["context"],
            "gold_titles": gold_titles,
        })
    return out


CONVERTERS = {
    "hotpotqa": _convert_hotpotqa,
    "musique": _convert_musique,
    "2wikimultihopqa": _convert_2wikimultihopqa,
}


def setup_one(name: str, out_name: str, force: bool) -> None:
    print(f"=== {name} -> dataset/linearrag/{out_name} ===")
    questions_raw, corpus_raw = _download(name, force)
    chunks = _build_chunks(corpus_raw)
    questions = CONVERTERS[name](questions_raw)

    out_dir = OUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (out_dir / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(chunks)} chunks, {len(questions)} questions -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-download even if raw cache exists")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        setup_one(name, DATASETS[name], args.force)
    print("\nDone. Next: python scripts/preprocess_data.py --dataset <name> --num_clients 5")


if __name__ == "__main__":
    main()
