"""Data loading, partitioning and preprocessing for FedCondGraphRAG.

Consolidates four previously separate modules:
  - hotpot_loader    : HotpotQA raw-JSON ingestion
  - linearrag_loader : LinearRAG chunks.json / questions.json ingestion
  - corpus_index     : bidirectional passage/sentence lookup index
  - federated_partition : deterministic passage/chunk splitting across clients

Also provides a CLI entry-point (``python -m fedcond_grag.dataloader.data_preprocess``)
that replaces ``scripts/preprocess_data.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ===========================================================================
# HotpotQA loader
# ===========================================================================

@dataclass
class HotpotPassage:
    title: str
    passage_id: str
    passage_text: str
    sentences: list[str]


@dataclass
class HotpotQuestion:
    question_id: str
    question: str
    answer: str
    supporting_facts: list[tuple[str, int]]
    passage_ids: list[str]


@dataclass
class HotpotCorpus:
    passages: list[HotpotPassage] = field(default_factory=list)
    questions: list[HotpotQuestion] = field(default_factory=list)
    _title_map: dict[str, HotpotPassage] = field(default_factory=dict, repr=False)

    def get_by_title(self, title: str) -> HotpotPassage | None:
        return self._title_map.get(title)

    def get_by_id(self, passage_id: str) -> HotpotPassage | None:
        for p in self.passages:
            if p.passage_id == passage_id:
                return p
        return None

    def passage_texts(self) -> list[str]:
        return [p.passage_text for p in self.passages]


def _hotpot_passage_id(title: str) -> str:
    return hashlib.sha1(title.encode()).hexdigest()


def _make_hotpot_passage(title: str, sentences: list[str]) -> HotpotPassage:
    passage_text = f"Title: {title}\n" + "\n".join(s.strip() for s in sentences if s.strip())
    return HotpotPassage(
        title=title,
        passage_id=_hotpot_passage_id(title),
        passage_text=passage_text,
        sentences=[s.strip() for s in sentences if s.strip()],
    )


def load_hotpot(path: str | Path, *, max_samples: int | None = None) -> HotpotCorpus:
    """Load raw HotpotQA distractor JSON into a HotpotCorpus."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if max_samples is not None:
        raw = raw[:max_samples]

    corpus = HotpotCorpus()
    for item in raw:
        passage_ids: list[str] = []
        for title, sentences in item["context"]:
            pid = _hotpot_passage_id(title)
            passage_ids.append(pid)
            if title not in corpus._title_map:
                passage = _make_hotpot_passage(title, sentences)
                corpus.passages.append(passage)
                corpus._title_map[title] = passage
        corpus.questions.append(HotpotQuestion(
            question_id=item["_id"],
            question=item["question"],
            answer=item["answer"],
            supporting_facts=[(t, int(i)) for t, i in item["supporting_facts"]],
            passage_ids=passage_ids,
        ))
    return corpus


def load_hotpot_split(
    train_path: str | Path,
    dev_path: str | Path,
    *,
    max_train: int | None = None,
    max_dev: int | None = None,
) -> tuple[HotpotCorpus, HotpotCorpus]:
    return load_hotpot(train_path, max_samples=max_train), load_hotpot(dev_path, max_samples=max_dev)


# ===========================================================================
# LinearRAG loader
# ===========================================================================

_INDEX_RE = re.compile(r"^(\d+):(.*)", re.DOTALL)


@dataclass
class LinearRAGChunk:
    index: int
    text: str
    body: str


@dataclass
class LinearRAGQuestion:
    question_id: str
    source: str
    question: str
    answer: str
    question_type: str
    evidence: list[list]


@dataclass
class LinearRAGDataset:
    name: str
    chunks: list[LinearRAGChunk] = field(default_factory=list)
    questions: list[LinearRAGQuestion] = field(default_factory=list)

    def chunk_texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def question_titles(self) -> list[str]:
        seen: set[str] = set()
        titles: list[str] = []
        for q in self.questions:
            for title, _ in q.evidence:
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        return titles


def _parse_chunk(text: str) -> LinearRAGChunk:
    m = _INDEX_RE.match(text)
    if m:
        return LinearRAGChunk(index=int(m.group(1)), text=text, body=m.group(2).strip())
    return LinearRAGChunk(index=-1, text=text, body=text.strip())


def _parse_question(q: dict) -> LinearRAGQuestion:
    return LinearRAGQuestion(
        question_id=q["id"],
        source=q.get("source", ""),
        question=q["question"],
        answer=q["answer"],
        question_type=q.get("question_type", ""),
        evidence=q.get("evidence", []),
    )


def load_linearrag(
    chunks_path: str | Path,
    questions_path: str | Path,
    *,
    name: str = "",
    max_chunks: int | None = None,
    max_questions: int | None = None,
) -> LinearRAGDataset:
    """Load a LinearRAG-format dataset from chunks.json + questions.json."""
    chunks_path = Path(chunks_path)
    questions_path = Path(questions_path)
    with chunks_path.open("r", encoding="utf-8") as f:
        raw_chunks: list[str] = json.load(f)
    if max_chunks is not None:
        raw_chunks = raw_chunks[:max_chunks]
    with questions_path.open("r", encoding="utf-8") as f:
        raw_questions: list[dict] = json.load(f)
    if max_questions is not None:
        raw_questions = raw_questions[:max_questions]
    return LinearRAGDataset(
        name=name or chunks_path.parent.name,
        chunks=[_parse_chunk(t) for t in raw_chunks],
        questions=[_parse_question(q) for q in raw_questions],
    )


def load_linearrag_dataset(dataset_root: str | Path, dataset_name: str, **kwargs) -> LinearRAGDataset:
    """Load using canonical layout: {root}/{name}/chunks.json + questions.json."""
    root = Path(dataset_root) / dataset_name
    return load_linearrag(root / "chunks.json", root / "questions.json", name=dataset_name, **kwargs)


def save_chunk_list(chunks: Sequence[LinearRAGChunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([c.text for c in chunks], f, ensure_ascii=False)


def save_question_list(questions: Sequence[LinearRAGQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"id": q.question_id, "source": q.source, "question": q.question,
         "answer": q.answer, "question_type": q.question_type, "evidence": q.evidence}
        for q in questions
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ===========================================================================
# Corpus index
# ===========================================================================

def _sentence_id(passage_id: str, sentence_index: int) -> str:
    return hashlib.sha1(f"{passage_id}::{sentence_index}".encode()).hexdigest()


@dataclass
class CorpusIndex:
    """Bidirectional lookup: id ↔ text for passages and sentences."""
    _passage_id_to_text: dict[str, str] = field(default_factory=dict, repr=False)
    _passage_id_to_title: dict[str, str] = field(default_factory=dict, repr=False)
    _sentence_id_to_text: dict[str, str] = field(default_factory=dict, repr=False)
    _title_to_id: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_corpus(cls, corpus: HotpotCorpus) -> "CorpusIndex":
        idx = cls()
        for passage in corpus.passages:
            idx._passage_id_to_text[passage.passage_id] = passage.passage_text
            idx._passage_id_to_title[passage.passage_id] = passage.title
            idx._title_to_id[passage.title] = passage.passage_id
            for i, sent in enumerate(passage.sentences):
                idx._sentence_id_to_text[_sentence_id(passage.passage_id, i)] = sent
        return idx

    def get_passage_text(self, passage_id: str) -> str | None:
        return self._passage_id_to_text.get(passage_id)

    def get_passage_title(self, passage_id: str) -> str | None:
        return self._passage_id_to_title.get(passage_id)

    def get_sentence_text(self, sentence_id: str) -> str | None:
        return self._sentence_id_to_text.get(sentence_id)

    def passage_id_for_title(self, title: str) -> str | None:
        return self._title_to_id.get(title)

    def num_passages(self) -> int:
        return len(self._passage_id_to_text)

    def num_sentences(self) -> int:
        return len(self._sentence_id_to_text)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "passage_id_to_text": self._passage_id_to_text,
                "passage_id_to_title": self._passage_id_to_title,
                "sentence_id_to_text": self._sentence_id_to_text,
                "title_to_id": self._title_to_id,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls()
        idx._passage_id_to_text = data["passage_id_to_text"]
        idx._passage_id_to_title = data["passage_id_to_title"]
        idx._sentence_id_to_text = data["sentence_id_to_text"]
        idx._title_to_id = data["title_to_id"]
        return idx


# ===========================================================================
# Federated partition
# ===========================================================================

@dataclass
class ClientCorpus:
    client_id: int
    num_clients: int
    passages: list[HotpotPassage] = field(default_factory=list)

    def passage_texts(self) -> list[str]:
        return [p.passage_text for p in self.passages]

    def passage_ids(self) -> list[str]:
        return [p.passage_id for p in self.passages]

    def titles(self) -> list[str]:
        return [p.title for p in self.passages]


def _client_for_title(title: str, num_clients: int) -> int:
    return int(hashlib.md5(title.encode()).hexdigest(), 16) % num_clients


def federated_partition(corpus: HotpotCorpus, num_clients: int = 5) -> list[ClientCorpus]:
    """Partition corpus passages across num_clients by hash(title)."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    clients = [ClientCorpus(client_id=i, num_clients=num_clients) for i in range(num_clients)]
    for passage in corpus.passages:
        clients[_client_for_title(passage.title, num_clients)].passages.append(passage)
    return clients


def partition_stats(clients: list[ClientCorpus]) -> dict:
    sizes = [len(c.passages) for c in clients]
    total = sum(sizes)
    return {
        "num_clients": len(clients),
        "total_passages": total,
        "per_client": sizes,
        "min": min(sizes),
        "max": max(sizes),
        "no_overlap": total == len({p.passage_id for c in clients for p in c.passages}),
    }


@dataclass
class ClientChunks:
    client_id: int
    num_clients: int
    chunks: list

    def chunk_texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def indices(self) -> list[int]:
        return [c.index for c in self.chunks]


def partition_linearrag_chunks(chunks, num_clients: int = 5) -> list[ClientChunks]:
    """Partition LinearRAG chunks across clients by chunk.index % num_clients."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    clients = [ClientChunks(client_id=i, num_clients=num_clients, chunks=[]) for i in range(num_clients)]
    for chunk in chunks:
        clients[(chunk.index if chunk.index >= 0 else 0) % num_clients].chunks.append(chunk)
    return clients


def chunk_partition_stats(clients: list[ClientChunks]) -> dict:
    sizes = [len(c.chunks) for c in clients]
    total = sum(sizes)
    all_indices = [c.index for client in clients for c in client.chunks]
    return {
        "num_clients": len(clients),
        "total_chunks": total,
        "per_client": sizes,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "no_overlap": total == len(set(all_indices)),
    }


# ===========================================================================
# Preprocessing CLI  (replaces scripts/preprocess_data.py)
# ===========================================================================

_ALL_DATASETS = ["hotpotqa", "2wikimultihop", "musique", "medical"]


def preprocess_one(dataset_name: str, num_clients: int, dataset_root: Path, out_root: Path) -> dict:
    src = dataset_root / dataset_name
    if not src.exists():
        print(f"  [SKIP] {dataset_name}: source not found at {src}")
        return {}

    print(f"\n=== {dataset_name} ===")
    dataset = load_linearrag_dataset(dataset_root, dataset_name)
    print(f"  Loaded {len(dataset.chunks)} chunks, {len(dataset.questions)} questions")

    clients = partition_linearrag_chunks(dataset.chunks, num_clients=num_clients)
    stats = chunk_partition_stats(clients)
    print(f"  Partition stats: {stats}")

    assert stats["no_overlap"], "BUG: duplicate chunk index across clients"
    assert stats["total_chunks"] == len(dataset.chunks), "BUG: chunks lost in partition"
    assert stats["min"] > 0, "WARNING: some client has 0 chunks"

    ds_out = out_root / dataset_name
    for client in clients:
        out_dir = ds_out / f"client_{client.client_id}"
        save_chunk_list(client.chunks, out_dir / "chunks.json")
        print(f"  client_{client.client_id}: {len(client.chunks)} chunks → {out_dir}/chunks.json")

    save_question_list(dataset.questions, ds_out / "questions.json")
    print(f"  questions → {ds_out}/questions.json")
    return stats


def main(argv: list[str] | None = None) -> None:
    _root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Preprocess LinearRAG datasets into per-client chunks.")
    parser.add_argument("--dataset", default="hotpotqa", choices=_ALL_DATASETS + ["all"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--dataset-root", default=str(_root / "dataset" / "linearrag"))
    parser.add_argument("--out-root", default=str(_root / "processed"))
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    datasets = _ALL_DATASETS if args.dataset == "all" else [args.dataset]

    print(f"Preprocessing {datasets} → {out_root}  (num_clients={args.num_clients})")
    for ds in datasets:
        preprocess_one(ds, num_clients=args.num_clients, dataset_root=dataset_root, out_root=out_root)
    print(f"\nDone. Verify with:\n  ls {out_root}/{datasets[0]}/")


if __name__ == "__main__":
    main()
