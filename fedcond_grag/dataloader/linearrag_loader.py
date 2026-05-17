"""Loader for LinearRAG pre-processed dataset format.

LinearRAG provides two files per dataset:
  chunks.json   — list of passage strings, each prefixed with an integer
                  index: "N:passage text..." (lowercase, LinearRAG style)
  questions.json — list of question dicts:
                  { id, source, question, answer, question_type,
                    evidence: [[title, [sentence, ...]], ...] }

This is the input format we use for all datasets instead of raw HotpotQA
JSON, because the data was downloaded directly from LinearRAG's repository.

Canonical data location: dataset/linearrag/{dataset_name}/
Processed output:        processed/{dataset_name}/client_{m}/chunks.json

Schema reference: docs/plan/02_DATA_AND_TRIGRAPH.md §9.1 (adapted)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_INDEX_RE = re.compile(r"^(\d+):(.*)", re.DOTALL)


@dataclass
class LinearRAGChunk:
    """One passage in LinearRAG format."""
    index: int           # numeric prefix (used by LinearRAG for sequential edges)
    text: str            # full original string including prefix ("N:text...")
    body: str            # text after stripping the "N:" prefix


@dataclass
class LinearRAGQuestion:
    question_id: str
    source: str
    question: str
    answer: str
    question_type: str
    evidence: list[list]   # [[title, [sent1, sent2, ...]], ...]


@dataclass
class LinearRAGDataset:
    name: str
    chunks: list[LinearRAGChunk] = field(default_factory=list)
    questions: list[LinearRAGQuestion] = field(default_factory=list)

    def chunk_texts(self) -> list[str]:
        """Return the raw chunk strings as passed to LinearRAG.index()."""
        return [c.text for c in self.chunks]

    def question_titles(self) -> list[str]:
        """All unique document titles referenced in evidence fields."""
        seen: set[str] = set()
        titles: list[str] = []
        for q in self.questions:
            for title, _ in q.evidence:
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        return titles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_linearrag(
    chunks_path: str | Path,
    questions_path: str | Path,
    *,
    name: str = "",
    max_chunks: int | None = None,
    max_questions: int | None = None,
) -> LinearRAGDataset:
    """Load a LinearRAG-format dataset from two JSON files.

    Args:
        chunks_path:    Path to chunks.json.
        questions_path: Path to questions.json.
        name:           Dataset name (hotpotqa, musique, …).
        max_chunks:     Limit number of chunks loaded (for smoke tests).
        max_questions:  Limit number of questions loaded.

    Returns:
        LinearRAGDataset with parsed chunks and questions.
    """
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

    chunks = [_parse_chunk(text) for text in raw_chunks]
    questions = [_parse_question(q) for q in raw_questions]

    return LinearRAGDataset(name=name or chunks_path.parent.name, chunks=chunks, questions=questions)


def load_linearrag_dataset(
    dataset_root: str | Path,
    dataset_name: str,
    **kwargs,
) -> LinearRAGDataset:
    """Convenience wrapper using the canonical directory layout.

    Expects:
        {dataset_root}/{dataset_name}/chunks.json
        {dataset_root}/{dataset_name}/questions.json
    """
    root = Path(dataset_root) / dataset_name
    return load_linearrag(
        root / "chunks.json",
        root / "questions.json",
        name=dataset_name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def save_chunk_list(chunks: Sequence[LinearRAGChunk], path: str | Path) -> None:
    """Save a subset of chunks as a chunks.json (preserves LinearRAG format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([c.text for c in chunks], f, ensure_ascii=False)


def save_question_list(questions: Sequence[LinearRAGQuestion], path: str | Path) -> None:
    """Save a list of questions as questions.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "id": q.question_id,
            "source": q.source,
            "question": q.question,
            "answer": q.answer,
            "question_type": q.question_type,
            "evidence": q.evidence,
        }
        for q in questions
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_chunk(text: str) -> LinearRAGChunk:
    m = _INDEX_RE.match(text)
    if m:
        return LinearRAGChunk(index=int(m.group(1)), text=text, body=m.group(2).strip())
    # Fallback: no index prefix (shouldn't happen with LinearRAG data)
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
