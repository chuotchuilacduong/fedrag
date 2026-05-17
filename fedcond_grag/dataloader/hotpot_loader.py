"""HotpotQA distractor-set loader.

Normalises the raw HotpotQA JSON into three clean structures:
- HotpotPassage  – one document (title + sentence list)
- HotpotQuestion – one QA example with supporting-fact refs
- HotpotCorpus   – holds all passages deduped by title

Schema reference: docs/plan/02_DATA_AND_TRIGRAPH.md §9.1
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class HotpotPassage:
    title: str
    passage_id: str          # sha1(title)
    passage_text: str        # sentences joined by newline
    sentences: list[str]


@dataclass
class HotpotQuestion:
    question_id: str
    question: str
    answer: str
    supporting_facts: list[tuple[str, int]]   # (title, sentence_index)
    passage_ids: list[str]                    # all passage_ids referenced in context


@dataclass
class HotpotCorpus:
    passages: list[HotpotPassage] = field(default_factory=list)
    questions: list[HotpotQuestion] = field(default_factory=list)
    # title → passage (dedup)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passage_id(title: str) -> str:
    return hashlib.sha1(title.encode()).hexdigest()


def _make_passage(title: str, sentences: list[str]) -> HotpotPassage:
    passage_text = f"Title: {title}\n" + "\n".join(s.strip() for s in sentences if s.strip())
    return HotpotPassage(
        title=title,
        passage_id=_passage_id(title),
        passage_text=passage_text,
        sentences=[s.strip() for s in sentences if s.strip()],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_hotpot(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> HotpotCorpus:
    """Load raw HotpotQA distractor JSON into a HotpotCorpus.

    Args:
        path: Path to ``hotpot_train_v1.1.json`` or
              ``hotpot_dev_distractor_v1.json``.
        max_samples: If set, only read the first N questions.

    Returns:
        HotpotCorpus with deduplicated passages and all questions.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if max_samples is not None:
        raw = raw[:max_samples]

    corpus = HotpotCorpus()

    for item in raw:
        qid = item["_id"]
        question = item["question"]
        answer = item["answer"]
        supporting_facts: list[tuple[str, int]] = [
            (title, int(sent_idx)) for title, sent_idx in item["supporting_facts"]
        ]

        context: list[tuple[str, list[str]]] = item["context"]
        passage_ids: list[str] = []

        for title, sentences in context:
            pid = _passage_id(title)
            passage_ids.append(pid)
            if title not in corpus._title_map:
                passage = _make_passage(title, sentences)
                corpus.passages.append(passage)
                corpus._title_map[title] = passage

        corpus.questions.append(
            HotpotQuestion(
                question_id=qid,
                question=question,
                answer=answer,
                supporting_facts=supporting_facts,
                passage_ids=passage_ids,
            )
        )

    return corpus


def load_hotpot_split(
    train_path: str | Path,
    dev_path: str | Path,
    *,
    max_train: int | None = None,
    max_dev: int | None = None,
) -> tuple[HotpotCorpus, HotpotCorpus]:
    """Convenience wrapper that loads both train and dev splits."""
    train = load_hotpot(train_path, max_samples=max_train)
    dev = load_hotpot(dev_path, max_samples=max_dev)
    return train, dev
