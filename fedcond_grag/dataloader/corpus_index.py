"""Bidirectional corpus index: passage_id / sentence_id ↔ text.

Used during inference to look up text by ID without holding the full
corpus in memory during GNN encoding.

Schema reference: docs/plan/02_DATA_AND_TRIGRAPH.md §9.1
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .hotpot_loader import HotpotCorpus, HotpotPassage


@dataclass
class CorpusIndex:
    """Bidirectional lookup: id ↔ text for passages and sentences."""

    _passage_id_to_text: dict[str, str] = field(default_factory=dict, repr=False)
    _passage_id_to_title: dict[str, str] = field(default_factory=dict, repr=False)
    _sentence_id_to_text: dict[str, str] = field(default_factory=dict, repr=False)
    # title → passage_id
    _title_to_id: dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(cls, corpus: HotpotCorpus) -> "CorpusIndex":
        idx = cls()
        for passage in corpus.passages:
            idx._passage_id_to_text[passage.passage_id] = passage.passage_text
            idx._passage_id_to_title[passage.passage_id] = passage.title
            idx._title_to_id[passage.title] = passage.passage_id
            for sent_idx, sent_text in enumerate(passage.sentences):
                sid = _sentence_id(passage.passage_id, sent_idx)
                idx._sentence_id_to_text[sid] = sent_text
        return idx

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "passage_id_to_text": self._passage_id_to_text,
            "passage_id_to_title": self._passage_id_to_title,
            "sentence_id_to_text": self._sentence_id_to_text,
            "title_to_id": self._title_to_id,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls()
        idx._passage_id_to_text = data["passage_id_to_text"]
        idx._passage_id_to_title = data["passage_id_to_title"]
        idx._sentence_id_to_text = data["sentence_id_to_text"]
        idx._title_to_id = data["title_to_id"]
        return idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentence_id(passage_id: str, sentence_index: int) -> str:
    """Stable sentence ID derived from passage_id + position."""
    raw = f"{passage_id}::{sentence_index}"
    return hashlib.sha1(raw.encode()).hexdigest()
