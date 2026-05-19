"""Thin wrapper around LinearRAG's SpacyNER for standalone NER use."""

from __future__ import annotations

import os

from fedcond_grag.linearrag.ner import SpacyNER


class EntityExtractor:
    """Named entity recognition backed by LinearRAG's SpacyNER."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._ner = SpacyNER(spacy_model)

    def extract_from_passages(
        self,
        hash_id_to_passage: dict[str, str],
        max_workers: int | None = None,
    ) -> tuple[dict, dict]:
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 16)
        return self._ner.batch_ner(hash_id_to_passage, max_workers)

    def extract_from_question(self, question: str) -> set[str]:
        return self._ner.question_ner(question)
