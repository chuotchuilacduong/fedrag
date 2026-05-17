"""Thin wrapper around LinearRAG's SpacyNER for standalone NER use."""

from __future__ import annotations

from fedcond_grag.linearrag.ner import SpacyNER


class EntityExtractor:
    """Named entity recognition backed by LinearRAG's SpacyNER."""

    def __init__(self, spacy_model: str = "en_core_web_trf"):
        self._ner = SpacyNER(spacy_model)

    def extract_from_passages(
        self,
        hash_id_to_passage: dict[str, str],
        max_workers: int = 4,
    ) -> tuple[dict, dict]:
        """Extract entities from passages.

        Returns:
            (passage_hash_id_to_entities, sentence_to_entities)
        """
        return self._ner.batch_ner(hash_id_to_passage, max_workers)

    def extract_from_question(self, question: str) -> set[str]:
        """Extract named entities from a question string."""
        return self._ner.question_ner(question)
