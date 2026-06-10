"""EvidenceLinearRAG — captures intermediate retrieval state per question.

Uses a private _CaptureLinearRAG subclass to save actived_entities and
sorted_passage_hash_ids after each graph_search_with_seed_entities call.
EvidenceLinearRAG wraps this and provides retrieve_with_evidence().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fedcond_grag.linearrag import LinearRAG, LinearRAGConfig
from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder
from fedcond_grag.client.stage_d_retrieve.linearrag_retriever import LinearRAGRetriever


# ---------------------------------------------------------------------------
# EvidenceRetrievalResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRetrievalResult:
    """All retrieval evidence for a single question.

    Attributes:
        question: The question string.
        gold_answer: Reference answer (may be empty string).
        actived_entities: BFS-activated entities.
            Mapping from hash_id -> (entity_idx, score, tier).
        sorted_passage_hash_ids: PPR-ranked passage hash_ids (all of them).
        sorted_passage_scores: PPR scores corresponding to sorted_passage_hash_ids.
        top_k_passages: Final top-k passage texts (already sliced).
    """

    question: str
    gold_answer: str
    # BFS-activated entities: hash_id -> (entity_idx, score, tier)
    actived_entities: dict[str, tuple[int, float, int]]
    # PPR-ranked passages (all of them, not just top-k)
    sorted_passage_hash_ids: list[str]
    sorted_passage_scores: list[float]
    # Final top-k passage texts (already sliced)
    top_k_passages: list[str]


# ---------------------------------------------------------------------------
# _CaptureLinearRAG
# ---------------------------------------------------------------------------


class _CaptureLinearRAG(LinearRAG):
    """LinearRAG subclass that saves actived_entities after each query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_actived_entities: dict = {}
        self._last_sorted_passage_hash_ids: list[str] = []

    def graph_search_with_seed_entities(
        self,
        question,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        self._last_actived_entities = actived_entities
        passage_weights = self.calculate_passage_scores(
            question, question_embedding, actived_entities
        )
        node_weights = entity_weights + passage_weights
        sorted_passage_hash_ids, sorted_passage_scores = self.run_ppr(node_weights)
        self._last_sorted_passage_hash_ids = sorted_passage_hash_ids
        return sorted_passage_hash_ids, sorted_passage_scores


# ---------------------------------------------------------------------------
# EvidenceLinearRAG
# ---------------------------------------------------------------------------


class EvidenceLinearRAG(LinearRAGRetriever):
    """LinearRAGRetriever subclass that exposes intermediate retrieval state.

    Replaces the internal LinearRAG with _CaptureLinearRAG so that
    actived_entities and sorted_passage_hash_ids are available after each
    per-question graph_search_with_seed_entities call.
    """

    def __init__(
        self,
        working_dir,
        dataset_name,
        encoder=None,
        spacy_model: str = "en_core_web_sm",
        llm_model: Any = None,
        **config_kwargs,
    ):
        # Build config (same as parent)
        enc = encoder or load_encoder(DEFAULT_MODEL)
        cfg = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=enc,
            llm_model=llm_model,
            spacy_model=spacy_model,
            working_dir=str(working_dir),
            **config_kwargs,
        )
        # Use _CaptureLinearRAG instead of LinearRAG
        self._rag = _CaptureLinearRAG(global_config=cfg)
        self._indexed = False
        self._prepared = False  # True after first retrieve call builds lookup arrays

    def _prepare(self) -> None:
        """Build lookup arrays from embedding stores — called once after index()."""
        rag = self._rag
        rag.entity_hash_ids = list(rag.entity_embedding_store.hash_id_to_text.keys())
        rag.entity_embeddings = np.array(rag.entity_embedding_store.embeddings)
        rag.passage_hash_ids = list(rag.passage_embedding_store.hash_id_to_text.keys())
        rag.passage_embeddings = np.array(rag.passage_embedding_store.embeddings)
        rag.sentence_hash_ids = list(rag.sentence_embedding_store.hash_id_to_text.keys())
        rag.sentence_embeddings = np.array(rag.sentence_embedding_store.embeddings)
        rag.node_name_to_vertex_idx = {
            v["name"]: v.index for v in rag.graph.vs if "name" in v.attributes()
        }
        rag.vertex_idx_to_node_name = {
            v.index: v["name"] for v in rag.graph.vs if "name" in v.attributes()
        }
        self._prepared = True

    def retrieve_with_evidence(
        self,
        questions: list[dict],
    ) -> list[EvidenceRetrievalResult]:
        """Like retrieve() but returns EvidenceRetrievalResult with intermediate state.

        Must call index() first.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve_with_evidence()")

        if not self._prepared:
            self._prepare()

        rag = self._rag

        results = []
        for q_info in questions:
            question = q_info["question"]
            q_emb = rag.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=rag.config.batch_size,
            )
            seed_idx, seed_ents, seed_hids, seed_scores = rag.get_seed_entities(question)

            if seed_ents:
                sorted_hids, sorted_scores = rag.graph_search_with_seed_entities(
                    question, q_emb, seed_idx, seed_ents, seed_hids, seed_scores
                )
                # _CaptureLinearRAG saves actived_entities here
                actived = dict(rag._last_actived_entities)
                top_k = rag.config.retrieval_top_k
                top_hids = sorted_hids[:top_k]
                top_passages = [
                    rag.passage_embedding_store.hash_id_to_text[h] for h in top_hids
                ]
            else:
                sorted_idx, sorted_scores = rag.dense_passage_retrieval(q_emb)
                top_k = rag.config.retrieval_top_k
                top_hids = [rag.passage_hash_ids[i] for i in sorted_idx[:top_k]]
                top_passages = [
                    rag.passage_embedding_store.texts[i] for i in sorted_idx[:top_k]
                ]
                sorted_hids = [rag.passage_hash_ids[i] for i in sorted_idx]
                sorted_scores = sorted_scores
                actived = {}

            results.append(
                EvidenceRetrievalResult(
                    question=question,
                    gold_answer=q_info.get("answer", ""),
                    actived_entities=actived,
                    sorted_passage_hash_ids=sorted_hids,
                    sorted_passage_scores=list(sorted_scores),
                    top_k_passages=top_passages,
                )
            )
        return results
