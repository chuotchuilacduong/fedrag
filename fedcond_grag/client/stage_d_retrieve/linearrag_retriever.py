"""Thin wrapper around LinearRAG for retrieval and graph building.

Delegates all indexing and retrieval to LinearRAG's existing implementation.
Adds to_pyg() to expose the Tri-Graph as a PyG Data object after indexing.

LinearRAG.index()    — NER + embedding + graph construction (igraph)
LinearRAG.retrieve() — seed entities → entity propagation → PPR → top-k passages
LinearRAG.qa()       — retrieve() + LLM answer generation

Usage:
    retriever = LinearRAGRetriever(
        working_dir="processed/linearrag_cache",
        dataset_name="hotpotqa_client_0",
    )
    retriever.index(chunk_texts)                 # builds igraph + embedding stores
    graph = retriever.to_pyg()                   # PyG Tri-Graph for GNN
    results = retriever.retrieve(questions)      # list[dict] with sorted_passage
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch_geometric.data import Data

from fedcond_grag.linearrag import LinearRAG, LinearRAGConfig
from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder
from fedcond_grag.client.stage_a_trigraph.trigraph_builder import _rag_to_pyg


class LinearRAGRetriever:
    """Federated-client retriever backed by LinearRAG.

    All retrieval logic lives in LinearRAG — this class only:
    1. Constructs LinearRAG with the right config.
    2. Exposes index / retrieve / qa with the same signatures.
    3. Adds to_pyg() to convert the indexed state to a PyG Data object.
    """

    def __init__(
        self,
        working_dir: str | Path,
        dataset_name: str,
        encoder=None,
        spacy_model: str = "en_core_web_trf",
        llm_model: Any = None,
        **config_kwargs,
    ) -> None:
        enc = encoder or load_encoder(DEFAULT_MODEL)
        cfg = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=enc,
            llm_model=llm_model,
            spacy_model=spacy_model,
            working_dir=str(working_dir),
            **config_kwargs,
        )
        self._rag = LinearRAG(global_config=cfg)
        self._indexed = False

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def index(self, passages: list[str]) -> None:
        """Index passages using LinearRAG.index() directly.

        Builds NER cache, embedding parquet files, and the igraph graph.
        Must be called before retrieve(), qa(), or to_pyg().
        """
        self._rag.index(passages)
        self._indexed = True

    def to_pyg(self) -> Data:
        """Convert indexed LinearRAG state to a PyG Tri-Graph Data object.

        Returns:
            PyG Data with x, edge_index, edge_type, node_type, node_text.
            See trigraph_builder._rag_to_pyg for field details.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before to_pyg()")
        return _rag_to_pyg(self._rag)

    # ------------------------------------------------------------------
    # Retrieval — direct delegation to LinearRAG
    # ------------------------------------------------------------------

    def retrieve(self, questions: list[dict]) -> list[dict]:
        """Retrieve top passages per question.

        Args:
            questions: list of dicts with at least {"question": str, "answer": str}.

        Returns:
            Same format as LinearRAG.retrieve(): list of dicts with
            question, sorted_passage, sorted_passage_scores, gold_answer.
        """
        return self._rag.retrieve(questions)

    def qa(self, questions: list[dict]) -> list[dict]:
        """Answer questions with the LLM. Delegates to LinearRAG.qa()."""
        return self._rag.qa(questions)

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------

    @property
    def rag(self) -> LinearRAG:
        """The underlying LinearRAG instance (for advanced use)."""
        return self._rag
