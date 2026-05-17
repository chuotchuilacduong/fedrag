"""Frozen text encoder for Tri-Graph node embeddings.

All clients must use the same encoder so embeddings are comparable.
Default: all-MiniLM-L6-v2 (d=384) per docs/plan/02_DATA_AND_TRIGRAPH.md §10.3.

The encoder is passed directly to LinearRAGConfig.embedding_model, which
expects a SentenceTransformer instance (not a string).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Lazy import so tests that mock this can patch before import
_SentenceTransformer = None


def _get_st():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class NodeEncoder:
    """Thin wrapper around SentenceTransformer that stays frozen (no grad)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = _get_st()(model_name)
        # Ensure all parameters are frozen (LinearRAG rule: encoder must not train)
        for p in self._model.parameters():
            p.requires_grad_(False)

    # SentenceTransformer-compatible interface so we can pass this directly
    # to LinearRAGConfig.embedding_model without any wrapping.
    def encode(
        self,
        sentences,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        return self._model.encode(
            sentences,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            **kwargs,
        )

    def parameters(self):
        return self._model.parameters()

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()


# One shared instance per process (encoder is stateless and heavy to load)
@lru_cache(maxsize=8)
def load_encoder(model_name: str = DEFAULT_MODEL) -> NodeEncoder:
    return NodeEncoder(model_name)
