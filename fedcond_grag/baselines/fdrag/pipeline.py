"""End-to-end FD-RAG pipeline: index (Stages 1+2) + query (Stage 3).

Kept as a thin ``FDRAG`` class so ``client_runner.py`` can drive it exactly
like ``LinearRAG`` -- ``FDRAG(cfg).index(corpus)`` then
``FDRAG.answer_batch(questions)``.

Federation (Stage 3 §4) is orchestrated *between* clients by the runner,
not by this class -- the class just exposes ``export_bundle`` and
``ingest_global`` hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.data_types import Hyperedge, QAMemory
from fedcond_grag.baselines.fdrag.federation import (
    ClientBundle,
    GlobalKnowledge,
    export_local,
)
from fedcond_grag.baselines.fdrag.inference import AnswerResult, FDRAGInference
from fedcond_grag.baselines.fdrag.memory import build_memories
from fedcond_grag.baselines.fdrag.sahl import learn_hypergraph
from fedcond_grag.baselines.fdrag.segment import segment_corpus, split_by_granularity

_log = logging.getLogger("fdrag.pipeline")


@dataclass
class IndexStats:
    n_docs: int
    n_paragraph_units: int
    n_sentence_units: int
    n_hyperedges: int
    n_memories: int
    build_seconds: float


class FDRAG:
    def __init__(
        self,
        cfg: FDRAGConfig,
        encoder,
        llm_infer: Optional[Callable[[str], str]] = None,
        client_id: int = 0,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.llm_infer = llm_infer
        self.client_id = client_id

        self.hyperedges: list[Hyperedge] = []
        self.edges_by_id: dict[str, Hyperedge] = {}
        self.memories: list[QAMemory] = []
        self.inference: Optional[FDRAGInference] = None
        self.stats: Optional[IndexStats] = None
        self.local_edge_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Stage 1 + Stage 2
    # ------------------------------------------------------------------
    def index(self, corpus: Iterable[dict]) -> IndexStats:
        """Corpus items: ``{title, text}`` (or ``{title, text, idx}``)."""
        t0 = time.perf_counter()
        corpus = list(corpus)

        units = segment_corpus(corpus)
        units_p, units_s = split_by_granularity(units)
        _log.info(
            "[fdrag c=%d] segmented %d docs -> %d paragraphs, %d sentences",
            self.client_id, len(corpus), len(units_p), len(units_s),
        )

        X_p = self._encode_units(units_p)
        X_s = self._encode_units(units_s)
        self.cfg.encoder_dim = int(X_p.shape[1]) if X_p.size else int(X_s.shape[1] if X_s.size else 0)

        _log.info("[fdrag c=%d] running SAHL for %d steps", self.client_id, self.cfg.opt_steps)
        sahl = learn_hypergraph(
            units_p, units_s, X_p, X_s,
            lam=self.cfg.lam, gamma=self.cfg.gamma, mu=self.cfg.mu,
            m_p_ratio=self.cfg.m_p_ratio, m_s_ratio=self.cfg.m_s_ratio,
            steps=self.cfg.opt_steps, lr=self.cfg.learning_rate,
            max_pairs=self.cfg.max_pairs,
            sparsify_mode=self.cfg.sparsify, sparsify_top_r=self.cfg.sparsify_top_r,
            device=self.cfg.device, seed=self.cfg.seed + self.client_id,
        )
        self.hyperedges = sahl.hyperedges
        self.edges_by_id = {e.id: e for e in self.hyperedges}
        self.local_edge_ids = set(self.edges_by_id.keys())
        _log.info("[fdrag c=%d] SAHL produced %d hyperedges", self.client_id, len(self.hyperedges))

        _log.info("[fdrag c=%d] building QA memories (Alg 2)", self.client_id)
        self.memories = build_memories(
            self.hyperedges, self.cfg,
            llm_infer=self.llm_infer, encoder=self.encoder,
            client_id=self.client_id,
        )
        _log.info("[fdrag c=%d] memory bank size: %d", self.client_id, len(self.memories))

        self._rebuild_inference()
        elapsed = time.perf_counter() - t0
        self.stats = IndexStats(
            n_docs=len(corpus),
            n_paragraph_units=len(units_p),
            n_sentence_units=len(units_s),
            n_hyperedges=len(self.hyperedges),
            n_memories=len(self.memories),
            build_seconds=elapsed,
        )
        return self.stats

    def _encode_units(self, units) -> np.ndarray:
        if not units:
            return np.zeros((0, self.cfg.encoder_dim or 384), dtype=np.float32)
        vecs = self.encoder.encode(
            [u.text for u in units],
            batch_size=self.cfg.encoder_batch_size,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        # Cache back onto units for downstream reuse
        for u, v in zip(units, vecs):
            u.vec = v
        return vecs

    def _rebuild_inference(self) -> None:
        self.inference = FDRAGInference(
            cfg=self.cfg,
            memories=self.memories,
            edges_by_id=self.edges_by_id,
            encoder=self.encoder,
            llm_infer=self.llm_infer,
            local_edge_ids=self.local_edge_ids,
        )

    # ------------------------------------------------------------------
    # Stage 3 hooks (see federation.py for the coordinator side)
    # ------------------------------------------------------------------
    def export_bundle(self, candidate_vocab: dict[str, list[str]]) -> ClientBundle:
        return export_local(
            client_id=self.client_id,
            memories=self.memories,
            edges_by_id=self.edges_by_id,
            cfg=self.cfg,
            candidate_vocab=candidate_vocab,
        )

    def ingest_global(self, gk: GlobalKnowledge) -> None:
        """Append foreign memories + facts-only foreign edges, then rebuild
        the query-time index."""
        # De-dup: skip foreign memories that also originated from THIS client.
        for m in gk.memories:
            if m.origin_client == self.client_id:
                continue
            # Re-encode the sanitized question against our local encoder
            # (foreign vec may have come from a different encoder version).
            v = np.asarray(self.encoder.encode([m.question], batch_size=1, show_progress_bar=False))[0]
            m.q_vec = v.astype(np.float32)
            self.memories.append(m)
        for e_id, edge in gk.edges.items():
            if e_id not in self.edges_by_id:
                self.edges_by_id[e_id] = edge
        # local_edge_ids stays as-is -- foreign edges are NOT local.
        self._rebuild_inference()

    # ------------------------------------------------------------------
    # Stage 3 query
    # ------------------------------------------------------------------
    def answer(self, question: str) -> AnswerResult:
        if self.inference is None:
            raise RuntimeError("call FDRAG.index() before FDRAG.answer()")
        return self.inference.answer(question)

    def answer_batch(self, questions: list[str]) -> list[AnswerResult]:
        return [self.answer(q) for q in questions]
