"""Algorithm 3: Dual-System (Memorizer / Cognizer) inference for FD-RAG.

Directly implements §3 of the spec:
  1. encode the query and extract A_q (spaCy anchors);
  2. dense-shortlist top ``ann_shortlist_mult * top_k`` memories, rescore
     with Eq.11+12;
  3. FAST PATH (Memorizer): if the top score ≥ δ, return that memory's
     answer -- 0 LLM calls;
  4. SLOW PATH (Cognizer): union the supports of the top-K memories into
     E_q, build Z_q as (C(e), F(e)) per edge, ask the LLM once. For foreign
     edges with no local ``C(e)``, use facts-only evidence per [GAP-8].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.data_types import Hyperedge, QAMemory
from fedcond_grag.baselines.fdrag.facts import query_anchors
from fedcond_grag.baselines.fdrag.memory import DenseIndex
from fedcond_grag.baselines.fdrag.prompts import build_rag_prompt
from fedcond_grag.baselines.fdrag.scoring import score_shortlist


@dataclass
class AnswerResult:
    answer: str
    llm_calls: int
    path: str                              # "fast" or "slow"
    top_score: float
    top_memory_id: Optional[str] = None
    evidence_edge_ids: list[str] = field(default_factory=list)
    latency_sec: float = 0.0


class FDRAGInference:
    """Query-time engine. Built once per client after Stages 1+2."""

    def __init__(
        self,
        *,
        cfg: FDRAGConfig,
        memories: list[QAMemory],
        edges_by_id: dict[str, Hyperedge],
        encoder,
        llm_infer: Optional[Callable[[str], str]] = None,
        local_edge_ids: Optional[set[str]] = None,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.llm_infer = llm_infer
        self.memories: dict[str, QAMemory] = {m.id: m for m in memories}
        self.edges_by_id = edges_by_id
        # A local edge_id is one for which we still have raw C(e). Foreign
        # edges (shipped in via Γ^g in Alg 4/§4.3) only have F(e). Default
        # to "every edge we hold in edges_by_id is local" if not told
        # otherwise -- the federation module overrides this.
        self.local_edge_ids = local_edge_ids or set(edges_by_id.keys())

        dim = int(memories[0].q_vec.shape[0]) if memories else 0
        self.index = DenseIndex(dim)
        self.index.build(memories)
        self.id_to_anchors = {m.id: m.anchors for m in memories}

    # ------------------------------------------------------------------
    def answer(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()
        if not self.memories:
            latency = time.perf_counter() - t0
            return AnswerResult(answer="", llm_calls=0, path="empty", top_score=0.0, latency_sec=latency)

        q_vec = np.asarray(
            self.encoder.encode([question], batch_size=1, show_progress_bar=False)
        )[0].astype(np.float32)
        a_q = query_anchors(question, self.cfg.spacy_model)

        shortlist_k = max(self.cfg.top_k * self.cfg.ann_shortlist_mult, self.cfg.top_k)
        dense = self.index.shortlist(q_vec, shortlist_k)
        rescored = score_shortlist(
            q_vec, a_q, dense, self.id_to_anchors, alpha=self.cfg.alpha
        )
        if not rescored:
            latency = time.perf_counter() - t0
            return AnswerResult(answer="", llm_calls=0, path="empty", top_score=0.0, latency_sec=latency)

        top_id, top_score, _, _ = rescored[0]

        # ------------------- FAST PATH (Memorizer) -------------------
        if top_score >= self.cfg.delta:
            m = self.memories[top_id]
            latency = time.perf_counter() - t0
            return AnswerResult(
                answer=m.answer,
                llm_calls=0,
                path="fast",
                top_score=top_score,
                top_memory_id=top_id,
                evidence_edge_ids=m.support_edge_ids,
                latency_sec=latency,
            )

        # ------------------- SLOW PATH (Cognizer) -------------------
        top_k_ids = [mid for mid, _, _, _ in rescored[: self.cfg.top_k]]
        top_k_mems = [self.memories[mid] for mid in top_k_ids]

        # E_q = ∪ S_r
        edge_ids = []
        seen = set()
        for m in top_k_mems:
            for e_id in m.support_edge_ids:
                if e_id not in seen:
                    seen.add(e_id)
                    edge_ids.append(e_id)

        # Z_q = {(C(e), F(e))} with foreign-edge handling per [GAP-8]
        evidence: list[str] = []
        for e_id in edge_ids:
            edge = self.edges_by_id.get(e_id)
            if edge is None:
                continue
            is_local = e_id in self.local_edge_ids
            if is_local:
                ctx = "\n".join(edge.context)
                facts_str = "; ".join(f"{s} [{l}]" for s, l in edge.facts)
                evidence.append(f"[edge {e_id[:6]}] {ctx}\n(facts: {facts_str})")
            else:
                if self.cfg.foreign_evidence_mode == "drop":
                    continue
                sanitized = edge.facts  # facts_only for foreign
                facts_str = "; ".join(f"{s} [{l}]" for s, l in sanitized)
                evidence.append(f"[edge {e_id[:6]} FOREIGN, facts-only] {facts_str}")

        ref_qa = [(m.question, m.answer) for m in top_k_mems]
        prompt = build_rag_prompt(question=question, ref_qa=ref_qa, evidence=evidence)
        if self.llm_infer is None:
            # Deterministic fallback for smoke tests: return the best memory
            # answer even though score < δ.
            latency = time.perf_counter() - t0
            return AnswerResult(
                answer=self.memories[top_id].answer,
                llm_calls=0,
                path="slow_no_llm",
                top_score=top_score,
                top_memory_id=top_id,
                evidence_edge_ids=edge_ids,
                latency_sec=latency,
            )
        try:
            raw = self.llm_infer(prompt)
        except Exception:
            raw = self.memories[top_id].answer
        latency = time.perf_counter() - t0
        return AnswerResult(
            answer=(raw or "").strip(),
            llm_calls=1,
            path="slow",
            top_score=top_score,
            top_memory_id=top_id,
            evidence_edge_ids=edge_ids,
            latency_sec=latency,
        )
