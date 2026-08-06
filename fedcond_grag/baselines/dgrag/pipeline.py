"""DGRAG end-to-end pipeline: index (Phase A) + query (Phase B).

Mirrors the FDRAG.index() / .answer() shape so client_runner.py can drive
both baselines the same way. Phase A runs offline; Phase B is online.

Federation (cross-edge retrieval) is orchestrated in federation.py by the
runner — this class exposes publish_summaries() / receive_peer_summaries()
hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fedcond_grag.baselines.dgrag.config import AblationFlag, DGRAGConfig
from fedcond_grag.baselines.dgrag.chunker import chunk_corpus
from fedcond_grag.baselines.dgrag.data_types import AnswerResult, Evidence, Subgraph, SummaryRecord
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.offline.extract import extract_all
from fedcond_grag.baselines.dgrag.offline.partition import partition_graph
from fedcond_grag.baselines.dgrag.offline.summarize import (
    subgraph_to_summary_record,
    summarize_subgraphs,
)
from fedcond_grag.baselines.dgrag.online.cross_edge import cross_edge_query
from fedcond_grag.baselines.dgrag.online.gate import run_gate
from fedcond_grag.baselines.dgrag.online.local_query import local_query, retrieve_only
from fedcond_grag.baselines.dgrag.store import EdgeKnowledgeBase, SummaryVectorStore

_log = logging.getLogger("dgrag.pipeline")


@dataclass
class IndexStats:
    edge_id: str
    n_docs: int
    n_chunks: int
    n_entities: int
    n_relations: int
    n_subgraphs: int
    n_subgraph_sizes_p50: int
    build_seconds: float


class DGRAG:
    def __init__(
        self,
        cfg: DGRAGConfig,
        encoder,
        edge_id: str = "edge-0",
        slm: Optional[DGRAGModel] = None,
        cloud_llm: Optional[DGRAGModel] = None,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.edge_id = edge_id
        # In §9 no-cloud adaptation SLM == cloud_llm (same model)
        self.slm = slm
        self.cloud_llm = cloud_llm or slm

        self.kb = EdgeKnowledgeBase(edge_id=edge_id, encoder=encoder)
        self.subgraphs: list[Subgraph] = []
        self.summary_vdb: Optional[SummaryVectorStore] = None  # set by ingest_peer_summaries
        self.stats: Optional[IndexStats] = None

    # ------------------------------------------------------------------
    # Phase A — offline construction
    # ------------------------------------------------------------------

    def index(self, corpus: list[dict]) -> IndexStats:
        t0 = time.perf_counter()

        # ① Chunking
        chunks = chunk_corpus(
            corpus,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        _log.info("[%s] chunked %d docs -> %d chunks", self.edge_id, len(corpus), len(chunks))

        # ① Entity/relation extraction (may call LLM many times)
        if self.slm is not None:
            entity_store, relation_store = extract_all(
                chunks, self.slm,
                glean_rounds=self.cfg.glean_rounds,
            )
        else:
            # No-LLM smoke-test path: empty KG, retrieval falls back to chunk VDB only
            entity_store, relation_store = {}, {}

        # Add chunks, entities, relations to KB
        self.kb.add_chunks_batch(chunks)
        self.kb.add_entities_batch(list(entity_store.values()))
        self.kb.add_relations_batch(list(relation_store.values()))

        _log.info("[%s] KB: %d entities, %d relations", self.edge_id,
                  len(self.kb.entities), len(self.kb.relations))

        # ② Partition + summarize
        if self.kb.graph.number_of_nodes() > 0 and self.slm is not None:
            communities, lin_texts = partition_graph(
                self.kb.graph, self.edge_id,
                entity_store, relation_store,
                target=self.cfg.target_entities_per_subgraph,
                min_size=self.cfg.subgraph_min_size,
                resolutions=self.cfg.leiden_resolutions,
                n_iterations=self.cfg.leiden_n_iterations,
                seed=self.cfg.leiden_seed,
            )
            self.subgraphs = summarize_subgraphs(
                communities, lin_texts,
                edge_id=self.edge_id,
                entities=entity_store,
                relations=relation_store,
                slm=self.slm,
                encoder=self.encoder,
                summary_max_tokens=self.cfg.summary_max_tokens,
            )
        else:
            self.subgraphs = []

        self.kb.subgraphs = self.subgraphs

        elapsed = time.perf_counter() - t0
        sizes = [len(sg.entity_names) for sg in self.subgraphs]
        p50 = sorted(sizes)[len(sizes) // 2] if sizes else 0
        self.stats = IndexStats(
            edge_id=self.edge_id,
            n_docs=len(corpus),
            n_chunks=len(chunks),
            n_entities=len(self.kb.entities),
            n_relations=len(self.kb.relations),
            n_subgraphs=len(self.subgraphs),
            n_subgraph_sizes_p50=p50,
            build_seconds=elapsed,
        )
        return self.stats

    # ------------------------------------------------------------------
    # Federation hooks
    # ------------------------------------------------------------------

    def publish_summaries(self) -> list[SummaryRecord]:
        return [subgraph_to_summary_record(sg) for sg in self.subgraphs
                if sg.summary_vec is not None]

    def ingest_peer_summaries(self, vdb: SummaryVectorStore) -> None:
        """Accept the global summary VDB built by the coordinator."""
        self.summary_vdb = vdb

    def retrieve_only_fn(self) -> Callable[[str], Evidence]:
        """Returns a callable for cross-edge fan-out (no generation)."""
        cfg = self.cfg
        enc = self.encoder
        slm = self.slm

        def _fn(query: str) -> Evidence:
            if slm is None:
                from fedcond_grag.baselines.dgrag.data_types import Evidence
                return Evidence(entities=[], relations=[], chunks=[], edge_id=self.edge_id)
            return retrieve_only(
                query, self.kb, slm, enc,
                top_ent=cfg.top_ent, top_rel=cfg.top_rel,
                top_ent_expanded=cfg.top_ent_expanded, top_rel_expanded=cfg.top_rel_expanded,
                top_chunk=cfg.top_chunk, ctx_max_tokens=cfg.ctx_max_tokens,
                use_kg=(cfg.ablation != "w/o_KG"),
            )
        return _fn

    # ------------------------------------------------------------------
    # Phase B — online query
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        peer_retrieve_fns: Optional[dict[str, Callable[[str], Evidence]]] = None,
    ) -> AnswerResult:
        t0 = time.perf_counter()
        llm_calls = 0
        cfg = self.cfg

        if self.slm is None:
            # Smoke-test path: no LLM, return chunk-match heuristic
            latency = time.perf_counter() - t0
            return AnswerResult(
                answer=self._chunk_fallback(question),
                route="fallback",
                gate_score=0.0,
                llm_calls=0,
                latency_sec=latency,
            )

        # Effective batch_b for w/o_BQ ablation
        b = 1 if cfg.ablation == "w/o_BQ" else cfg.batch_b

        # 4.1.1-4.1.5 local query: keyword + retrieve + B-batch generate
        answers, evidence = local_query(
            question, self.kb, self.slm, self.encoder,
            batch_b=b,
            batch_temperature=cfg.batch_temperature,
            batch_top_p=cfg.batch_top_p,
            top_ent=cfg.top_ent, top_rel=cfg.top_rel,
            top_ent_expanded=cfg.top_ent_expanded, top_rel_expanded=cfg.top_rel_expanded,
            top_chunk=cfg.top_chunk, ctx_max_tokens=cfg.ctx_max_tokens,
            short_answer=True,
            use_kg=(cfg.ablation != "w/o_KG"),
        )
        llm_calls += 1 + b  # 1 keyword call + B generation calls

        # Step ⑤ Gate
        skip_cd = (cfg.ablation == "w/o_CD")
        skip_se = (cfg.ablation == "w/o_SE") or (cfg.ablation == "w/o_BQ")
        gate_decision, gate_llm_calls = run_gate(
            question, answers, self.slm, self.encoder,
            gate_threshold=cfg.gate_threshold,
            skip_confidence=skip_cd,
            skip_similarity=skip_se,
        )
        llm_calls += gate_llm_calls

        if gate_decision.route == "local":
            latency = time.perf_counter() - t0
            return AnswerResult(
                answer=gate_decision.best_answer or (answers[0] if answers else ""),
                route="local",
                gate_score=gate_decision.score,
                llm_calls=llm_calls,
                latency_sec=latency,
                peer_edges_contacted=0,
            )

        # Step ⑥ Cross-edge escalation
        if peer_retrieve_fns and self.summary_vdb is not None:
            cloud_answer, cloud_calls, n_peers = cross_edge_query(
                question,
                summary_vdb=self.summary_vdb,
                peer_retrieve_fn=peer_retrieve_fns,
                cloud_llm=self.cloud_llm,
                encoder=self.encoder,
                top_m=cfg.top_m,
                thr_summary=cfg.thr_summary,
                thr_global=cfg.thr_global,
                top_k_edges=cfg.top_k_edges,
                cloud_ctx_max_tokens=cfg.cloud_ctx_max_tokens,
                short_answer=True,
            )
            # Each peer also runs 1 keyword-extraction call per retrieve_only call
            llm_calls += n_peers + cloud_calls  # peer keyword×n_peers + 1 cloud gen
            latency = time.perf_counter() - t0
            return AnswerResult(
                answer=cloud_answer.strip(),
                route="global",
                gate_score=gate_decision.score,
                llm_calls=llm_calls,
                latency_sec=latency,
                peer_edges_contacted=n_peers,
            )

        # Fallback if no peers available (local mode)
        latency = time.perf_counter() - t0
        best = answers[0] if answers else ""
        return AnswerResult(
            answer=best,
            route="local_fallback",
            gate_score=gate_decision.score,
            llm_calls=llm_calls,
            latency_sec=latency,
            peer_edges_contacted=0,
        )

    def _chunk_fallback(self, question: str) -> str:
        """No-LLM heuristic: return the chunk with highest word overlap with question."""
        if not self.kb.chunks:
            return ""
        q_words = set(question.lower().split())
        best_chunk = max(
            self.kb.chunks.values(),
            key=lambda c: len(q_words & set(c.text.lower().split())),
        )
        return " ".join(best_chunk.text.split()[:40])
