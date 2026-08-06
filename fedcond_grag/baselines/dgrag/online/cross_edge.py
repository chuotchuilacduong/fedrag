"""Step ⑥: Cross-edge Retrieval Mechanism (spec §4.3).

In the §9 no-cloud adaptation, the cloud server is replaced by a coordinator
process running the same model as the edges. Steps:

1. Summary Matching: query the global summary VDB, threshold at THR_SUMMARY.
2. Fan-out: for each of the top-K selected peer edges, call retrieve_only()
   (no generation on the peer side — spec invariant).
3. Filter + rerank evidence by cosine similarity, threshold at THR_GLOBAL.
4. Cloud (coordinator) generates the final answer from the aggregated evidence.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from fedcond_grag.baselines.dgrag.data_types import AnswerResult, Chunk, Entity, Evidence, Relation
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.prompts import build_rag_prompt
from fedcond_grag.baselines.dgrag.store import SummaryVectorStore

_log = logging.getLogger("dgrag.cross_edge")


def _format_entities(entities: list[Entity]) -> str:
    if not entities:
        return "(none)"
    return "\n".join(f"{e.name} | type: {e.type} | {e.description}" for e in entities)


def _format_relations(relations: list[Relation]) -> str:
    if not relations:
        return "(none)"
    return "\n".join(f"({r.source}, {r.target}) | {r.keyword}: {r.description}" for r in relations)


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n---\n".join(c.text for c in chunks)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" [truncated]" if len(words) > max_words else "")


def cross_edge_query(
    query: str,
    summary_vdb: SummaryVectorStore,
    peer_retrieve_fn: dict[str, Callable[[str], Evidence]],
    cloud_llm: DGRAGModel,
    encoder,
    *,
    top_m: int = 1,
    thr_summary: float = 0.4,
    thr_global: float = 0.4,
    top_k_edges: int = 1,
    cloud_ctx_max_tokens: int = 24000,
    short_answer: bool = True,
) -> tuple[str, int, int]:
    """Run the global query path. Returns (answer, llm_calls, n_peer_edges_contacted)."""
    llm_calls = 0

    # 1. Summary matching
    q_vec = np.asarray(encoder.encode([query], batch_size=1, show_progress_bar=False))[0]
    summary_hits = summary_vdb.search(q_vec, top_m=top_m * 10, threshold=thr_summary)

    # Unique edge ids, limited to top_k_edges
    seen_edges: set[str] = set()
    target_edge_ids: list[str] = []
    for record in summary_hits:
        eid = record.edge_id
        if eid not in seen_edges:
            seen_edges.add(eid)
            target_edge_ids.append(eid)
            if len(target_edge_ids) >= top_k_edges:
                break

    if not target_edge_ids:
        _log.debug("No summary hits above threshold=%.2f; using all available peers", thr_summary)
        target_edge_ids = list(peer_retrieve_fn.keys())[:top_k_edges]

    # 2. Fan-out: retrieve_only at each selected peer edge (no generation)
    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    all_chunks: list[Chunk] = []
    n_contacted = 0

    for edge_id in target_edge_ids:
        retrieve_fn = peer_retrieve_fn.get(edge_id)
        if retrieve_fn is None:
            continue
        n_contacted += 1
        try:
            evidence = retrieve_fn(query)
            all_entities.extend(evidence.entities)
            all_relations.extend(evidence.relations)
            all_chunks.extend(evidence.chunks)
        except Exception as exc:
            _log.warning("Peer retrieval failed for edge %s: %s", edge_id, exc)

    # 3. Filter + rerank by cosine(embed(query), embed(evidence_text))
    # Filter chunks by similarity threshold
    if all_chunks:
        chunk_texts = [c.text for c in all_chunks]
        chunk_vecs = np.asarray(encoder.encode(chunk_texts, batch_size=64, show_progress_bar=False))
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        chunk_vecs_norm = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-8)
        sims = chunk_vecs_norm @ q_norm
        order = np.argsort(-sims)
        filtered_chunks = [all_chunks[i] for i in order if sims[i] >= thr_global]
        if not filtered_chunks:  # relax threshold if nothing passes
            filtered_chunks = [all_chunks[i] for i in order[:5]]
    else:
        filtered_chunks = []

    # 4. Cloud generation (same model as edge in §9 adaptation)
    budget = cloud_ctx_max_tokens // 3
    ents_text = _truncate_words(_format_entities(all_entities), budget)
    rels_text = _truncate_words(_format_relations(all_relations), budget)
    chunks_text = _truncate_words(_format_chunks(filtered_chunks), budget)

    rag_prompt = build_rag_prompt(
        query=query,
        entities_text=ents_text,
        relationships_text=rels_text,
        chunks_text=chunks_text,
        short_answer=short_answer,
    )
    answer = cloud_llm.infer(rag_prompt, temperature=0.0)
    llm_calls += 1

    return answer, llm_calls, n_contacted
