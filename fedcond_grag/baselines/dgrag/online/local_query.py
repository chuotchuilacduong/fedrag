"""Step ④: Local Query — dual-level retrieval + batch generation (spec §4.1).

Also exposes retrieve_only() for cross-edge fan-out: returns the formatted
evidence context without invoking the SLM for generation. The cross-edge
retrieval calls this on peer edges (spec §4.3: "each runs its LOCAL
RETRIEVAL only").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fedcond_grag.baselines.dgrag.data_types import Chunk, Entity, Evidence, Relation
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.prompts import (
    build_keywords_prompt,
    build_rag_prompt,
)
from fedcond_grag.baselines.dgrag.store import EdgeKnowledgeBase

_log = logging.getLogger("dgrag.local_query")


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
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [truncated]"


def extract_keywords(query: str, slm: DGRAGModel) -> tuple[list[str], list[str]]:
    """Return (high_level_keywords, low_level_keywords) from PROMPT_KEYWORDS."""
    prompt = build_keywords_prompt(query)
    raw = slm.infer_json(prompt, temperature=0.0, default={})
    high = raw.get("high_level_keywords") or []
    low = raw.get("low_level_keywords") or []
    if not isinstance(high, list):
        high = [str(high)]
    if not isinstance(low, list):
        low = [str(low)]
    return high, low


def retrieve_only(
    query: str,
    kb: EdgeKnowledgeBase,
    slm: DGRAGModel,
    encoder,
    *,
    top_ent: int = 60,
    top_rel: int = 60,
    top_ent_expanded: int = 100,
    top_rel_expanded: int = 100,
    top_chunk: int = 5,
    ctx_max_tokens: int = 8000,
    use_kg: bool = True,
) -> Evidence:
    """Steps 4.1.1-4.1.4: keyword extract -> VDB match -> 1-hop expand -> chunks.

    Returns raw Entity/Relation/Chunk lists. No generation.
    """
    # 4.1.1 keyword extraction
    high_kw, low_kw = extract_keywords(query, slm)
    high_text = " ".join(high_kw) or query
    low_text = " ".join(low_kw) or query

    # 4.1.2 dual-level VDB matching
    high_vec = np.asarray(encoder.encode([high_text], batch_size=1, show_progress_bar=False))[0]
    low_vec = np.asarray(encoder.encode([low_text], batch_size=1, show_progress_bar=False))[0]

    ent_results = kb.vdb_entities.search(low_vec, top_k=top_ent)
    rel_results = kb.vdb_relations.search(high_vec, top_k=top_rel)

    ents: list[Entity] = [r for _, r in ent_results]
    rels: list[Relation] = [r for _, r in rel_results]

    # 4.1.3 1-hop graph expansion (disabled for w/o_KG ablation)
    if use_kg and kb.graph.number_of_nodes() > 0:
        ent_names = {e.name.strip().upper() for e in ents}
        expanded_names: set[str] = set(ent_names)
        for name in list(ent_names):
            if name in kb.graph:
                for nbr in kb.graph.neighbors(name):
                    expanded_names.add(nbr)
                    if len(expanded_names) >= top_ent_expanded:
                        break
            if len(expanded_names) >= top_ent_expanded:
                break
        for name in expanded_names - ent_names:
            e = kb.entities.get(name)
            if e:
                ents.append(e)
        ents = ents[:top_ent_expanded]

        # incident relations for expanded entities
        expanded_rel_ids: set[str] = {f"{r.source.upper()}___{r.target.upper()}" for r in rels}
        for name in expanded_names:
            if name in kb.graph:
                for nbr in kb.graph.neighbors(name):
                    key = (name, nbr)
                    r = kb.relations.get(key) or kb.relations.get((nbr, name))
                    if r:
                        rid = f"{r.source.upper()}___{r.target.upper()}"
                        if rid not in expanded_rel_ids:
                            expanded_rel_ids.add(rid)
                            rels.append(r)
                            if len(rels) >= top_rel_expanded:
                                break
                if len(rels) >= top_rel_expanded:
                    break

    # 4.1.4 source chunks
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for e in ents:
        for cid in e.source_chunks:
            if cid not in seen:
                seen.add(cid)
                chunk_ids.append(cid)
    for r in rels:
        for cid in r.source_chunks:
            if cid not in seen:
                seen.add(cid)
                chunk_ids.append(cid)
    chunks: list[Chunk] = [kb.chunks[cid] for cid in chunk_ids[:top_chunk] if cid in kb.chunks]

    return Evidence(entities=ents, relations=rels, chunks=chunks, edge_id=kb.edge_id)


def local_query(
    query: str,
    kb: EdgeKnowledgeBase,
    slm: DGRAGModel,
    encoder,
    *,
    batch_b: int = 3,
    batch_temperature: float = 0.8,
    batch_top_p: float = 0.95,
    top_ent: int = 60,
    top_rel: int = 60,
    top_ent_expanded: int = 100,
    top_rel_expanded: int = 100,
    top_chunk: int = 5,
    ctx_max_tokens: int = 8000,
    short_answer: bool = True,
    use_kg: bool = True,
) -> tuple[list[str], Evidence]:
    """Steps 4.1.1-4.1.5: retrieve context + generate B candidates.

    Returns (answers, evidence). The gate mechanism consumes `answers`.
    """
    evidence = retrieve_only(
        query, kb, slm, encoder,
        top_ent=top_ent, top_rel=top_rel,
        top_ent_expanded=top_ent_expanded, top_rel_expanded=top_rel_expanded,
        top_chunk=top_chunk, ctx_max_tokens=ctx_max_tokens,
        use_kg=use_kg,
    )

    ents_text = _format_entities(evidence.entities)
    rels_text = _format_relations(evidence.relations)
    chunks_text = _format_chunks(evidence.chunks)

    # Rough token budget: truncate at ctx_max_tokens words
    budget = ctx_max_tokens // 3  # split equally across the three parts
    ents_text = _truncate_words(ents_text, budget)
    rels_text = _truncate_words(rels_text, budget)
    chunks_text = _truncate_words(chunks_text, budget)

    rag_prompt = build_rag_prompt(
        query=query,
        entities_text=ents_text,
        relationships_text=rels_text,
        chunks_text=chunks_text,
        short_answer=short_answer,
    )

    # 4.1.5 batch generation (B candidates)
    answers = slm.generate_batch(
        rag_prompt, n=batch_b,
        temperature=batch_temperature, top_p=batch_top_p,
    )
    # Normalise: strip empty strings
    answers = [a.strip() for a in answers]
    return answers, evidence
