"""Step ①: Edge Graph Knowledge Extraction (spec §3.1).

SLM extracts entities and relations from each chunk using PROMPT_EXTRACT.
Entities are merged by canonical (upper-cased) name; relation descriptions
accumulate. An optional gleaning pass re-prompts the SLM for missed items.
"""

from __future__ import annotations

import logging
from typing import Optional

from fedcond_grag.baselines.dgrag.data_types import Chunk, Entity, Relation
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.prompts import (
    build_extract_continue_prompt,
    build_extract_prompt,
)

_log = logging.getLogger("dgrag.extract")

_VALID_TYPES = {"person", "place", "event", "object", "organization", "category", "concept"}


def _parse_entity(raw: dict) -> Optional[Entity]:
    name = (raw.get("entity_name") or raw.get("name") or "").strip()
    if not name:
        return None
    return Entity(
        name=name,
        type=(raw.get("entity_type") or raw.get("type") or "concept").lower().strip(),
        description=(raw.get("entity_description") or raw.get("description") or "").strip(),
    )


def _parse_relation(raw: dict) -> Optional[Relation]:
    src = (raw.get("source_entity") or raw.get("source") or "").strip()
    tgt = (raw.get("target_entity") or raw.get("target") or "").strip()
    if not src or not tgt:
        return None
    return Relation(
        source=src,
        target=tgt,
        keyword=(raw.get("relationship_keyword") or raw.get("keyword") or "related to").strip(),
        description=(raw.get("relationship_description") or raw.get("description") or "").strip(),
    )


def _merge_entity(store: dict[str, Entity], entity: Entity, chunk_id: str) -> None:
    key = entity.name.strip().upper()
    if key in store:
        existing = store[key]
        if entity.description and entity.description not in existing.description:
            existing.description = f"{existing.description}; {entity.description}"
        if chunk_id not in existing.source_chunks:
            existing.source_chunks.append(chunk_id)
    else:
        entity.source_chunks = [chunk_id]
        store[key] = entity


def _merge_relation(store: dict[tuple[str, str], Relation], relation: Relation, chunk_id: str) -> None:
    key = (relation.source.strip().upper(), relation.target.strip().upper())
    if key in store:
        existing = store[key]
        existing.weight += 1.0
        if relation.description and relation.description not in existing.description:
            existing.description = f"{existing.description}; {relation.description}"
        if chunk_id not in existing.source_chunks:
            existing.source_chunks.append(chunk_id)
    else:
        relation.source_chunks = [chunk_id]
        store[key] = relation


def extract_chunk(
    chunk: Chunk,
    slm: DGRAGModel,
    entity_store: dict[str, Entity],
    relation_store: dict[tuple[str, str], Relation],
    glean_rounds: int = 1,
) -> None:
    """Extract entities + relations from one chunk, merging into stores."""
    prompt = build_extract_prompt(chunk.text)
    raw = slm.infer_json(prompt, temperature=0.0, default={})
    prior_text = str(raw)

    for e_raw in raw.get("entities", []):
        e = _parse_entity(e_raw)
        if e:
            _merge_entity(entity_store, e, chunk.chunk_id)
    for r_raw in raw.get("relations", []):
        r = _parse_relation(r_raw)
        if r:
            _merge_relation(relation_store, r, chunk.chunk_id)

    # Gleaning: ask SLM for missed items
    for _ in range(glean_rounds):
        prompt2 = build_extract_continue_prompt(chunk.text, prior_text)
        raw2 = slm.infer_json(prompt2, temperature=0.0, default={})
        if not raw2:
            break
        for e_raw in raw2.get("entities", []):
            e = _parse_entity(e_raw)
            if e:
                _merge_entity(entity_store, e, chunk.chunk_id)
        for r_raw in raw2.get("relations", []):
            r = _parse_relation(r_raw)
            if r:
                _merge_relation(relation_store, r, chunk.chunk_id)
        prior_text = str(raw2)


def extract_all(
    chunks: list[Chunk],
    slm: DGRAGModel,
    *,
    glean_rounds: int = 1,
) -> tuple[dict[str, Entity], dict[tuple[str, str], Relation]]:
    """Extract from all chunks, return merged entity + relation stores."""
    entity_store: dict[str, Entity] = {}
    relation_store: dict[tuple[str, str], Relation] = {}
    for i, chunk in enumerate(chunks):
        _log.debug("Extracting chunk %d/%d", i + 1, len(chunks))
        try:
            extract_chunk(chunk, slm, entity_store, relation_store, glean_rounds=glean_rounds)
        except Exception as exc:
            _log.warning("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
    _log.info("Extracted %d entities, %d relations from %d chunks",
              len(entity_store), len(relation_store), len(chunks))
    return entity_store, relation_store
