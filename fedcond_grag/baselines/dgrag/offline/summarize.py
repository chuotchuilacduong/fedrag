"""Step ②③: Subgraph summarization + summary upload (spec §3.2-3.3)."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from fedcond_grag.baselines.dgrag.data_types import Entity, Relation, Subgraph, SummaryRecord
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.prompts import build_summarize_prompt

_log = logging.getLogger("dgrag.summarize")


def summarize_subgraphs(
    communities: list[list[str]],
    lin_texts: list[str],
    edge_id: str,
    entities: dict[str, "Entity"],
    relations: dict[tuple[str, str], "Relation"],
    slm: DGRAGModel,
    encoder,
    *,
    summary_max_tokens: int = 200,
) -> list[Subgraph]:
    """For each community: SLM-generate a summary, embed it, return Subgraphs."""
    subgraphs: list[Subgraph] = []
    for idx, (community, kg_text) in enumerate(zip(communities, lin_texts)):
        prompt = build_summarize_prompt(kg_text, max_tokens=summary_max_tokens)
        try:
            summary = slm.infer(prompt, temperature=0.0, max_tokens=summary_max_tokens + 100)
        except Exception as exc:
            _log.warning("Summarize failed for community %d: %s", idx, exc)
            summary = kg_text[:500]   # raw fallback

        summary = summary.strip()
        if not summary:
            summary = kg_text[:500]

        nodes_set = {n.strip().upper() for n in community}
        rel_keys = [(src, tgt) for (src, tgt) in relations if src in nodes_set and tgt in nodes_set]

        sg = Subgraph(
            subgraph_id=f"{edge_id}-{idx}",
            edge_id=edge_id,
            entity_names=list(community),
            relation_keys=rel_keys,
            summary=summary,
        )
        subgraphs.append(sg)

    # batch-embed all summaries
    if subgraphs:
        vecs = np.asarray(
            encoder.encode([s.summary for s in subgraphs], batch_size=64, show_progress_bar=False),
            dtype=np.float32,
        )
        for sg, vec in zip(subgraphs, vecs):
            sg.summary_vec = vec

    _log.info("[edge %s] produced %d subgraph summaries", edge_id, len(subgraphs))
    return subgraphs


def subgraph_to_summary_record(sg: Subgraph) -> SummaryRecord:
    return SummaryRecord(
        subgraph_id=sg.subgraph_id,
        edge_id=sg.edge_id,
        summary=sg.summary,
        embedding=sg.summary_vec,
    )
