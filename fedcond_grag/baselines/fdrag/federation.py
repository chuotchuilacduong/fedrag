"""Eq.17-18: local export and global fusion for FD-RAG's federated setting.

Split into two functions per §4.2:

- ``export_local(...)`` runs on each client -- calls ``anonymize_memories``
  and packages the outbound bundle. Raw C(e) is dropped here (§4.1 line 11).
- ``fuse_and_broadcast(bundles)`` runs on the coordinator -- concatenates
  every client's Γ̃_k into Γ^g, dedupes near-identical (q, a) pairs, then
  returns Γ^g for redistribution.

The resulting foreign memories are appended to each client's local Γ, and
their ``support_edge_ids`` reference *foreign* hyperedges that live only
inside ``fused_edges`` (facts-only, no raw C(e)). The inference module uses
``local_edge_ids`` to distinguish the two cases at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.data_types import Hyperedge, QAMemory
from fedcond_grag.baselines.fdrag.facts import (
    anchors_from_facts,
    normalize_anchor,
)
from fedcond_grag.baselines.fdrag.privacy import (
    anonymize_memories,
    build_candidate_vocab,
)


@dataclass
class ClientBundle:
    """What each client uploads to the coordinator."""
    client_id: int
    memories: list[QAMemory]                # sanitized; NO C(e)
    facts_by_edge: dict[str, list[tuple[str, str]]]  # sanitized F(e)


@dataclass
class GlobalKnowledge:
    """What the coordinator sends back to every client."""
    memories: list[QAMemory]                          # merged Γ^g
    edges: dict[str, Hyperedge]                       # foreign edges (facts-only)
    foreign_edge_ids: set[str] = field(default_factory=set)


def export_local(
    client_id: int,
    memories: list[QAMemory],
    edges_by_id: dict[str, Hyperedge],
    cfg: FDRAGConfig,
    candidate_vocab: dict[str, list[str]],
) -> ClientBundle:
    """Alg 4 wrapper -- returns the sanitized outbound bundle only."""
    san = anonymize_memories(
        memories,
        edges_by_id,
        candidate_vocab=candidate_vocab,
        epsilon=cfg.epsilon,
        c=cfg.ldp_c,
        sensitive_types=cfg.sensitive_types,
        seed=cfg.seed + client_id,
    )
    facts_by_edge: dict[str, list[tuple[str, str]]] = {}
    for m in san:
        facts_by_edge.setdefault("_bundle", []).extend(m.sanitized_facts or [])
        # We also keep per-edge sanitized facts so the coordinator can rebuild
        # foreign Hyperedge stubs. Consistency with per-memory mapping means
        # the same edge may appear with slightly different substitutions
        # across memories -- take the first occurrence.
        for e_id in m.support_edge_ids:
            if e_id not in facts_by_edge:
                # The sanitized facts we emitted concatenate ALL support edges
                # per memory; re-attribute using the source edge's fact count.
                src = edges_by_id.get(e_id)
                if src is None:
                    continue
                # Take a slice of san_facts corresponding to this edge.
                start = 0
                for other_id in m.support_edge_ids:
                    other = edges_by_id.get(other_id)
                    if other is None:
                        continue
                    n = len(other.facts)
                    if other_id == e_id:
                        facts_by_edge[e_id] = list(m.sanitized_facts or [])[start:start + n]
                        break
                    start += n
    facts_by_edge.pop("_bundle", None)
    return ClientBundle(client_id=client_id, memories=san, facts_by_edge=facts_by_edge)


def fuse_and_broadcast(
    bundles: Iterable[ClientBundle],
    cfg: FDRAGConfig,
) -> GlobalKnowledge:
    """Eq.18: Γ^g = ∪_k Γ̃_k. Dedupe near-identical (q, a) pairs by
    normalized string.
    """
    seen: set[tuple[str, str]] = set()
    merged_memories: list[QAMemory] = []
    foreign_edges: dict[str, Hyperedge] = {}
    foreign_edge_ids: set[str] = set()

    for b in bundles:
        for e_id, facts in b.facts_by_edge.items():
            # Prefix by client id so foreign edge ids never collide with a
            # different client's local edge ids.
            f_id = f"f{b.client_id}:{e_id}"
            foreign_edges[f_id] = Hyperedge(
                id=f_id,
                granularity="paragraph",
                member_unit_ids=[],
                weights=np.zeros((0,), dtype=np.float32),
                prototype=np.zeros((0,), dtype=np.float32),
                context=[],                              # facts-only, no C(e)
                facts=list(facts),
                anchors=anchors_from_facts(facts),
            )
            foreign_edge_ids.add(f_id)

        for m in b.memories:
            key = (normalize_anchor(m.question), normalize_anchor(m.answer))
            if key in seen:
                continue
            seen.add(key)
            # Remap support_edge_ids to their prefixed foreign ids.
            m.support_edge_ids = [f"f{b.client_id}:{e_id}" for e_id in m.support_edge_ids]
            merged_memories.append(m)

    return GlobalKnowledge(
        memories=merged_memories,
        edges=foreign_edges,
        foreign_edge_ids=foreign_edge_ids,
    )


def build_shared_vocab(
    per_client_edges: Iterable[list[Hyperedge]],
    sensitive_types: tuple[str, ...],
    encoder=None,
) -> dict[str, list[str]]:
    """Pool sensitive-typed spans from every client's edges into one shared
    vocabulary. Called by the coordinator (or by a driver that has access to
    every client's edges) *before* ``export_local`` runs on each client.

    This is the "candidate vocabulary" [GAP-6] -- shared across devices,
    never drawn from a single device's local corpus."""
    all_edges: list[Hyperedge] = []
    for edges in per_client_edges:
        all_edges.extend(edges)
    return build_candidate_vocab(all_edges, sensitive_types, encoder=encoder)
