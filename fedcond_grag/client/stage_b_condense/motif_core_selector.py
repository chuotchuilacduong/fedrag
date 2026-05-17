"""Query-agnostic S-E-P motif selection for client condensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .neighbor_gating import build_undirected_neighbors


ENTITY = 0
SENTENCE = 1
PASSAGE = 2


@dataclass(frozen=True)
class MotifSelectorConfig:
    entity_ratio: float = 0.05
    sentence_budget: int = 3
    passage_budget: int = 3
    lambda_idf: float = 1.0
    lambda_pr: float = 0.5
    lambda_mmr: float = 0.3
    min_entities: int = 1
    pagerank_iters: int = 20
    pagerank_damping: float = 0.85


@dataclass(frozen=True)
class MotifSelection:
    core_node_ids: Tensor
    core_edge_index: Tensor
    selected_motifs: list[tuple[int, list[int], list[int]]]
    original_to_core: dict[int, int]


def _typed_nodes(node_type: Tensor, value: int) -> list[int]:
    return (node_type.detach().cpu() == value).nonzero(as_tuple=False).flatten().tolist()


def _entity_pagerank(entity_ids: Sequence[int], neighbors: Sequence[set[int]], node_type: Tensor, cfg: MotifSelectorConfig) -> dict[int, float]:
    """PageRank on the entity projection induced by shared S/P neighbors."""

    entity_set = set(int(e) for e in entity_ids)
    if not entity_set:
        return {}
    projected = {e: set() for e in entity_set}
    for e in entity_set:
        for nbr in neighbors[e]:
            if int(node_type[nbr]) not in (SENTENCE, PASSAGE):
                continue
            projected[e].update(int(x) for x in neighbors[nbr] if x in entity_set and x != e)

    n = len(entity_set)
    rank = {e: 1.0 / n for e in entity_set}
    base = (1.0 - cfg.pagerank_damping) / n
    for _ in range(cfg.pagerank_iters):
        next_rank = {e: base for e in entity_set}
        for e, outs in projected.items():
            if not outs:
                share = cfg.pagerank_damping * rank[e] / n
                for target in entity_set:
                    next_rank[target] += share
                continue
            share = cfg.pagerank_damping * rank[e] / len(outs)
            for target in outs:
                next_rank[target] += share
        rank = next_rank
    return rank


def _score_entities(data, entity_ids: Sequence[int], neighbors: Sequence[set[int]], cfg: MotifSelectorConfig) -> dict[int, float]:
    node_type = data.node_type.detach().cpu()
    num_passages = max(1, int((node_type == PASSAGE).sum().item()))
    pagerank = _entity_pagerank(entity_ids, neighbors, node_type, cfg)
    scores: dict[int, float] = {}
    for e in entity_ids:
        deg_s = sum(1 for n in neighbors[e] if int(node_type[n]) == SENTENCE)
        deg_p = sum(1 for n in neighbors[e] if int(node_type[n]) == PASSAGE)
        score = log(1.0 + deg_s) + log(1.0 + deg_p)
        score += cfg.lambda_idf * log(num_passages / (1.0 + deg_p))
        score += cfg.lambda_pr * pagerank.get(int(e), 0.0)
        scores[int(e)] = float(score)
    return scores


def _mmr_select(entity_ids: Sequence[int], scores: dict[int, float], x: Tensor | None, k: int, cfg: MotifSelectorConfig) -> list[int]:
    remaining = [int(e) for e in entity_ids]
    selected: list[int] = []
    x_norm = F.normalize(x, p=2, dim=-1) if x is not None and x.numel() > 0 else None

    while remaining and len(selected) < k:
        best_e = None
        best_score = float("-inf")
        for e in remaining:
            diversity_penalty = 0.0
            if selected and x_norm is not None:
                sim = x_norm[e].unsqueeze(0) @ x_norm[selected].T
                diversity_penalty = float(sim.max().item())
            final = scores[e] - cfg.lambda_mmr * diversity_penalty
            if final > best_score:
                best_score = final
                best_e = e
        selected.append(int(best_e))
        remaining.remove(int(best_e))
    return selected


def _neighbor_score(node_id: int, anchor_id: int, selected_entities: Sequence[int], neighbors: Sequence[set[int]], x: Tensor | None) -> float:
    mention_count = sum(1 for n in neighbors[node_id] if n in selected_entities)
    centrality = len(neighbors[node_id])
    semantic = 0.0
    if x is not None and x.numel() > 0:
        entity_nbrs = [n for n in neighbors[node_id] if n in selected_entities]
        if entity_nbrs:
            target = F.normalize(x[entity_nbrs].mean(dim=0), p=2, dim=0)
        else:
            target = F.normalize(x[anchor_id], p=2, dim=0)
        semantic = float((F.normalize(x[node_id], p=2, dim=0) * target).sum().item())
    return float(mention_count + 0.1 * centrality + semantic)


def _select_typed_neighbors(
    anchor_id: int,
    target_type: int,
    budget: int,
    selected_entities: Sequence[int],
    neighbors: Sequence[set[int]],
    node_type: Tensor,
    x: Tensor | None,
) -> list[int]:
    candidates = [n for n in neighbors[anchor_id] if int(node_type[n]) == target_type]
    scored = [
        (_neighbor_score(int(n), anchor_id, selected_entities, neighbors, x), int(n))
        for n in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [node_id for _, node_id in scored[: max(int(budget), 0)]]


def _induced_sep_edges(edge_index: Tensor, core_ids: Sequence[int], node_type: Tensor) -> tuple[Tensor, dict[int, int]]:
    core_set = set(int(i) for i in core_ids)
    mapping = {int(node_id): i for i, node_id in enumerate(core_ids)}
    edges: set[tuple[int, int]] = set()
    src = edge_index[0].detach().cpu().tolist() if edge_index.numel() else []
    dst = edge_index[1].detach().cpu().tolist() if edge_index.numel() else []
    for u, v in zip(src, dst):
        u_i, v_i = int(u), int(v)
        if u_i not in core_set or v_i not in core_set:
            continue
        type_pair = {int(node_type[u_i]), int(node_type[v_i])}
        if type_pair not in ({ENTITY, SENTENCE}, {ENTITY, PASSAGE}):
            continue
        a, b = mapping[u_i], mapping[v_i]
        if a != b:
            edges.add((a, b))
            edges.add((b, a))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), mapping
    edge_tensor = torch.tensor(sorted(edges), dtype=torch.long, device=edge_index.device).T.contiguous()
    return edge_tensor, mapping


def select_motif_core(data, config: MotifSelectorConfig | None = None, x: Tensor | None = None) -> MotifSelection:
    """Select entity-anchored S-E-P motifs and return a remapped core graph."""

    cfg = config or MotifSelectorConfig()
    if not hasattr(data, "node_type"):
        raise ValueError("data.node_type is required for S-E-P motif selection")
    node_type = data.node_type.detach().cpu().long()
    x = x if x is not None else getattr(data, "x", None)
    if x is not None:
        x = x.detach().cpu().float()

    entity_ids = _typed_nodes(node_type, ENTITY)
    if not entity_ids:
        raise ValueError("Tri-Graph contains no entity nodes; cannot select S-E-P motifs")
    neighbors = build_undirected_neighbors(data.edge_index, int(node_type.numel()))
    scores = _score_entities(data, entity_ids, neighbors, cfg)
    k_entities = max(int(cfg.min_entities), int(ceil(cfg.entity_ratio * len(entity_ids))))
    selected_entities = _mmr_select(entity_ids, scores, x, min(k_entities, len(entity_ids)), cfg)

    selected_core: list[int] = []
    motifs: list[tuple[int, list[int], list[int]]] = []
    selected_entity_set = set(selected_entities)
    for entity_id in selected_entities:
        s_nodes = _select_typed_neighbors(
            entity_id, SENTENCE, cfg.sentence_budget, selected_entity_set, neighbors, node_type, x
        )
        p_nodes = _select_typed_neighbors(
            entity_id, PASSAGE, cfg.passage_budget, selected_entity_set, neighbors, node_type, x
        )
        motifs.append((entity_id, s_nodes, p_nodes))
        selected_core.extend([entity_id, *s_nodes, *p_nodes])

    # Preserve first-seen order so anchor-local motifs remain readable.
    deduped = list(dict.fromkeys(int(node_id) for node_id in selected_core))
    core_edge_index, mapping = _induced_sep_edges(data.edge_index, deduped, node_type)
    core_node_ids = torch.tensor(deduped, dtype=torch.long, device=data.edge_index.device)
    return MotifSelection(
        core_node_ids=core_node_ids,
        core_edge_index=core_edge_index,
        selected_motifs=motifs,
        original_to_core=mapping,
    )
