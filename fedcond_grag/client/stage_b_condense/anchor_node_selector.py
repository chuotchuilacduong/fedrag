"""Query-agnostic S-E-P anchor node selection for client condensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


ENTITY = 0
SENTENCE = 1
PASSAGE = 2


@dataclass(frozen=True)
class AnchorSelectorConfig:
    entity_ratio: float = 0.05
    sentence_budget: int = 3
    passage_budget: int = 3
    lambda_idf: float = 1.0
    lambda_pr: float = 0.5
    lambda_mmr: float = 0.3
    min_entities: int = 1
    pagerank_iters: int = 20
    pagerank_damping: float = 0.85
    # Cap entity pool before PageRank/MMR to handle very large graphs.
    # 0 = no cap (original behaviour). Entities are sampled by degree before scoring.
    max_entity_pool: int = 50_000


@dataclass(frozen=True)
class AnchorSelection:
    core_node_ids: Tensor
    core_edge_index: Tensor
    selected_motifs: list[tuple[int, list[int], list[int]]]
    original_to_core: dict[int, int]


def _typed_nodes(node_type: Tensor, value: int) -> list[int]:
    return (node_type.detach().cpu() == value).nonzero(as_tuple=False).flatten().tolist()


def _entity_pagerank_vectorized(
    entity_ids: Sequence[int],
    neighbors: Sequence[set[int]],
    node_type: Tensor,
    cfg: AnchorSelectorConfig,
    *,
    _node_type_list: list[int] | None = None,
) -> dict[int, float]:
    """PageRank on the entity projection using sparse tensor ops (no Python loops over N)."""

    entity_arr = [int(e) for e in entity_ids]
    n = len(entity_arr)
    if n == 0:
        return {}

    entity_set = set(entity_arr)
    entity_to_local = {e: i for i, e in enumerate(entity_arr)}

    # Convert to plain list for fast O(1) indexing in Python loops
    nt_list: list[int] = _node_type_list if _node_type_list is not None else node_type.detach().cpu().tolist()

    # Build projected entity-entity edges via shared S/P neighbors
    src_local: list[int] = []
    dst_local: list[int] = []
    for local_i, e in enumerate(entity_arr):
        for nbr in neighbors[e]:
            if nt_list[nbr] not in (SENTENCE, PASSAGE):
                continue
            for x in neighbors[nbr]:
                xi = int(x)
                if xi == e or xi not in entity_set:
                    continue
                src_local.append(local_i)
                dst_local.append(entity_to_local[xi])

    # Sparse out-degree for each entity
    if src_local:
        src_t = torch.tensor(src_local, dtype=torch.long)
        dst_t = torch.tensor(dst_local, dtype=torch.long)
        out_deg = torch.zeros(n, dtype=torch.float32)
        out_deg.scatter_add_(0, src_t, torch.ones(len(src_t), dtype=torch.float32))
    else:
        src_t = dst_t = torch.empty(0, dtype=torch.long)
        out_deg = torch.zeros(n, dtype=torch.float32)

    rank = torch.full((n,), 1.0 / n, dtype=torch.float32)
    base = (1.0 - cfg.pagerank_damping) / n
    dangling_mask = out_deg == 0  # [N] bool

    for _ in range(cfg.pagerank_iters):
        # Dangling mass distributed uniformly — O(N) not O(D*N)
        dangling_mass = float((cfg.pagerank_damping * rank * dangling_mask.float()).sum().item())
        next_rank = torch.full((n,), base + dangling_mass / n, dtype=torch.float32)

        # Propagate from non-dangling nodes via sparse scatter
        if src_t.numel() > 0:
            nz = out_deg[src_t]
            contrib = cfg.pagerank_damping * rank[src_t] / nz
            next_rank.scatter_add_(0, dst_t, contrib)

        rank = next_rank

    return {entity_arr[i]: float(rank[i].item()) for i in range(n)}


def _score_entities_vectorized(
    data,
    entity_ids: Sequence[int],
    neighbors: Sequence[set[int]],
    cfg: AnchorSelectorConfig,
) -> dict[int, float]:
    """Vectorized entity scoring using degree counts and PageRank."""

    entity_arr = [int(e) for e in entity_ids]
    n = len(entity_arr)
    node_type_cpu = data.node_type.detach().cpu().long()
    num_passages = max(1, int((node_type_cpu == PASSAGE).sum().item()))

    # Plain list for fast O(1) indexing in Python inner loops
    nt_list: list[int] = node_type_cpu.tolist()

    # Degree computation: single pass, plain lists to avoid per-element tensor indexing
    deg_s_list = [0.0] * n
    deg_p_list = [0.0] * n
    for i, e in enumerate(entity_arr):
        for nbr in neighbors[e]:
            t = nt_list[nbr]
            if t == SENTENCE:
                deg_s_list[i] += 1.0
            elif t == PASSAGE:
                deg_p_list[i] += 1.0
    deg_s = torch.tensor(deg_s_list, dtype=torch.float32)
    deg_p = torch.tensor(deg_p_list, dtype=torch.float32)

    pagerank = _entity_pagerank_vectorized(entity_arr, neighbors, node_type_cpu, cfg, _node_type_list=nt_list)
    pr_tensor = torch.tensor([pagerank.get(e, 0.0) for e in entity_arr], dtype=torch.float32)

    idf = torch.log(torch.tensor(num_passages, dtype=torch.float32) / (1.0 + deg_p))
    score_tensor = (
        torch.log1p(deg_s)
        + torch.log1p(deg_p)
        + cfg.lambda_idf * idf
        + cfg.lambda_pr * pr_tensor
    )
    return {entity_arr[i]: float(score_tensor[i].item()) for i in range(n)}


def _mmr_select_vectorized(
    entity_ids: Sequence[int],
    scores: dict[int, float],
    x: Tensor | None,
    k: int,
    cfg: AnchorSelectorConfig,
) -> list[int]:
    """Vectorized MMR selection — O(K × N) tensor ops instead of O(K × N) Python loops."""

    entity_arr = [int(e) for e in entity_ids]
    n = len(entity_arr)
    k = min(k, n)
    if k == 0:
        return []

    score_tensor = torch.tensor([scores[e] for e in entity_arr], dtype=torch.float32)

    x_norm: Tensor | None = None
    if x is not None and x.numel() > 0:
        x_sub = x[entity_arr].float()  # [N, d]
        x_norm = F.normalize(x_sub, p=2, dim=-1)  # [N, d]

    selected_mask = torch.zeros(n, dtype=torch.bool)
    selected_local: list[int] = []
    # Track max similarity to any selected entity per candidate
    max_sim = torch.zeros(n, dtype=torch.float32)

    for step in range(k):
        if x_norm is not None and step > 0:
            last = selected_local[-1]
            sim_to_last = x_norm @ x_norm[last]  # [N]
            max_sim = torch.maximum(max_sim, sim_to_last)

        final = score_tensor - cfg.lambda_mmr * max_sim
        final[selected_mask] = float("-inf")
        best = int(final.argmax().item())
        selected_mask[best] = True
        selected_local.append(best)

    return [entity_arr[i] for i in selected_local]


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
    """Filter edge_index to core-only S-E/P-E edges using tensor ops."""

    core_list = [int(i) for i in core_ids]
    mapping = {node_id: i for i, node_id in enumerate(core_list)}
    n_total = int(node_type.numel())

    if edge_index.numel() == 0 or not core_list:
        return torch.empty((2, 0), dtype=torch.long), mapping

    # Normalise everything to CPU — this is index-only work, no matmul
    ei = edge_index.cpu()
    nt = node_type.cpu().long()
    in_core = torch.zeros(n_total, dtype=torch.bool)
    core_tensor = torch.tensor(core_list, dtype=torch.long)
    in_core[core_tensor] = True

    src = ei[0]
    dst = ei[1]

    both_in_core = in_core[src] & in_core[dst]
    src_f = src[both_in_core]
    dst_f = dst[both_in_core]

    # Keep only S-E and P-E pairs
    src_type = nt[src_f]
    dst_type = nt[dst_f]

    valid = (
        ((src_type == SENTENCE) & (dst_type == ENTITY)) |
        ((src_type == ENTITY) & (dst_type == SENTENCE)) |
        ((src_type == PASSAGE) & (dst_type == ENTITY)) |
        ((src_type == ENTITY) & (dst_type == PASSAGE))
    )
    src_f = src_f[valid].detach().cpu().tolist()
    dst_f = dst_f[valid].detach().cpu().tolist()

    edges: set[tuple[int, int]] = set()
    for u, v in zip(src_f, dst_f):
        a, b = mapping[int(u)], mapping[int(v)]
        if a != b:
            edges.add((a, b))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long), mapping
    edge_tensor = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
    return edge_tensor, mapping


def select_anchor_nodes(data, config: AnchorSelectorConfig | None = None, x: Tensor | None = None) -> AnchorSelection:
    """Select entity-anchored S-E-P nodes and return a remapped core graph."""

    from fedcond_grag.client.stage_b_condense.neighbor_gating import build_undirected_neighbors

    cfg = config or AnchorSelectorConfig()
    if not hasattr(data, "node_type"):
        raise ValueError("data.node_type is required for S-E-P anchor node selection")
    node_type = data.node_type.detach().cpu().long()
    x = x if x is not None else getattr(data, "x", None)
    if x is not None:
        x = x.detach().cpu().float()

    entity_ids = _typed_nodes(node_type, ENTITY)
    if not entity_ids:
        raise ValueError("Tri-Graph contains no entity nodes; cannot select S-E-P anchor nodes")
    neighbors = build_undirected_neighbors(data.edge_index, int(node_type.numel()))

    # Subsample large entity pools by degree (high-degree entities first) so
    # PageRank/MMR stays tractable on full-scale datasets.
    if cfg.max_entity_pool > 0 and len(entity_ids) > cfg.max_entity_pool:
        import random
        # Sort by degree descending, keep top half by degree + random sample the rest
        half = cfg.max_entity_pool // 2
        by_deg = sorted(entity_ids, key=lambda e: len(neighbors[e]), reverse=True)
        top_deg = by_deg[:half]
        rest = by_deg[half:]
        random.shuffle(rest)
        entity_ids = top_deg + rest[: cfg.max_entity_pool - half]
        print(f"    [B] Entity pool capped: {len(entity_ids)} / {(node_type == 0).sum().item()} entities used for anchor scoring")

    scores = _score_entities_vectorized(data, entity_ids, neighbors, cfg)
    k_entities = max(int(cfg.min_entities), int(ceil(cfg.entity_ratio * len(entity_ids))))
    selected_entities = _mmr_select_vectorized(entity_ids, scores, x, min(k_entities, len(entity_ids)), cfg)

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

    deduped = list(dict.fromkeys(int(node_id) for node_id in selected_core))
    core_edge_index, mapping = _induced_sep_edges(data.edge_index, deduped, node_type)
    core_node_ids = torch.tensor(deduped, dtype=torch.long, device=data.edge_index.device)
    return AnchorSelection(
        core_node_ids=core_node_ids,
        core_edge_index=core_edge_index,
        selected_motifs=motifs,
        original_to_core=mapping,
    )
