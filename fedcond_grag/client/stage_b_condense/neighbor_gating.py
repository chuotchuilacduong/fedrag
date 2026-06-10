"""Budgeted neighbor gating for DANCE-style hierarchical text condensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .chunk_selection import ChunkSelection, select_chunks, topk_softmax


@dataclass(frozen=True)
class HopSelection:
    """Selected neighbors and weights for one hop."""

    node_ids: list[int]
    weights: Tensor


@dataclass(frozen=True)
class NodeEvidenceTrace:
    """Local-only audit trace for one condensed core node."""

    core_node_id: int
    hops: dict[int, HopSelection]
    chunks: ChunkSelection


def build_undirected_neighbors(edge_index: Tensor, num_nodes: int) -> list[set[int]]:
    """Build adjacency sets from a PyG-style ``edge_index``."""

    neighbors = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return neighbors
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for u, v in zip(src, dst):
        u_i, v_i = int(u), int(v)
        if u_i == v_i:
            continue
        neighbors[u_i].add(v_i)
        neighbors[v_i].add(u_i)
    return neighbors


def hop_neighbors(neighbors: Sequence[set[int]], node_id: int, hop: int) -> list[int]:
    """Return unique neighbors at exact hop distance 0, 1, or 2."""

    node_id = int(node_id)
    if hop == 0:
        return [node_id]
    if hop == 1:
        return sorted(neighbors[node_id])
    if hop != 2:
        raise ValueError(f"Unsupported hop: {hop}")

    one_hop = set(neighbors[node_id])
    two_hop: set[int] = set()
    for nbr in one_hop:
        two_hop.update(neighbors[nbr])
    two_hop.discard(node_id)
    two_hop.difference_update(one_hop)
    return sorted(two_hop)


def degree_difficulty_scores(candidates: Sequence[int], neighbors: Sequence[set[int]], device: torch.device) -> Tensor:
    """Adapt DANCE Eq. 21 with the plan's no-label degree heuristic."""

    if not candidates:
        return torch.empty(0, device=device)
    scores = [1.0 / (1.0 + len(neighbors[int(node_id)])) for node_id in candidates]
    return torch.tensor(scores, dtype=torch.float32, device=device)


def score_and_select(
    g_v: Tensor,
    neighbor_text_embs: Tensor,
    *,
    budget: int,
) -> Tensor:
    """Relevance-based selection: cosine(core node, neighbor text) → top-budget.

    Scores each candidate by cosine similarity to the core node's graph
    embedding (both live in the MiniLM space), then keeps the top-budget
    candidates with softmax weights over their scores.
    """

    if neighbor_text_embs.numel() == 0:
        return neighbor_text_embs.new_zeros((0,))
    query = torch.nn.functional.normalize(g_v.reshape(-1).float(), dim=0)
    keys = torch.nn.functional.normalize(neighbor_text_embs.float(), dim=-1)
    scores = keys @ query
    return topk_softmax(scores, int(budget)).to(neighbor_text_embs.dtype)


def hierarchical_text_condensation(
    *,
    core_node_ids: Sequence[int],
    edge_index: Tensor,
    graph_embeddings: Tensor,
    node_text_embeddings: Tensor,
    chunk_embeddings: Sequence[Tensor],
    hop_weights: Tensor | None = None,
    budgets: tuple[int, int, int] = (1, 3, 2),
    chunk_budget: int = 8,
    two_hop_prefetch_factor: int = 4,
) -> tuple[Tensor, dict[int, Tensor], dict[int, NodeEvidenceTrace]]:
    """Condense text evidence for every core node.

    Returns ``t_tilde`` aligned to ``core_node_ids``, hierarchical contexts
    ``c_v`` keyed by original node id, and local-only traces.
    """

    if graph_embeddings.dim() != 2:
        raise ValueError("graph_embeddings must be [num_nodes, d]")
    num_nodes = graph_embeddings.size(0)
    neighbors = build_undirected_neighbors(edge_index, num_nodes)
    device = graph_embeddings.device

    if hop_weights is None:
        hop_weights = graph_embeddings.new_tensor([0.4, 0.4, 0.2])
    hop_weights = hop_weights.to(device=device, dtype=graph_embeddings.dtype)
    hop_weights = hop_weights / hop_weights.sum().clamp_min(1e-12)

    t_tilde_rows: list[Tensor] = []
    contexts: dict[int, Tensor] = {}
    traces: dict[int, NodeEvidenceTrace] = {}

    for core_id in tqdm([int(node_id) for node_id in core_node_ids], desc="Stage B condense", unit="node"):
        g_v = graph_embeddings[core_id]
        selected_by_hop: dict[int, HopSelection] = {}
        selected_nodes_for_chunks: list[int] = []
        context = graph_embeddings.new_zeros((node_text_embeddings.size(1),))

        for hop, budget in enumerate(budgets):
            candidates = hop_neighbors(neighbors, core_id, hop)
            if hop == 2 and candidates:
                prefetch = max(int(budget) * int(two_hop_prefetch_factor), int(budget))
                diff = degree_difficulty_scores(candidates, neighbors, device=device)
                keep = min(prefetch, len(candidates))
                idx = torch.topk(diff, k=keep).indices.detach().cpu().tolist()
                candidates = [candidates[i] for i in idx]

            if not candidates:
                selected_by_hop[hop] = HopSelection(node_ids=[], weights=graph_embeddings.new_zeros((0,)))
                continue

            text = node_text_embeddings[candidates].to(device=device, dtype=graph_embeddings.dtype)
            weights = score_and_select(g_v, text, budget=budget)
            selected_idx = (weights > 0).nonzero(as_tuple=False).flatten().tolist()
            selected_nodes = [int(candidates[i]) for i in selected_idx]
            selected_weights = weights[selected_idx]
            selected_by_hop[hop] = HopSelection(
                node_ids=selected_nodes,
                weights=selected_weights.detach().cpu(),
            )

            if selected_nodes:
                selected_nodes_for_chunks.extend(selected_nodes)
                selected_text = node_text_embeddings[selected_nodes].to(device=device, dtype=graph_embeddings.dtype)
                context = context + hop_weights[hop] * (selected_weights.to(device).unsqueeze(0) @ selected_text).squeeze(0)

        t_tilde_v, chunk_trace = select_chunks(
            g_v,
            selected_nodes_for_chunks,
            chunk_embeddings,
            budget=chunk_budget,
        )
        t_tilde_rows.append(t_tilde_v)
        contexts[core_id] = context.detach()
        traces[core_id] = NodeEvidenceTrace(core_node_id=core_id, hops=selected_by_hop, chunks=chunk_trace)

    if t_tilde_rows:
        t_tilde = torch.stack(t_tilde_rows, dim=0)
    else:
        t_tilde = graph_embeddings.new_zeros((0, node_text_embeddings.size(1)))
    return t_tilde, contexts, traces
