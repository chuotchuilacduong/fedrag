"""Topology reconstruction for Stage B client condensation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


ENTITY = 0
SENTENCE = 1
PASSAGE = 2


@dataclass(frozen=True)
class TopologyResult:
    adjacency: Tensor
    edge_index: Tensor
    edge_weight: Tensor


def sep_topology_mask(node_type: Tensor, *, allow_entity_entity: bool = False) -> Tensor:
    """Return a mask that preserves the S-E-P invariant.

    The main condensed graph never receives direct Sentence-Passage edges.
    By default it also excludes same-type edges, leaving only E-S and E-P.
    """

    t_i = node_type.view(-1, 1)
    t_j = node_type.view(1, -1)
    one_entity = (t_i == ENTITY) ^ (t_j == ENTITY)
    non_entity_other = ((t_i == SENTENCE) | (t_i == PASSAGE) | (t_j == SENTENCE) | (t_j == PASSAGE))
    mask = one_entity & non_entity_other
    if allow_entity_entity:
        mask = mask | ((t_i == ENTITY) & (t_j == ENTITY))
    mask.fill_diagonal_(False)
    return mask


def evidence_prior(text_embeddings: Tensor) -> Tensor:
    """Cosine prior S_ij from DANCE §14.1, scaled to [0, 1]."""

    if text_embeddings.numel() == 0:
        return text_embeddings.new_zeros((0, 0))
    normed = F.normalize(text_embeddings, p=2, dim=-1)
    sim = normed @ normed.T
    sim.fill_diagonal_(0)
    min_v = sim.min()
    max_v = sim.max()
    return (sim - min_v) / (max_v - min_v + 1e-8)


def topk_per_row(weights: Tensor, k: int, mask: Tensor | None = None) -> Tensor:
    """Keep at most k values per row and zero the rest."""

    if weights.numel() == 0:
        return weights.clone()
    scores = weights.clone()
    scores.fill_diagonal_(0)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    out = torch.zeros_like(weights)
    k_eff = min(int(k), max(scores.size(1) - 1, 0))
    if k_eff <= 0:
        return out
    for row in range(scores.size(0)):
        finite = torch.isfinite(scores[row])
        if not finite.any():
            continue
        row_k = min(k_eff, int(finite.sum().item()))
        vals, idx = torch.topk(scores[row], k=row_k)
        keep = torch.isfinite(vals) & (vals > 0)
        if keep.any():
            out[row, idx[keep]] = vals[keep]
    return out


def symmetric_degree_capped_topk(weights: Tensor, k: int, mask: Tensor | None = None) -> Tensor:
    """Build a symmetric graph while capping every row degree at ``k``."""

    if weights.numel() == 0:
        return weights.clone()
    scores = weights.clone()
    scores.fill_diagonal_(0)
    if mask is not None:
        scores = scores.masked_fill(~mask, 0)
    scores = torch.maximum(scores, scores.T)

    rows, cols = torch.triu_indices(scores.size(0), scores.size(1), offset=1, device=scores.device)
    values = scores[rows, cols]
    keep = values > 0
    rows, cols, values = rows[keep], cols[keep], values[keep]
    order = torch.argsort(values, descending=True)

    out = torch.zeros_like(weights)
    degree = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    for idx in order.tolist():
        i, j = int(rows[idx]), int(cols[idx])
        if degree[i] >= k or degree[j] >= k:
            continue
        value = values[idx]
        out[i, j] = value
        out[j, i] = value
        degree[i] += 1
        degree[j] += 1
    return out


def dense_to_edge_index(adjacency: Tensor) -> tuple[Tensor, Tensor]:
    """Convert dense adjacency to PyG edge_index and edge_weight."""

    rows, cols = (adjacency > 0).nonzero(as_tuple=True)
    if rows.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=adjacency.device),
            torch.empty((0,), dtype=adjacency.dtype, device=adjacency.device),
        )
    edge_index = torch.stack([rows, cols], dim=0).long()
    edge_weight = adjacency[rows, cols]
    return edge_index, edge_weight


def knn_topology(
    x_fused: Tensor,
    *,
    node_type: Tensor | None = None,
    text_embeddings: Tensor | None = None,
    k: int = 8,
    prior_weight: float = 0.0,
    preserve_sep: bool = True,
) -> TopologyResult:
    """Phase-1 KNN topology baseline required by plan 10 §38 Point 1."""

    if x_fused.dim() != 2:
        raise ValueError("x_fused must be [num_nodes, dim]")
    if x_fused.size(0) == 0:
        adj = x_fused.new_zeros((0, 0))
        edge_index, edge_weight = dense_to_edge_index(adj)
        return TopologyResult(adj, edge_index, edge_weight)

    sim = F.normalize(x_fused, p=2, dim=-1) @ F.normalize(x_fused, p=2, dim=-1).T
    sim = (sim + 1.0) / 2.0
    sim.fill_diagonal_(0)
    if text_embeddings is not None and prior_weight > 0:
        sim = sim + float(prior_weight) * evidence_prior(text_embeddings)

    mask = None
    if preserve_sep and node_type is not None:
        mask = sep_topology_mask(node_type.to(device=x_fused.device))
    row_topk = topk_per_row(sim, k=k, mask=mask)
    adj = symmetric_degree_capped_topk(row_topk, k=k, mask=mask)
    adj.fill_diagonal_(0)
    edge_index, edge_weight = dense_to_edge_index(adj)
    return TopologyResult(adj, edge_index, edge_weight)


def soft_threshold(values: Tensor, threshold: Tensor) -> Tensor:
    return torch.sign(values) * torch.relu(values.abs() - threshold)


def self_expressive_topology(
    x_fused: Tensor,
    text_embeddings: Tensor,
    *,
    node_type: Tensor | None = None,
    alpha_recon: float = 8.0,
    beta_l1: float = 5.0,
    candidate_size: int = 16,
    iterations: int = 50,
    step_size: float = 1e-2,
    final_k: int = 8,
    preserve_sep: bool = True,
) -> TopologyResult:
    """DANCE Algo 4 ISTA reconstruction with S-E-P masking."""

    K = x_fused.size(0)
    if K == 0:
        adj = x_fused.new_zeros((0, 0))
        edge_index, edge_weight = dense_to_edge_index(adj)
        return TopologyResult(adj, edge_index, edge_weight)

    S = evidence_prior(text_embeddings).to(device=x_fused.device, dtype=x_fused.dtype)
    sim_x = x_fused @ x_fused.T
    sim_x.fill_diagonal_(float("-inf"))

    type_mask = None
    if preserve_sep and node_type is not None:
        type_mask = sep_topology_mask(node_type.to(device=x_fused.device))
        sim_x = sim_x.masked_fill(~type_mask, float("-inf"))
        S_for_candidates = S.masked_fill(~type_mask, float("-inf"))
    else:
        S_for_candidates = S.clone()
        S_for_candidates.fill_diagonal_(float("-inf"))

    support = torch.zeros((K, K), dtype=torch.bool, device=x_fused.device)
    q = min(int(candidate_size), max(K - 1, 0))
    for i in range(K):
        if q <= 0:
            continue
        for scores in (sim_x[i], S_for_candidates[i]):
            finite = torch.isfinite(scores)
            if finite.any():
                idx = torch.topk(scores, k=min(q, int(finite.sum().item()))).indices
                support[i, idx] = True
    support.fill_diagonal_(False)
    if type_mask is not None:
        support = support & type_mask

    Z = x_fused.new_zeros((K, K))
    lam_1 = float(beta_l1) / float(alpha_recon)
    lam_3 = 1.0 / float(alpha_recon)
    support_f = support.to(dtype=x_fused.dtype)

    for _ in range(int(iterations)):
        # Row-wise PyG features are [K, d], so each row i is reconstructed
        # from rows j through (Z @ X)[i]. This is the dimensionally correct
        # equivalent of DANCE Algo 4's XZ notation.
        grad = (Z @ x_fused - x_fused) @ x_fused.T
        grad = grad * support_f
        grad.fill_diagonal_(0)
        Z = Z - float(step_size) * grad
        tau = float(step_size) * (lam_1 + lam_3 * (1.0 - S))
        Z = soft_threshold(Z, tau)
        Z = Z * support_f
        Z.fill_diagonal_(0)

    weights = Z.abs() + Z.abs().T
    row_topk = topk_per_row(weights, k=final_k, mask=type_mask)
    adj = symmetric_degree_capped_topk(row_topk, k=final_k, mask=type_mask)
    adj.fill_diagonal_(0)
    edge_index, edge_weight = dense_to_edge_index(adj)
    return TopologyResult(adj, edge_index, edge_weight)
