"""Representation-space alignment loss for Stage C server optimization.

Implements the Lalign objective from the paper:
    Am  = softmax(Hm @ Hsyn^T / sqrt(H), dim=1)
    Lalign = sum_m (|Vm|/sum_j|Vj|) * (1/|Vm|) * ||Hm - Am @ Hsyn||^2_F

where Hm = fproj(phiGNN(G_m)) is computed per-node (not graph-pooled).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data


def encode_nodes(
    x: Tensor,
    edge_index: Tensor,
    encoder: nn.Module,
    projector: nn.Module,
) -> Tensor:
    """Per-node projection through GNN + projector: [N, hidden] -> [N, H].

    Keeps input dtype compatible with encoder (bfloat16 / float32).
    Output is cast back to float32 for loss computation.
    """
    _param = next(encoder.parameters(), None)
    enc_dtype = _param.dtype if _param is not None else x.dtype
    n_embeds, _ = encoder(x.to(enc_dtype), edge_index.long(), None)
    return projector(n_embeds).float()


def encode_nodes_with_edge_weight(
    x: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor,
    encoder: nn.Module,
    projector: nn.Module,
) -> Tensor:
    """Per-node projection through GCN + projector using soft edge weights.

    GCNConv supports edge_weight as a continuous scalar per edge, so gradient
    flows: Lalign → h_syn → edge_weight → adj_soft[rows,cols] → adj_soft → PGE.

    edge_weight: [E] continuous values from adj_soft[rows, cols] (has grad).
    """
    _param = next(encoder.parameters(), None)
    enc_dtype = _param.dtype if _param is not None else x.dtype
    n_embeds, _ = encoder(x.to(enc_dtype), edge_index.long(), edge_weight.to(enc_dtype))
    return projector(n_embeds).float()


def precompute_anchor_reprs(
    anchor_graphs: list[Data],
    encoder: nn.Module,
    projector: nn.Module,
    device: torch.device,
) -> list[Tensor]:
    """Detached per-node H_m for all anchor graphs (computed once per round).

    Uses edge_weight from the anchor graph if present (Stage B ISTA weights),
    matching Hm = fproj(phiGNN(G_m)) with the full graph structure.
    """
    encoder.eval()
    projector.eval()
    result: list[Tensor] = []
    with torch.no_grad():
        for graph in anchor_graphs:
            graph = graph.to(device)
            ew = getattr(graph, "edge_weight", None)
            if ew is not None:
                h = encode_nodes_with_edge_weight(
                    graph.x, graph.edge_index, ew, encoder, projector
                )
            else:
                h = encode_nodes(graph.x, graph.edge_index, encoder, projector)
            result.append(h.detach())
    return result


def representation_alignment_loss(h_syn: Tensor, anchor_h_list: list[Tensor]) -> Tensor:
    """Lalign from paper §3.2:

        Am     = softmax(Hm @ Hsyn^T / sqrt(H), dim=1)       [Nm, Kg]
        Lalign = sum_m (|Vm| / sum_j|Vj|) * (1/|Vm|) * ||Hm - Am @ Hsyn||^2_F
               = (1 / sum_j|Vj|) * sum_m ||Hm - Am @ Hsyn||^2_F

    Uses sum-over-nodes Frobenius norm divided by |Vm|, NOT mean over elements
    (i.e. does not divide by H — matches the paper formula exactly).

    h_syn:          [Kg, H]  — synthetic node projections (has grad)
    anchor_h_list:  list of [Nm, H]  — anchor node projections (detached)
    """
    total = sum(h.size(0) for h in anchor_h_list)
    if total == 0 or h_syn.size(0) == 0:
        return h_syn.new_zeros(())

    scale = math.sqrt(h_syn.size(-1))
    # weight * (1/n_m) = (n_m/total) * (1/n_m) = 1/total — constant across loop
    inv_total = 1.0 / total
    loss = h_syn.new_zeros(())

    for h_m in anchor_h_list:
        logits = h_m @ h_syn.T / scale                   # [Nm, Kg]
        a_m = torch.softmax(logits, dim=1)               # [Nm, Kg]
        h_reconstructed = a_m @ h_syn                    # [Nm, H]
        frob_sq = (h_reconstructed - h_m).pow(2).sum()   # ||...||^2_F
        loss = loss + inv_total * frob_sq

    return loss


def diversity_loss(h_syn: Tensor) -> Tensor:
    """Penalize pairwise cosine similarity between synthetic nodes.

    Encourages synthetic nodes to cover diverse semantic regions.
    Uses mean of |cos_sim_{i≠j}| via the Gram matrix. O(Kg^2) but
    Kg=1024 is fine (~4 MB).
    """
    if h_syn.size(0) < 2:
        return h_syn.new_zeros(())
    h_norm = F.normalize(h_syn, dim=-1)                              # [Kg, H]
    gram = h_norm @ h_norm.T                                         # [Kg, Kg]
    K = h_syn.size(0)
    off_diag = gram[~torch.eye(K, dtype=torch.bool, device=h_syn.device)]
    return off_diag.abs().mean()


def degree_regularization(adj: Tensor, target_degree: float) -> Tensor:
    """MSE between per-node degree and target average degree.

    Prevents degenerate topologies (all isolated or fully connected).
    """
    if adj.numel() == 0:
        return adj.new_zeros(())
    degree = adj.sum(dim=1).float()
    return F.mse_loss(degree, torch.full_like(degree, target_degree))


def compute_target_degree(anchor_graphs: list[Data]) -> float:
    """Average node degree across all anchor graphs."""
    total_edges = 0
    total_nodes = 0
    for graph in anchor_graphs:
        total_nodes += int(graph.x.size(0))
        if graph.edge_index.numel() > 0:
            total_edges += int(graph.edge_index.size(1))
    if total_nodes == 0:
        return 8.0
    return total_edges / total_nodes
