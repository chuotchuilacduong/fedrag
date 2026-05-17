"""Query-agnostic surrogate losses for Stage C anchor gradient matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ENTITY = 0
SENTENCE = 1
PASSAGE = 2


@dataclass(frozen=True)
class SurrogateOutput:
    loss: Tensor
    type_loss: Tensor
    link_loss: Tensor
    type_accuracy: Tensor


def edge_index_to_dense(edge_index: Tensor, num_nodes: int, edge_weight: Tensor | None = None) -> Tensor:
    """Build a dense symmetric adjacency matrix from a PyG edge_index."""

    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float32
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    if edge_index.numel() == 0:
        return adj
    src, dst = edge_index[0].long(), edge_index[1].long()
    weight = edge_weight.to(device=device, dtype=dtype) if edge_weight is not None else torch.ones(src.numel(), device=device, dtype=dtype)
    adj[src, dst] = weight
    adj[dst, src] = torch.maximum(adj[dst, src], weight)
    adj.fill_diagonal_(0)
    return adj


def normalize_dense_adjacency(adj: Tensor, *, add_self_loops: bool = True) -> Tensor:
    """Symmetric GCN normalization for dense adjacency."""

    if add_self_loops:
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
    degree = adj.sum(dim=1).clamp_min(1e-12)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)


class SurrogateGNN(nn.Module):
    """Small dense-adjacency GCN for node-type and link surrogate gradients."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        dims = [input_dim]
        if num_layers == 1:
            dims.append(output_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(output_dim)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        h = x
        adj_norm = normalize_dense_adjacency(adj)
        embedding = h
        for idx, layer in enumerate(self.layers):
            h = adj_norm @ h
            h = layer(h)
            if idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                embedding = h
        if len(self.layers) == 1:
            embedding = x
        return embedding, h


def node_type_ce(logits: Tensor, node_type: Tensor) -> Tensor:
    """Node-type CE over entity/sentence/passage labels."""

    return F.cross_entropy(logits, node_type.long())


def _negative_edges(num_nodes: int, positive_adj: Tensor, num_samples: int) -> Tensor:
    if num_nodes <= 1 or num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=positive_adj.device)
    samples: list[Tensor] = []
    attempts = 0
    while sum(sample.size(1) for sample in samples) < num_samples and attempts < 20:
        attempts += 1
        src = torch.randint(0, num_nodes, (num_samples,), device=positive_adj.device)
        dst = torch.randint(0, num_nodes, (num_samples,), device=positive_adj.device)
        keep = (src != dst) & (positive_adj[src, dst] <= 0)
        if keep.any():
            samples.append(torch.stack([src[keep], dst[keep]], dim=0))
    if not samples:
        return torch.empty((2, 0), dtype=torch.long, device=positive_adj.device)
    out = torch.cat(samples, dim=1)[:, :num_samples]
    return out.long()


def link_prediction_bce(embeddings: Tensor, adj: Tensor, edge_index: Tensor | None = None) -> Tensor:
    """Balanced dot-product link prediction BCE."""

    num_nodes = embeddings.size(0)
    if edge_index is None:
        edge_index = (torch.triu(adj, diagonal=1) > 0).nonzero(as_tuple=False).T.contiguous()
    if edge_index.numel() == 0:
        return embeddings.new_zeros(())

    src, dst = edge_index[0].long(), edge_index[1].long()
    pos_logits = (embeddings[src] * embeddings[dst]).sum(dim=-1)
    neg = _negative_edges(num_nodes, adj, pos_logits.numel())
    if neg.numel() == 0:
        return F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

    neg_logits = (embeddings[neg[0]] * embeddings[neg[1]]).sum(dim=-1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def surrogate_loss(
    model: SurrogateGNN,
    x: Tensor,
    adj: Tensor,
    node_type: Tensor,
    *,
    edge_index: Tensor | None = None,
    type_weight: float = 1.0,
    link_weight: float = 0.5,
) -> SurrogateOutput:
    """Compute the Stage C surrogate from plan 04 §16 and 10_DANCE §39.3."""

    embeddings, logits = model(x, adj)
    loss_type = node_type_ce(logits, node_type)
    loss_link = link_prediction_bce(embeddings, adj, edge_index=edge_index)
    loss = float(type_weight) * loss_type + float(link_weight) * loss_link
    pred = logits.argmax(dim=-1)
    acc = (pred == node_type.long()).float().mean()
    return SurrogateOutput(loss=loss, type_loss=loss_type, link_loss=loss_link, type_accuracy=acc)


def parameter_gradients(loss: Tensor, parameters: Sequence[nn.Parameter], *, create_graph: bool = False) -> list[Tensor]:
    """Return gradients for all parameters, replacing unused grads with zeros."""

    grads = torch.autograd.grad(loss, parameters, create_graph=create_graph, allow_unused=True)
    out: list[Tensor] = []
    for grad, parameter in zip(grads, parameters):
        if grad is None:
            out.append(torch.zeros_like(parameter))
        else:
            out.append(grad)
    return out


def gradient_match_loss(global_grads: Sequence[Tensor], anchor_grads: Sequence[Tensor], *, norm_weight: float = 0.0) -> Tensor:
    """Cosine gradient matching loss from plan 04 §18."""

    flat_global = torch.cat([grad.reshape(-1) for grad in global_grads])
    flat_anchor = torch.cat([grad.detach().reshape(-1).to(flat_global.device) for grad in anchor_grads])
    cosine = F.cosine_similarity(flat_global.unsqueeze(0), flat_anchor.unsqueeze(0), dim=1).squeeze(0)
    loss = 1.0 - cosine
    if norm_weight > 0:
        loss = loss + float(norm_weight) * F.mse_loss(flat_global, flat_anchor)
    return loss
