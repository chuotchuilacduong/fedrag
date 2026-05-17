"""Graph-text gated fusion for client condensed node features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphTextFusion(nn.Module):
    """Implement DANCE Eq. 10 with separate graph/text projections."""

    def __init__(self, graph_dim: int, text_dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = int(out_dim or graph_dim)
        self.graph_proj = nn.Linear(graph_dim, out_dim)
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.gate = nn.Linear(graph_dim + text_dim, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph_embeddings: Tensor, text_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        if graph_embeddings.dim() != 2 or text_embeddings.dim() != 2:
            raise ValueError("graph_embeddings and text_embeddings must be rank-2 tensors")
        if graph_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError("graph/text embeddings must have the same row count")

        gate = torch.sigmoid(self.gate(torch.cat([graph_embeddings, text_embeddings], dim=-1)))
        fused = self.norm(self.graph_proj(graph_embeddings) + gate * self.text_proj(text_embeddings))
        return fused, gate.squeeze(-1)


def fuse(graph_embeddings: Tensor, text_embeddings: Tensor, module: GraphTextFusion | None = None) -> tuple[Tensor, Tensor]:
    """Functional helper for one-off fusion."""

    if module is None:
        module = GraphTextFusion(graph_embeddings.size(1), text_embeddings.size(1), graph_embeddings.size(1)).to(
            graph_embeddings.device
        )
    return module(graph_embeddings, text_embeddings)
