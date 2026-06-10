"""Graph-text gated fusion for client condensed node features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphTextFusion(nn.Module):
    """Fixed-gate additive fusion of graph and text embeddings.

    Stage B runs entirely under torch.no_grad(), so any learned component
    (gate, projections) would only ever carry its random initialization.
    Earlier versions projected both branches through random nn.Linear layers,
    which rotated the fused features out of the MiniLM embedding space and
    broke the cosine retrieval against them in Stage D. Both branches are now
    combined identity-wise so the anchor-graph features stay in the same
    embedding space as the Tri-Graph / query embeddings:

        x_fused = LayerNorm(x_graph + gate * x_text),  gate fixed (default 0.5)

    LayerNorm's affine parameters are never trained (γ=1, β=0), so it acts as
    a deterministic normalization.
    """

    def __init__(self, graph_dim: int, text_dim: int, gate_value: float = 0.5):
        super().__init__()
        if not 0.0 <= gate_value <= 1.0:
            raise ValueError("gate_value must lie in [0, 1]")
        if int(graph_dim) != int(text_dim):
            raise ValueError(
                f"graph_dim ({graph_dim}) must equal text_dim ({text_dim}) — "
                "both branches must live in the same embedding space "
                "(identity fusion, no projections)"
            )
        self.register_buffer("gate_value", torch.tensor(float(gate_value)))
        self.norm = nn.LayerNorm(int(graph_dim))

    def forward(self, graph_embeddings: Tensor, text_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        if graph_embeddings.dim() != 2 or text_embeddings.dim() != 2:
            raise ValueError("graph_embeddings and text_embeddings must be rank-2 tensors")
        if graph_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError("graph/text embeddings must have the same row count")

        gate = self.gate_value.to(graph_embeddings.dtype)
        fused = self.norm(graph_embeddings + gate * text_embeddings)
        gate_per_node = gate.expand(graph_embeddings.size(0))
        return fused, gate_per_node


def fuse(graph_embeddings: Tensor, text_embeddings: Tensor, module: GraphTextFusion | None = None) -> tuple[Tensor, Tensor]:
    """Functional helper for one-off fusion."""

    if module is None:
        module = GraphTextFusion(graph_embeddings.size(1), text_embeddings.size(1)).to(
            graph_embeddings.device
        )
    return module(graph_embeddings, text_embeddings)
