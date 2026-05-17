"""Type-aware PGE adjacency for FedCondGraphRAG synthetic graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fedcond_grag.client.stage_b_condense.topology_reconstruction import (
    sep_topology_mask,
    symmetric_degree_capped_topk,
)

_N_PART = 5  # number of chunks for pairwise computation — avoids [K,K,d] OOM


class TypeAwarePGE(nn.Module):
    """Parameterized Graph Estimator with node-type embeddings.

    Forward pass is chunked (n_part=5) to avoid building the full [K,K,d]
    pairwise tensor in memory — same strategy as the vendored PGE.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        type_emb_dim: int = 16,
        topk: int = 8,
        preserve_sep: bool = True,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(3, type_emb_dim)
        self.topk = int(topk)
        self.preserve_sep = bool(preserve_sep)
        in_dim = feature_dim * 2 + type_emb_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, node_type: Tensor, *, sparsify: bool = True) -> Tensor:
        K = x.size(0)
        device = x.device

        # Type embeddings — cheap [K, type_emb_dim]
        t_emb = self.type_emb(node_type.long())

        # All (i,j) pair indices — [K²]
        idx = torch.arange(K, device=device)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        src = ii.reshape(-1)  # [K²]
        dst = jj.reshape(-1)  # [K²]

        # Chunked MLP forward — avoids [K,K,d] tensor in memory
        chunk_size = max(1, (K * K + _N_PART - 1) // _N_PART)
        scores_parts = []
        for start in range(0, K * K, chunk_size):
            end = min(start + chunk_size, K * K)
            s = src[start:end]
            d = dst[start:end]
            xi = x[s]
            xj = x[d]
            ti = t_emb[s]
            tj = t_emb[d]
            feat = torch.cat([(xi - xj).abs(), xi * xj, ti, tj], dim=-1)
            scores_parts.append(self.mlp(feat).squeeze(-1))

        scores = torch.cat(scores_parts).reshape(K, K)
        scores = torch.sigmoid(scores)
        scores = (scores + scores.T) / 2.0
        scores.fill_diagonal_(0.0)

        mask = None
        if self.preserve_sep:
            mask = sep_topology_mask(node_type.to(device=device))
            scores = scores.masked_fill(~mask, 0.0)

        if not sparsify:
            return scores
        return symmetric_degree_capped_topk(scores, k=self.topk, mask=mask)

    @torch.no_grad()
    def inference(self, x: Tensor, node_type: Tensor) -> Tensor:
        return self.forward(x, node_type, sparsify=True)
