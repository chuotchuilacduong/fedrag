"""Chunk-level evidence selection for DANCE-style text condensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ChunkSelection:
    """Selected chunk indices and weights for one core node."""

    node_ids: list[int]
    chunk_ids: list[int]
    weights: Tensor


def topk_softmax(scores: Tensor, k: int) -> Tensor:
    """Softmax over the top-k scores and zero everywhere else.

    This is the phase-1 approximation to DANCE's entmax + hard top-B STE
    operator from plan 10_DANCE_REFERENCE §38 Point 1.
    """

    if scores.numel() == 0:
        return scores.new_zeros(scores.shape)
    if k <= 0:
        return scores.new_zeros(scores.shape)

    k_eff = min(int(k), scores.numel())
    values, indices = torch.topk(scores, k=k_eff)
    weights = torch.softmax(values, dim=0)
    out = scores.new_zeros(scores.shape)
    out.scatter_(0, indices, weights)
    return out


def score_chunks(g_v: Tensor, chunks: Tensor, scorer: nn.Module | None = None) -> Tensor:
    """Score chunks against a graph-side query vector."""

    if chunks.numel() == 0:
        return chunks.new_zeros((0,))
    query = scorer(g_v) if scorer is not None else g_v
    if query.dim() != 1:
        query = query.reshape(-1)
    return chunks @ query / sqrt(max(query.numel(), 1))


def select_chunks(
    g_v: Tensor,
    candidate_node_ids: Sequence[int],
    chunk_bank: Sequence[Tensor],
    *,
    budget: int = 8,
) -> tuple[Tensor, ChunkSelection]:
    """Select up to ``budget`` chunks and return ``t_tilde_v`` plus trace.

    ``chunk_bank[u]`` must be a ``[num_chunks_u, d]`` tensor. The trace contains
    integer ids only; callers that need raw spans should keep them in local-only
    storage outside the upload object.
    """

    flat_chunks: list[Tensor] = []
    flat_node_ids: list[int] = []
    flat_chunk_ids: list[int] = []

    for node_id in candidate_node_ids:
        chunks = chunk_bank[int(node_id)]
        if chunks.dim() == 1:
            chunks = chunks.unsqueeze(0)
        for chunk_id in range(chunks.size(0)):
            flat_chunks.append(chunks[chunk_id])
            flat_node_ids.append(int(node_id))
            flat_chunk_ids.append(chunk_id)

    if not flat_chunks:
        dim = g_v.numel()
        empty = g_v.new_zeros((dim,))
        trace = ChunkSelection(node_ids=[], chunk_ids=[], weights=g_v.new_zeros((0,)))
        return empty, trace

    chunks_tensor = torch.stack(flat_chunks, dim=0).to(device=g_v.device, dtype=g_v.dtype)
    scores = score_chunks(g_v, chunks_tensor)
    weights = topk_softmax(scores, int(budget))
    selected = weights > 0

    if selected.any():
        t_tilde = weights @ chunks_tensor
    else:
        t_tilde = chunks_tensor.new_zeros((chunks_tensor.size(1),))

    trace = ChunkSelection(
        node_ids=[flat_node_ids[i] for i in selected.nonzero(as_tuple=False).flatten().tolist()],
        chunk_ids=[flat_chunk_ids[i] for i in selected.nonzero(as_tuple=False).flatten().tolist()],
        weights=weights[selected].detach().cpu(),
    )
    return t_tilde, trace


def count_selected(weights: Tensor) -> int:
    """Return the number of non-zero selected weights."""

    return int((weights > 0).sum().item())
