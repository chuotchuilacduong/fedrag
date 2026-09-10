"""Diversity diagnostics for X_syn / H_syn snapshots.

Two metrics, computed on L2-row-normalized (never centered) representations
after excluding near-zero rows:

  mean_offdiag_cosine -- mean of the off-diagonal Gram matrix of row-normalized
      features. Rises toward 1 as nodes collapse onto the same direction.

  effective_rank -- entropy effective rank of the singular value spectrum of
      the (uncentered, row-normalized) representation matrix:
          p_i = sigma_i / sum(sigma), r_eff = exp(-sum(p_i log p_i))
      Falls toward 1 as the representation collapses onto fewer directions.

Both operate on a single representation matrix R (X_syn or H_syn) and report
the number of rows excluded as near-zero, so redundancy and pure gain-scaling
degeneracies are visible separately.
"""

from __future__ import annotations

import torch


def compute_diversity_metrics(rep: torch.Tensor, eps: float = 1e-8) -> dict:
    """Diversity diagnostics for one [N, D] representation matrix.

    Returns a dict with num_nodes, num_valid_rows, num_near_zero_rows,
    mean_offdiag_cosine, effective_rank. The last two are NaN when fewer than
    2 valid rows remain (undefined).
    """
    rep = rep.detach().to(torch.float32)
    num_nodes = int(rep.size(0))
    if num_nodes == 0:
        return {
            "num_nodes": 0, "num_valid_rows": 0, "num_near_zero_rows": 0,
            "mean_offdiag_cosine": float("nan"), "effective_rank": float("nan"),
        }

    norms = rep.norm(dim=1)
    valid_mask = norms > eps
    num_valid = int(valid_mask.sum().item())
    num_near_zero = num_nodes - num_valid

    if num_valid < 2:
        return {
            "num_nodes": num_nodes, "num_valid_rows": num_valid,
            "num_near_zero_rows": num_near_zero,
            "mean_offdiag_cosine": float("nan"), "effective_rank": float("nan"),
        }

    valid = rep[valid_mask]
    normalized = valid / valid.norm(dim=1, keepdim=True).clamp_min(eps)

    gram = normalized @ normalized.T
    k = normalized.size(0)
    offdiag_mask = ~torch.eye(k, dtype=torch.bool, device=normalized.device)
    mean_offdiag_cosine = float(gram[offdiag_mask].mean().item())

    # Uncentered SVD of the row-normalized matrix (never mean-subtracted).
    singular_values = torch.linalg.svdvals(normalized.double())
    s_sum = singular_values.sum().clamp_min(eps)
    p = (singular_values / s_sum).clamp_min(eps)
    entropy = -(p * p.log()).sum()
    effective_rank = float(torch.exp(entropy).item())

    return {
        "num_nodes": num_nodes,
        "num_valid_rows": num_valid,
        "num_near_zero_rows": num_near_zero,
        "mean_offdiag_cosine": round(mean_offdiag_cosine, 6),
        "effective_rank": round(effective_rank, 6),
    }


def rows_for_pca(rep: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, int]:
    """Drop near-zero rows and L2-row-normalize, WITHOUT centering -- the
    preprocessing shared by the diversity metrics and the PCA visualization.

    Returns (normalized_valid_rows [N', D], num_near_zero_rows).
    """
    rep = rep.detach().to(torch.float32)
    norms = rep.norm(dim=1)
    valid_mask = norms > eps
    valid = rep[valid_mask]
    normalized = valid / valid.norm(dim=1, keepdim=True).clamp_min(eps)
    return normalized, int((~valid_mask).sum().item())
