"""Semantic-Aware Hypergraph Learning (Algorithm 1) for FD-RAG.

Direct implementation of ``implementfd_rag.md`` §2.1: Eq.4-8 with projected
gradient descent onto the simplex, Duchi et al.-style projection, and the
critical implementation notes (stop-gradient on ρ, soft `Ĥ`-weighted N(e)
during training, orphan-fallback after sparsification, pluggable sparsify
rule per [GAP-3]).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from fedcond_grag.baselines.fdrag.data_types import Hyperedge, TextUnit


# ---------------------------------------------------------------------------
# Simplex projection (Duchi et al. 2008, O(n log n) sort-based)
# ---------------------------------------------------------------------------
def _project_rows_onto_simplex(V: torch.Tensor) -> torch.Tensor:
    """Row-wise Euclidean projection of `V` (N x M) onto the probability
    simplex Δ^M. Standard sort-based algorithm."""
    N, M = V.shape
    U, _ = torch.sort(V, dim=1, descending=True)         # (N, M)
    cssv = torch.cumsum(U, dim=1) - 1.0                  # (N, M)
    ind = torch.arange(1, M + 1, device=V.device, dtype=V.dtype).unsqueeze(0)
    cond = U - cssv / ind > 0
    # Rightmost True index per row
    rho = cond.float().cumsum(dim=1).argmax(dim=1)       # last True idx
    theta = cssv.gather(1, rho.unsqueeze(1)) / (rho.unsqueeze(1).to(V.dtype) + 1.0)
    W = torch.clamp(V - theta, min=0.0)
    return W


# ---------------------------------------------------------------------------
# SAHL result container
# ---------------------------------------------------------------------------
@dataclass
class SAHLResult:
    hyperedges: list[Hyperedge]
    H_p: np.ndarray           # sparsified incidence, paragraph
    H_s: np.ndarray           # sparsified incidence, sentence
    grad_norms: list[float]   # convergence trace ‖Ĥ_k − Ĥ_{k+1}‖ / η


# ---------------------------------------------------------------------------
# Loss pieces (Eq. 5-8)
# ---------------------------------------------------------------------------
def _prototypes(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Eq.5 (soft form): E = (H^T X) / colsum(H)."""
    denom = H.sum(dim=0, keepdim=True).clamp_min(1e-8)   # (1, M)
    return (H.transpose(0, 1) @ X) / denom.transpose(0, 1)


def _intra_loss(H: torch.Tensor, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Eq.6 (soft/differentiable form using Ĥ weights)."""
    # D2[n, m] = ||x_n - e_m||^2
    x2 = (X * X).sum(dim=1, keepdim=True)                # (N, 1)
    e2 = (E * E).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, M)
    D2 = x2 + e2 - 2.0 * (X @ E.transpose(0, 1))
    D2 = D2.clamp_min(0.0)
    num = (H * D2).sum(dim=0)                            # (M,)
    denom = H.sum(dim=0).clamp_min(1e-8)                 # (M,)
    return (num / denom).mean()


def _inter_loss(
    E: torch.Tensor, gamma: float, max_pairs: Optional[int] = None
) -> torch.Tensor:
    """Eq.7 with ρ_ij = cos(e_i, e_j) STOP-GRAD.

    Full pair enumeration is O(M^2 D); if `max_pairs` is set we subsample
    ordered pairs uniformly per note §2.1.4.
    """
    M = E.shape[0]
    if M <= 1:
        return torch.zeros((), device=E.device, dtype=E.dtype)
    E_norm = E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)

    if max_pairs is None or max_pairs >= M * M:
        rho = (E_norm @ E_norm.transpose(0, 1)).detach()  # (M, M) stop-grad
        e2 = (E * E).sum(dim=1, keepdim=True)
        D2 = e2 + e2.transpose(0, 1) - 2.0 * (E @ E.transpose(0, 1))
        D2 = D2.clamp_min(0.0)
        term = rho * D2 + (1.0 - rho) * torch.relu(gamma - D2)
        return term.mean()

    idx_i = torch.randint(0, M, (max_pairs,), device=E.device)
    idx_j = torch.randint(0, M, (max_pairs,), device=E.device)
    Ei, Ej = E[idx_i], E[idx_j]
    rho = (E_norm[idx_i] * E_norm[idx_j]).sum(dim=1).detach()
    diff = Ei - Ej
    D2 = (diff * diff).sum(dim=1).clamp_min(0.0)
    term = rho * D2 + (1.0 - rho) * torch.relu(gamma - D2)
    return term.mean()


# ---------------------------------------------------------------------------
# Initialization + sparsification
# ---------------------------------------------------------------------------
def _init_simplex(N: int, M: int, generator: torch.Generator) -> torch.Tensor:
    v = torch.rand(N, M, generator=generator).clamp_min(1e-4)
    v = v / v.sum(dim=1, keepdim=True)
    return v


def _sparsify(H_hat: np.ndarray, mu: float, mode: str, top_r: int) -> np.ndarray:
    N, M = H_hat.shape
    if mode == "absolute":
        keep = H_hat >= mu
    elif mode == "relative":
        row_max = H_hat.max(axis=1, keepdims=True).clip(min=1e-8)
        keep = H_hat >= (mu * row_max)
    elif mode == "top_r":
        r = max(1, top_r)
        order = np.argsort(-H_hat, axis=1)[:, :r]
        keep = np.zeros_like(H_hat, dtype=bool)
        rows = np.arange(N)[:, None]
        keep[rows, order] = True
    else:
        raise ValueError(f"unknown sparsify mode: {mode}")

    H = H_hat * keep
    # Orphan fallback: every row must own at least one hyperedge (spec §2.1 line 29)
    zero_rows = ~keep.any(axis=1)
    if zero_rows.any():
        argmax = H_hat[zero_rows].argmax(axis=1)
        H[zero_rows, argmax] = H_hat[zero_rows, argmax]
    return H


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def learn_hypergraph(
    units_p: list[TextUnit],
    units_s: list[TextUnit],
    X_p: np.ndarray,
    X_s: np.ndarray,
    *,
    lam: float,
    gamma: float,
    mu: float,
    m_p_ratio: float,
    m_s_ratio: float,
    steps: int = 300,
    lr: float = 0.05,
    max_pairs: Optional[int] = None,
    sparsify_mode: str = "absolute",
    sparsify_top_r: int = 1,
    device: str = "cpu",
    seed: int = 42,
) -> SAHLResult:
    """Algorithm 1: run SAHL over paragraph + sentence granularities jointly.

    See ``implementfd_rag.md`` §2.1 for the full pseudocode and equations. The
    two granularities share the same objective (`L_total` is summed over
    `t ∈ {p, s}`) and step together, but keep separate incidence matrices
    and prototype sets.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    torch_device = torch.device(device)

    N_p, N_s = X_p.shape[0], X_s.shape[0]
    if N_p == 0 and N_s == 0:
        return SAHLResult(hyperedges=[], H_p=np.zeros((0, 0)), H_s=np.zeros((0, 0)), grad_norms=[])

    M_p = max(1, math.ceil(N_p * m_p_ratio)) if N_p > 0 else 0
    M_s = max(1, math.ceil(N_s * m_s_ratio)) if N_s > 0 else 0

    Xp_t = torch.as_tensor(X_p, dtype=torch.float32, device=torch_device) if N_p else None
    Xs_t = torch.as_tensor(X_s, dtype=torch.float32, device=torch_device) if N_s else None

    Hp = _init_simplex(N_p, M_p, gen).to(torch_device).requires_grad_(True) if N_p else None
    Hs = _init_simplex(N_s, M_s, gen).to(torch_device).requires_grad_(True) if N_s else None

    grad_norms: list[float] = []

    for step in range(steps):
        loss = torch.zeros((), device=torch_device)
        if Hp is not None:
            Ep = _prototypes(Hp, Xp_t)
            loss = loss + lam * _intra_loss(Hp, Xp_t, Ep) + (1.0 - lam) * _inter_loss(Ep, gamma, max_pairs)
        if Hs is not None:
            Es = _prototypes(Hs, Xs_t)
            loss = loss + lam * _intra_loss(Hs, Xs_t, Es) + (1.0 - lam) * _inter_loss(Es, gamma, max_pairs)

        grads = torch.autograd.grad(loss, [h for h in (Hp, Hs) if h is not None])
        with torch.no_grad():
            new_norm_sq = 0.0
            i = 0
            for H in (Hp, Hs):
                if H is None:
                    continue
                g = grads[i]; i += 1
                H_new = _project_rows_onto_simplex(H - lr * g)
                new_norm_sq += float(((H - H_new) / lr).pow(2).sum().item())
                H.copy_(H_new)
            grad_norms.append(math.sqrt(new_norm_sq))

    # ----- Sparsify + build final hyperedge objects (Eq. 4 sparsification) -----
    hyperedges: list[Hyperedge] = []

    def _finalize(
        H_soft: torch.Tensor,
        units: list[TextUnit],
        X_np: np.ndarray,
        granularity: str,
    ) -> np.ndarray:
        H_hat = H_soft.detach().cpu().numpy()
        H = _sparsify(H_hat, mu=mu, mode=sparsify_mode, top_r=sparsify_top_r)
        M = H.shape[1]
        for m in range(M):
            col = H[:, m]
            member_idx = np.flatnonzero(col > 0)
            if member_idx.size == 0:
                continue
            weights = col[member_idx]
            proto = (weights[:, None] * X_np[member_idx]).sum(axis=0) / weights.sum()
            hyperedges.append(Hyperedge(
                id=f"{granularity[0]}:{uuid.uuid4().hex[:8]}",
                granularity=granularity,
                member_unit_ids=[units[i].id for i in member_idx],
                weights=weights,
                prototype=proto.astype(np.float32),
                context=[units[i].text for i in member_idx],
            ))
        return H

    H_p_final = _finalize(Hp, units_p, X_p, "paragraph") if Hp is not None else np.zeros((0, 0))
    H_s_final = _finalize(Hs, units_s, X_s, "sentence") if Hs is not None else np.zeros((0, 0))

    return SAHLResult(hyperedges=hyperedges, H_p=H_p_final, H_s=H_s_final, grad_norms=grad_norms)
