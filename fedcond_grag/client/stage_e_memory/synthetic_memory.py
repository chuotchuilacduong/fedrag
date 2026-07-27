"""Stage E — client-side query-conditioned synthetic memory adaptation.

Implements FedRAG Phase 1 (paper §3.5, Appendix B.7): each round the client
receives the broadcast synthetic memory Θ_syn = {X_syn, θ_PGE}, adapts a local
copy for K_mem steps by minimizing

    L_mem = L_QA + λ_gm · L_GM + λ_align · L_align + λ_reg · L_reg,

and returns only the parameter delta Δ_m = Θ_syn,m^(K_mem) − Θ_syn to the
server. Raw queries, answers, evidence graphs, and retrieval traces stay local.

L_QA uses differentiable soft synthetic retrieval (B.5.2): the pooled evidence
prompt representation z̄_e attends over the projected synthetic nodes,
z_c = Σ_i α_i z_i^syn, so gradients flow from the frozen-LLM QA loss back to
X_syn and θ_PGE through the soft context token.

L_GM follows the paper's first-order approximation: the prompt-module gradient
induced by private local evidence (g_loc) is detached, and g_syn is computed
through the synthetic context branch only — via a token-gradient surrogate
(∂L/∂z_c is detached and re-contracted with z_c), which avoids a second-order
backward through the LLM while keeping the context-branch gradient exact.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE
from fedcond_grag.server.stage_c_aggregate.repr_align import (
    degree_regularization,
    diversity_loss,
    encode_nodes,
    encode_nodes_with_edge_weight,
)


class LocalSyntheticMemory(nn.Module):
    """Client-local copy of the synthetic memory Θ_syn = {X_syn, θ_PGE}.

    Built from the server broadcast, adapted with private QA feedback, and
    exported as a parameter delta. Never holds client text or queries.
    """

    def __init__(
        self,
        x: Tensor,
        node_type: Tensor,
        pge: TypeAwarePGE,
        target_degree: float = 8.0,
    ):
        super().__init__()
        self.x = nn.Parameter(x.detach().clone().float())
        self.register_buffer("node_type", node_type.detach().clone().long())
        self.pge = pge
        for p in self.pge.parameters():
            p.requires_grad_(True)
        self.target_degree = float(target_degree)

    @classmethod
    def from_broadcast(cls, state: dict, device: torch.device) -> "LocalSyntheticMemory":
        """Rebuild Θ_syn from the server's `synthetic_state` broadcast dict."""
        pge = TypeAwarePGE(**state["pge_config"])
        pge.load_state_dict({k: v.clone() for k, v in state["pge_state"].items()})
        mem = cls(
            state["x"].to(device),
            state["node_type"].to(device),
            pge.to(device),
            target_degree=float(state.get("target_degree", 8.0)),
        )
        return mem.to(device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def soft_edges(self) -> tuple[Tensor, Tensor, Tensor]:
        """Soft synthetic topology with gradient to both X_syn and θ_PGE.

        Returns (adj_soft [Kg,Kg], edge_index [2,E], edge_weight [E]); the
        edge weights are entries of adj_soft so gradient flows to the PGE.
        """
        adj_soft = self.pge(self.x, self.node_type, sparsify=False)
        rows, cols = (adj_soft.detach() > 0).nonzero(as_tuple=True)
        edge_index = torch.stack([rows, cols], dim=0).long()
        edge_weight = adj_soft[rows, cols]
        return adj_soft, edge_index, edge_weight

    def encode(self, encoder: nn.Module, projector: nn.Module) -> tuple[Tensor, Tensor]:
        """Project synthetic nodes into the LLM prompt space.

        Returns (z_syn [Kg, H] float32 with grad, adj_soft [Kg, Kg]).
        """
        adj_soft, edge_index, edge_weight = self.soft_edges()
        if edge_index.numel() > 0:
            z_syn = encode_nodes_with_edge_weight(
                self.x, edge_index, edge_weight, encoder, projector
            )
        else:
            z_syn = encode_nodes(self.x, edge_index, encoder, projector)
        return z_syn, adj_soft

    def soft_context(self, query: Tensor, z_syn: Tensor, tau: float = 0.1) -> Tensor:
        """Differentiable soft synthetic retrieval (paper B.5.2).

        query: [B, H] pooled evidence prompt representations z̄_e.
        z_syn: [Kg, H] projected synthetic nodes.
        Returns z_c = Σ_i α_i(q) z_i^syn as [B, H] (float32).
        """
        q = F.normalize(query.float(), dim=-1)
        z = F.normalize(z_syn.float(), dim=-1)
        alpha = torch.softmax(q @ z.T / max(tau, 1e-6), dim=-1)  # [B, Kg]
        return alpha @ z_syn.float()

    # ------------------------------------------------------------------
    # Local losses
    # ------------------------------------------------------------------

    def alignment_loss(self, z_client: Tensor, z_syn: Tensor) -> Tensor:
        """Client-side L_align (paper B.7).

            P_m     = RowSoftmax(Z_m Z_syn^T / sqrt(d))
            L_align = ||Z_m − P_m Z_syn||²_F / |Ṽ_m|

        z_client: [K, H] detached projections of the client-condensed graph.
        z_syn:    [Kg, H] projections of the local synthetic copy (has grad).
        """
        if z_client.numel() == 0 or z_syn.numel() == 0:
            return z_syn.new_zeros(())
        scale = math.sqrt(z_syn.size(-1))
        logits = z_client.float() @ z_syn.T / scale
        p = torch.softmax(logits, dim=-1)
        frob_sq = (p @ z_syn - z_client.float()).pow(2).sum()
        return frob_sq / max(z_client.size(0), 1)

    def regularization(
        self,
        z_syn: Tensor,
        adj_soft: Tensor,
        lambda_syn_div: float = 0.1,
        lambda_deg: float = 0.05,
    ) -> Tensor:
        """L_reg = λ_synDiv · L_synDiv + λ_deg · L_deg (paper B.4.2)."""
        loss = z_syn.new_zeros(())
        if lambda_syn_div > 0:
            loss = loss + lambda_syn_div * diversity_loss(z_syn)
        if lambda_deg > 0:
            loss = loss + lambda_deg * degree_regularization(adj_soft, self.target_degree)
        return loss

    # ------------------------------------------------------------------
    # Delta export
    # ------------------------------------------------------------------

    def delta(self, state: dict) -> dict:
        """Δ_m = Θ_syn,m^(K_mem) − Θ_syn^(r) against the broadcast state."""
        with torch.no_grad():
            dx = self.x.detach().cpu().float() - state["x"].detach().cpu().float()
            dpge = {
                k: v.detach().cpu().float() - state["pge_state"][k].detach().cpu().float()
                for k, v in self.pge.state_dict().items()
            }
        return {"x": dx, "pge": dpge}


def gradient_matching_loss(
    g_loc: list[Tensor | None], g_syn: list[Tensor | None]
) -> Tensor:
    """L_GM = 1 − cos(g_loc, g_syn) over flattened prompt-module gradients.

    g_loc entries are treated as detached targets. None entries (allow_unused)
    are skipped on both sides in lockstep.
    """
    flat_loc, flat_syn = [], []
    for gl, gs in zip(g_loc, g_syn):
        if gl is None or gs is None:
            continue
        flat_loc.append(gl.detach().float().reshape(-1))
        flat_syn.append(gs.float().reshape(-1))
    if not flat_syn:
        return torch.zeros((), requires_grad=True)
    vl = torch.cat(flat_loc)
    vs = torch.cat(flat_syn)
    cos = (vl * vs).sum() / (vl.norm() * vs.norm() + 1e-8)
    return 1.0 - cos


def aggregate_syn_deltas(entries: list[tuple[dict, int]]) -> dict | None:
    """Server-side Eq. (18): Δ = Σ_m (n_m / Σ_j n_j) · Δ_m.

    entries: list of (delta_dict, num_samples). Returns an averaged delta with
    the same {"x", "pge": {...}} structure, or None if entries is empty.
    """
    entries = [(d, n) for d, n in entries if d is not None and n > 0]
    if not entries:
        return None
    total = float(sum(n for _, n in entries))
    dx = sum(d["x"].float() * (n / total) for d, n in entries)
    pge_keys = set().union(*(d["pge"].keys() for d, _ in entries))
    dpge = {}
    for key in pge_keys:
        parts = [(d["pge"][key].float(), n) for d, n in entries if key in d["pge"]]
        w = float(sum(n for _, n in parts))
        dpge[key] = sum(t * (n / w) for t, n in parts)
    return {"x": dx, "pge": dpge}
