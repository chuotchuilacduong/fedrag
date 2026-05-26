"""Stage B client-side graph condensation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover - fallback for import-only environments.
    Data = None

from .graph_text_fusion import GraphTextFusion
from .anchor_node_selector import AnchorSelection, AnchorSelectorConfig, select_anchor_nodes
from .neighbor_gating import NodeEvidenceTrace, hierarchical_text_condensation
from .node_text_embedder import NodeTextBank, build_text_bank
from .topology_reconstruction import TopologyResult, knn_topology, self_expressive_topology


@dataclass(frozen=True)
class ClientCondensationConfig:
    motif: AnchorSelectorConfig = field(default_factory=AnchorSelectorConfig)
    text_budgets: tuple[int, int, int] = (1, 3, 2)
    chunk_budget: int = 8
    hop_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    topology_method: str = "knn"
    knn_k: int = 8
    prior_weight: float = 0.0
    self_expr_candidate_size: int = 16
    self_expr_iterations: int = 50
    self_expr_step_size: float = 1e-2
    self_expr_alpha: float = 8.0
    self_expr_beta: float = 5.0
    out_dim: int | None = None
    preserve_sep_topology: bool = True


@dataclass
class ClientCondensedGraph:
    """Numeric-only upload object for one client."""

    x: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    node_type: Tensor
    hashed_local_ids: Tensor | None = None

    def to_pyg_data(self):
        if Data is None:
            raise RuntimeError("torch_geometric is required to create a PyG Data object")
        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            node_type=self.node_type,
        )
        if self.hashed_local_ids is not None:
            data.hashed_local_ids = self.hashed_local_ids
        return data


@dataclass
class CondensationArtifacts:
    """Local-only artifacts that must not be uploaded."""

    motif_selection: AnchorSelection
    contexts: dict[int, Tensor]
    evidence_traces: dict[int, NodeEvidenceTrace]
    fusion_gate: Tensor
    topology: TopologyResult


class ClientCondensor(nn.Module):
    """Condense one local Tri-Graph into a numeric anchor graph C_m."""

    def __init__(self, graph_dim: int, text_dim: int, config: ClientCondensationConfig | None = None):
        super().__init__()
        self.config = config or ClientCondensationConfig()
        out_dim = self.config.out_dim or graph_dim
        self.W_q = nn.Linear(graph_dim, text_dim)
        self.W_k = nn.Linear(text_dim, text_dim)
        self.W_s = nn.Linear(graph_dim, text_dim)
        self.fusion = GraphTextFusion(graph_dim=graph_dim, text_dim=text_dim, out_dim=out_dim)

    def forward(
        self,
        tri_graph,
        *,
        text_bank: NodeTextBank,
        graph_embeddings: Tensor | None = None,
        motif_selection: AnchorSelection | None = None,
        return_artifacts: bool = False,
    ):
        cfg = self.config
        if graph_embeddings is None:
            graph_embeddings = tri_graph.x
        graph_embeddings = graph_embeddings.float()
        node_text_embeddings = text_bank.node_embeddings.to(
            device=graph_embeddings.device, dtype=graph_embeddings.dtype
        )
        chunk_embeddings = [
            chunks.to(device=graph_embeddings.device, dtype=graph_embeddings.dtype)
            for chunks in text_bank.chunk_embeddings
        ]

        motif = motif_selection or select_anchor_nodes(tri_graph, config=cfg.motif, x=graph_embeddings)
        core_ids = motif.core_node_ids.to(device=graph_embeddings.device)
        core_graph_embeddings = graph_embeddings[core_ids]
        core_node_type = tri_graph.node_type[core_ids].long().to(device=graph_embeddings.device)

        hop_weights = torch.tensor(cfg.hop_weights, dtype=graph_embeddings.dtype, device=graph_embeddings.device)
        t_tilde, contexts, traces = hierarchical_text_condensation(
            core_node_ids=core_ids.detach().cpu().tolist(),
            edge_index=tri_graph.edge_index.to(device=graph_embeddings.device),
            graph_embeddings=graph_embeddings,
            node_text_embeddings=node_text_embeddings,
            chunk_embeddings=chunk_embeddings,
            W_q=self.W_q,
            W_k=self.W_k,
            W_s=self.W_s,
            hop_weights=hop_weights,
            budgets=cfg.text_budgets,
            chunk_budget=cfg.chunk_budget,
        )

        x_fused, gate = self.fusion(core_graph_embeddings, t_tilde)
        topology_method = cfg.topology_method.lower()
        if topology_method == "knn":
            topology = knn_topology(
                x_fused,
                node_type=core_node_type,
                text_embeddings=t_tilde,
                k=cfg.knn_k,
                prior_weight=cfg.prior_weight,
                preserve_sep=cfg.preserve_sep_topology,
            )
        elif topology_method in {"self_expression", "self-expressive", "self_expr"}:
            topology = self_expressive_topology(
                x_fused,
                t_tilde,
                node_type=core_node_type,
                alpha_recon=cfg.self_expr_alpha,
                beta_l1=cfg.self_expr_beta,
                candidate_size=cfg.self_expr_candidate_size,
                iterations=cfg.self_expr_iterations,
                step_size=cfg.self_expr_step_size,
                final_k=cfg.knn_k,
                preserve_sep=cfg.preserve_sep_topology,
            )
        else:
            raise ValueError(f"Unsupported topology_method: {cfg.topology_method}")

        condensed = ClientCondensedGraph(
            x=x_fused,
            edge_index=topology.edge_index,
            edge_weight=topology.edge_weight,
            node_type=core_node_type,
            hashed_local_ids=_hash_local_ids(core_ids),
        )
        if not return_artifacts:
            return condensed
        artifacts = CondensationArtifacts(
            motif_selection=motif,
            contexts=contexts,
            evidence_traces=traces,
            fusion_gate=gate.detach(),
            topology=topology,
        )
        return condensed, artifacts


def _hash_local_ids(node_ids: Tensor) -> Tensor:
    """Cheap stable numeric ids for audit without exposing raw strings."""

    ids = node_ids.detach().cpu().long()
    return ((ids * 1_103_515_245 + 12_345) % 2_147_483_647).to(device=node_ids.device)


def condense_client_graph(
    tri_graph,
    *,
    node_texts: Sequence[str] | None = None,
    text_bank: NodeTextBank | None = None,
    graph_embeddings: Tensor | None = None,
    config: ClientCondensationConfig | None = None,
    return_artifacts: bool = False,
):
    """Convenience wrapper for one-shot Stage B condensation."""

    if graph_embeddings is None:
        graph_embeddings = tri_graph.x
    if text_bank is None:
        if node_texts is None:
            node_texts = [f"node {i}" for i in range(int(graph_embeddings.size(0)))]
        text_bank = build_text_bank(node_texts, dim=int(graph_embeddings.size(1)), device=graph_embeddings.device)

    condensor = ClientCondensor(
        graph_dim=int(graph_embeddings.size(1)),
        text_dim=int(text_bank.node_embeddings.size(1)),
        config=config,
    ).to(graph_embeddings.device)
    return condensor(
        tri_graph,
        text_bank=text_bank,
        graph_embeddings=graph_embeddings,
        return_artifacts=return_artifacts,
    )
