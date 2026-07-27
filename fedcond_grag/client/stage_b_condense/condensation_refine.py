"""Stage B — retrieval-preserving condensation refinement (paper B.3.5).

After the constructive initialization (anchor selection B.3.1, motif expansion
B.3.2, text condensation B.3.3, fusion + topology B.3.4), the condensed node
features X̃ are refined by minimizing the client-side condensation objective

    L_cond = L_ret + λ_rep · L_rep + λ_div · L_div            (paper Eq. 3)

where
    L_ret = E_{q∼Q_m} KL[ r_Gm(·|q) ‖ r̂_G̃m(·|q) ]           (paper Eq. 2)
        with r̂ the condensed passage-retrieval distribution lifted back to
        the original passage space through the soft coverage map
        A_m(p, p̃) = softmax_p̃( cos(x_p, x̃_p̃) / τ_a ),
    L_rep = mean_{v∈B_m} ‖ h_v − Σ_ũ Π_m(v, ũ) h̃_ũ ‖²        (paper B.3.5)
        with H = f_GNN(G_m; θ⁰), H̃ = f_GNN(G̃_m; θ⁰) under a frozen θ⁰,
    L_div = (1/|Ṽ|²) Σ_{ũ≠ṽ} max(0, cos(x̃_ũ, x̃_ṽ) − δ).

Only X̃ is optimized. Queries are the client's own local training questions
embedded with the same frozen sentence encoder that produced node features —
they never leave the client. When no queries are available (e.g. a bootstrap
run without QA data) L_ret is skipped and only L_rep + L_div refine X̃.
After refinement the condensed topology is rebuilt from the refined features
(similarity kNN, paper B.3.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fedcond_grag.client.stage_b_condense.topology_reconstruction import knn_topology

_PASSAGE_TYPE = 2  # node_type convention: 0=entity, 1=sentence, 2=passage
_EPS = 1e-8


@dataclass(frozen=True)
class RetrievalRefineConfig:
    iterations: int = 100
    lr: float = 1e-2
    lambda_rep: float = 1.0       # λ_rep
    lambda_div: float = 0.1       # λ_div
    delta_margin: float = 0.5     # δ in L_div
    tau_ret: float = 0.1          # retrieval softmax temperature
    tau_cov: float = 0.1          # τ_a soft coverage map temperature
    max_queries: int = 256        # cap on |Q_m| used for L_ret
    rep_sample_nodes: int = 1024  # |B_m| sampled full-graph nodes per step
    rep_hidden_dim: int = 128     # frozen θ⁰ GNN width
    rep_num_layers: int = 2
    seed: int = 0
    rebuild_topology: bool = True
    knn_k: int = 8
    preserve_sep_topology: bool = True


def _frozen_rep_gnn(in_dim: int, config: RetrievalRefineConfig, device) -> nn.Module:
    """f_GNN(·; θ⁰): a fixed randomly-initialized GCN shared by both graphs."""
    from fedcond_grag.model.gnn import GCN

    generator_state = torch.random.get_rng_state()
    torch.manual_seed(config.seed)
    gnn = GCN(
        in_channels=in_dim,
        hidden_channels=config.rep_hidden_dim,
        out_channels=config.rep_hidden_dim,
        num_layers=max(2, config.rep_num_layers),
        dropout=0.0,
    ).to(device)
    torch.random.set_rng_state(generator_state)
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    return gnn


def _retrieval_distribution(queries: Tensor, passages: Tensor, tau: float) -> Tensor:
    """softmax_p( cos(e_q, x_p) / τ ) as [Q, P]."""
    q = F.normalize(queries.float(), dim=-1)
    p = F.normalize(passages.float(), dim=-1)
    return torch.softmax(q @ p.T / max(tau, _EPS), dim=-1)


def refine_condensed_graph(
    tri_graph,
    condensed,
    *,
    query_embeddings: Tensor | None = None,
    config: RetrievalRefineConfig | None = None,
):
    """Refine condensed features by minimizing L_cond; returns (Data, history).

    tri_graph / condensed: objects with x, edge_index, node_type (and the
    condensed one edge_weight) — PyG ``Data`` works for both. The full private
    graph is used read-only; only the condensed features are optimized.
    """
    from torch_geometric.data import Data

    cfg = config or RetrievalRefineConfig()
    device = condensed.x.device
    x_tilde = nn.Parameter(condensed.x.detach().clone().float())
    optimizer = torch.optim.Adam([x_tilde], lr=cfg.lr)

    full_x = tri_graph.x.detach().float().to(device)
    full_type = tri_graph.node_type.detach().long().to(device)
    cond_type = condensed.node_type.detach().long().to(device)
    cond_edge_index = condensed.edge_index.detach().to(device)
    cond_edge_weight = getattr(condensed, "edge_weight", None)
    if cond_edge_weight is not None:
        cond_edge_weight = cond_edge_weight.detach().float().to(device)

    # --- L_ret targets (fixed): full-graph passage retrieval distribution ---
    full_p_mask = full_type == _PASSAGE_TYPE
    cond_p_mask = cond_type == _PASSAGE_TYPE
    use_ret = (
        query_embeddings is not None
        and query_embeddings.numel() > 0
        and bool(full_p_mask.any())
        and bool(cond_p_mask.any())
    )
    if use_ret:
        queries = query_embeddings.detach().float().to(device)[: cfg.max_queries]
        x_p_full = full_x[full_p_mask]                                   # [P, d]
        r_full = _retrieval_distribution(queries, x_p_full, cfg.tau_ret).detach()
        x_p_full_norm = F.normalize(x_p_full, dim=-1)
        log_r_full = torch.log(r_full.clamp_min(_EPS))

    # --- L_rep targets (fixed): full-graph node representations under θ⁰ ---
    rep_gnn = _frozen_rep_gnn(full_x.size(1), cfg, device)
    with torch.no_grad():
        h_full, _ = rep_gnn(full_x, tri_graph.edge_index.to(device), None)
        h_full = h_full.detach()
    rep_scale = float(h_full.size(1)) ** 0.5

    history: dict[str, list[float]] = {"total": [], "ret": [], "rep": [], "div": []}
    num_full = full_x.size(0)
    for _ in range(max(1, cfg.iterations)):
        # L_ret — KL between full retrieval and the lifted condensed retrieval
        loss_ret = x_tilde.new_zeros(())
        if use_ret:
            x_p_cond = x_tilde[cond_p_mask]                              # [P̃, d]
            r_cond = _retrieval_distribution(queries, x_p_cond, cfg.tau_ret)
            x_p_cond_norm = F.normalize(x_p_cond, dim=-1)
            coverage = torch.softmax(                                    # A_m [P, P̃]
                x_p_full_norm @ x_p_cond_norm.T / max(cfg.tau_cov, _EPS), dim=-1
            )
            r_hat = r_cond @ coverage.T                                  # [Q, P]
            # Renormalize the lifted scores into a proper distribution over p
            # — without this the KL can go negative and the optimizer exploits
            # the unnormalized mass instead of matching the retrieval shape.
            r_hat = r_hat / r_hat.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            loss_ret = (
                (r_full * (log_r_full - torch.log(r_hat.clamp_min(_EPS))))
                .sum(dim=-1)
                .mean()
            )

        # L_rep — soft-assignment reconstruction of sampled full-graph reprs
        h_tilde, _ = rep_gnn(x_tilde, cond_edge_index, cond_edge_weight)
        n_sample = min(cfg.rep_sample_nodes, num_full)
        idx = torch.randperm(num_full, device=device)[:n_sample]
        h_v = h_full[idx]                                                # [B, H]
        assign = torch.softmax(h_v @ h_tilde.T / rep_scale, dim=-1)      # Π_m
        loss_rep = (assign @ h_tilde - h_v).pow(2).sum(dim=-1).mean()

        # L_div — hinge on pairwise cosine similarity of condensed features
        x_norm = F.normalize(x_tilde, dim=-1)
        gram = x_norm @ x_norm.T
        k = x_tilde.size(0)
        off_diag = ~torch.eye(k, dtype=torch.bool, device=device)
        loss_div = F.relu(gram[off_diag] - cfg.delta_margin).sum() / max(k * k, 1)

        total = loss_ret + cfg.lambda_rep * loss_rep + cfg.lambda_div * loss_div
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        history["total"].append(float(total.detach().cpu()))
        history["ret"].append(float(loss_ret.detach().cpu()))
        history["rep"].append(float(loss_rep.detach().cpu()))
        history["div"].append(float(loss_div.detach().cpu()))

    x_refined = x_tilde.detach()

    # --- Topology reconstruction from refined features (paper B.3.4) ---
    if cfg.rebuild_topology:
        topology = knn_topology(
            x_refined,
            node_type=cond_type,
            k=cfg.knn_k,
            preserve_sep=cfg.preserve_sep_topology,
        )
        edge_index, edge_weight = topology.edge_index, topology.edge_weight
    else:
        edge_index, edge_weight = cond_edge_index, cond_edge_weight

    refined = Data(
        x=x_refined,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_type=cond_type,
    )
    if getattr(condensed, "hashed_local_ids", None) is not None:
        refined.hashed_local_ids = condensed.hashed_local_ids
    return refined, history
