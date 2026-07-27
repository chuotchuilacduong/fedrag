"""Stage B retrieval-preserving refinement tests (paper B.3.5).

Verifies the condensation objective L_cond = L_ret(KL) + λ_rep·L_rep +
λ_div·L_div on toy tri-graphs: loss decreases, features move, topology is
rebuilt, and the no-query fallback (L_ret skipped) still works.
"""

import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_b_condense import (
    RetrievalRefineConfig,
    refine_condensed_graph,
)

DIM = 24


def make_tri_graph(n_per_type: int = 12, seed: int = 0) -> Data:
    torch.manual_seed(seed)
    n = 3 * n_per_type
    node_type = torch.arange(3).repeat_interleave(n_per_type)  # 0/1/2 blocks
    x = torch.randn(n, DIM)
    edge_index = torch.randint(0, n, (2, 6 * n))
    return Data(x=x, edge_index=edge_index, node_type=node_type)


def make_condensed(k_per_type: int = 4, seed: int = 1) -> Data:
    torch.manual_seed(seed)
    k = 3 * k_per_type
    node_type = torch.arange(3).repeat_interleave(k_per_type)
    x = torch.randn(k, DIM)
    edge_index = torch.randint(0, k, (2, 4 * k))
    edge_weight = torch.rand(4 * k)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, node_type=node_type)


CFG = RetrievalRefineConfig(
    iterations=60,
    lr=5e-2,
    rep_sample_nodes=36,
    rep_hidden_dim=16,
    max_queries=8,
    knn_k=4,
    preserve_sep_topology=False,
)


def test_refine_reduces_l_cond_with_queries():
    tri = make_tri_graph()
    cond = make_condensed()
    queries = torch.randn(8, DIM)
    refined, hist = refine_condensed_graph(
        tri, cond, query_embeddings=queries, config=CFG
    )
    assert hist["total"][-1] < hist["total"][0]
    # KL retrieval term is active and finite
    assert all(v == v for v in hist["ret"])  # no NaNs
    assert any(v > 0 for v in hist["ret"])
    # Features actually moved; shapes preserved
    assert refined.x.shape == cond.x.shape
    assert (refined.x - cond.x).abs().sum() > 0
    assert torch.equal(refined.node_type, cond.node_type)


def test_refine_without_queries_skips_l_ret():
    tri = make_tri_graph(seed=2)
    cond = make_condensed(seed=3)
    refined, hist = refine_condensed_graph(
        tri, cond, query_embeddings=None, config=CFG
    )
    assert all(v == 0.0 for v in hist["ret"])
    assert hist["total"][-1] < hist["total"][0]
    assert refined.x.shape == cond.x.shape


def test_refine_rebuilds_topology_from_refined_features():
    tri = make_tri_graph(seed=4)
    cond = make_condensed(seed=5)
    refined, _ = refine_condensed_graph(
        tri, cond, query_embeddings=torch.randn(4, DIM), config=CFG
    )
    # kNN topology: symmetric edge list with weights, at least K edges
    assert refined.edge_index.size(0) == 2
    assert refined.edge_weight is not None
    assert refined.edge_index.size(1) == refined.edge_weight.size(0)
    assert refined.edge_index.size(1) > 0


def test_refine_keep_topology_when_disabled():
    tri = make_tri_graph(seed=6)
    cond = make_condensed(seed=7)
    cfg = RetrievalRefineConfig(
        iterations=10, lr=1e-2, rep_sample_nodes=36, rep_hidden_dim=16,
        rebuild_topology=False, preserve_sep_topology=False,
    )
    refined, _ = refine_condensed_graph(
        tri, cond, query_embeddings=None, config=cfg
    )
    assert torch.equal(refined.edge_index, cond.edge_index)


def test_kl_term_pulls_condensed_passages_toward_full_retrieval():
    """With only L_ret active, condensed passage features should drift toward
    the full passage cluster the queries retrieve from."""
    torch.manual_seed(10)
    tri = make_tri_graph(seed=10)
    # Make full passage features clustered around a direction; queries aligned
    direction = torch.randn(DIM)
    p_mask = tri.node_type == 2
    tri.x[p_mask] = direction + 0.05 * torch.randn(int(p_mask.sum()), DIM)
    queries = direction.unsqueeze(0) + 0.05 * torch.randn(6, DIM)

    cond = make_condensed(seed=11)
    cfg = RetrievalRefineConfig(
        iterations=150, lr=5e-2, lambda_rep=0.0, lambda_div=0.0,
        rep_sample_nodes=8, rep_hidden_dim=16, preserve_sep_topology=False,
    )
    refined, hist = refine_condensed_graph(
        tri, cond, query_embeddings=queries, config=cfg
    )
    # Proper KL: non-negative throughout and substantially reduced
    assert all(v >= -1e-6 for v in hist["ret"])
    assert hist["ret"][-1] < 0.5 * hist["ret"][0]
    assert (refined.x - cond.x).abs().sum() > 0
