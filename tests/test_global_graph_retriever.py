"""Unit tests for GlobalGraphRetriever and related helpers.

All tests are self-contained (no disk I/O).
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_d_retrieve import (
    GlobalGraphRetriever,
    GlobalRetrievalResult,
    retrieve_global_subgraph,
)


# ---------------------------------------------------------------------------
# Helper to build a synthetic graph for testing
# ---------------------------------------------------------------------------

def _make_graph(K: int, d: int = 8, edges: list[tuple[int, int]] | None = None) -> Data:
    """Build a minimal synthetic graph Data object."""
    torch.manual_seed(42)
    x = torch.randn(K, d) if K > 0 else torch.zeros(0, d)
    node_type = torch.zeros(K, dtype=torch.long)

    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.ones(len(edges))
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, node_type=node_type)


# ---------------------------------------------------------------------------
# Test 1: isolated nodes — no edges — seeds equal top_r highest-cos nodes
# ---------------------------------------------------------------------------

def test_top_r_seeds_no_neighbors():
    """Isolated nodes (no edges) — seeds should be exactly top_r highest-cosine nodes."""
    torch.manual_seed(42)
    K, d, top_r = 10, 8, 3
    graph = _make_graph(K, d, edges=None)

    query = torch.randn(d)
    retriever = GlobalGraphRetriever(graph, top_r=top_r)
    result = retriever.retrieve(query)

    # With no edges, kept_indices == seed_indices (sorted)
    assert result.seed_indices.shape[0] == top_r
    assert result.seed_scores.shape[0] == top_r
    # seed_scores should be in descending order (topk returns sorted desc)
    scores = result.seed_scores
    assert (scores[:-1] >= scores[1:]).all(), "Seed scores should be in descending order"
    # kept_indices sorted ascending
    ki = result.kept_indices
    assert (ki[:-1] < ki[1:]).all() or ki.numel() <= 1, "kept_indices must be sorted ascending"
    # Since no edges, kept == seeds
    assert set(result.seed_indices.tolist()).issubset(set(result.kept_indices.tolist()))


# ---------------------------------------------------------------------------
# Test 2: star graph — seeds include hub — all spokes should be kept
# ---------------------------------------------------------------------------

def test_1hop_expansion():
    """Star graph: hub=0 connected to nodes 1..4. If hub is a seed, all spokes must be kept."""
    torch.manual_seed(42)
    d = 8
    K = 5
    # Hub 0 connected to spokes 1, 2, 3, 4
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    graph = _make_graph(K, d, edges=edges)

    # Craft query that is identical to node 0's embedding (hub will be top seed)
    query = graph.x[0].clone()

    retriever = GlobalGraphRetriever(graph, top_r=1)
    result = retriever.retrieve(query)

    # Hub (node 0) should be a seed
    assert 0 in result.seed_indices.tolist(), "Hub node 0 should be selected as seed"
    # All spokes (1-4) plus hub (0) should be in kept_indices
    kept = set(result.kept_indices.tolist())
    for spoke in [1, 2, 3, 4]:
        assert spoke in kept, f"Spoke {spoke} should be kept via 1-hop expansion"


# ---------------------------------------------------------------------------
# Test 3: max_nodes budget — expansion clamped
# ---------------------------------------------------------------------------

def test_max_nodes_budget():
    """If 1-hop expansion exceeds max_nodes, result must be clamped."""
    torch.manual_seed(42)
    d = 8
    K = 10
    # Fully connect 0 to all others (hub → 9 spokes)
    edges = [(0, i) for i in range(1, K)]
    graph = _make_graph(K, d, edges=edges)

    # Hub will be seed, expansion would keep all K nodes
    query = graph.x[0].clone()
    max_nodes = 4

    result = retrieve_global_subgraph(graph, query, top_r=1, max_nodes=max_nodes)

    assert result.kept_indices.numel() <= max_nodes, (
        f"kept_indices {result.kept_indices.numel()} exceeds max_nodes={max_nodes}"
    )
    assert result.data.x.size(0) <= max_nodes


# ---------------------------------------------------------------------------
# Test 4: feature preservation — subgraph rows match original x rows
# ---------------------------------------------------------------------------

def test_feature_preservation():
    """kept_indices[i] → original x row must equal output data.x[i]."""
    torch.manual_seed(42)
    K, d, top_r = 20, 16, 5
    graph = _make_graph(K, d, edges=[(i, (i + 1) % K) for i in range(K)])

    query = torch.randn(d)
    retriever = GlobalGraphRetriever(graph, top_r=top_r)
    result = retriever.retrieve(query)

    for local_i, global_i in enumerate(result.kept_indices.tolist()):
        assert torch.allclose(result.data.x[local_i], graph.x[global_i]), (
            f"Feature mismatch at local={local_i}, global={global_i}"
        )


# ---------------------------------------------------------------------------
# Test 5: edge index values are valid local indices
# ---------------------------------------------------------------------------

def test_local_edge_indices():
    """After relabeling, all edge_index values must be < len(kept_indices)."""
    torch.manual_seed(42)
    K, d, top_r = 15, 8, 4
    edges = [(i, j) for i in range(K) for j in range(i + 1, min(i + 3, K))]
    graph = _make_graph(K, d, edges=edges)

    query = torch.randn(d)
    retriever = GlobalGraphRetriever(graph, top_r=top_r)
    result = retriever.retrieve(query)

    M = result.kept_indices.numel()
    if result.data.edge_index.numel() > 0:
        assert result.data.edge_index.max().item() < M, (
            f"edge_index max {result.data.edge_index.max().item()} >= M={M}"
        )
        assert result.data.edge_index.min().item() >= 0, "edge_index has negative values"


# ---------------------------------------------------------------------------
# Test 6: batch retrieval returns list of correct length
# ---------------------------------------------------------------------------

def test_batch_retrieval_length():
    """retrieve_batch([q1, q2]) must return a list of length 2."""
    torch.manual_seed(42)
    K, d = 10, 8
    graph = _make_graph(K, d)

    query_embeddings = torch.randn(2, d)
    retriever = GlobalGraphRetriever(graph, top_r=3)
    results = retriever.retrieve_batch(query_embeddings)

    assert isinstance(results, list), "retrieve_batch should return a list"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for r in results:
        assert isinstance(r, GlobalRetrievalResult), "Each result should be a GlobalRetrievalResult"


# ---------------------------------------------------------------------------
# Test 7: empty graph — no exception, result has 0 nodes
# ---------------------------------------------------------------------------

def test_empty_graph():
    """K_g=0 Data() must not raise and must return 0-node result."""
    torch.manual_seed(42)
    # Build empty graph
    empty_graph = Data(
        x=torch.zeros(0, 8),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_weight=torch.zeros(0),
        node_type=torch.zeros(0, dtype=torch.long),
    )
    query = torch.randn(8)

    result = retrieve_global_subgraph(empty_graph, query, top_r=5)

    assert result.kept_indices.numel() == 0, "Empty graph: kept_indices should be empty"
    assert result.seed_indices.numel() == 0, "Empty graph: seed_indices should be empty"
    assert result.seed_scores.numel() == 0, "Empty graph: seed_scores should be empty"
    assert result.data.x.size(0) == 0, "Empty graph: data.x should have 0 rows"


# ---------------------------------------------------------------------------
# Test 8: top_r > K_g — seeds silently clamped to K_g
# ---------------------------------------------------------------------------

def test_top_r_gt_kg_clamp():
    """top_r=100 with K_g=5 should silently clamp seeds to 5."""
    torch.manual_seed(42)
    K, d = 5, 8
    graph = _make_graph(K, d)

    query = torch.randn(d)
    result = retrieve_global_subgraph(graph, query, top_r=100)

    # Should have at most K seeds
    assert result.seed_indices.numel() <= K, (
        f"seed_indices.numel()={result.seed_indices.numel()} should be <= K_g={K}"
    )
    assert result.seed_scores.numel() <= K
    # All seed indices should be valid
    assert result.seed_indices.max().item() < K
