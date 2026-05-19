"""Global graph retriever for the FedCondGraphRAG server-side synthetic graph.

Retrieves a subgraph of the server's synthetic global graph for a query by:
  1. Normalizing query and node embeddings for cosine similarity.
  2. Picking TopR seed nodes by cosine similarity.
  3. Expanding seeds 1-hop along edge_index (treated as undirected).
  4. Taking the induced subgraph with relabeled node indices.
  5. Optionally clamping to max_nodes by keeping highest-scoring nodes first.

The synthetic graph is produced by FedCondQAServer.export_synthetic_graph():
    Data(x [K_g,d], edge_index [2,E], edge_weight [E], node_type [K_g])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


@dataclass(frozen=True)
class GlobalRetrievalResult:
    """Result of a global graph retrieval operation.

    Attributes:
        data: PyG Data with fields x, edge_index, edge_weight, node_type.
              Node indices are relabeled 0..M-1 where M = len(kept_indices).
        seed_indices: shape [top_r] — original X_global indices of seed nodes.
        kept_indices: shape [M] — original X_global indices of all kept nodes
                      (seeds + 1-hop neighbours), sorted ascending.
        seed_scores: shape [top_r] — cosine similarity scores for each seed.
    """

    data: Data
    seed_indices: Tensor
    kept_indices: Tensor
    seed_scores: Tensor


class GlobalGraphRetriever:
    """Retrieves a subgraph of the server's synthetic global graph for a query.

    The synthetic graph is produced by FedCondQAServer.export_synthetic_graph():
        Data(x [K_g,d], edge_index [2,E], edge_weight [E], node_type [K_g])

    Retrieval:
        1. Normalize query and X_global rows.
        2. Pick TopR seeds by cosine similarity.
        3. Expand seeds 1-hop along edge_index (undirected).
        4. Take induced subgraph with relabeled nodes.
        5. Optionally clamp to max_nodes by keeping highest-scoring seeds first.

    Adjacency lists and normalized embeddings are precomputed in __init__ so
    repeated retrieve() calls are O(top_r × avg_degree) instead of O(E).
    """

    def __init__(
        self,
        synthetic_graph: Data,
        *,
        top_r: int = 16,
        max_nodes: int | None = None,
    ) -> None:
        self._graph = synthetic_graph
        self._top_r = top_r
        self._max_nodes = max_nodes

        # Precompute normalized embeddings once — avoids per-query F.normalize
        x = synthetic_graph.x
        if x is not None and x.numel() > 0:
            self._x_norm = F.normalize(x.float(), dim=-1)  # [K, d]
        else:
            self._x_norm = None

        # Precompute adjacency lists for O(deg) 1-hop expansion
        K = x.size(0) if x is not None and x.numel() > 0 else 0
        adj: list[list[int]] = [[] for _ in range(K)]
        edge_index = synthetic_graph.edge_index
        if edge_index is not None and edge_index.numel() > 0:
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            for u, v in zip(src_list, dst_list):
                adj[u].append(v)
                adj[v].append(u)
        self._adj = adj

    def retrieve(self, query_embedding: Tensor) -> GlobalRetrievalResult:
        """Retrieve a subgraph for a single query embedding.

        Args:
            query_embedding: shape [d] or [1, d] — the query vector.

        Returns:
            GlobalRetrievalResult with the extracted subgraph and metadata.
        """
        return _retrieve_impl(
            self._graph,
            query_embedding,
            top_r=self._top_r,
            max_nodes=self._max_nodes,
            x_norm=self._x_norm,
            adj=self._adj,
        )

    def retrieve_batch(self, query_embeddings: Tensor) -> list[GlobalRetrievalResult]:
        """Retrieve subgraphs for a batch of query embeddings.

        Args:
            query_embeddings: shape [B, d] — batch of query vectors.

        Returns:
            List of length B, one GlobalRetrievalResult per query.
        """
        return [self.retrieve(query_embeddings[i]) for i in range(query_embeddings.size(0))]


def retrieve_global_subgraph(
    synthetic_graph: Data,
    query_embedding: Tensor,
    *,
    top_r: int = 16,
    max_nodes: int | None = None,
) -> GlobalRetrievalResult:
    """Convenience wrapper — one-shot call without constructing a retriever.

    Args:
        synthetic_graph: PyG Data from FedCondQAServer.export_synthetic_graph().
        query_embedding: shape [d] or [1, d].
        top_r: number of seed nodes to select by cosine similarity.
        max_nodes: if set, clamp the kept set to this many nodes.

    Returns:
        GlobalRetrievalResult with the extracted subgraph and metadata.
    """
    return _retrieve_impl(
        synthetic_graph,
        query_embedding,
        top_r=top_r,
        max_nodes=max_nodes,
        x_norm=None,
        adj=None,
    )


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _retrieve_impl(
    graph: Data,
    query_embedding: Tensor,
    *,
    top_r: int,
    max_nodes: int | None,
    x_norm: Tensor | None = None,
    adj: list[list[int]] | None = None,
) -> GlobalRetrievalResult:
    """Core retrieval logic shared by class method and convenience function."""
    x: Tensor = graph.x  # [K_g, d]
    edge_index: Tensor = graph.edge_index  # [2, E]
    node_type: Tensor = graph.node_type  # [K_g]

    # Handle missing edge_weight gracefully
    if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
        edge_weight: Tensor = graph.edge_weight  # [E]
    else:
        E = edge_index.size(1) if edge_index is not None and edge_index.numel() > 0 else 0
        edge_weight = torch.ones(E, dtype=torch.float32)

    K_g = x.size(0) if x is not None and x.numel() > 0 else 0

    # Handle empty graph
    if K_g == 0:
        empty_data = Data(
            x=torch.zeros(0, 0),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_weight=torch.zeros(0),
            node_type=torch.zeros(0, dtype=torch.long),
        )
        return GlobalRetrievalResult(
            data=empty_data,
            seed_indices=torch.zeros(0, dtype=torch.long),
            kept_indices=torch.zeros(0, dtype=torch.long),
            seed_scores=torch.zeros(0),
        )

    # Clamp top_r to K_g
    top_r = min(top_r, K_g)

    # Step 1: Normalize for cosine similarity (use precomputed if available)
    q = F.normalize(query_embedding.flatten().unsqueeze(0).float(), dim=-1)  # [1, d]
    if x_norm is None:
        x_norm = F.normalize(x.float(), dim=-1)  # [K_g, d]

    # Step 2: Cosine similarity scores for ALL nodes
    scores = (x_norm @ q.T).squeeze(-1)  # [K_g]

    # Step 3: Pick top_r seeds
    seed_scores_vals, seed_idx = torch.topk(scores, top_r)
    # seed_idx: [top_r] — original indices, in descending score order

    # Step 4: 1-hop expansion using precomputed adjacency (O(deg)) or tensor scan (O(E))
    seed_set = set(seed_idx.tolist())
    if adj is not None:
        # Fast path: adjacency list lookup — O(top_r × avg_degree)
        neighbor_set: set[int] = set()
        for s in seed_set:
            neighbor_set.update(adj[s])
        kept_set = seed_set | neighbor_set
    elif edge_index.numel() > 0:
        # Slow fallback: tensor scan for each seed
        neighbor_set = set()
        src = edge_index[0]
        dst = edge_index[1]
        for s in seed_set:
            neighbor_set.update(dst[src == s].tolist())
            neighbor_set.update(src[dst == s].tolist())
        kept_set = seed_set | neighbor_set
    else:
        kept_set = seed_set

    dev = x.device  # keep all index tensors on the same device as the graph

    # Step 5: Optionally clamp to max_nodes
    if max_nodes is not None and len(kept_set) > max_nodes:
        # Keep the max_nodes nodes with the highest cosine scores
        all_kept = torch.tensor(sorted(kept_set), dtype=torch.long, device=dev)
        kept_scores = scores[all_kept]
        _, top_idx = torch.topk(kept_scores, min(max_nodes, len(kept_set)))
        kept_set = set(all_kept[top_idx].tolist())

        # Re-derive seed_indices as intersection of kept with original top seeds
        # (preserve original score order)
        new_seeds_ordered = [i for i in seed_idx.tolist() if i in kept_set]
        seed_idx = torch.tensor(new_seeds_ordered, dtype=torch.long, device=dev)
        seed_scores_vals = scores[seed_idx]

    # Build sorted kept_indices (must be on same device as edge_index)
    kept_indices = torch.sort(torch.tensor(sorted(kept_set), dtype=torch.long, device=dev)).values

    # Step 6: Extract induced subgraph with relabeled nodes
    if edge_index.numel() > 0:
        sub_edge_index, sub_edge_weight = subgraph(
            kept_indices,
            edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=K_g,
        )
    else:
        sub_edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
        sub_edge_weight = torch.zeros(0, device=dev)

    sub_x = x[kept_indices]
    sub_node_type = node_type[kept_indices]

    out_data = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        node_type=sub_node_type,
    )

    return GlobalRetrievalResult(
        data=out_data,
        seed_indices=seed_idx,
        kept_indices=kept_indices,
        seed_scores=seed_scores_vals,
    )
