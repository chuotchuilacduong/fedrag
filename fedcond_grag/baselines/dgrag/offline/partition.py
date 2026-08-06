"""Step ②: Topology-based Graph Partitioning (spec §3.2).

Uses python-igraph's community_leiden (igraph ≥ 0.10 built-in) to partition
the knowledge graph. Runs a resolution sweep to find the cut whose mean
community size is closest to TARGET_ENTITIES_PER_SUBGRAPH (40). Then applies
merge-small / split-large post-processing per the spec.

python-igraph 1.0.0 (already in requirements.txt) has community_leiden.
leidenalg is NOT required here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import igraph as ig
import networkx as nx

from fedcond_grag.baselines.dgrag.data_types import Entity, Relation, Subgraph

_log = logging.getLogger("dgrag.partition")


# ---------------------------------------------------------------------------
# NetworkX -> igraph conversion
# ---------------------------------------------------------------------------

def _nx_to_igraph(G: nx.Graph) -> tuple[ig.Graph, list[str]]:
    nodes = list(G.nodes())
    if not nodes:
        return ig.Graph(n=0), []
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges.append((node_idx[u], node_idx[v]))
        weights.append(float(data.get("weight", 1.0)))
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.vs["name"] = nodes
    ig_graph.es["weight"] = weights if weights else [1.0] * len(edges)
    return ig_graph, nodes


# ---------------------------------------------------------------------------
# Resolution sweep: find cut nearest to target mean size
# ---------------------------------------------------------------------------

def _run_leiden_at_resolution(
    ig_graph: ig.Graph,
    resolution: float,
    n_iterations: int,
    seed: int,
) -> Optional[ig.VertexClustering]:
    try:
        return ig_graph.community_leiden(
            objective_function="modularity",
            weights="weight" if ig_graph.ecount() > 0 else None,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed,
        )
    except Exception as exc:
        _log.debug("Leiden at res=%.3f failed: %s", resolution, exc)
        return None


def _mean_size(partition: ig.VertexClustering) -> float:
    sizes = partition.sizes()
    if not sizes:
        return 0.0
    return sum(sizes) / len(sizes)


# ---------------------------------------------------------------------------
# Merge small / split large post-processing
# ---------------------------------------------------------------------------

def _merge_small_communities(
    communities: list[list[str]],
    G: nx.Graph,
    min_size: int,
) -> list[list[str]]:
    """Merge communities smaller than min_size into their most-connected neighbor."""
    changed = True
    while changed:
        changed = False
        small_idx = [i for i, c in enumerate(communities) if len(c) < min_size]
        if not small_idx:
            break
        node_to_community = {n: i for i, c in enumerate(communities) for n in c}
        for si in small_idx:
            if si >= len(communities):
                continue
            small = communities[si]
            if len(small) >= min_size:
                continue
            # Count inter-community edges to each neighbor community
            neighbor_weight: dict[int, float] = {}
            for node in small:
                for nbr in G.neighbors(node):
                    ci = node_to_community.get(nbr, -1)
                    if ci != si and ci >= 0:
                        neighbor_weight[ci] = neighbor_weight.get(ci, 0.0) + G[node][nbr].get("weight", 1.0)
            if not neighbor_weight:
                continue
            target = max(neighbor_weight, key=neighbor_weight.__getitem__)
            communities[target] = communities[target] + small
            communities[si] = []
            node_to_community = {n: i for i, c in enumerate(communities) for n in c}
            changed = True
        communities = [c for c in communities if c]
    return communities


def _split_large_communities(
    communities: list[list[str]],
    G: nx.Graph,
    max_size: int,
    n_iterations: int,
    seed: int,
) -> list[list[str]]:
    """Recursively Leiden-split communities larger than max_size."""
    result: list[list[str]] = []
    for community in communities:
        if len(community) <= max_size:
            result.append(community)
            continue
        subgraph = G.subgraph(community)
        sub_ig, sub_nodes = _nx_to_igraph(subgraph)
        if sub_ig.vcount() == 0:
            result.append(community)
            continue
        # try a higher resolution to force finer split
        for res in [2.0, 5.0, 10.0, 50.0]:
            part = _run_leiden_at_resolution(sub_ig, res, n_iterations, seed)
            if part is None:
                continue
            if len(part) > 1:
                sub_communities = [[sub_nodes[i] for i in c] for c in part]
                # recurse if still too large
                sub_communities = _split_large_communities(
                    sub_communities, subgraph, max_size, n_iterations, seed
                )
                result.extend(sub_communities)
                break
        else:
            result.append(community)  # can't split further
    return result


# ---------------------------------------------------------------------------
# Linearize subgraph text (spec §3.2)
# ---------------------------------------------------------------------------

def linearize(
    node_names: list[str],
    entities: dict[str, "Entity"],
    relations: dict[tuple[str, str], "Relation"],
) -> str:
    """Format a subgraph as plain text for the SLM summarizer."""
    lines = ["Entities:"]
    for name in node_names:
        key = name.strip().upper()
        e = entities.get(key)
        if e:
            lines.append(f"{e.name} | type: {e.type} | description: {e.description}")
        else:
            lines.append(f"{name}")
    lines.append("Relationships:")
    nodes_set = {n.strip().upper() for n in node_names}
    for (src, tgt), r in relations.items():
        if src in nodes_set and tgt in nodes_set:
            lines.append(f"({r.source}, {r.target}) | keyword: {r.keyword} | description: {r.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def partition_graph(
    G: nx.Graph,
    edge_id: str,
    entities: dict[str, "Entity"],
    relations: dict[tuple[str, str], "Relation"],
    *,
    target: int = 40,
    min_size: int = 10,
    resolutions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    n_iterations: int = 10,
    seed: int = 42,
) -> tuple[list[list[str]], list[str]]:
    """Partition G into communities, return (communities, linearized_texts).

    Each community is a list of node (entity) names. Linearized texts are
    formatted for `offline/summarize.py`.
    """
    nodes = list(G.nodes())
    if not nodes:
        return [], []

    ig_graph, node_list = _nx_to_igraph(G)

    # ---- resolution sweep ----
    best_part: Optional[ig.VertexClustering] = None
    best_diff = float("inf")
    for res in resolutions:
        part = _run_leiden_at_resolution(ig_graph, res, n_iterations, seed)
        if part is None:
            continue
        diff = abs(_mean_size(part) - target)
        if diff < best_diff:
            best_diff = diff
            best_part = part

    if best_part is None:
        # Fallback: treat the whole graph as one community
        communities = [nodes]
    else:
        communities = [[node_list[i] for i in c] for c in best_part]

    # ---- post-processing ----
    communities = _merge_small_communities(communities, G, min_size=min_size)
    communities = _split_large_communities(communities, G, max_size=2 * target,
                                           n_iterations=n_iterations, seed=seed)

    sizes = [len(c) for c in communities]
    p50 = sorted(sizes)[len(sizes) // 2] if sizes else 0
    _log.info("[edge %s] %d communities, p50 size=%d (target=%d)", edge_id, len(communities), p50, target)

    lin_texts = [linearize(c, entities, relations) for c in communities]
    return communities, lin_texts
