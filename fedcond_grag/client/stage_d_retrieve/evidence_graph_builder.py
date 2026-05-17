"""Build an evidence subgraph E_q from a Tri-Graph + EvidenceRetrievalResult.

E_q contains:
  * All activated entity nodes (from EvidenceRetrievalResult.actived_entities).
  * All sentence nodes connected to any activated entity via S-E edges.
  * Top-K passage nodes (from EvidenceRetrievalResult.sorted_passage_hash_ids[:top_k]).
  * All S-E and P-E edges between these nodes (inherited via subgraph()).
  * NO S-P edges (S-E-P invariant).

Hash-IDs are reconstructed from trigraph.node_text + trigraph.node_type using
the md5 prefix convention: entity -> "entity-", sentence -> "sentence-",
passage -> "passage-".
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

if TYPE_CHECKING:
    from .evidence_linearrag import EvidenceRetrievalResult


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceGraph:
    """Induced subgraph E_q for one question.

    Attributes:
        data: PyG Data with fields x, edge_index, edge_type, node_type, node_text.
        kept_indices: 1-D LongTensor of shape [M] — original trigraph node
            indices sorted ascending.
    """

    data: Data
    kept_indices: Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rebuild_hash_map(trigraph: Data) -> dict[str, int]:
    """Build hash_id -> PyG node index from node_text + node_type.

    Node layout (must match trigraph_builder):
      [0 .. ne-1]          entity nodes  (node_type == 0)
      [ne .. ne+ns-1]      sentence nodes (node_type == 1)
      [ne+ns .. ne+ns+np-1] passage nodes (node_type == 2)
    """
    ne = int((trigraph.node_type == 0).sum())
    ns = int((trigraph.node_type == 1).sum())

    def hid(prefix: str, text: str) -> str:
        return prefix + md5(text.encode()).hexdigest()

    mapping: dict[str, int] = {}
    for i, t in enumerate(trigraph.node_text[:ne]):
        mapping[hid("entity-", t)] = i
    for i, t in enumerate(trigraph.node_text[ne : ne + ns]):
        mapping[hid("sentence-", t)] = ne + i
    for i, t in enumerate(trigraph.node_text[ne + ns :]):
        mapping[hid("passage-", t)] = ne + ns + i
    return mapping


# ---------------------------------------------------------------------------
# Public builder function
# ---------------------------------------------------------------------------


def build_evidence_graph(
    trigraph: Data,
    result: "EvidenceRetrievalResult",
    *,
    top_k: int = 5,
) -> EvidenceGraph:
    """Build the induced evidence subgraph E_q.

    Args:
        trigraph: PyG Data produced by trigraph_builder.  Must have fields
                  x, edge_index, edge_type, node_type, node_text.
        result: EvidenceRetrievalResult for the question of interest.
        top_k: Number of top passages to include from sorted_passage_hash_ids.

    Returns:
        EvidenceGraph with the induced subgraph.
    """
    # ------------------------------------------------------------------
    # 1. Rebuild hash_id -> PyG node index map from node_text.
    # ------------------------------------------------------------------
    hash_map = _rebuild_hash_map(trigraph)

    # ------------------------------------------------------------------
    # 2. Collect entity nodes from actived_entities keys.
    # ------------------------------------------------------------------
    entity_pyg: list[int] = [
        hash_map[h] for h in result.actived_entities if h in hash_map
    ]

    # ------------------------------------------------------------------
    # 3. Collect passage nodes from top-k sorted_passage_hash_ids.
    # ------------------------------------------------------------------
    passage_pyg: list[int] = [
        hash_map[h]
        for h in result.sorted_passage_hash_ids[:top_k]
        if h in hash_map
    ]

    # ------------------------------------------------------------------
    # 4. Collect sentence nodes via S-E edges adjacent to activated entities.
    # ------------------------------------------------------------------
    # edge_type == 0 means S-E edges
    se_mask = trigraph.edge_type == 0
    se_ei = trigraph.edge_index[:, se_mask]

    if entity_pyg and se_ei.shape[1] > 0:
        entity_set = torch.tensor(list(entity_pyg), dtype=torch.long)

        src_is_sent = trigraph.node_type[se_ei[0]] == 1
        dst_is_ent = torch.isin(se_ei[1], entity_set)
        dst_is_sent = trigraph.node_type[se_ei[1]] == 1
        src_is_ent = torch.isin(se_ei[0], entity_set)

        sent_nodes_a = se_ei[0][src_is_sent & dst_is_ent]
        sent_nodes_b = se_ei[1][dst_is_sent & src_is_ent]
        sentence_pyg = torch.cat([sent_nodes_a, sent_nodes_b]).unique().tolist()
    else:
        sentence_pyg = []

    # ------------------------------------------------------------------
    # 5. Union all, sort into kept tensor.
    # ------------------------------------------------------------------
    kept_set = set(entity_pyg) | set(sentence_pyg) | set(passage_pyg)

    if not kept_set:
        # Empty edge case
        feat_dim = trigraph.x.shape[1] if trigraph.x.dim() > 1 else 1
        out = Data(
            x=torch.zeros((0, feat_dim), dtype=trigraph.x.dtype),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            node_type=torch.zeros((0,), dtype=torch.long),
            node_text=[],
        )
        return EvidenceGraph(data=out, kept_indices=torch.zeros((0,), dtype=torch.long))

    kept = torch.tensor(sorted(kept_set), dtype=torch.long)

    # ------------------------------------------------------------------
    # 6. Extract subgraph with relabeled nodes.
    # ------------------------------------------------------------------
    sub_ei, sub_etype = subgraph(
        kept,
        trigraph.edge_index,
        edge_attr=trigraph.edge_type,
        relabel_nodes=True,
        num_nodes=trigraph.num_nodes,
    )

    # ------------------------------------------------------------------
    # 7. Build output Data — include node_text for LLM prompting.
    # ------------------------------------------------------------------
    sub_node_text = [trigraph.node_text[i] for i in kept.tolist()]
    out = Data(
        x=trigraph.x[kept],
        edge_index=sub_ei,
        edge_type=sub_etype,
        node_type=trigraph.node_type[kept],
        node_text=sub_node_text,
    )
    return EvidenceGraph(data=out, kept_indices=kept)
