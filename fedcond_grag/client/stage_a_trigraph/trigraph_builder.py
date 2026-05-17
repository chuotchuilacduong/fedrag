"""Build a PyG Tri-Graph (Entity / Sentence / Passage) from a client corpus.

Strategy: call LinearRAG.index() on the client's passage texts, then
extract its internal state (embedding stores + adjacency dicts) and
convert to a torch_geometric.data.Data object.

Design invariants (docs/plan/02_DATA_AND_TRIGRAPH.md §10.1):
  - ONLY S-E and P-E edges go into edge_index. Never S-P.
  - P-P sequential edges produced by LinearRAG.add_adjacent_passage_edges()
    are filtered out here; they are not part of the Tri-Graph topology.
  - edge_type: 0 = S-E, 1 = P-E (both stored as undirected pairs).

Integration notes (docs/plan/09_INT_HOST_REPO.md §29 Step 5):
  - embedding_model must be a SentenceTransformer instance, not a string.
  - llm_model=None is safe when only calling index() (not qa()).
  - Attribute names verified by reading LinearRAG.py directly:
      entity:   rag.entity_embedding_store
      sentence: rag.sentence_embedding_store
      passage:  rag.passage_embedding_store
      S-E adj:  rag.sentence_hash_id_to_entity_hash_ids
      P-E adj:  rag.node_to_node_stats (filter to entity keys only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from .node_encoder import DEFAULT_MODEL, NodeEncoder, load_encoder

# Node-type constants — must match graph_condensation constants exactly.
ENTITY = 0
SENTENCE = 1
PASSAGE = 2

_SE_EDGE = 0   # edge_type value for Sentence–Entity
_PE_EDGE = 1   # edge_type value for Passage–Entity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_trigraph_for_client(
    passages: Sequence[str],
    *,
    working_dir: str | Path,
    dataset_name: str,
    encoder: NodeEncoder | None = None,
    spacy_model: str = "en_core_web_trf",
) -> Data:
    """Build a Tri-Graph PyG Data object for one client's corpus.

    Args:
        passages: Plain-text passage strings (already formatted with title).
        working_dir: LinearRAG cache root (embedding parquet files live here).
        dataset_name: Sub-directory under working_dir used by LinearRAG.
        encoder: NodeEncoder to use. Defaults to all-MiniLM-L6-v2.
        spacy_model: spaCy NER model. Default en_core_web_trf.

    Returns:
        PyG Data with fields:
            x          [N, d]  float32 L2-normalised node embeddings
            edge_index [2, E]  int64  undirected edges (both directions stored)
            edge_type  [E]     int64  0=S-E, 1=P-E
            node_type  [N]     int64  0=entity, 1=sentence, 2=passage
            node_text  list[str]      raw text per node (local-only, not uploaded)
    """
    builder = TriGraphBuilder(
        working_dir=working_dir,
        dataset_name=dataset_name,
        encoder=encoder,
        spacy_model=spacy_model,
    )
    return builder.build(passages)


@dataclass
class TriGraphBuilder:
    """Stateful builder; reuse to index multiple clients in one process."""

    working_dir: str | Path
    dataset_name: str
    encoder: NodeEncoder | None = None
    spacy_model: str = "en_core_web_trf"

    def build(self, passages: Sequence[str]) -> Data:
        from fedcond_grag.linearrag import LinearRAG, LinearRAGConfig

        enc = self.encoder or load_encoder(DEFAULT_MODEL)
        cfg = LinearRAGConfig(
            dataset_name=self.dataset_name,
            embedding_model=enc,
            llm_model=None,
            spacy_model=self.spacy_model,
            working_dir=str(self.working_dir),
        )
        rag = LinearRAG(global_config=cfg)
        rag.index(list(passages))
        return _rag_to_pyg(rag)


# ---------------------------------------------------------------------------
# Internal conversion
# ---------------------------------------------------------------------------


def _rag_to_pyg(rag) -> Data:
    e_store = rag.entity_embedding_store
    s_store = rag.sentence_embedding_store
    p_store = rag.passage_embedding_store

    entity_hids: list[str] = e_store.hash_ids
    sentence_hids: list[str] = s_store.hash_ids
    passage_hids: list[str] = p_store.hash_ids

    ne, ns, np_ = len(entity_hids), len(sentence_hids), len(passage_hids)

    # Global node indices
    e_gidx = {h: i for i, h in enumerate(entity_hids)}
    s_gidx = {h: ne + i for i, h in enumerate(sentence_hids)}
    p_gidx = {h: ne + ns + i for i, h in enumerate(passage_hids)}

    # Node features — embeddings are lists of lists/arrays after parquet round-trip
    x = torch.tensor(
        _stack_embeddings(e_store.embeddings)
        + _stack_embeddings(s_store.embeddings)
        + _stack_embeddings(p_store.embeddings),
        dtype=torch.float32,
    )  # [N, d]

    node_type = torch.cat([
        torch.full((ne,), ENTITY, dtype=torch.long),
        torch.full((ns,), SENTENCE, dtype=torch.long),
        torch.full((np_,), PASSAGE, dtype=torch.long),
    ])

    node_text: list[str] = list(e_store.texts) + list(s_store.texts) + list(p_store.texts)

    # Build edges
    src, dst, etype = _build_edges(
        rag, e_gidx, s_gidx, p_gidx,
        entity_hid_set=set(entity_hids),
    )

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        node_text=node_text,
    )


def _build_edges(
    rag,
    e_gidx: dict[str, int],
    s_gidx: dict[str, int],
    p_gidx: dict[str, int],
    entity_hid_set: set[str],
) -> tuple[list[int], list[int], list[int]]:
    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []

    # S-E edges (undirected) from sentence_hash_id_to_entity_hash_ids
    for s_hid, e_hids in rag.sentence_hash_id_to_entity_hash_ids.items():
        if s_hid not in s_gidx:
            continue
        si = s_gidx[s_hid]
        for e_hid in e_hids:
            if e_hid not in e_gidx:
                continue
            ei = e_gidx[e_hid]
            src += [si, ei]
            dst += [ei, si]
            etype += [_SE_EDGE, _SE_EDGE]

    # P-E edges (undirected) from node_to_node_stats
    # node_to_node_stats contains both P-E and P-P sequential edges;
    # filter by checking if neighbor is an entity hash_id.
    for p_hid, neighbors in rag.node_to_node_stats.items():
        if p_hid not in p_gidx:
            continue
        pi = p_gidx[p_hid]
        for neighbor_hid in neighbors:
            if neighbor_hid not in entity_hid_set:
                continue  # skip P-P sequential edges
            if neighbor_hid not in e_gidx:
                continue
            ei = e_gidx[neighbor_hid]
            src += [pi, ei]
            dst += [ei, pi]
            etype += [_PE_EDGE, _PE_EDGE]

    return src, dst, etype


def _stack_embeddings(embeddings: list) -> list:
    """Convert list-of-arrays/lists to a flat Python list of lists for torch.tensor()."""
    result = []
    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            result.append(emb.tolist())
        else:
            result.append(list(emb))
    return result
