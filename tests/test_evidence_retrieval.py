"""Unit tests for EvidenceRetrievalResult and build_evidence_graph.

6 self-contained tests — no disk I/O, no real RAG indexing.
"""

from __future__ import annotations

from hashlib import md5

import pytest
import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_d_retrieve.evidence_linearrag import EvidenceRetrievalResult
from fedcond_grag.client.stage_d_retrieve.evidence_graph_builder import (
    EvidenceGraph,
    build_evidence_graph,
    _rebuild_hash_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hid(prefix: str, text: str) -> str:
    return prefix + md5(text.encode()).hexdigest()


def _make_trigraph(
    entity_texts: list[str],
    sentence_texts: list[str],
    passage_texts: list[str],
    edges: list[tuple[int, int, int]],  # (src, dst, edge_type)
    embed_dim: int = 8,
) -> Data:
    """Build a synthetic trigraph with real hash-able node_text."""
    ne = len(entity_texts)
    ns = len(sentence_texts)
    np_ = len(passage_texts)
    n = ne + ns + np_

    torch.manual_seed(0)
    x = torch.randn(n, embed_dim)

    node_type = torch.cat([
        torch.zeros(ne, dtype=torch.long),
        torch.ones(ns, dtype=torch.long),
        torch.full((np_,), 2, dtype=torch.long),
    ])
    node_text = entity_texts + sentence_texts + passage_texts

    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst])
        edge_type = torch.tensor([e[2] for e in edges], dtype=torch.long)
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


def _make_result(
    actived_entities: dict | None = None,
    sorted_passage_hash_ids: list[str] | None = None,
    sorted_passage_scores: list[float] | None = None,
    top_k_passages: list[str] | None = None,
    question: str = "test question?",
    gold_answer: str = "test answer",
) -> EvidenceRetrievalResult:
    return EvidenceRetrievalResult(
        question=question,
        gold_answer=gold_answer,
        actived_entities=actived_entities or {},
        sorted_passage_hash_ids=sorted_passage_hash_ids or [],
        sorted_passage_scores=sorted_passage_scores or [],
        top_k_passages=top_k_passages or [],
    )


# ---------------------------------------------------------------------------
# Test 1: EvidenceRetrievalResult has required fields
# ---------------------------------------------------------------------------


def test_evidence_result_fields():
    """EvidenceRetrievalResult can be instantiated with all required fields."""
    torch.manual_seed(0)

    actived = {"entity-abc123": (0, 0.9, 1), "entity-def456": (1, 0.5, 2)}
    result = EvidenceRetrievalResult(
        question="Who wrote Hamlet?",
        gold_answer="Shakespeare",
        actived_entities=actived,
        sorted_passage_hash_ids=["passage-aaa", "passage-bbb"],
        sorted_passage_scores=[0.8, 0.6],
        top_k_passages=["Hamlet was written by Shakespeare.", "A famous play."],
    )

    assert result.question == "Who wrote Hamlet?"
    assert result.gold_answer == "Shakespeare"
    assert isinstance(result.actived_entities, dict)
    assert len(result.actived_entities) == 2
    assert isinstance(result.sorted_passage_hash_ids, list)
    assert isinstance(result.sorted_passage_scores, list)
    assert isinstance(result.top_k_passages, list)

    # Verify tuple structure: (entity_idx, score, tier)
    for hid, val in result.actived_entities.items():
        assert isinstance(hid, str)
        assert len(val) == 3
        idx, score, tier = val
        assert isinstance(idx, int)
        assert isinstance(score, float)
        assert isinstance(tier, int)


# ---------------------------------------------------------------------------
# Test 2: known activated entities appear in kept_indices
# ---------------------------------------------------------------------------


def test_build_evidence_graph_nodes():
    """Activated entity nodes appear in kept_indices."""
    torch.manual_seed(0)

    entity_texts = ["cat", "dog", "bird"]
    sentence_texts = ["the cat sat", "the dog ran"]
    passage_texts = ["cats are pets", "dogs are loyal"]

    # Global indices: ent0=0,ent1=1,ent2=2, sent0=3,sent1=4, pass0=5,pass1=6
    # S-E edges: sent0->ent0, sent1->ent1
    # P-E edges: pass0->ent0, pass1->ent2
    edges = [
        (3, 0, 0), (0, 3, 0),   # S-E: sent0 <-> ent0
        (4, 1, 0), (1, 4, 0),   # S-E: sent1 <-> ent1
        (5, 0, 1), (0, 5, 1),   # P-E: pass0 <-> ent0
        (6, 2, 1), (2, 6, 1),   # P-E: pass1 <-> ent2
    ]
    trigraph = _make_trigraph(entity_texts, sentence_texts, passage_texts, edges)

    # Activate ent0 and ent1
    e0_hid = _hid("entity-", "cat")
    e1_hid = _hid("entity-", "dog")
    p0_hid = _hid("passage-", "cats are pets")

    result = _make_result(
        actived_entities={e0_hid: (0, 0.9, 1), e1_hid: (1, 0.5, 2)},
        sorted_passage_hash_ids=[p0_hid],
    )

    eg = build_evidence_graph(trigraph, result, top_k=1)

    assert isinstance(eg, EvidenceGraph)
    assert isinstance(eg.kept_indices, torch.Tensor)
    assert eg.kept_indices.dtype == torch.long

    kept = eg.kept_indices.tolist()
    # ent0 (idx=0) and ent1 (idx=1) must be in kept
    assert 0 in kept, "ent0 (cat) must be in kept_indices"
    assert 1 in kept, "ent1 (dog) must be in kept_indices"


# ---------------------------------------------------------------------------
# Test 3: sentences adjacent to activated entities are included
# ---------------------------------------------------------------------------


def test_build_evidence_graph_sentences():
    """Sentence nodes connected to activated entities via S-E edges are included."""
    torch.manual_seed(0)

    entity_texts = ["cat", "dog", "bird"]
    sentence_texts = ["the cat sat", "the dog ran"]
    passage_texts = ["cats are pets", "dogs are loyal"]

    # Global indices: ent0=0,ent1=1,ent2=2, sent0=3,sent1=4, pass0=5,pass1=6
    edges = [
        (3, 0, 0), (0, 3, 0),   # S-E: sent0 <-> ent0
        (4, 1, 0), (1, 4, 0),   # S-E: sent1 <-> ent1
        (5, 0, 1), (0, 5, 1),   # P-E: pass0 <-> ent0
        (6, 2, 1), (2, 6, 1),   # P-E: pass1 <-> ent2
    ]
    trigraph = _make_trigraph(entity_texts, sentence_texts, passage_texts, edges)

    # Activate only ent0 ("cat"); sent0 is adjacent via S-E, sent1 is not
    e0_hid = _hid("entity-", "cat")
    p0_hid = _hid("passage-", "cats are pets")

    result = _make_result(
        actived_entities={e0_hid: (0, 0.9, 1)},
        sorted_passage_hash_ids=[p0_hid],
    )

    eg = build_evidence_graph(trigraph, result, top_k=1)
    kept = eg.kept_indices.tolist()

    # sent0 (idx=3) is adjacent to activated ent0 (idx=0) via S-E — must be included
    assert 3 in kept, "sent0 (adj to activated ent0) must be in kept_indices"
    # sent1 (idx=4) is adjacent to ent1 which is NOT activated — must be excluded
    assert 4 not in kept, "sent1 (adj to non-activated ent1) must NOT be in kept_indices"


# ---------------------------------------------------------------------------
# Test 4: top-k passages are included
# ---------------------------------------------------------------------------


def test_build_evidence_graph_passages():
    """Top-k passage nodes appear in kept_indices."""
    torch.manual_seed(0)

    entity_texts = ["cat", "dog", "bird"]
    sentence_texts = ["the cat sat", "the dog ran"]
    passage_texts = ["cats are pets", "dogs are loyal"]

    # Global indices: ent0=0,ent1=1,ent2=2, sent0=3,sent1=4, pass0=5,pass1=6
    edges = [
        (5, 0, 1), (0, 5, 1),   # P-E: pass0 <-> ent0
        (6, 2, 1), (2, 6, 1),   # P-E: pass1 <-> ent2
    ]
    trigraph = _make_trigraph(entity_texts, sentence_texts, passage_texts, edges)

    p0_hid = _hid("passage-", "cats are pets")
    p1_hid = _hid("passage-", "dogs are loyal")

    # No activated entities, just passages
    result = _make_result(
        actived_entities={},
        sorted_passage_hash_ids=[p0_hid, p1_hid],
    )

    eg = build_evidence_graph(trigraph, result, top_k=2)
    kept = eg.kept_indices.tolist()

    # pass0 (idx=5) and pass1 (idx=6) must be in kept
    assert 5 in kept, "pass0 must be in kept_indices"
    assert 6 in kept, "pass1 must be in kept_indices"

    # top_k=1 should only include pass0
    eg1 = build_evidence_graph(trigraph, result, top_k=1)
    kept1 = eg1.kept_indices.tolist()
    assert 5 in kept1, "pass0 must be in kept_indices with top_k=1"
    assert 6 not in kept1, "pass1 must NOT be in kept_indices with top_k=1"


# ---------------------------------------------------------------------------
# Test 5: no S-P edges in the output evidence graph
# ---------------------------------------------------------------------------


def test_build_evidence_graph_no_sp_edges():
    """No Sentence-Passage edges appear in the output (S-E-P invariant)."""
    torch.manual_seed(0)

    entity_texts = ["cat"]
    sentence_texts = ["the cat sat"]
    passage_texts = ["cats are pets"]

    # Global indices: ent0=0, sent0=1, pass0=2
    # Only legitimate edges: S-E and P-E
    edges = [
        (1, 0, 0), (0, 1, 0),   # S-E: sent0 <-> ent0
        (2, 0, 1), (0, 2, 1),   # P-E: pass0 <-> ent0
    ]
    trigraph = _make_trigraph(entity_texts, sentence_texts, passage_texts, edges)

    e0_hid = _hid("entity-", "cat")
    p0_hid = _hid("passage-", "cats are pets")

    result = _make_result(
        actived_entities={e0_hid: (0, 0.9, 1)},
        sorted_passage_hash_ids=[p0_hid],
    )

    eg = build_evidence_graph(trigraph, result, top_k=1)

    # Verify no S-P edges in the output
    if eg.data.edge_index.shape[1] > 0:
        src_types = eg.data.node_type[eg.data.edge_index[0]]
        dst_types = eg.data.node_type[eg.data.edge_index[1]]
        sp_mask = (
            ((src_types == 1) & (dst_types == 2)) |
            ((src_types == 2) & (dst_types == 1))
        )
        assert not sp_mask.any(), "S-P edges must not appear in evidence graph"


# ---------------------------------------------------------------------------
# Test 6: empty actived_entities + no passage hash_ids -> valid empty EvidenceGraph
# ---------------------------------------------------------------------------


def test_build_evidence_graph_empty_result():
    """actived_entities={} and empty hash_ids returns a valid (empty-ish) EvidenceGraph."""
    torch.manual_seed(0)

    entity_texts = ["cat", "dog"]
    sentence_texts = ["the cat sat"]
    passage_texts = ["cats are pets"]

    edges = [
        (2, 0, 0), (0, 2, 0),   # S-E
        (3, 0, 1), (0, 3, 1),   # P-E
    ]
    trigraph = _make_trigraph(entity_texts, sentence_texts, passage_texts, edges)

    # Completely empty result
    result = _make_result(
        actived_entities={},
        sorted_passage_hash_ids=[],
    )

    eg = build_evidence_graph(trigraph, result, top_k=5)

    # Must return a valid EvidenceGraph (not crash)
    assert isinstance(eg, EvidenceGraph)
    assert isinstance(eg.data, Data)
    assert isinstance(eg.kept_indices, torch.Tensor)

    # No nodes should be kept
    assert eg.data.num_nodes == 0
    assert eg.kept_indices.shape[0] == 0
