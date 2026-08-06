"""Tests for retrieval R@k metrics (PPR passages that feed the LLM)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedcond_grag.client.client import FedCondQAClient
from fedcond_grag.trainer import FedTrainer
from fedcond_grag.utils.evaluate import norm_passage_title, retrieval_recall_at_k


def test_norm_passage_title():
    assert norm_passage_title("Move (1970 film): Move is a 1970 film.") == "move (1970 film)"
    # Optional "<int>:" trigraph node prefix is stripped first
    assert norm_passage_title("12:Stuart Rosenberg: was a director.") == "stuart rosenberg"
    assert norm_passage_title("  No Colon Here  ") == "no colon here"


def test_retrieval_recall_at_k():
    gold = {"a", "b"}
    retrieved = ["a", "x", "b", None, "y"]
    rec = retrieval_recall_at_k(gold, retrieved, ks=(1, 2, 5))
    assert rec == {1: 0.5, 2: 0.5, 5: 1.0}

    # unmappable slots (None) count as misses
    rec = retrieval_recall_at_k(gold, [None, "a"], ks=(1, 2))
    assert rec == {1: 0.0, 2: 0.5}

    # no gold → cannot score
    assert retrieval_recall_at_k(set(), ["a"]) == {}


def test_sample_retrieval_recall():
    node_text = [
        "0:Move (1970 film): Move is a 1970 American comedy film.",
        "1:Stuart Rosenberg: was an American director.",
        "2:Unrelated: something else entirely.",
    ]
    client = SimpleNamespace(
        _ppr_node_map=torch.tensor([[0, 2, -1, -1, 1]]),
        tri_graph=SimpleNamespace(node_text=node_text),
    )
    sample = {
        "idx": 0,
        "retrieved_passages": [  # gold evidence passages of the record
            "Move (1970 film): Move is a 1970 American comedy film.",
            "Stuart Rosenberg: was an American director.",
        ],
    }
    rec = FedTrainer._sample_retrieval_recall(client, sample, ks=(1, 2, 5))
    assert rec == {1: 0.5, 2: 0.5, 5: 1.0}

    # missing map / bad idx / no gold → {}
    assert FedTrainer._sample_retrieval_recall(
        SimpleNamespace(_ppr_node_map=None, tri_graph=SimpleNamespace(node_text=node_text)),
        sample,
    ) == {}
    assert FedTrainer._sample_retrieval_recall(client, {**sample, "idx": 5}) == {}
    assert FedTrainer._sample_retrieval_recall(client, {**sample, "retrieved_passages": []}) == {}


def _fake_client(desc_source: str):
    """Minimal stand-in exposing the attrs _attach_evidence_graphs touches."""
    node_text = [
        "0:Move (1970 film): Move is a 1970 American comedy film.",
        "1:Stuart Rosenberg: was an American director.",
        "2:Unrelated: something else entirely.",
    ]
    tri_graph = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_type=torch.tensor([2, 2, 2]),
    )
    tri_graph.node_text = node_text
    return SimpleNamespace(
        args=SimpleNamespace(desc_source=desc_source),
        client_id=0,
        tri_graph=tri_graph,
        _local_adj=None,
        # rank order: node 1 first, then node 0; -1 slots are unmapped passages
        _ppr_node_map=torch.tensor([[1, -1, 0, -1, -1]]),
    )


def test_attach_evidence_graphs_ppr_desc():
    client = _fake_client("ppr")
    sample = {"idx": 0, "id": "q0", "desc": "GOLD DESC"}
    out = FedCondQAClient._attach_evidence_graphs(client, [sample])
    assert out[0]["desc"] == (
        "Stuart Rosenberg: was an American director.\n\n"
        "Move (1970 film): Move is a 1970 American comedy film."
    )
    assert out[0]["graph"].x.size(0) >= 2  # anchors + 1-hop neighbours
    # original sample dict untouched
    assert sample["desc"] == "GOLD DESC"


def test_attach_evidence_graphs_gold_desc():
    client = _fake_client("gold")
    out = FedCondQAClient._attach_evidence_graphs(
        client, [{"idx": 0, "id": "q0", "desc": "GOLD DESC"}]
    )
    assert out[0]["desc"] == "GOLD DESC"
