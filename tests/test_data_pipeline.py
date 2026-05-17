"""Tests for the data pipeline modules.

Covers:
- hotpot_loader: HotpotCorpus construction, dedup, passage format
- federated_partition: no overlap, full coverage, determinism
- corpus_index: round-trip save/load, lookup correctness
- trigraph_builder: PyG Data format, S-E-P invariant (no S-P edges),
                    correct edge_type values, node ordering
- graph_store: save/load round-trip
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Data

from fedcond_grag.dataloader.corpus_index import CorpusIndex
from fedcond_grag.dataloader.federated_partition import federated_partition, partition_stats
from fedcond_grag.dataloader.hotpot_loader import HotpotCorpus, load_hotpot
from fedcond_grag.client.stage_a_trigraph.graph_store import load_trigraph, save_trigraph
from fedcond_grag.client.stage_a_trigraph.trigraph_builder import (
    ENTITY,
    PASSAGE,
    SENTENCE,
    _rag_to_pyg,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_hotpot_json(num_questions: int = 3) -> list[dict]:
    """Build minimal in-memory HotpotQA-format data."""
    titles = [f"Doc{i}" for i in range(num_questions * 2)]
    records = []
    for q in range(num_questions):
        t0, t1 = titles[q * 2], titles[q * 2 + 1]
        records.append({
            "_id": f"q{q}",
            "question": f"Question {q}?",
            "answer": f"Answer {q}",
            "supporting_facts": [[t0, 0], [t1, 0]],
            "context": [
                [t0, [f"Sentence A of {t0}.", f"Sentence B of {t0}."]],
                [t1, [f"Sentence A of {t1}.", f"Sentence B of {t1}."]],
                ["Distractor", ["This is a distractor passage."]],
            ],
            "type": "bridge",
            "level": "medium",
        })
    return records


def _load_corpus_from_list(records: list[dict]) -> HotpotCorpus:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(records, f)
        path = f.name
    return load_hotpot(path)


def _make_fake_rag(num_entities: int = 3, num_sentences: int = 4, num_passages: int = 2, dim: int = 8):
    """Construct a fake LinearRAG object mimicking post-index() state."""
    torch.manual_seed(42)
    rag = MagicMock()

    def _store(prefix, n):
        store = MagicMock()
        hids = [f"{prefix}-{i:04x}" for i in range(n)]
        store.hash_ids = hids
        store.texts = [f"{prefix} text {i}" for i in range(n)]
        store.embeddings = [torch.randn(dim).numpy().tolist() for _ in range(n)]
        store.hash_id_to_text = dict(zip(hids, store.texts))
        store.hash_id_to_idx = {h: i for i, h in enumerate(hids)}
        store.text_to_hash_id = {t: h for t, h in zip(store.texts, hids)}
        return store

    rag.entity_embedding_store = _store("entity", num_entities)
    rag.sentence_embedding_store = _store("sentence", num_sentences)
    rag.passage_embedding_store = _store("passage", num_passages)

    e_hids = rag.entity_embedding_store.hash_ids
    s_hids = rag.sentence_embedding_store.hash_ids
    p_hids = rag.passage_embedding_store.hash_ids

    # S-E adjacency: each sentence mentions 1-2 entities (wrap-around)
    rag.sentence_hash_id_to_entity_hash_ids = {
        s_hids[i]: [e_hids[i % num_entities], e_hids[(i + 1) % num_entities]]
        for i in range(num_sentences)
    }
    rag.entity_hash_id_to_sentence_hash_ids = {
        e_hids[i]: [s_hids[i % num_sentences]]
        for i in range(num_entities)
    }

    # P-E adjacency (in node_to_node_stats) + optional P-P edge (should be filtered)
    pe_dict: dict = {}
    if num_passages > 0 and num_entities > 0:
        pe_dict[p_hids[0]] = {e_hids[0]: 0.5}
        if num_entities > 1:
            pe_dict[p_hids[0]][e_hids[1]] = 0.3
        if num_passages > 1:
            pe_dict[p_hids[0]][p_hids[1]] = 1.0   # P-P — must be filtered out
            pe_dict[p_hids[1]] = {e_hids[num_entities - 1]: 0.7}
    rag.node_to_node_stats = pe_dict

    return rag


# ---------------------------------------------------------------------------
# hotpot_loader
# ---------------------------------------------------------------------------


def test_load_hotpot_basic():
    records = _make_hotpot_json(num_questions=3)
    corpus = _load_corpus_from_list(records)
    assert len(corpus.questions) == 3


def test_load_hotpot_deduplicates_passages():
    records = _make_hotpot_json(num_questions=3)
    # All questions share the "Distractor" passage → should appear only once
    corpus = _load_corpus_from_list(records)
    titles = [p.title for p in corpus.passages]
    assert len(titles) == len(set(titles)), "Passages must be deduplicated by title"


def test_load_hotpot_passage_format():
    records = _make_hotpot_json(num_questions=1)
    corpus = _load_corpus_from_list(records)
    for p in corpus.passages:
        assert p.passage_text.startswith("Title:"), (
            f"Passage must start with 'Title:' prefix, got: {p.passage_text[:40]!r}"
        )
        assert p.passage_id == hashlib.sha1(p.title.encode()).hexdigest()
        assert len(p.sentences) >= 1


def test_load_hotpot_max_samples():
    records = _make_hotpot_json(num_questions=5)
    corpus = _load_corpus_from_list(records)
    assert len(corpus.questions) == 5

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(records, f)
        path = f.name
    corpus2 = load_hotpot(path, max_samples=2)
    assert len(corpus2.questions) == 2


def test_load_hotpot_supporting_facts_mapped():
    records = _make_hotpot_json(num_questions=2)
    corpus = _load_corpus_from_list(records)
    for q in corpus.questions:
        for title, sent_idx in q.supporting_facts:
            assert isinstance(title, str)
            assert isinstance(sent_idx, int)


# ---------------------------------------------------------------------------
# federated_partition
# ---------------------------------------------------------------------------


def test_partition_no_overlap():
    records = _make_hotpot_json(num_questions=5)
    corpus = _load_corpus_from_list(records)
    clients = federated_partition(corpus, num_clients=3)
    assert len(clients) == 3
    stats = partition_stats(clients)
    assert stats["no_overlap"], "Passages must appear on exactly one client"
    assert stats["total_passages"] == len(corpus.passages)


def test_partition_full_coverage():
    records = _make_hotpot_json(num_questions=4)
    corpus = _load_corpus_from_list(records)
    clients = federated_partition(corpus, num_clients=5)
    assigned = sum(len(c.passages) for c in clients)
    assert assigned == len(corpus.passages)


def test_partition_deterministic():
    records = _make_hotpot_json(num_questions=3)
    corpus = _load_corpus_from_list(records)
    c1 = federated_partition(corpus, num_clients=3)
    c2 = federated_partition(corpus, num_clients=3)
    for a, b in zip(c1, c2):
        assert [p.title for p in a.passages] == [p.title for p in b.passages]


def test_partition_single_client():
    records = _make_hotpot_json(num_questions=2)
    corpus = _load_corpus_from_list(records)
    clients = federated_partition(corpus, num_clients=1)
    assert len(clients) == 1
    assert len(clients[0].passages) == len(corpus.passages)


# ---------------------------------------------------------------------------
# corpus_index
# ---------------------------------------------------------------------------


def test_corpus_index_lookup():
    records = _make_hotpot_json(num_questions=2)
    corpus = _load_corpus_from_list(records)
    idx = CorpusIndex.from_corpus(corpus)

    assert idx.num_passages() == len(corpus.passages)
    assert idx.num_sentences() > 0

    for p in corpus.passages:
        assert idx.get_passage_text(p.passage_id) == p.passage_text
        assert idx.get_passage_title(p.passage_id) == p.title
        assert idx.passage_id_for_title(p.title) == p.passage_id


def test_corpus_index_save_load(tmp_path):
    records = _make_hotpot_json(num_questions=2)
    corpus = _load_corpus_from_list(records)
    idx = CorpusIndex.from_corpus(corpus)

    path = tmp_path / "corpus_index.json"
    idx.save(path)

    idx2 = CorpusIndex.load(path)
    assert idx2.num_passages() == idx.num_passages()
    assert idx2.num_sentences() == idx.num_sentences()
    for p in corpus.passages:
        assert idx2.get_passage_text(p.passage_id) == p.passage_text


# ---------------------------------------------------------------------------
# trigraph_builder (_rag_to_pyg — tested without real LinearRAG/spaCy)
# ---------------------------------------------------------------------------


def test_rag_to_pyg_node_counts():
    rag = _make_fake_rag(num_entities=3, num_sentences=4, num_passages=2, dim=8)
    g = _rag_to_pyg(rag)
    assert g.x.shape == (9, 8)   # 3+4+2
    assert g.node_type.shape == (9,)
    assert g.node_type[:3].eq(ENTITY).all()
    assert g.node_type[3:7].eq(SENTENCE).all()
    assert g.node_type[7:].eq(PASSAGE).all()


def test_rag_to_pyg_no_sp_edges():
    rag = _make_fake_rag(num_entities=3, num_sentences=4, num_passages=2, dim=8)
    g = _rag_to_pyg(rag)

    assert g.edge_index.shape[0] == 2
    e_set = set(range(3))         # node indices 0-2 are entities
    s_set = set(range(3, 7))      # node indices 3-6 are sentences
    p_set = set(range(7, 9))      # node indices 7-8 are passages

    for col in range(g.edge_index.shape[1]):
        a, b = int(g.edge_index[0, col]), int(g.edge_index[1, col])
        # S-P direct edges are forbidden
        assert not (a in s_set and b in p_set), f"Forbidden S-P edge: {a}→{b}"
        assert not (a in p_set and b in s_set), f"Forbidden P-S edge: {a}→{b}"
        # P-P edges must not appear
        assert not (a in p_set and b in p_set), f"Forbidden P-P edge: {a}→{b}"


def test_rag_to_pyg_edge_types():
    rag = _make_fake_rag(num_entities=3, num_sentences=4, num_passages=2, dim=8)
    g = _rag_to_pyg(rag)

    e_set = set(range(3))
    s_set = set(range(3, 7))
    p_set = set(range(7, 9))

    for col in range(g.edge_index.shape[1]):
        a, b = int(g.edge_index[0, col]), int(g.edge_index[1, col])
        et = int(g.edge_type[col])
        if (a in s_set and b in e_set) or (a in e_set and b in s_set):
            assert et == 0, f"S-E edge should have edge_type=0, got {et}"
        elif (a in p_set and b in e_set) or (a in e_set and b in p_set):
            assert et == 1, f"P-E edge should have edge_type=1, got {et}"
        else:
            raise AssertionError(f"Unexpected edge {a}→{b} (type {et})")


def test_rag_to_pyg_undirected():
    rag = _make_fake_rag(num_entities=2, num_sentences=2, num_passages=1, dim=4)
    g = _rag_to_pyg(rag)
    # Every (a→b) must have a matching (b→a)
    edges = set(
        zip(g.edge_index[0].tolist(), g.edge_index[1].tolist())
    )
    for a, b in list(edges):
        assert (b, a) in edges, f"Edge {a}→{b} has no reverse"


def test_rag_to_pyg_dtype():
    rag = _make_fake_rag()
    g = _rag_to_pyg(rag)
    assert g.x.dtype == torch.float32
    assert g.edge_index.dtype == torch.long
    assert g.edge_type.dtype == torch.long
    assert g.node_type.dtype == torch.long


def test_rag_to_pyg_empty_graph():
    """Empty embedding stores should produce a valid Data with zero nodes."""
    rag = _make_fake_rag(num_entities=0, num_sentences=0, num_passages=0, dim=4)
    rag.sentence_hash_id_to_entity_hash_ids = {}
    rag.node_to_node_stats = {}
    g = _rag_to_pyg(rag)
    assert g.edge_index.shape == (2, 0)


# ---------------------------------------------------------------------------
# graph_store
# ---------------------------------------------------------------------------


def test_graph_store_roundtrip(tmp_path):
    rag = _make_fake_rag(num_entities=3, num_sentences=3, num_passages=2, dim=8)
    g = _rag_to_pyg(rag)
    path = tmp_path / "client_0" / "trigraph.pt"
    save_trigraph(g, path)
    g2 = load_trigraph(path)

    assert torch.equal(g.x, g2.x)
    assert torch.equal(g.edge_index, g2.edge_index)
    assert torch.equal(g.edge_type, g2.edge_type)
    assert torch.equal(g.node_type, g2.node_type)
    assert g2.node_text == g.node_text
