"""Tests for the LinearRAG-format data loader and chunk partitioning.

Covers:
- linearrag_loader: parse index prefix, parse questions, save/load round-trip
- partition_linearrag_chunks: no overlap, full coverage, determinism, balance
- Integration: real processed files (hotpotqa) if available
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fedcond_grag.dataloader.linearrag_loader import (
    LinearRAGChunk,
    LinearRAGDataset,
    load_linearrag,
    save_chunk_list,
    save_question_list,
    _parse_chunk,
)
from fedcond_grag.dataloader.federated_partition import (
    partition_linearrag_chunks,
    chunk_partition_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    "0:vaada poda nanbargal is a 2011 indian tamil language film.",
    "1:##dhal desam ( 1996 ) and kadhalar dhinam ( 1999 ).",
    "2:wilsberg is a german television crime series.",
    "3:the second spanish republic was proclaimed in 1931.",
    "4:the film stars newcomers in the lead roles.",
    "5:einstein was born in ulm, germany, in 1879.",
    "6:##some heading with no preceding passage",
    "7:another passage about physics and relativity.",
    "8:medicine and biology are closely related fields.",
    "9:the olympic games were held in tokyo in 2021.",
]

SAMPLE_QUESTIONS = [
    {
        "id": "q001",
        "source": "hotpotqa",
        "question": "What year was the film released?",
        "answer": "2011",
        "question_type": "bridge",
        "evidence": [["Vaada Poda Nanbargal", ["Vaada Poda Nanbargal is a 2011 film."]]],
    },
    {
        "id": "q002",
        "source": "hotpotqa",
        "question": "Who directed the film?",
        "answer": "Manikai",
        "question_type": "bridge",
        "evidence": [["Vaada Poda Nanbargal", ["Directed by Manikai."]]],
    },
]


def _write_temp_dataset(chunks=SAMPLE_CHUNKS, questions=SAMPLE_QUESTIONS):
    """Write sample data to a temp dir and return (chunks_path, questions_path)."""
    tmp = tempfile.mkdtemp()
    cp = Path(tmp) / "chunks.json"
    qp = Path(tmp) / "questions.json"
    cp.write_text(json.dumps(chunks), encoding="utf-8")
    qp.write_text(json.dumps(questions), encoding="utf-8")
    return cp, qp


# ---------------------------------------------------------------------------
# _parse_chunk
# ---------------------------------------------------------------------------


def test_parse_chunk_with_index():
    c = _parse_chunk("42:some text about einstein")
    assert c.index == 42
    assert c.body == "some text about einstein"
    assert c.text == "42:some text about einstein"


def test_parse_chunk_multiline():
    c = _parse_chunk("7:line one\nline two")
    assert c.index == 7
    assert c.body == "line one\nline two"


def test_parse_chunk_no_prefix():
    c = _parse_chunk("text without any index prefix")
    assert c.index == -1
    assert c.body == "text without any index prefix"


def test_parse_chunk_zero_index():
    c = _parse_chunk("0:first chunk")
    assert c.index == 0


# ---------------------------------------------------------------------------
# load_linearrag
# ---------------------------------------------------------------------------


def test_load_linearrag_count():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp, name="test")
    assert len(ds.chunks) == len(SAMPLE_CHUNKS)
    assert len(ds.questions) == len(SAMPLE_QUESTIONS)
    assert ds.name == "test"


def test_load_linearrag_chunk_fields():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    c0 = ds.chunks[0]
    assert c0.index == 0
    assert "vaada poda nanbargal" in c0.body


def test_load_linearrag_question_fields():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    q = ds.questions[0]
    assert q.question_id == "q001"
    assert q.question == "What year was the film released?"
    assert q.answer == "2011"
    assert q.question_type == "bridge"
    assert len(q.evidence) == 1
    assert q.evidence[0][0] == "Vaada Poda Nanbargal"


def test_load_linearrag_max_chunks():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp, max_chunks=3)
    assert len(ds.chunks) == 3


def test_load_linearrag_chunk_texts():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    texts = ds.chunk_texts()
    assert texts == SAMPLE_CHUNKS


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_chunk_list_roundtrip(tmp_path):
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    out = tmp_path / "out_chunks.json"
    save_chunk_list(ds.chunks, out)
    with out.open() as f:
        restored = json.load(f)
    assert restored == SAMPLE_CHUNKS


def test_save_question_list_roundtrip(tmp_path):
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    out = tmp_path / "out_questions.json"
    save_question_list(ds.questions, out)
    with out.open() as f:
        restored = json.load(f)
    assert len(restored) == len(SAMPLE_QUESTIONS)
    assert restored[0]["id"] == "q001"
    assert restored[0]["answer"] == "2011"


# ---------------------------------------------------------------------------
# partition_linearrag_chunks
# ---------------------------------------------------------------------------


def test_partition_no_overlap():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    clients = partition_linearrag_chunks(ds.chunks, num_clients=3)
    stats = chunk_partition_stats(clients)
    assert stats["no_overlap"], "Each chunk index must appear exactly once"
    assert stats["total_chunks"] == len(SAMPLE_CHUNKS)


def test_partition_full_coverage():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    clients = partition_linearrag_chunks(ds.chunks, num_clients=5)
    total = sum(len(c.chunks) for c in clients)
    assert total == len(SAMPLE_CHUNKS)


def test_partition_deterministic():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    c1 = partition_linearrag_chunks(ds.chunks, num_clients=3)
    c2 = partition_linearrag_chunks(ds.chunks, num_clients=3)
    for a, b in zip(c1, c2):
        assert a.indices() == b.indices()


def test_partition_num_clients():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    for n in [1, 3, 5, 10]:
        clients = partition_linearrag_chunks(ds.chunks, num_clients=n)
        assert len(clients) == n


def test_partition_correct_assignment():
    """Chunk with index N goes to client N % num_clients."""
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    clients = partition_linearrag_chunks(ds.chunks, num_clients=5)
    for client in clients:
        for chunk in client.chunks:
            expected_cid = chunk.index % 5
            assert client.client_id == expected_cid, (
                f"Chunk {chunk.index} should be on client {expected_cid}, "
                f"got client {client.client_id}"
            )


def test_partition_chunk_texts():
    cp, qp = _write_temp_dataset()
    ds = load_linearrag(cp, qp)
    clients = partition_linearrag_chunks(ds.chunks, num_clients=5)
    for client in clients:
        for text in client.chunk_texts():
            assert text in SAMPLE_CHUNKS


# ---------------------------------------------------------------------------
# Integration: real processed files (skip if not present)
# ---------------------------------------------------------------------------

_PROCESSED = Path(__file__).resolve().parent.parent / "processed" / "hotpotqa"


@pytest.mark.skipif(
    not (_PROCESSED / "questions.json").exists(),
    reason="processed/hotpotqa not found — run scripts/preprocess_data.py first",
)
def test_processed_hotpotqa_structure():
    for m in range(5):
        client_dir = _PROCESSED / f"client_{m}"
        assert (client_dir / "chunks.json").exists(), f"Missing client_{m}/chunks.json"
        with (client_dir / "chunks.json").open() as f:
            chunks = json.load(f)
        assert len(chunks) > 0, f"client_{m} has 0 chunks"
        # Verify LinearRAG index prefix present on all chunks
        import re
        prefix_re = re.compile(r"^\d+:")
        for c in chunks[:5]:
            assert prefix_re.match(c), f"Chunk missing index prefix: {c[:50]!r}"


@pytest.mark.skipif(
    not (_PROCESSED / "questions.json").exists(),
    reason="processed/hotpotqa not found",
)
def test_processed_hotpotqa_questions():
    with (_PROCESSED / "questions.json").open() as f:
        qs = json.load(f)
    assert len(qs) == 1000
    assert "id" in qs[0] and "question" in qs[0] and "answer" in qs[0]
