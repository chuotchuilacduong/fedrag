"""Corpus → Chunk segmentation for DGRAG (spec §3.1).

Uses word count as a token proxy (1 word ≈ 1 token). This avoids a tiktoken
dependency while staying within the ballpark of the spec's 1200/100 token
chunk_size/overlap defaults.
"""

from __future__ import annotations

import uuid

from fedcond_grag.baselines.dgrag.data_types import Chunk


def chunk_document(
    doc: dict,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split one ``{title, text[, idx]}`` corpus item into Chunks."""
    title = doc.get("title", "") or ""
    body = doc.get("text", "") or ""
    doc_id = str(doc.get("idx", doc.get("id", uuid.uuid4().hex[:8])))
    full_text = f"{title}\n{body}".strip() if title else body.strip()
    if not full_text:
        return []

    words = full_text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    for i, start in enumerate(range(0, len(words), step)):
        word_slice = words[start: start + chunk_size]
        text = " ".join(word_slice)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}-{i}",
            text=text,
            doc_id=doc_id,
            tokens=len(word_slice),
        ))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_corpus(
    corpus: list[dict],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in corpus:
        out.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return out
