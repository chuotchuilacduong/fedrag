"""Corpus -> TextUnit segmentation for FD-RAG.

Two granularities per §2 of the spec: ``paragraph`` (t = p) and ``sentence``
(t = s). Every corpus item ``{title, text}`` becomes one paragraph unit and
one sentence unit per split sentence, with the paragraph as ``parent_id``
(§1.3 TextUnit).

Sentence splitting is intentionally regex-based here so this module has no
hard dep on spaCy for the segment stage (Stage 1 only *needs* spaCy for
Alg 2 fact extraction).
"""

from __future__ import annotations

import re
from typing import Iterable

from fedcond_grag.baselines.fdrag.data_types import TextUnit


# Simple sentence boundary: end-of-sentence punctuation followed by whitespace.
# Not perfect (Wikipedia-style abbreviations will over-split) but sufficient
# for §1.2's D=1024/BGE-M3 sentence embeddings and the "N_s/4" hyperedge budget.
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"'\(])")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def segment_corpus(corpus: Iterable[dict]) -> list[TextUnit]:
    """Turn a list of ``{title, text}`` (or ``{title, text, idx}``) into
    interleaved paragraph + sentence TextUnits with parent links.
    """
    units: list[TextUnit] = []
    for i, item in enumerate(corpus):
        title = item.get("title", "") or ""
        body = item.get("text", "") or ""
        doc_id = str(item.get("idx", i))
        para_text = (f"{title}\n{body}" if title else body).strip()
        if not para_text:
            continue
        para_id = f"p:{doc_id}"
        units.append(TextUnit(
            id=para_id,
            text=para_text,
            granularity="paragraph",
            doc_id=doc_id,
            parent_id=None,
        ))
        for si, sent in enumerate(_split_sentences(body)):
            units.append(TextUnit(
                id=f"s:{doc_id}:{si}",
                text=sent,
                granularity="sentence",
                doc_id=doc_id,
                parent_id=para_id,
            ))
    return units


def split_by_granularity(units: list[TextUnit]) -> tuple[list[TextUnit], list[TextUnit]]:
    p = [u for u in units if u.granularity == "paragraph"]
    s = [u for u in units if u.granularity == "sentence"]
    return p, s
