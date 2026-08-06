"""Typed-fact extraction (Eq.9) and anchor normalization for FD-RAG.

Per §2.2 of the spec the fact extractor is spaCy (``en_core_web_*``): we take
``doc.ents`` for named entities plus ``doc.noun_chunks`` as ``NOUN_CHUNK`` and
deduplicate by lowercased, whitespace-normalized span. Anchor normalization
must match on both the query side and the hyperedge side (see §3 notes) or
the Dice cover in Eq.12 silently reads 0.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Iterable

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_anchor(span: str) -> str:
    """Same normalization the fedcond_grag eval uses so anchors line up with
    gold answers when appropriate. Lowercase, strip punctuation, strip
    articles, collapse whitespace."""
    if not span:
        return ""
    s = span.lower().translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


@lru_cache(maxsize=4)
def _load_spacy(model_name: str = "en_core_web_sm"):
    """Load spaCy once. If neither spacy nor the model is available, return
    ``None`` -- callers must handle that and fall back to a heuristic
    extractor. This keeps the baseline importable in envs where spaCy is not
    installed (see requirements.txt: spaCy is a project dep but not part of
    the sandbox we may be invoked from).
    """
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load(model_name, disable=["lemmatizer"])
    except Exception:
        try:
            return spacy.load(model_name)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Heuristic fallback: capitalized-run detector (used only when spaCy is
# unavailable). Type tag is "CHUNK" to distinguish from real NER labels.
# ---------------------------------------------------------------------------
_CAP_RUN = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b")
_NUMBER = re.compile(r"\b\d{1,4}(?:[-/]\d{1,4}){0,2}\b")


def _heuristic_facts(text: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    for m in _CAP_RUN.finditer(text):
        span = m.group(0).strip()
        if len(span) > 2:
            spans.append((span, "CHUNK"))
    for m in _NUMBER.finditer(text):
        spans.append((m.group(0), "CARDINAL"))
    return spans


def extract_facts(text: str, spacy_model: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """Eq.9: return the typed fact set F_m = ⟨(u_1, v_1), …⟩ for one text
    unit. Deduplicated by normalized span, order preserved."""
    if not text or not text.strip():
        return []

    nlp = _load_spacy(spacy_model)
    if nlp is None:
        raw = _heuristic_facts(text)
    else:
        raw = []
        try:
            doc = nlp(text)
            for ent in doc.ents:
                raw.append((ent.text.strip(), ent.label_))
            for chunk in doc.noun_chunks:
                raw.append((chunk.text.strip(), "NOUN_CHUNK"))
        except Exception:
            raw = _heuristic_facts(text)

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for span, label in raw:
        key = normalize_anchor(span)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((span, label))
    return out


def extract_facts_batch(texts: Iterable[str], spacy_model: str = "en_core_web_sm") -> list[list[tuple[str, str]]]:
    """Same as ``extract_facts`` but processes many strings in one pipe --
    ~5-10x faster with spaCy for long doc lists."""
    texts = list(texts)
    nlp = _load_spacy(spacy_model)
    if nlp is None:
        return [extract_facts(t, spacy_model) for t in texts]

    out: list[list[tuple[str, str]]] = []
    try:
        for doc in nlp.pipe(texts, batch_size=64):
            raw = [(ent.text.strip(), ent.label_) for ent in doc.ents]
            raw.extend((chunk.text.strip(), "NOUN_CHUNK") for chunk in doc.noun_chunks)
            seen: set[str] = set()
            per: list[tuple[str, str]] = []
            for span, label in raw:
                key = normalize_anchor(span)
                if not key or key in seen:
                    continue
                seen.add(key)
                per.append((span, label))
            out.append(per)
    except Exception:
        return [extract_facts(t, spacy_model) for t in texts]
    return out


def anchors_from_facts(facts: Iterable[tuple[str, str]]) -> set[str]:
    return {normalize_anchor(span) for span, _ in facts if span and span.strip()}


def query_anchors(question: str, spacy_model: str = "en_core_web_sm") -> set[str]:
    """A_q for §3 inference. Same extractor as hyperedge anchors so the two
    sets are directly comparable in Eq.12."""
    return anchors_from_facts(extract_facts(question, spacy_model))
