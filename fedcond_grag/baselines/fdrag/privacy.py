"""Algorithm 4: ε-LDP anonymization of exported QA memories.

Implements Eq.16 (randomized response with a per-type candidate set):

    Pr[e' = e]   = e^ε / (e^ε + c - 1)
    Pr[e' = w]   = 1   / (e^ε + c - 1),   w ∈ W \\ {e}

Design decisions taken here (per the spec's [GAP-6]/[GAP-7]):

- The candidate vocabulary is SHARED and DETERMINISTIC across devices --
  built once at fusion time by the coordinator from the union of typed
  spans seen across all clients. This side-steps the "sample W from local
  corpus leaks local content" trap the spec warns about while still not
  requiring an external entity list.
- Substitution consistency is per-memory-item (not per-device), matching
  the spec's recommendation for [GAP-7].
- Raw C(e) is NEVER exported (line 11 of the pseudocode) -- only the
  sanitized fact tuples travel.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

import numpy as np

from fedcond_grag.baselines.fdrag.data_types import Hyperedge, QAMemory
from fedcond_grag.baselines.fdrag.facts import normalize_anchor


# ---------------------------------------------------------------------------
# Candidate vocabulary
# ---------------------------------------------------------------------------
def build_candidate_vocab(
    edges: list[Hyperedge],
    sensitive_types: tuple[str, ...],
    *,
    encoder=None,
) -> dict[str, list[str]]:
    """Group all seen spans of sensitive types into a per-type vocabulary.

    Deterministic order (alphabetical). If ``encoder`` is passed, entries
    within a type are further sorted by embedding-space proximity to the
    type's centroid so ``nearest_c`` below returns semantically similar
    surrogates (§4.1 line 6-7)."""
    per_type: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        for span, label in e.facts:
            if label not in sensitive_types:
                continue
            key = normalize_anchor(span)
            if not key or key in seen[label]:
                continue
            seen[label].add(key)
            per_type[label].append(span)
    for label in per_type:
        per_type[label].sort()

    if encoder is not None:
        for label, spans in per_type.items():
            if len(spans) < 2:
                continue
            vecs = np.asarray(encoder.encode(spans, batch_size=64, show_progress_bar=False)).astype(np.float32)
            centroid = vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            order = np.argsort(-(vecs @ centroid))
            per_type[label] = [spans[i] for i in order]
    return dict(per_type)


def _nearest_c(vocab: list[str], target: str, c: int) -> list[str]:
    """Deterministic 'nearest neighbours' fallback: use the first c items
    from the vocab (which was pre-sorted by centroid similarity if an
    encoder was passed to ``build_candidate_vocab``). Always includes
    ``target`` in position 0."""
    if target not in vocab:
        pool = vocab
    else:
        pool = [target] + [v for v in vocab if v != target]
    return pool[:c] if len(pool) >= c else pool


# ---------------------------------------------------------------------------
# Eq.16: randomized response sampler
# ---------------------------------------------------------------------------
def _sample_replacement(
    e: str, W: list[str], epsilon: float, c: int, rng: random.Random
) -> str:
    """Draw e' ~ RR(ε, W)."""
    if not W:
        return e
    if len(W) < c:
        # Degrade gracefully -- use the smaller pool, keep the ratio.
        c = len(W)
    p_keep = math.exp(epsilon) / (math.exp(epsilon) + c - 1)
    p_other = 1.0 / (math.exp(epsilon) + c - 1)
    weights = [p_keep if w == e else p_other for w in W]
    total = sum(weights)
    weights = [w / total for w in weights]
    r, acc = rng.random(), 0.0
    for w, prob in zip(W, weights):
        acc += prob
        if r <= acc:
            return w
    return W[-1]


# ---------------------------------------------------------------------------
# Anonymize one client's memory bank (Alg 4)
# ---------------------------------------------------------------------------
def anonymize_memories(
    memories: list[QAMemory],
    edges_by_id: dict[str, Hyperedge],
    *,
    candidate_vocab: dict[str, list[str]],
    epsilon: float,
    c: int,
    sensitive_types: tuple[str, ...],
    seed: int = 0,
) -> list[QAMemory]:
    """Return sanitized copies of each memory. Raw C(e) is not copied --
    outbound memories carry only ``sanitized_facts`` per support edge.

    Substitution is *consistent within one memory item*: a given entity `e`
    is replaced by the same `e'` in the question, answer, and every
    supporting fact tuple. This is what the spec's [GAP-7] resolution asks
    for."""
    rng = random.Random(seed)
    out: list[QAMemory] = []

    for i, m in enumerate(memories):
        # Collect all sensitive entities appearing anywhere in this memory
        # item (question, answer, supporting fact tuples).
        mapping: dict[str, str] = {}
        pending: list[tuple[str, str]] = []

        for e_id in m.support_edge_ids:
            edge = edges_by_id.get(e_id)
            if edge is None:
                continue
            for span, label in edge.facts:
                if label in sensitive_types and span not in mapping:
                    pending.append((span, label))

        # Also entities that appear as literal substrings in q/a even if
        # they aren't in the facts -- keeps a/q consistent after subs.
        # (Deferred: without a per-item NER pass we can't do this cheaply.
        # Fact spans cover the [PERSON, ORG, GPE, LOC] case dominant in the
        # HotPotQA/2Wiki/MuSiQue benchmarks.)

        for span, label in pending:
            W = _nearest_c(candidate_vocab.get(label, []), span, c=c)
            mapping[span] = _sample_replacement(span, W, epsilon, c, rng)

        san_q = _apply_map(m.question, mapping)
        san_a = _apply_map(m.answer, mapping)
        san_facts: list[tuple[str, str]] = []
        for e_id in m.support_edge_ids:
            edge = edges_by_id.get(e_id)
            if edge is None:
                continue
            for span, label in edge.facts:
                san_facts.append((mapping.get(span, span), label))

        anchors = {normalize_anchor(s) for s, _ in san_facts}
        out.append(QAMemory(
            id=f"f{m.origin_client}:{m.id}",
            question=san_q,
            answer=san_a,
            support_edge_ids=list(m.support_edge_ids),
            q_vec=m.q_vec.copy(),          # encoder re-runs anyway; harmless to copy
            anchors=anchors,
            origin_client=m.origin_client,
            is_foreign=True,
            sanitized_facts=san_facts,
        ))
    return out


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    if not mapping or not text:
        return text
    # Longest span first to avoid substring collisions ("New York" before "York").
    for span in sorted(mapping.keys(), key=len, reverse=True):
        if span and span in text:
            text = text.replace(span, mapping[span])
    return text
