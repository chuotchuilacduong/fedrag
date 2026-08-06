"""Algorithm 2: QA memory construction for FD-RAG.

Per §2.2 of the spec, for every hyperedge:
  1. run spaCy fact extraction over its member contexts (Eq.9) and store
     normalized anchors A(e_m);
  2. for each of R_m memories, sample a question type, gather any additional
     supporting hyperedges (compositional types only, per [GAP-4]), and ask
     the LLM to write one grounded QA pair (Table 9);
  3. embed every accepted question and build an in-memory dense index.

Validation follows the spec's "not in the paper, but necessary" list:
non-empty q/a, short answer, answer text grounded in context or facts, no
near-duplicate questions per edge.
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from typing import Callable, Iterable, Optional

import numpy as np

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.data_types import Hyperedge, QAMemory
from fedcond_grag.baselines.fdrag.facts import (
    anchors_from_facts,
    extract_facts_batch,
    normalize_anchor,
)
from fedcond_grag.baselines.fdrag.prompts import build_qa_prompt

_log = logging.getLogger("fdrag.memory")

_QA_RE = re.compile(
    r"Question\s*:\s*(?P<q>.+?)\s*Answer\s*:\s*(?P<a>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _parse_qa(response: str) -> Optional[tuple[str, str]]:
    if not response:
        return None
    text = response.strip()
    # First try the exact 2-line format
    m = _QA_RE.search(text)
    if m:
        q, a = m.group("q").strip(), m.group("a").strip()
        # Strip anything after a newline in the answer (spec-strict output)
        a = a.splitlines()[0].strip() if a else a
        return (q, a) if q and a else None
    # Fallback: two non-empty lines "Q\nA"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0].lstrip("Q:").strip(), lines[1].lstrip("A:").strip()
    return None


def _valid(
    q: str, a: str, ctx: str, facts_norm: set[str], allow_ungrounded: bool
) -> bool:
    if not q or not a:
        return False
    if len(a.split()) > 40:
        return False
    if allow_ungrounded:
        return True
    a_norm = normalize_anchor(a)
    if a_norm and a_norm in normalize_anchor(ctx):
        return True
    if a_norm in facts_norm:
        return True
    return False


def _select_neighbors(
    edge: Hyperedge,
    edges: list[Hyperedge],
    edge_index: dict[str, int],
    qtype: str,
    cfg: FDRAGConfig,
) -> list[Hyperedge]:
    if qtype not in cfg.compositional_types or cfg.neighbor_selection == "none":
        return []
    if cfg.neighbor_selection == "prototype_cosine":
        proto = edge.prototype / (np.linalg.norm(edge.prototype) + 1e-8)
        sims = []
        for other in edges:
            if other.id == edge.id:
                continue
            v = other.prototype / (np.linalg.norm(other.prototype) + 1e-8)
            sims.append((float(np.dot(proto, v)), other))
        sims.sort(key=lambda t: -t[0])
        return [o for _, o in sims[: cfg.max_neighbors]]
    # anchor_overlap (default)
    overlaps = []
    for other in edges:
        if other.id == edge.id:
            continue
        share = len(edge.anchors & other.anchors)
        if share > 0:
            overlaps.append((share, other))
    overlaps.sort(key=lambda t: -t[0])
    return [o for _, o in overlaps[: cfg.max_neighbors]]


def build_memories(
    hyperedges: list[Hyperedge],
    cfg: FDRAGConfig,
    *,
    llm_infer: Optional[Callable[[str], str]],
    encoder,
    client_id: int = 0,
) -> list[QAMemory]:
    """Algorithm 2. If ``llm_infer`` is None the LLM step is skipped and one
    trivial "cloze" memory is created per edge (span-copy questions) so
    downstream pieces are still testable without an LLM endpoint."""

    rng = random.Random(cfg.seed + client_id * 1000)

    # ---- (line 1-3) Populate per-edge facts + anchors ----
    fact_lists = extract_facts_batch(
        [" ".join(e.context) for e in hyperedges], cfg.spacy_model
    )
    for edge, facts in zip(hyperedges, fact_lists):
        edge.facts = facts
        edge.anchors = anchors_from_facts(facts)

    edge_index = {e.id: i for i, e in enumerate(hyperedges)}

    # ---- (line 5-13) Generate QA memories ----
    memories: list[QAMemory] = []
    per_edge_norm_q: dict[str, set[str]] = {}
    total_cap = cfg.max_memories_total

    for edge in hyperedges:
        if total_cap is not None and len(memories) >= total_cap:
            break
        for _ in range(cfg.memories_per_edge):
            if total_cap is not None and len(memories) >= total_cap:
                break
            qtype = rng.choice(cfg.question_types)
            support = [edge] + _select_neighbors(edge, hyperedges, edge_index, qtype, cfg)
            ctx = "\n\n".join(t for e in support for t in e.context)
            all_facts = [f for e in support for f in e.facts]
            facts_norm = {normalize_anchor(s) for s, _ in all_facts}

            if llm_infer is None:
                qa = _fallback_cloze(edge, rng)
                if qa is None:
                    continue
                q, a = qa
                allow_ungrounded = False
            else:
                prompt = build_qa_prompt(qtype=qtype, facts=all_facts, text=ctx)
                try:
                    raw = llm_infer(prompt)
                except Exception as exc:
                    _log.warning("QA-gen LLM call failed on edge %s: %s", edge.id, exc)
                    continue
                parsed = _parse_qa(raw)
                if not parsed:
                    continue
                q, a = parsed
                allow_ungrounded = qtype == "False Premise"

            if not _valid(q, a, ctx, facts_norm, allow_ungrounded):
                continue

            q_norm = normalize_anchor(q)
            seen_here = per_edge_norm_q.setdefault(edge.id, set())
            if q_norm in seen_here:
                continue
            seen_here.add(q_norm)

            support_ids = [e.id for e in support]
            anchors = set().union(*(e.anchors for e in support))
            memories.append(QAMemory(
                id=f"m:{uuid.uuid4().hex[:8]}",
                question=q,
                answer=a,
                support_edge_ids=support_ids,
                q_vec=np.zeros((0,), dtype=np.float32),  # filled below
                anchors=anchors,
                origin_client=client_id,
                is_foreign=False,
            ))

    # ---- (line 16-18) Embed questions in one batch ----
    if memories:
        vecs = encoder.encode(
            [m.question for m in memories], batch_size=cfg.encoder_batch_size, show_progress_bar=False
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        for m, v in zip(memories, vecs):
            m.q_vec = v

    return memories


# ---------------------------------------------------------------------------
# Fallback QA when no LLM endpoint is provided. Deterministic and grounded --
# picks the longest named span from the edge's facts and generates a "What is
# mentioned about X?" style cloze answered by that same span. This is only
# used for smoke tests / offline dev; real evaluation must set an LLM.
# ---------------------------------------------------------------------------
def _fallback_cloze(edge: Hyperedge, rng: random.Random) -> Optional[tuple[str, str]]:
    if not edge.facts:
        return None
    spans = sorted(edge.facts, key=lambda t: -len(t[0]))
    span, label = spans[0]
    q = f"What is mentioned about {span}?"
    # Answer = the shortest sentence in the context that contains the span
    sentences = [s.strip() for c in edge.context for s in re.split(r"(?<=[\.\!\?])\s+", c) if span.lower() in s.lower()]
    if not sentences:
        return None
    a = min(sentences, key=len)
    if len(a.split()) > 40:
        a = " ".join(a.split()[:40])
    return q, a


# ---------------------------------------------------------------------------
# Dense retrieval index (numpy inner-product, since encoder returns
# L2-normalized embeddings by default per node_encoder.encode).
# ---------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.matrix: Optional[np.ndarray] = None
        self.ids: list[str] = []

    def build(self, items: Iterable[QAMemory]) -> None:
        items = list(items)
        if not items:
            self.matrix = np.zeros((0, self.dim), dtype=np.float32)
            self.ids = []
            return
        self.matrix = np.stack([m.q_vec for m in items]).astype(np.float32)
        self.ids = [m.id for m in items]

    def shortlist(self, q_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.matrix is None or len(self.ids) == 0:
            return []
        sims = self.matrix @ q_vec.astype(np.float32)
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]
