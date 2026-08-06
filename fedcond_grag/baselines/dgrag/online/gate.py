"""Step ⑤: Gate Mechanism (spec §4.2).

Three sequential checks, order matters — confidence runs FIRST and
short-circuits to GLOBAL if any candidate expresses uncertainty. The
spec is explicit: "Order matters."

Similarity score = unweighted mean of exactly 3 sub-metrics:
  s_cos  = mean pairwise cosine of embedded answers (vector level)
  s_jac  = mean pairwise Jaccard of word-token sets (lexical level)
  s_sem  = SLM semantic consistency judgment in [0,1]

If score >= GATE_THRESHOLD: pick best answer via SLM; return LOCAL.
Otherwise: return GLOBAL.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import numpy as np

from fedcond_grag.baselines.dgrag.data_types import GateDecision
from fedcond_grag.baselines.dgrag.llm import DGRAGModel
from fedcond_grag.baselines.dgrag.prompts import (
    build_confidence_prompt,
    build_select_best_prompt,
    build_semantic_consistency_prompt,
)

_log = logging.getLogger("dgrag.gate")

# Phrases that indicate low confidence (matched case-insensitively)
_LOW_CONF_PHRASES = (
    "insufficient information",
    "i don't know",
    "i do not know",
    "the provided data does not",
    "need more details",
    "cannot determine",
    "not enough information",
    "information is insufficient",
    "does not contain",
    "unable to answer",
)


def _fast_confidence_check(answers: list[str]) -> bool:
    """Heuristic pre-check before calling the SLM: if any answer contains
    low-confidence phrases, return False immediately."""
    for a in answers:
        lower = a.lower()
        if any(phrase in lower for phrase in _LOW_CONF_PHRASES):
            return False
    return True


def _mean_pairwise_cosine(answers: list[str], encoder) -> float:
    if len(answers) < 2:
        return 1.0
    vecs = np.asarray(encoder.encode(answers, batch_size=len(answers), show_progress_bar=False))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    vecs = vecs / norms
    pairs = list(combinations(range(len(vecs)), 2))
    if not pairs:
        return 1.0
    sims = [float(np.dot(vecs[i], vecs[j])) for i, j in pairs]
    return sum(sims) / len(sims)


def _mean_pairwise_jaccard(answers: list[str]) -> float:
    if len(answers) < 2:
        return 1.0
    token_sets = [set(a.lower().split()) for a in answers]
    pairs = list(combinations(range(len(token_sets)), 2))
    if not pairs:
        return 1.0
    scores = []
    for i, j in pairs:
        union = token_sets[i] | token_sets[j]
        inter = token_sets[i] & token_sets[j]
        scores.append(len(inter) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def run_gate(
    query: str,
    answers: list[str],
    slm: DGRAGModel,
    encoder,
    *,
    gate_threshold: float = 0.75,
    skip_confidence: bool = False,   # w/o_CD ablation
    skip_similarity: bool = False,   # w/o_SE ablation
) -> tuple[GateDecision, int]:
    """Execute the gate mechanism. Returns (GateDecision, llm_calls_used)."""
    llm_calls = 0

    # Non-empty answers only
    valid = [a for a in answers if a.strip()]
    if not valid:
        return GateDecision(route="global", reason="empty answers", score=0.0), 0

    # ── (1) Confidence Detection ─────────────────────────────────────
    if not skip_confidence:
        # Fast heuristic check first (no LLM call)
        if not _fast_confidence_check(valid):
            return GateDecision(route="global", reason="low confidence (heuristic)", score=0.0), 0

        conf_prompt = build_confidence_prompt(query, valid)
        conf_raw = slm.infer_json(conf_prompt, temperature=0.0, default={"confident": True})
        llm_calls += 1
        confident = conf_raw.get("confident", True)
        if isinstance(confident, str):
            confident = confident.lower() not in ("false", "no", "0")
        if not confident:
            reason = conf_raw.get("reason", "low confidence")
            return GateDecision(route="global", reason=reason, score=0.0), llm_calls

    # ── (2) Similarity Evaluation ────────────────────────────────────
    if skip_similarity:
        # w/o_SE ablation: no similarity eval; pick candidate 0
        best = valid[0] if valid else ""
        return GateDecision(
            route="local", reason="similarity skipped (w/o_SE)", score=1.0,
            best_answer=best, s_cos=1.0, s_jac=1.0, s_sem=1.0,
        ), llm_calls

    s_cos = _mean_pairwise_cosine(valid, encoder)
    s_jac = _mean_pairwise_jaccard(valid)

    sem_prompt = build_semantic_consistency_prompt(query, valid)
    sem_raw = slm.infer_json(sem_prompt, temperature=0.0, default={"consistency": 0.5})
    llm_calls += 1
    s_sem_raw = sem_raw.get("consistency", 0.5)
    try:
        s_sem = float(s_sem_raw)
    except (TypeError, ValueError):
        s_sem = 0.5
    s_sem = max(0.0, min(1.0, s_sem))

    score = (s_cos + s_jac + s_sem) / 3.0

    # ── (3) Similarity-based Selection ───────────────────────────────
    if score >= gate_threshold:
        sel_prompt = build_select_best_prompt(query, valid)
        sel_raw = slm.infer_json(sel_prompt, temperature=0.0, default={"index": 0})
        llm_calls += 1
        idx_raw = sel_raw.get("index", 0)
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(idx, len(valid) - 1))
        return GateDecision(
            route="local", reason="passed gate",
            score=score, best_answer=valid[idx],
            s_cos=s_cos, s_jac=s_jac, s_sem=s_sem,
        ), llm_calls
    else:
        return GateDecision(
            route="global", reason=f"score {score:.3f} < threshold {gate_threshold:.3f}",
            score=score, s_cos=s_cos, s_jac=s_jac, s_sem=s_sem,
        ), llm_calls
