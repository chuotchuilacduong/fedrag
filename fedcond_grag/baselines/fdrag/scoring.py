"""Eq.11-13 scoring: dense similarity + Dice anchor cover.

Kept pure so it can be unit-tested in isolation from any encoder / LLM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def dice_cover(a_q: set[str], a_r: set[str]) -> float:
    """Eq.12: Cover(q, S_r) = 2 |A_q ∩ A_r| / (|A_q| + |A_r|)."""
    denom = len(a_q) + len(a_r)
    if denom == 0:
        return 0.0
    return 2.0 * len(a_q & a_r) / denom


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Assumes both inputs are already L2-normalized (encoder default)."""
    if u.size == 0 or v.size == 0:
        return 0.0
    return float(np.dot(u, v))


def combined_score(
    sim: float, cover: float, alpha: float
) -> float:
    """Eq.11: Score(q, γ_r) = α * Sim + (1 - α) * Cover."""
    return alpha * sim + (1.0 - alpha) * cover


def score_shortlist(
    q_vec: np.ndarray,
    q_anchors: set[str],
    candidates: Iterable[tuple[str, float]],  # (memory_id, dense_sim from shortlist)
    id_to_anchors: dict[str, set[str]],
    alpha: float,
) -> list[tuple[str, float, float, float]]:
    """Rescore an ANN dense shortlist with Eq.11+12.

    Returns: [(memory_id, combined_score, sim, cover)] sorted by combined_score desc.
    """
    out = []
    for mid, sim in candidates:
        cover = dice_cover(q_anchors, id_to_anchors.get(mid, set()))
        out.append((mid, combined_score(sim, cover, alpha), sim, cover))
    out.sort(key=lambda t: -t[1])
    return out
