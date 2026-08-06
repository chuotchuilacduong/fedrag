"""FD-RAG configuration.

Mirrors §1.2 of ``implementfd_rag.md`` (the ``fd-rag.pdf`` implementation
spec extracted for this repo). Every value here is either directly named in
the spec's YAML block or is one of the ``[GAP-*]`` decisions the spec
explicitly asks the implementer to expose as config with a recommended
default (see the trailing decision table in §7 for the recommended defaults
used below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FDRAGConfig:
    # --- structure learning (SAHL, Alg 1) ---
    mu: float = 0.5                       # incidence sparsification threshold
    lam: float = 0.6                      # intra vs inter loss balance (Eq.8)
    gamma: float = 1.0                    # inter margin, Eq.7 [GAP-2]
    opt_steps: int = 300
    learning_rate: float = 0.05
    m_p_ratio: float = 0.25               # M_p = ceil(N_p * ratio) (spec: N_p / 4)
    m_s_ratio: float = 0.25               # M_s = ceil(N_s * ratio) (spec: N_s / 4)
    max_pairs: int | None = None          # cap inter-hyperedge pair count per step
    sparsify: Literal["absolute", "relative", "top_r"] = "absolute"  # [GAP-3]
    sparsify_top_r: int = 1               # only used when sparsify == "top_r"

    # --- inference (Alg 3) ---
    alpha: float = 0.7                    # Eq.11 dense/structural balance [GAP-2]
    delta: float = 0.8                    # Memorizer confidence threshold
    top_k: int = 5                        # Cognizer top-K memory retrieval [GAP-1]
    ann_shortlist_mult: int = 10          # dense shortlist size = top_k * this

    # --- QA memory synthesis (Alg 2) ---
    memories_per_edge: int = 1            # R_m; [GAP-5]
    question_types: tuple[str, ...] = (
        "Set", "Comparison", "Aggregation",
        "Multi-hop", "Post-processing Heavy", "False Premise",
    )
    neighbor_selection: Literal["anchor_overlap", "prototype_cosine", "none"] = "anchor_overlap"  # [GAP-4]
    max_neighbors: int = 2
    compositional_types: tuple[str, ...] = (
        "Multi-hop", "Comparison", "Aggregation", "Set",
    )

    # --- federation (Alg 4) ---
    num_clients: int = 5                  # [GAP-1] split from top_k
    epsilon: float = 1.0                  # LDP budget
    ldp_c: int = 5                        # candidate set size for perturbation
    sensitive_types: tuple[str, ...] = ("PERSON", "ORG", "GPE", "LOC")
    foreign_evidence_mode: Literal["facts_only", "drop"] = "facts_only"  # [GAP-8]

    # --- runtime knobs (not spec-visible) ---
    device: str = "cpu"                   # torch device for SAHL
    encoder_batch_size: int = 64
    max_memories_total: int | None = None  # hard cap on QA memory count per client
    spacy_model: str = "en_core_web_sm"
    seed: int = 42

    # cached derived state (set at runtime)
    encoder_dim: int | None = field(default=None)
