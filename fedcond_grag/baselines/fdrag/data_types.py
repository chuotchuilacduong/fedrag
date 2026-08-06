"""Core FD-RAG data structures.

Verbatim from the ``implementfd_rag.md`` §1.3 table -- these are the objects
that flow through Stages 1/2/3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class TextUnit:
    id: str
    text: str
    granularity: Literal["paragraph", "sentence"]
    doc_id: str
    parent_id: Optional[str] = None
    vec: Optional[np.ndarray] = None


@dataclass
class Hyperedge:
    id: str
    granularity: Literal["paragraph", "sentence"]
    member_unit_ids: list[str]
    weights: np.ndarray               # H[n, m] over members
    prototype: np.ndarray             # Eq.5
    context: list[str] = field(default_factory=list)          # C(e_m)
    facts: list[tuple[str, str]] = field(default_factory=list)  # F_m
    anchors: set[str] = field(default_factory=set)              # A(e), normalized


@dataclass
class QAMemory:
    id: str
    question: str
    answer: str
    support_edge_ids: list[str]       # S^r_m; must contain the owning edge
    q_vec: np.ndarray                 # f(q_r)
    anchors: set[str] = field(default_factory=set)  # A_r = ∪_{e ∈ S_r} A(e)
    origin_client: int = 0
    is_foreign: bool = False
    sanitized_facts: Optional[list[tuple[str, str]]] = None  # populated by Alg 4
