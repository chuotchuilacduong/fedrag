"""DGRAG data structures (spec §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    tokens: int


@dataclass
class Entity:
    name: str                          # canonical upper-cased key
    type: str
    description: str
    source_chunks: list[str] = field(default_factory=list)
    vec: Optional[np.ndarray] = None   # embed(name + " " + description)


@dataclass
class Relation:
    source: str                        # entity name
    target: str                        # entity name
    keyword: str
    description: str
    weight: float = 1.0                # [FREE] accumulate on duplicates
    source_chunks: list[str] = field(default_factory=list)
    vec: Optional[np.ndarray] = None   # embed(keyword + " " + description)


@dataclass
class Subgraph:
    subgraph_id: str                   # f"{edge_id}-{idx}"
    edge_id: str
    entity_names: list[str]
    relation_keys: list[tuple[str, str]]  # (source, target)
    summary: str
    summary_vec: Optional[np.ndarray] = None


@dataclass
class SummaryRecord:
    subgraph_id: str
    edge_id: str
    summary: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Evidence:
    """Formatted retrieval result returned by retrieve_only()."""
    entities: list[Entity]
    relations: list[Relation]
    chunks: list[Chunk]
    edge_id: str


@dataclass
class GateDecision:
    route: str                 # "local" | "global"
    reason: str
    score: float = 0.0
    best_answer: Optional[str] = None
    s_cos: float = 0.0
    s_jac: float = 0.0
    s_sem: float = 0.0


@dataclass
class AnswerResult:
    answer: str
    route: str                 # "local" | "global" | "fallback"
    gate_score: float
    llm_calls: int
    latency_sec: float = 0.0
    peer_edges_contacted: int = 0
