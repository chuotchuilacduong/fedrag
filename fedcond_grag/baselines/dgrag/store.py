"""Per-edge storage: three vector DBs + one graph DB (spec §2.1).

NumpyVectorStore is a lightweight cosine-similarity store (same pattern as
fdrag's DenseIndex) — no external vector DB dependency. Each edge holds three
instances: vdb_entities, vdb_relations, vdb_chunks. The cloud coordinator
holds a fourth for subgraph summaries.

NetworkX Graph wraps entities as nodes and relations as directed edges.
"""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx
import numpy as np

from fedcond_grag.baselines.dgrag.data_types import (
    Chunk,
    Entity,
    Relation,
    Subgraph,
    SummaryRecord,
)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class NumpyVectorStore:
    """Cosine-similarity store backed by a numpy matrix.

    Records are arbitrary dicts; ``vectors`` are their L2-normalised embeddings.
    """

    def __init__(self):
        self._ids: list[str] = []
        self._records: list[Any] = []
        self._matrix: Optional[np.ndarray] = None   # (N, D)

    def upsert(self, record_id: str, record: Any, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        if record_id in self._ids:
            idx = self._ids.index(record_id)
            self._records[idx] = record
            self._matrix[idx] = v
        else:
            self._ids.append(record_id)
            self._records.append(record)
            if self._matrix is None:
                self._matrix = v[np.newaxis, :]
            else:
                self._matrix = np.vstack([self._matrix, v[np.newaxis, :]])

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[float, Any]]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = self._matrix @ q
        k = min(top_k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(float(sims[i]), self._records[i]) for i in top]

    def search_with_threshold(self, query_vec: np.ndarray, top_k: int, threshold: float) -> list[tuple[float, Any]]:
        results = self.search(query_vec, top_k)
        return [(s, r) for s, r in results if s >= threshold]

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Per-edge knowledge base
# ---------------------------------------------------------------------------

class EdgeKnowledgeBase:
    """Holds the 3 vector DBs + graph DB for one edge node (spec §2.1)."""

    def __init__(self, edge_id: str, encoder):
        self.edge_id = edge_id
        self.encoder = encoder
        self.vdb_entities = NumpyVectorStore()
        self.vdb_relations = NumpyVectorStore()
        self.vdb_chunks = NumpyVectorStore()
        self.graph: nx.Graph = nx.Graph()
        # In-memory record lookup
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str], Relation] = {}
        self.chunks: dict[str, Chunk] = {}
        self.subgraphs: list[Subgraph] = []

    def _entity_text(self, e: Entity) -> str:
        return f"{e.name} {e.description}"

    def _relation_text(self, r: Relation) -> str:
        return f"{r.keyword} {r.description}"

    def add_entity(self, entity: Entity) -> None:
        key = entity.name.strip().upper()
        self.entities[key] = entity
        vec = np.asarray(
            self.encoder.encode([self._entity_text(entity)], batch_size=1, show_progress_bar=False)
        )[0]
        entity.vec = vec.astype(np.float32)
        self.vdb_entities.upsert(key, entity, vec)
        self.graph.add_node(key, type=entity.type, description=entity.description)

    def add_relation(self, relation: Relation) -> None:
        key = (relation.source.strip().upper(), relation.target.strip().upper())
        if key in self.relations:
            existing = self.relations[key]
            existing.weight += relation.weight
            existing.description += f"; {relation.description}"
            existing.source_chunks = list(set(existing.source_chunks + relation.source_chunks))
        else:
            self.relations[key] = relation
        vec = np.asarray(
            self.encoder.encode([self._relation_text(relation)], batch_size=1, show_progress_bar=False)
        )[0]
        relation.vec = vec.astype(np.float32)
        rel_id = f"{key[0]}___{key[1]}"
        self.vdb_relations.upsert(rel_id, relation, vec)
        if key[0] in self.graph and key[1] in self.graph:
            if self.graph.has_edge(key[0], key[1]):
                self.graph[key[0]][key[1]]["weight"] = self.relations[key].weight
            else:
                self.graph.add_edge(key[0], key[1], weight=self.relations[key].weight,
                                    keyword=relation.keyword)

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        vec = np.asarray(
            self.encoder.encode([chunk.text], batch_size=1, show_progress_bar=False)
        )[0]
        self.vdb_chunks.upsert(chunk.chunk_id, chunk, vec)

    def add_entities_batch(self, entities: list[Entity]) -> None:
        if not entities:
            return
        texts = [self._entity_text(e) for e in entities]
        vecs = np.asarray(self.encoder.encode(texts, batch_size=64, show_progress_bar=False))
        for entity, vec in zip(entities, vecs):
            key = entity.name.strip().upper()
            entity.vec = vec.astype(np.float32)
            self.entities[key] = entity
            self.vdb_entities.upsert(key, entity, vec)
            self.graph.add_node(key, type=entity.type, description=entity.description)

    def add_relations_batch(self, relations: list[Relation]) -> None:
        if not relations:
            return
        texts = [self._relation_text(r) for r in relations]
        vecs = np.asarray(self.encoder.encode(texts, batch_size=64, show_progress_bar=False))
        for relation, vec in zip(relations, vecs):
            key = (relation.source.strip().upper(), relation.target.strip().upper())
            relation.vec = vec.astype(np.float32)
            if key in self.relations:
                existing = self.relations[key]
                existing.weight += relation.weight
                existing.description += f"; {relation.description}"
                existing.source_chunks = list(set(existing.source_chunks + relation.source_chunks))
            else:
                self.relations[key] = relation
            rel_id = f"{key[0]}___{key[1]}"
            self.vdb_relations.upsert(rel_id, self.relations[key], vec)
            if key[0] in self.graph.nodes and key[1] in self.graph.nodes:
                if self.graph.has_edge(key[0], key[1]):
                    self.graph[key[0]][key[1]]["weight"] = self.relations[key].weight
                else:
                    self.graph.add_edge(key[0], key[1], weight=self.relations[key].weight,
                                        keyword=relation.keyword)

    def add_chunks_batch(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vecs = np.asarray(self.encoder.encode(texts, batch_size=64, show_progress_bar=False))
        for chunk, vec in zip(chunks, vecs):
            self.chunks[chunk.chunk_id] = chunk
            self.vdb_chunks.upsert(chunk.chunk_id, chunk, vec)


# ---------------------------------------------------------------------------
# Cloud summary store
# ---------------------------------------------------------------------------

class SummaryVectorStore:
    """Global summary vector DB held by the coordinator (spec §2.2)."""

    def __init__(self):
        self._store = NumpyVectorStore()
        self._records: dict[str, SummaryRecord] = {}

    def upsert(self, record: SummaryRecord) -> None:
        assert record.embedding is not None
        self._store.upsert(record.subgraph_id, record, record.embedding)
        self._records[record.subgraph_id] = record

    def search(self, query_vec: np.ndarray, top_m: int, threshold: float = 0.0) -> list[SummaryRecord]:
        results = self._store.search_with_threshold(query_vec, top_m, threshold)
        return [r for _, r in results]

    def __len__(self) -> int:
        return len(self._records)
