"""Federated corpus partition.

Deterministically assigns each passage (document) to one of N clients
using hash(title) mod num_clients, as specified in:
    docs/plan/02_DATA_AND_TRIGRAPH.md §9.2

The partition is:
- Deterministic: same title always goes to the same client.
- Balanced in expectation: MD5 produces near-uniform hashes.
- Later extensible to topic/Louvain-based partition (see 11_INT_GFL.md §50.4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

from .hotpot_loader import HotpotCorpus, HotpotPassage


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ClientCorpus:
    client_id: int
    num_clients: int
    passages: list[HotpotPassage] = field(default_factory=list)

    def passage_texts(self) -> list[str]:
        return [p.passage_text for p in self.passages]

    def passage_ids(self) -> list[str]:
        return [p.passage_id for p in self.passages]

    def titles(self) -> list[str]:
        return [p.title for p in self.passages]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _client_for_title(title: str, num_clients: int) -> int:
    digest = hashlib.md5(title.encode()).hexdigest()
    return int(digest, 16) % num_clients


def federated_partition(
    corpus: HotpotCorpus,
    num_clients: int = 5,
) -> list[ClientCorpus]:
    """Partition corpus passages across num_clients by hash(title).

    Args:
        corpus: A loaded HotpotCorpus (deduplicated passages).
        num_clients: Number of federated clients (N). Default 5 per plan.

    Returns:
        List of ClientCorpus objects, one per client, ordered by client_id.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")

    clients: list[ClientCorpus] = [
        ClientCorpus(client_id=i, num_clients=num_clients)
        for i in range(num_clients)
    ]

    for passage in corpus.passages:
        cid = _client_for_title(passage.title, num_clients)
        clients[cid].passages.append(passage)

    return clients


def partition_stats(clients: list[ClientCorpus]) -> dict:
    """Return summary statistics for a partition (for logging/checkpointing)."""
    sizes = [len(c.passages) for c in clients]
    total = sum(sizes)
    return {
        "num_clients": len(clients),
        "total_passages": total,
        "per_client": sizes,
        "min": min(sizes),
        "max": max(sizes),
        "no_overlap": total == len({p.passage_id for c in clients for p in c.passages}),
    }


# ---------------------------------------------------------------------------
# LinearRAG-format chunk partition
# ---------------------------------------------------------------------------


@dataclass
class ClientChunks:
    """Per-client slice of a LinearRAG chunk list."""
    client_id: int
    num_clients: int
    chunks: list  # list[LinearRAGChunk]

    def chunk_texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def indices(self) -> list[int]:
        return [c.index for c in self.chunks]


def partition_linearrag_chunks(
    chunks,  # Sequence[LinearRAGChunk]
    num_clients: int = 5,
) -> list[ClientChunks]:
    """Partition LinearRAG chunks across num_clients.

    Chunks carry a numeric index prefix (from LinearRAG format "N:text").
    We assign each chunk by  client_id = chunk.index % num_clients.

    This is:
    - Deterministic: same index → same client always.
    - Round-robin: gives ~equal client sizes when indices are contiguous.
    - Sequential-friendly: consecutive chunks stay near each other within
      a client, which preserves LinearRAG's sequential passage edges.

    Args:
        chunks:      List of LinearRAGChunk objects.
        num_clients: Number of federated clients (default 5 per plan).

    Returns:
        List of ClientChunks, one per client, ordered by client_id.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")

    clients: list[ClientChunks] = [
        ClientChunks(client_id=i, num_clients=num_clients, chunks=[])
        for i in range(num_clients)
    ]

    for chunk in chunks:
        idx = chunk.index if chunk.index >= 0 else 0
        cid = idx % num_clients
        clients[cid].chunks.append(chunk)

    return clients


def chunk_partition_stats(clients: list[ClientChunks]) -> dict:
    sizes = [len(c.chunks) for c in clients]
    total = sum(sizes)
    all_indices = [c.index for client in clients for c in client.chunks]
    return {
        "num_clients": len(clients),
        "total_chunks": total,
        "per_client": sizes,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "no_overlap": total == len(set(all_indices)),
    }
