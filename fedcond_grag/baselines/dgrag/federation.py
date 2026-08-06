"""Peer coordinator for DGRAG's §9 no-cloud adaptation.

In the paper's edge-cloud model, the cloud holds a single summary VDB and
routes cross-edge retrieval. Here we simulate that as a coordinator object
that runs in the same process — same role, no HTTP.

Usage in client_runner:
1. Build each DGRAG client independently (Phase A).
2. Call build_coordinator(clients) to collect summaries and build the global
   summary VDB.
3. Distribute the summary VDB back to every client via ingest_peer_summaries.
4. At query time each client's answer() can call cross_edge_query through
   the per-client retrieve_only_fn() callbacks.
"""

from __future__ import annotations

import logging

from fedcond_grag.baselines.dgrag.pipeline import DGRAG
from fedcond_grag.baselines.dgrag.store import SummaryVectorStore

_log = logging.getLogger("dgrag.federation")


def build_coordinator(clients: list[DGRAG]) -> SummaryVectorStore:
    """Collect subgraph summary records from every client into a single VDB.

    Only (subgraph_id, edge_id, summary, embedding) is transmitted — raw
    chunks and full KG structures stay on the edge (spec §2.2 invariant).
    """
    summary_vdb = SummaryVectorStore()
    for client in clients:
        records = client.publish_summaries()
        for record in records:
            summary_vdb.upsert(record)
        _log.info("[coordinator] ingested %d summaries from edge %s",
                  len(records), client.edge_id)
    _log.info("[coordinator] global summary VDB: %d records total", len(summary_vdb))
    return summary_vdb


def distribute_summary_vdb(clients: list[DGRAG], summary_vdb: SummaryVectorStore) -> None:
    """Push the global summary VDB back to every client."""
    for client in clients:
        client.ingest_peer_summaries(summary_vdb)


def build_peer_retrieve_fns(clients: list[DGRAG], requesting_edge_id: str) -> dict:
    """Build the {edge_id: retrieve_fn} dict a client uses for cross-edge fan-out.

    Excludes the requesting client itself (a client doesn't fan out to itself).
    """
    return {
        c.edge_id: c.retrieve_only_fn()
        for c in clients
        if c.edge_id != requesting_edge_id
    }
