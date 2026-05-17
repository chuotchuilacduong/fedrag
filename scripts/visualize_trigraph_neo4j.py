"""Load a client's Tri-Graph into Neo4j for visual inspection.

Usage:
    # Default: load processed/hotpotqa/client_0/trigraph.pt
    python scripts/visualize_trigraph_neo4j.py

    # Custom path / connection / wipe-before-load:
    python scripts/visualize_trigraph_neo4j.py \
        --trigraph processed/hotpotqa/client_3/trigraph.pt \
        --uri bolt://localhost:7687 \
        --user neo4j --password your_password \
        --wipe \
        --limit-nodes 5000      # optional cap for huge graphs

After loading, open http://localhost:7474 and try the example Cypher queries
printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
from neo4j import GraphDatabase

TYPE_LABEL = {0: "Entity", 1: "Sentence", 2: "Passage"}
EDGE_TYPE_LABEL = {0: "SE", 1: "PE"}


def load_trigraph(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "x": payload["x"],
        "edge_index": payload["edge_index"],
        "edge_type": payload["edge_type"],
        "node_type": payload["node_type"],
        "node_text": payload.get("node_text", []),
    }


def push(driver, tg, *, batch_size: int = 2000, limit_nodes: int | None = None):
    node_type = tg["node_type"].tolist()
    node_text = tg["node_text"]
    N = len(node_type) if limit_nodes is None else min(int(limit_nodes), len(node_type))

    src = tg["edge_index"][0].tolist()
    dst = tg["edge_index"][1].tolist()
    etype = tg["edge_type"].tolist()

    with driver.session() as session:
        # Build constraints / indexes (idempotent — won't fail if exists).
        for label in ("Entity", "Sentence", "Passage"):
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

        # ---- Nodes ----
        print(f"[push] inserting {N} nodes…")
        t = time.time()
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = [
                {
                    "id": i,
                    "type": int(node_type[i]),
                    "label": TYPE_LABEL[int(node_type[i])],
                    "text": (node_text[i] if i < len(node_text) else ""),
                }
                for i in range(start, end)
            ]
            session.run(
                """
                UNWIND $rows AS row
                CALL apoc.merge.node([row.label],
                                     {id: row.id},
                                     {text: row.text, type: row.type})
                YIELD node
                RETURN count(node)
                """,
                rows=batch,
            ) if _has_apoc(session) else session.run(
                # APOC-free fallback: one query per label, MERGE on (label, id)
                """
                UNWIND $rows AS row
                CALL {
                    WITH row
                    WITH row WHERE row.label = "Entity"
                    MERGE (n:Entity {id: row.id})
                    SET n.text = row.text, n.type = row.type
                }
                CALL {
                    WITH row
                    WITH row WHERE row.label = "Sentence"
                    MERGE (n:Sentence {id: row.id})
                    SET n.text = row.text, n.type = row.type
                }
                CALL {
                    WITH row
                    WITH row WHERE row.label = "Passage"
                    MERGE (n:Passage {id: row.id})
                    SET n.text = row.text, n.type = row.type
                }
                """,
                rows=batch,
            )
        print(f"[push] nodes done in {time.time() - t:.1f}s")

        # ---- Edges ----
        # edge_index stores both directions; only keep src < dst to avoid duplicates.
        unique_edges = []
        seen = set()
        for s, d, et in zip(src, dst, etype):
            if limit_nodes is not None and (s >= N or d >= N):
                continue
            a, b = (s, d) if s < d else (d, s)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            unique_edges.append({"src": a, "dst": b, "kind": EDGE_TYPE_LABEL[int(et)]})

        print(f"[push] inserting {len(unique_edges)} edges…")
        t = time.time()
        for start in range(0, len(unique_edges), batch_size):
            end = min(start + batch_size, len(unique_edges))
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a {id: row.src}), (b {id: row.dst})
                CALL {
                    WITH a, b, row
                    WITH a, b, row WHERE row.kind = "SE"
                    MERGE (a)-[:SE]-(b)
                }
                CALL {
                    WITH a, b, row
                    WITH a, b, row WHERE row.kind = "PE"
                    MERGE (a)-[:PE]-(b)
                }
                """,
                rows=unique_edges[start:end],
            )
        print(f"[push] edges done in {time.time() - t:.1f}s")


def _has_apoc(session) -> bool:
    try:
        session.run("CALL apoc.help('apoc') YIELD name RETURN name LIMIT 1").consume()
        return True
    except Exception:
        return False


def wipe(driver) -> None:
    print("[wipe] deleting all nodes and relationships…")
    with driver.session() as session:
        # Batched delete to handle large graphs.
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(n) AS deleted"
            ).single()
            if result["deleted"] == 0:
                break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trigraph", default="processed/hotpotqa/client_0/trigraph.pt")
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="fedcondgrag")
    p.add_argument("--wipe", action="store_true", help="Delete existing data first")
    p.add_argument("--limit-nodes", type=int, default=None,
                   help="Cap number of nodes to load (for fast preview)")
    args = p.parse_args()

    path = Path(args.trigraph)
    if not path.exists():
        raise SystemExit(f"trigraph not found: {path}")

    print(f"[load] {path}")
    tg = load_trigraph(path)
    print(f"  nodes={tg['x'].shape[0]}, edges={tg['edge_index'].shape[1] // 2} (undirected)")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        if args.wipe:
            wipe(driver)
        push(driver, tg, limit_nodes=args.limit_nodes)
    finally:
        driver.close()

    print()
    print("=" * 60)
    print("Loaded. Open http://localhost:7474 and try:")
    print("=" * 60)
    print("// Quick counts by node type")
    print("MATCH (n) RETURN labels(n)[0] AS type, count(*) AS n;")
    print()
    print("// Top 20 entities by degree (most-mentioned topics)")
    print("MATCH (e:Entity)-[r]-() RETURN e.text, count(r) AS deg ORDER BY deg DESC LIMIT 20;")
    print()
    print("// 1-hop neighborhood around an entity")
    print("MATCH (e:Entity {text: 'einstein'})-[r]-(n) RETURN e, r, n LIMIT 50;")
    print()
    print("// One full S-E-P motif")
    print("MATCH (s:Sentence)-[:SE]-(e:Entity)-[:PE]-(p:Passage)")
    print("RETURN s, e, p LIMIT 25;")
    print()
    print("// All edges of a passage (anchor for its entities)")
    print("MATCH (p:Passage {id: 25000})-[r]-(n) RETURN p, r, n;")


if __name__ == "__main__":
    main()
