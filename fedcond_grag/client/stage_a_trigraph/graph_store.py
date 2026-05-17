"""Save and load client Tri-Graph .pt files.

Storage format per docs/plan/02_DATA_AND_TRIGRAPH.md §10.4:
    client_{m}/trigraph.pt = {
        x:           [N, d]  float32 node embeddings
        edge_index:  [2, E]  int64
        edge_type:   [E]     int64  (0=S-E, 1=P-E)
        node_type:   [N]     int64  (0=entity, 1=sentence, 2=passage)
        node_text:   list[str]  (local-only metadata, not uploaded to server)
    }
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


def save_trigraph(graph: Data, path: str | Path) -> None:
    """Persist a Tri-Graph Data object to disk.

    Args:
        graph: PyG Data object produced by trigraph_builder.
        path:  Destination file path (e.g. ``client_0/trigraph.pt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "edge_type": graph.edge_type,
        "node_type": graph.node_type,
        "node_text": graph.node_text if hasattr(graph, "node_text") else [],
    }
    torch.save(payload, path)


def load_trigraph(path: str | Path) -> Data:
    """Load a Tri-Graph Data object from disk.

    Args:
        path: File written by save_trigraph.

    Returns:
        PyG Data with x, edge_index, edge_type, node_type, node_text.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    return Data(
        x=payload["x"],
        edge_index=payload["edge_index"],
        edge_type=payload["edge_type"],
        node_type=payload["node_type"],
        node_text=payload.get("node_text", []),
    )
