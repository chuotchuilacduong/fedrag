"""Simulated per-round communication cost.

Client and server run in one process and talk through `message_pool` dicts
(no real network I/O) -- same premise `scripts/run_node_ratio_ablation.py`'s
`trigraph_bytes`/`condensed_bytes` already use for its own byte counts. This
generalizes that to arbitrary message payloads (tensors, PyG `Data`, nested
dicts of state-dict tensors) so round-level communication cost can be logged
without hand-enumerating each message's fields.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

# Keys whose payload represents the synthetic/condensed memory itself, as
# opposed to model weights or training deltas -- see client.py::send_message
# (anchor_graph) and server.py::send_message (synthetic_x/adj/node_type).
MEMORY_KEYS = {"anchor_graph", "synthetic_x", "synthetic_adj", "synthetic_node_type"}

# Keys that fully re-export data already counted elsewhere in the same message
# -- server.py's synthetic_graph is synthetic_x/adj/node_type repackaged as a
# PyG Data for reader-side retrieval, so it is dropped from the total entirely.
_REDUNDANT_KEYS = {"synthetic_graph"}

# server.py's synthetic_state dict re-wraps "x"/"node_type" (already counted
# via synthetic_x/synthetic_node_type above) alongside the PGE's own trained
# weights ("pge_state") and small config scalars. Only the duplicated fields
# are dropped -- pge_state is real per-round payload and stays in the total,
# just not classified as "memory" (it's model parameters, not memory content).
_DUPLICATE_SUBFIELDS = {"synthetic_state": {"x", "node_type"}}


def _bytes_of(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, Data):
        return sum(_bytes_of(obj[key]) for key in obj.keys())
    if isinstance(obj, dict):
        return sum(_bytes_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_bytes_of(v) for v in obj)
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, (int, float)):
        return 8
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    return 0


def _bytes_of_deduped(key: str, value) -> int:
    dupes = _DUPLICATE_SUBFIELDS.get(key)
    if dupes and isinstance(value, dict):
        return sum(_bytes_of(v) for k, v in value.items() if k not in dupes)
    return _bytes_of(value)


def message_bytes(msg: dict) -> tuple[int, int]:
    """Return (memory_bytes, total_bytes) for one client's or the server's
    outgoing message this round."""
    if not msg:
        return 0, 0
    memory = sum(_bytes_of_deduped(k, v) for k, v in msg.items() if k in MEMORY_KEYS)
    total = sum(
        _bytes_of_deduped(k, v) for k, v in msg.items() if k not in _REDUNDANT_KEYS
    )
    return memory, total
