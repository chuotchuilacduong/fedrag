"""Per-client local GRAG baseline (https://github.com/HuieL/GRAG).

Unlike `baselines/gretriever`, GRAG's GNN is genuinely different from this
project's own model (query-conditioned node/edge features), so its `model/`
and retrieval code are vendored under `_vendor/` (see `_vendor/VENDORED.md`).
`client_runner.py` drives it per federated client, mirroring the
`baselines/hipporag` / `baselines/gretriever` methodology.
"""

from .client_runner import run_client_baseline, run_all_clients

__all__ = ["run_client_baseline", "run_all_clients"]
