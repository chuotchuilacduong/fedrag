"""Per-client local G-Retriever baseline.

Unlike HippoRAG, G-Retriever isn't vendored separately: this project's own
`fedcond_grag/model/graph_llm.py` / `gnn.py` already *are* G-Retriever's
model (see `client_runner.py` module docstring for the diff against
https://github.com/XiaoxinHe/G-Retriever). This package just drives that
existing model independently per federated client, mirroring the HippoRAG
baseline's methodology.
"""

from .client_runner import run_client_baseline, run_all_clients

__all__ = ["run_client_baseline", "run_all_clients"]
