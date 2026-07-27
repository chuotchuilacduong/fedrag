"""Per-client local HippoRAG baseline.

Drives the real, pip-installed HippoRAG package (installed straight from
https://github.com/OSU-NLP-Group/HippoRAG, MIT licensed -- see
requirements.txt) so it can be run independently on each federated client's
own passage shard and evaluated against this project's benchmark question
sets. See `client_runner.py`.
"""

from .client_runner import run_client_baseline, run_all_clients

__all__ = ["run_client_baseline", "run_all_clients"]
