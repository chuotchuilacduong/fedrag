"""FD-RAG baseline (Federated Dual-System Retrieval-Augmented Generation).

Faithful reimplementation of the algorithms in ``implementfd_rag.md`` (extracted
from the FD-RAG anonymous ACL submission at ``fd-rag.pdf``). Structured as its
own baseline under ``fedcond_grag/baselines/fdrag/`` and driven per-client from
``client_runner.py`` in the same style as ``baselines/linearrag``,
``baselines/hipporag`` and ``baselines/comorag`` (own passage shard per client,
evaluated against the global question set for the dataset).
"""

from fedcond_grag.baselines.fdrag.config import FDRAGConfig
from fedcond_grag.baselines.fdrag.pipeline import FDRAG

__all__ = ["FDRAG", "FDRAGConfig"]
