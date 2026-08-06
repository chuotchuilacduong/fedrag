"""DGRAG baseline — Distributed Graph-based Retrieval-Augmented Generation.

Faithful implementation of Zhou/Yan/Yang arXiv:2505.19847v2, adapted for the
HotPotQA/2WikiMQA/MuSiQue short-answer evaluation protocol per spec §9.
"""

from fedcond_grag.baselines.dgrag.config import DGRAGConfig
from fedcond_grag.baselines.dgrag.pipeline import DGRAG

__all__ = ["DGRAG", "DGRAGConfig"]
