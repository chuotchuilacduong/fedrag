"""Stage E — query-conditioned synthetic memory adaptation (FedRAG Phase 1)."""

from fedcond_grag.client.stage_e_memory.synthetic_memory import (
    LocalSyntheticMemory,
    aggregate_syn_deltas,
    gradient_matching_loss,
)

__all__ = [
    "LocalSyntheticMemory",
    "aggregate_syn_deltas",
    "gradient_matching_loss",
]
