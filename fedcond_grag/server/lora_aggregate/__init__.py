"""Federated LoRA-adapter aggregation strategies for Stage D.

Selected via --lora-agg-method on `fedcond_grag fl-train`; only takes effect
when --llm-frozen False (LoRA fine-tuning of the shared LLM enabled -- by
default the LLM stays frozen and only the GNN encoder/projector are
federated). Ported from FedLLM-Factory
(https://github.com/boyi-liu/FedLLM-Factory, alg/*.py):

    fedit     -- plain weighted average of A and B independently (baseline)
    flexlora  -- stack A/B -> exact summed delta -> SVD-compress back to rank
    rolora    -- alternate training/aggregating A vs B by round parity
    flora     -- stack A/B -> exact summed delta -> merge into frozen base weight
"""
from __future__ import annotations

from .fedit import FedITAggregator
from .flexlora import FlexLoRAAggregator
from .flora import FLoRAAggregator
from .keys import has_lora, lora_state_dict
from .rolora import RoLoRAAggregator

_REGISTRY = {
    "fedit": FedITAggregator,
    "flexlora": FlexLoRAAggregator,
    "rolora": RoLoRAAggregator,
    "flora": FLoRAAggregator,
}


def build_lora_aggregator(name: str, rank: int = 8, scale: float = 2.0):
    key = str(name or "fedit").lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown --lora-agg-method '{name}', expected one of {sorted(_REGISTRY)}")
    if key == "flexlora":
        return FlexLoRAAggregator(rank=rank, scale=scale)
    return _REGISTRY[key]()


__all__ = [
    "build_lora_aggregator",
    "has_lora",
    "lora_state_dict",
    "FedITAggregator",
    "FlexLoRAAggregator",
    "RoLoRAAggregator",
    "FLoRAAggregator",
]
