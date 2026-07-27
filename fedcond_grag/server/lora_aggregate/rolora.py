"""RoLoRA (NeurIPS 2025, "Robust Federated Finetuning of LLMs via Alternating
Optimization of LoRA") -- alternate which half of the adapter is optimized
instead of training + averaging both every round.

Training A and B simultaneously then averaging both independently across
clients compounds FedIT's approximation error every round. RoLoRA sidesteps
it by never touching both halves in the same round: odd rounds only lora_B
is trainable/aggregated, even rounds only lora_A is -- the frozen half keeps
whatever the last aggregation produced. The client-side freeze toggle lives
in FedCondQAClient.local_train (gated on args.lora_agg_method == "rolora");
this class only does the server-side half of the alternation.
"""
from __future__ import annotations

from .fedit import FedITAggregator


class RoLoRAAggregator:
    name = "rolora"

    def __init__(self):
        self._avg = FedITAggregator()

    @staticmethod
    def active_half(round_id: int) -> str:
        return "lora_B" if round_id % 2 == 1 else "lora_A"

    def aggregate(self, client_loras, client_weights, global_lora, round_id, model=None):
        active = f".{self.active_half(round_id)}."
        active_slice = {k: v for k, v in global_lora.items() if active in k}
        updated = self._avg.aggregate(client_loras, client_weights, active_slice, round_id)
        aggregated = dict(global_lora)
        aggregated.update(updated)
        return aggregated
