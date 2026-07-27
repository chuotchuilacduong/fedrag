"""FedIT baseline: plain weighted average of LoRA A and B independently.

Reference baseline every federated-LoRA paper compares against (FedLLM-Factory's
FTBaseServer.aggregate, https://github.com/boyi-liu/FedLLM-Factory). Averaging
A and B separately is only a first-order approximation of averaging the true
weight delta B@A -- FlexLoRA/FLoRA exist specifically to fix the error this
introduces.
"""
from __future__ import annotations


class FedITAggregator:
    name = "fedit"

    def aggregate(self, client_loras, client_weights, global_lora, round_id, model=None):
        total = sum(client_weights)
        keys = global_lora.keys() if global_lora else client_loras[0].keys()
        aggregated = {}
        for k in keys:
            acc = None
            for lora, w in zip(client_loras, client_weights):
                if k not in lora:
                    continue
                term = lora[k].float() * (w / total)
                acc = term if acc is None else acc + term
            if acc is not None:
                aggregated[k] = acc
        return aggregated
