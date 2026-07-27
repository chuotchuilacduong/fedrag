"""FLoRA (NeurIPS 2024, "Federated Fine-Tuning Large Language Models with
Heterogeneous Low-Rank Adaptations") -- stack-then-merge instead of
averaging deltas.

Concatenating clients' A/B along the rank axis and multiplying reconstructs
the *exact* sum of their individual weight updates (no averaging, no SVD
approximation -- see flexlora.py's docstring for why that matters). FLoRA
takes this a step further: it merges that exact delta directly into the
frozen base weight and hands every client back the *same fixed* initial
adapter (random A / zero B) next round, so successive rounds' low-rank
updates compose into a full-rank change in the backbone instead of being
diluted by repeated averaging.

Requires a dense (non-quantized) base weight -- merging into a bitsandbytes
Params4bit/Int8Params tensor would need a dequantize/requantize round trip
this implementation doesn't do, so 4-bit/8-bit LLM loading is incompatible
with this aggregator (see the RuntimeError below).
"""
from __future__ import annotations

import torch

from .keys import b_key_of, base_key_of, is_quantized_param, lora_a_keys


class FLoRAAggregator:
    name = "flora"

    def aggregate(self, client_loras, client_weights, global_lora, round_id, model=None):
        if model is None:
            raise ValueError(
                "FLoRA aggregation needs the shared LLM (model=...) to merge deltas into base weights"
            )

        total = sum(client_weights)
        params = dict(model.named_parameters())

        with torch.no_grad():
            for a_key in lora_a_keys(global_lora):
                b_key = b_key_of(a_key)
                base_key = base_key_of(a_key)
                base_param = params.get(base_key)
                if base_param is None:
                    continue
                if is_quantized_param(base_param):
                    raise RuntimeError(
                        f"FLoRA merges the aggregated LoRA delta directly into '{base_key}', which "
                        "requires a dense base weight. This run has --llm-load-in-4bit/"
                        "--llm-load-in-8bit enabled, so that weight is a packed "
                        f"{type(base_param).__name__} tensor that can't be updated with a plain "
                        "in-place add -- disable quantization to use --lora-agg-method flora."
                    )

                a_list, b_list = [], []
                for lora, w in zip(client_loras, client_weights):
                    if a_key not in lora or b_key not in lora:
                        continue
                    a_list.append((w / total) * lora[a_key].float())
                    b_list.append(lora[b_key].float())
                if not a_list:
                    continue

                stacked_A = torch.cat(a_list, dim=0)
                stacked_B = torch.cat(b_list, dim=1)
                delta_W = torch.matmul(stacked_B, stacked_A)
                base_param.data.add_(delta_W.to(base_param.dtype))

        # global_lora is deliberately returned unchanged: every client restarts
        # next round from the same fixed initial adapter, on top of the newly
        # merged base weight.
        return dict(global_lora)
