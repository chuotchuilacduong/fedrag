"""FlexLoRA (NeurIPS 2024, "Federated Fine-tuning of LLMs under Heterogeneous
Tasks and Client Resources") -- reconstruct the *exact* summed weight update
before compressing back to a common rank, instead of averaging A and B
independently the way FedIT does.

For each target module: stack every sampled client's (data-weighted) A and
raw B along the rank axis, form the full-rank delta_W = B_stack @ A_stack
(mathematically the sum of each client's own B_k @ A_k update -- no averaging
error), then SVD-truncate it back down to `rank` so the aggregated adapter
keeps the shape clients started from. `scale` (the paper's redistribution
factor) inflates delta_W before the SVD and divides it back out of B
afterwards purely to keep the intermediate singular values numerically
stable; it cancels out in the reconstruction B @ A.
"""
from __future__ import annotations

import torch

from .keys import b_key_of, lora_a_keys


class FlexLoRAAggregator:
    name = "flexlora"

    def __init__(self, rank: int = 8, scale: float = 2.0):
        self.rank = rank
        self.scale = scale

    def aggregate(self, client_loras, client_weights, global_lora, round_id, model=None):
        total = sum(client_weights)
        aggregated = dict(global_lora)
        for a_key in lora_a_keys(global_lora):
            b_key = b_key_of(a_key)
            a_list, b_list = [], []
            for lora, w in zip(client_loras, client_weights):
                if a_key not in lora or b_key not in lora:
                    continue
                a_list.append((w / total) * lora[a_key].float())
                b_list.append(lora[b_key].float())
            if not a_list:
                continue

            stacked_A = torch.cat(a_list, dim=0)   # [sum(r_k), in]
            stacked_B = torch.cat(b_list, dim=1)   # [out, sum(r_k)]
            delta_W = torch.matmul(stacked_B, stacked_A) * self.scale

            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
            r = min(self.rank, S.numel())
            sqrt_S = torch.diag(torch.sqrt(S[:r]))

            new_B = torch.matmul(U[:, :r], sqrt_S) / self.scale
            new_A = Vh[:r, :]

            aggregated[b_key] = new_B.to(global_lora[b_key].dtype)
            aggregated[a_key] = new_A.to(global_lora[a_key].dtype)
        return aggregated
