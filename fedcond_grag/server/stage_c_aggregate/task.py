"""Minimal task wrapper for FedCondGraphRAG.

Replaces the heavy gfl Task abstraction with the few attributes the
FedCondQAClient / FedCondQAServer actually use:
    task.splitted_data["data"]   (client init reads its tri-graph from here)
    task.num_feats               (server SurrogateGNN input_dim)
    task.num_global_classes      (always 3: entity / sentence / passage)
    task.override_evaluate       (server installs its own evaluator)
"""

from __future__ import annotations

import torch


class CondensationQATask:
    def __init__(self, args, client_id, data, data_dir, device):
        self.args = args
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        # Keep the trigraph on CPU — it's multi-GB and only moved to device
        # lazily inside _condense_anchor_graph() via graph.to(self.device).
        self.data = data
        self.override_evaluate = None
        self.step_preprocess = None

        if self.data is not None:
            if not hasattr(self.data, "node_type") and hasattr(self.data, "y"):
                self.data.node_type = self.data.y
            if not hasattr(self.data, "node_type"):
                raise ValueError("CondensationQATask requires data.node_type or data.y")
            self.data.y = self.data.node_type.long()
            self.data.num_global_classes = 3

            num_nodes = int(self.data.x.size(0))
            ones = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
            self.train_mask = ones
            self.val_mask = ones
            self.test_mask = ones
            self.splitted_data = {
                "data": self.data,
                "train_mask": ones,
                "val_mask": ones,
                "test_mask": ones,
            }
            self.processed_data = self.splitted_data
        else:
            self.splitted_data = None
            self.processed_data = None

    @property
    def num_samples(self) -> int:
        return int(self.data.x.shape[0]) if self.data is not None else 0

    @property
    def num_feats(self) -> int:
        return int(self.data.x.shape[1])

    @property
    def num_global_classes(self) -> int:
        return int(getattr(self.data, "num_global_classes", 3))

    def evaluate(self, splitted_data=None, mute=False):
        if self.override_evaluate is None:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_train": zero,
                "loss_val": zero,
                "loss_test": zero,
                "accuracy_train": 0.0,
                "accuracy_val": 0.0,
                "accuracy_test": 0.0,
            }
        return self.override_evaluate(splitted_data, mute)
