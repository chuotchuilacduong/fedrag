"""fedcond_qa server: Stage C global graph condensation over C_m anchors."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import time
from typing import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from fedcond_grag.server.stage_c_aggregate.config import config as default_config
from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE
from fedcond_grag.server.stage_c_aggregate.task import CondensationQATask
from fedcond_grag.server.stage_c_aggregate.surrogate import (
    SurrogateGNN,
    edge_index_to_dense,
    gradient_match_loss,
    parameter_gradients,
    surrogate_loss,
)


@dataclass
class SyntheticGraphState:
    x: Tensor
    adj: Tensor
    node_type: Tensor


class FedCondQAServer:
    """Server-side global condensation for FedCondGraphRAG (Stage C).

    Aggregates anchor graphs {C_m} from clients via node-type/link surrogate
    gradient matching, then exports a synthetic global graph G_global.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        for key, value in default_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        self.args = args
        self.data_dir = data_dir
        self.message_pool = message_pool
        self.device = device
        self.task = CondensationQATask(args, None, global_data, data_dir, device)

        self.model = SurrogateGNN(
            input_dim=self.task.num_feats,
            hidden_dim=args.hid_dim,
            output_dim=3,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(self.device)
        self.synthetic_x: nn.Parameter | None = None
        self.synthetic_node_type: Tensor | None = None
        self.pge: TypeAwarePGE | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.best_loss = float("inf")
        self.best_state: SyntheticGraphState | None = None
        self.global_model_state: dict | None = None
        self.task.override_evaluate = self.get_override_evaluate()

    def send_message(self):
        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None:
            self.message_pool["server"] = {}
            return
        with torch.no_grad():
            adj = self.pge.inference(self.synthetic_x, self.synthetic_node_type)
        msg: dict = {
            "synthetic_x": self.synthetic_x.detach(),
            "synthetic_adj": adj.detach(),
            "synthetic_node_type": self.synthetic_node_type.detach(),
            "synthetic_graph": self.export_synthetic_graph(),
        }
        if self.global_model_state:
            msg["model_weights"] = self.global_model_state
        self.message_pool["server"] = msg

    def execute(self):
        start = time.perf_counter()
        anchor_graphs = self._collect_anchor_graphs()
        if not anchor_graphs:
            return

        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None:
            self.init_synthetic_graph(anchor_graphs)

        anchor_gradients = self.compute_anchor_gradients(anchor_graphs)
        num_steps = int(getattr(self.args, "server_condense_iters", 50))
        last_loss = None
        for _ in range(max(1, num_steps)):
            last_loss = self.server_condense_step(anchor_gradients)

        self.train_loss_match = float(last_loss.detach().cpu()) if last_loss is not None else 0.0
        self.message_pool["extra_server_compute"] = self.message_pool.get("extra_server_compute", 0.0) + time.perf_counter() - start
        self._fedavg_model_weights()
        self.send_message()

    def init_synthetic_graph(self, anchor_graphs: Sequence[Data]) -> None:
        feature_dim = int(anchor_graphs[0].x.size(1))
        total_nodes = int(getattr(self.args, "num_global_syn_nodes", 1024))
        type_counts = torch.zeros(3, dtype=torch.float32, device=self.device)
        by_type: list[list[Tensor]] = [[], [], []]
        for graph in anchor_graphs:
            graph = self._prepare_anchor_graph(graph)
            for type_id in range(3):
                mask = graph.node_type.long() == type_id
                count = int(mask.sum().item())
                type_counts[type_id] += count
                if count > 0:
                    by_type[type_id].append(graph.x[mask].detach())

        if type_counts.sum() == 0:
            type_counts[:] = 1
        ratios = type_counts / type_counts.sum()
        n_per_type = torch.floor(ratios * total_nodes).long()
        while int(n_per_type.sum().item()) < total_nodes:
            n_per_type[int(torch.argmax(ratios - n_per_type.float() / max(total_nodes, 1)).item())] += 1
        while int(n_per_type.sum().item()) > total_nodes:
            idx = int(torch.argmax(n_per_type).item())
            n_per_type[idx] -= 1

        xs: list[Tensor] = []
        types: list[Tensor] = []
        for type_id, count in enumerate(n_per_type.tolist()):
            if count <= 0:
                continue
            if by_type[type_id]:
                source = torch.cat(by_type[type_id], dim=0)
                sample_idx = torch.randint(0, source.size(0), (count,), device=self.device)
                init = source[sample_idx] + 0.01 * torch.randn(count, feature_dim, device=self.device)
            else:
                init = torch.randn(count, feature_dim, device=self.device) * 0.02
            xs.append(init)
            types.append(torch.full((count,), type_id, dtype=torch.long, device=self.device))

        self.synthetic_x = nn.Parameter(torch.cat(xs, dim=0))
        self.synthetic_node_type = torch.cat(types, dim=0)
        self.pge = TypeAwarePGE(
            feature_dim=feature_dim,
            hidden_dim=int(getattr(self.args, "pge_hidden", 256)),
            type_emb_dim=int(getattr(self.args, "type_emb_dim", 16)),
            topk=int(getattr(self.args, "pge_topk", 8)),
            preserve_sep=bool(getattr(self.args, "preserve_sep_topology", True)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [self.synthetic_x, *self.pge.parameters()],
            lr=float(getattr(self.args, "lr_feat", 1e-2)),
        )

    def compute_anchor_gradients(self, anchor_graphs: Sequence[Data]) -> list[Tensor]:
        self.model.reset_parameters()
        self.model.train()
        params = list(self.model.parameters())
        weighted_grads = [torch.zeros_like(param) for param in params]
        total_nodes = sum(max(1, int(graph.x.size(0))) for graph in anchor_graphs)

        for graph in anchor_graphs:
            graph = self._prepare_anchor_graph(graph)
            adj = edge_index_to_dense(graph.edge_index, graph.x.size(0), getattr(graph, "edge_weight", None)).to(self.device)
            output = surrogate_loss(
                self.model,
                graph.x,
                adj,
                graph.node_type,
                edge_index=graph.edge_index,
                type_weight=float(getattr(self.args, "surrogate_type_weight", 1.0)),
                link_weight=float(getattr(self.args, "surrogate_link_weight", 0.5)),
            )
            grads = parameter_gradients(output.loss, params, create_graph=False)
            coeff = float(graph.x.size(0)) / float(total_nodes)
            for idx, grad in enumerate(grads):
                weighted_grads[idx] = weighted_grads[idx] + coeff * grad.detach()
        return weighted_grads

    def server_condense_step(self, anchor_gradients: Sequence[Tensor]) -> Tensor:
        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None or self.optimizer is None:
            raise RuntimeError("Synthetic graph is not initialized")

        self.model.train()
        params = list(self.model.parameters())
        adj = self.pge(self.synthetic_x, self.synthetic_node_type)
        output = surrogate_loss(
            self.model,
            self.synthetic_x,
            adj,
            self.synthetic_node_type,
            type_weight=float(getattr(self.args, "surrogate_type_weight", 1.0)),
            link_weight=float(getattr(self.args, "surrogate_link_weight", 0.5)),
        )
        global_grads = parameter_gradients(output.loss, params, create_graph=True)
        loss = gradient_match_loss(
            global_grads,
            anchor_gradients,
            norm_weight=float(getattr(self.args, "match_norm_weight", 0.0)),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if float(loss.detach().cpu()) < self.best_loss:
            self.best_loss = float(loss.detach().cpu())
            self.best_state = SyntheticGraphState(
                x=self.synthetic_x.detach().clone(),
                adj=adj.detach().clone(),
                node_type=self.synthetic_node_type.detach().clone(),
            )
        return loss

    def _fedavg_model_weights(self) -> None:
        """FedAvg GNN encoder + projector weights from sampled clients."""
        sampled = self.message_pool.get("sampled_clients", [])
        _WEIGHT_KEYS = ("graph_encoder", "projector", "condensed_encoder", "projector_c")

        client_entries: list[tuple[dict, int]] = []
        for cid in sampled:
            msg = self.message_pool.get(f"client_{cid}", {})
            weights = msg.get("model_weights")
            n = int(msg.get("num_samples", 0))
            if weights and n > 0:
                client_entries.append((weights, n))

        if not client_entries:
            return

        total = sum(n for _, n in client_entries)
        aggregated: dict[str, dict] = {}
        for key in _WEIGHT_KEYS:
            entries = [(w[key], n) for w, n in client_entries if key in w]
            if not entries:
                continue
            avg: dict = {}
            for param_name in entries[0][0]:
                avg[param_name] = sum(
                    sd[param_name].float() * (n / total) for sd, n in entries
                )
            aggregated[key] = avg

        self.global_model_state = aggregated

    def export_synthetic_graph(self) -> Data:
        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None:
            raise RuntimeError("Synthetic graph is not initialized")
        with torch.no_grad():
            adj = self.pge.inference(self.synthetic_x, self.synthetic_node_type)
            rows, cols = (adj > 0).nonzero(as_tuple=True)
            edge_index = torch.stack([rows, cols], dim=0).long()
            edge_weight = adj[rows, cols]
        data = Data(
            x=self.synthetic_x.detach().clone(),
            edge_index=edge_index,
            edge_weight=edge_weight.detach().clone(),
            node_type=self.synthetic_node_type.detach().clone(),
            y=self.synthetic_node_type.detach().clone(),
        )
        data.num_global_classes = 3
        return data

    def get_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None:
                zero = torch.tensor(0.0, device=self.device)
                return {
                    "loss_train": zero,
                    "loss_val": zero,
                    "loss_test": zero,
                    "accuracy_train": 0.0,
                    "accuracy_val": 0.0,
                    "accuracy_test": 0.0,
                }
            with torch.no_grad():
                adj = self.pge.inference(self.synthetic_x, self.synthetic_node_type)
                output = surrogate_loss(self.model, self.synthetic_x, adj, self.synthetic_node_type)
            if not mute:
                print(f"[server]\tloss_match: {getattr(self, 'train_loss_match', 0.0):.4f}\ttype_acc: {float(output.type_accuracy):.4f}")
            return {
                "loss_train": output.loss.detach(),
                "loss_val": output.loss.detach(),
                "loss_test": output.loss.detach(),
                "accuracy_train": float(output.type_accuracy.detach().cpu()),
                "accuracy_val": float(output.type_accuracy.detach().cpu()),
                "accuracy_test": float(output.type_accuracy.detach().cpu()),
            }

        return override_evaluate

    def _collect_anchor_graphs(self) -> list[Data]:
        anchors: list[Data] = []
        for client_id in self.message_pool.get("sampled_clients", []):
            message = self.message_pool.get(f"client_{client_id}", {})
            graph = message.get("anchor_graph")
            if graph is not None:
                anchors.append(self._prepare_anchor_graph(graph))
        return anchors

    def _prepare_anchor_graph(self, graph: Data) -> Data:
        graph = copy.copy(graph).to(self.device)
        if not hasattr(graph, "node_type") and hasattr(graph, "y"):
            graph.node_type = graph.y
        if not hasattr(graph, "node_type"):
            raise ValueError("Anchor graph requires node_type")
        graph.node_type = graph.node_type.long().to(self.device)
        graph.y = graph.node_type
        graph.x = graph.x.float().to(self.device)
        graph.edge_index = graph.edge_index.long().to(self.device)
        if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
            graph.edge_weight = graph.edge_weight.float().to(self.device)
        graph.num_global_classes = 3
        return graph
