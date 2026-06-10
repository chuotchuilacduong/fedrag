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
from fedcond_grag.server.stage_c_aggregate.repr_align import (
    compute_target_degree,
    degree_regularization,
    diversity_loss,
    encode_nodes,
    encode_nodes_with_edge_weight,
    precompute_anchor_reprs,
    representation_alignment_loss,
)
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
        self._target_anchor_degree: float = 8.0
        # Repr-align encoder: initialized with random weights, refreshed each round via FedAvg
        self.repr_encoder: nn.Module | None = None
        self.repr_projector: nn.Module | None = None
        self._build_repr_encoder()
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

        mode = str(getattr(self.args, "server_stage_c_mode", "gradient_match"))

        num_steps = int(getattr(self.args, "server_condense_iters", 50))
        last_loss = None

        if self.global_model_state and mode in ("repr_align", "both"):
            self._load_repr_align_weights(self.global_model_state)

        if mode == "gradient_match":
            anchor_gradients = self.compute_anchor_gradients(anchor_graphs)
            for _ in range(max(1, num_steps)):
                last_loss = self.server_condense_step(anchor_gradients)

        elif mode == "repr_align":
            self._target_anchor_degree = compute_target_degree(anchor_graphs)
            anchor_h_list = precompute_anchor_reprs(
                anchor_graphs, self.repr_encoder, self.repr_projector, self.device
            )
            for _ in range(max(1, num_steps)):
                last_loss = self.server_repr_align_step(anchor_h_list)

        elif mode == "both":
            self._target_anchor_degree = compute_target_degree(anchor_graphs)
            anchor_gradients = self.compute_anchor_gradients(anchor_graphs)
            anchor_h_list = precompute_anchor_reprs(
                anchor_graphs, self.repr_encoder, self.repr_projector, self.device
            )
            for _ in range(max(1, num_steps)):
                last_loss = self.server_combined_step(anchor_gradients, anchor_h_list)

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

    def _syn_edge_index(self) -> Tensor:
        """Current synthetic graph topology as edge_index (no gradient, no PGE grad)."""
        with torch.no_grad():
            adj = self.pge.inference(self.synthetic_x.detach(), self.synthetic_node_type)
            rows, cols = (adj > 0).nonzero(as_tuple=True)
            return torch.stack([rows, cols], dim=0).long()

    def _update_best_state(self, loss: Tensor) -> None:
        if float(loss.detach().cpu()) < self.best_loss:
            self.best_loss = float(loss.detach().cpu())
            with torch.no_grad():
                adj = self.pge.inference(self.synthetic_x, self.synthetic_node_type)
            self.best_state = SyntheticGraphState(
                x=self.synthetic_x.detach().clone(),
                adj=adj.detach().clone(),
                node_type=self.synthetic_node_type.detach().clone(),
            )

    def _repr_align_aux_loss(self, h_syn: Tensor, adj_for_deg: Tensor | None = None) -> Tensor:
        """Auxiliary diversity + degree losses, zero if lambdas are 0."""
        loss = h_syn.new_zeros(())
        lambda_div = float(getattr(self.args, "lambda_div", 0.0))
        if lambda_div > 0:
            loss = loss + lambda_div * diversity_loss(h_syn)
        lambda_deg = float(getattr(self.args, "lambda_deg", 0.0))
        if lambda_deg > 0 and adj_for_deg is not None:
            loss = loss + lambda_deg * degree_regularization(adj_for_deg, self._target_anchor_degree)
        return loss

    def server_repr_align_step(self, anchor_h_list: list[Tensor]) -> Tensor:
        """One optimization step using representation alignment (Lalign + aux losses).

        Uses soft edge weights from PGE so gradient flows to both synthetic_x and PGE:
            adj_soft = PGE(synthetic_x, ...)          -- continuous [Kg,Kg]
            edge_weight = adj_soft[rows, cols]         -- [E], has grad
            h_syn = GCN(synthetic_x, edge_index,
                        edge_weight) → projector       -- gradient to x and edge_weight
            Lalign → backward → synthetic_x ✓, PGE ✓
        """
        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None or self.optimizer is None:
            raise RuntimeError("Synthetic graph is not initialized")

        adj_soft = self.pge(self.synthetic_x, self.synthetic_node_type, sparsify=False)
        rows, cols = (adj_soft.detach() > 0).nonzero(as_tuple=True)
        edge_index_syn = torch.stack([rows, cols], dim=0).long()
        edge_weight = adj_soft[rows, cols]  # [E], gradient flows to adj_soft → PGE

        h_syn = encode_nodes_with_edge_weight(
            self.synthetic_x, edge_index_syn, edge_weight,
            self.repr_encoder, self.repr_projector,
        )

        loss = representation_alignment_loss(h_syn, anchor_h_list)
        loss = loss + self._repr_align_aux_loss(h_syn, adj_soft)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_best_state(loss)
        return loss

    def server_combined_step(
        self,
        anchor_gradients: Sequence[Tensor],
        anchor_h_list: list[Tensor],
    ) -> Tensor:
        """One step combining gradient matching and representation alignment.

        Both losses are summed before a single backward + optimizer.step(),
        so PGE and synthetic_x each receive coherent gradients.
        """
        if self.synthetic_x is None or self.synthetic_node_type is None or self.pge is None or self.optimizer is None:
            raise RuntimeError("Synthetic graph is not initialized")

        # --- Gradient matching component (needs create_graph for 2nd-order) ---
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
        loss_gm = gradient_match_loss(
            global_grads,
            anchor_gradients,
            norm_weight=float(getattr(self.args, "match_norm_weight", 0.0)),
        )

        # --- Repr alignment component (soft edge weights → grad to PGE) ---
        adj_soft_ra = self.pge(self.synthetic_x, self.synthetic_node_type, sparsify=False)
        rows_ra, cols_ra = (adj_soft_ra.detach() > 0).nonzero(as_tuple=True)
        edge_index_ra = torch.stack([rows_ra, cols_ra], dim=0).long()
        edge_weight_ra = adj_soft_ra[rows_ra, cols_ra]
        h_syn = encode_nodes_with_edge_weight(
            self.synthetic_x, edge_index_ra, edge_weight_ra,
            self.repr_encoder, self.repr_projector,
        )
        loss_ra = representation_alignment_loss(h_syn, anchor_h_list)
        loss_ra = loss_ra + self._repr_align_aux_loss(h_syn, adj_soft_ra)

        w_gm = float(getattr(self.args, "grad_match_weight", 1.0))
        w_ra = float(getattr(self.args, "repr_align_weight", 1.0))
        loss = w_gm * loss_gm + w_ra * loss_ra

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_best_state(loss)
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
        if aggregated:
            self._load_repr_align_weights(aggregated)

    def _build_repr_encoder(self) -> None:
        """Initialize repr_encoder + repr_projector with random weights at startup.

        Uses args for GNN architecture and repr_proj_out_dim for the projector
        output size (defaults to 4096, matching typical LLM hidden size).
        Called once in __init__; _load_repr_align_weights() refreshes weights
        with FedAvg'd values each round without rebuilding the modules.
        """
        # Import gnn.py directly to avoid fedcond_grag/model/__init__.py pulling
        # in GraphLLM (which requires transformers — not available on a bare server).
        import importlib.util
        import pathlib
        _gnn_file = pathlib.Path(__file__).parent.parent / "model" / "gnn.py"
        _spec = importlib.util.spec_from_file_location("_fedrag_gnn_direct", _gnn_file)
        _gnn_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_gnn_mod)
        load_gnn_model = _gnn_mod.load_gnn_model

        args = self.args
        gnn_name = getattr(args, "gnn_model_name_c", getattr(args, "gnn_model_name", "gat"))
        num_layers = getattr(args, "gnn_num_layers_c", None) or getattr(args, "gnn_num_layers", 2)
        num_heads = getattr(args, "gnn_num_heads_c", None) or getattr(args, "gnn_num_heads", 4)
        hidden_dim = getattr(args, "gnn_hidden_dim_c", None) or getattr(args, "gnn_hidden_dim", 1024)
        in_dim = getattr(args, "gnn_in_dim_c", None) or getattr(args, "gnn_in_dim", 1024)
        dropout = float(getattr(args, "gnn_dropout", 0.0))
        proj_out = int(getattr(args, "repr_proj_out_dim", 4096))

        self.repr_encoder = load_gnn_model[gnn_name](
            in_channels=in_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
        ).to(self.device)

        self.repr_projector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_out),
        ).to(self.device)

        for p in self.repr_encoder.parameters():
            p.requires_grad_(False)
        for p in self.repr_projector.parameters():
            p.requires_grad_(False)

    def _load_repr_align_weights(self, state_dicts: dict) -> None:
        """Refresh repr_encoder + repr_projector from FedAvg'd client weights.

        Prefers condensed_encoder/projector_c (dual mode); falls back to
        graph_encoder/projector (shared-encoder mode).

        Rebuilds repr_projector if the state dict shape differs from the
        current module (e.g. repr_proj_out_dim default 4096 vs actual LLM
        hidden size 3584 for Qwen2.5-7B).
        """
        enc_key = "condensed_encoder" if "condensed_encoder" in state_dicts else "graph_encoder"
        proj_key = "projector_c" if "projector_c" in state_dicts else "projector"
        if enc_key not in state_dicts or proj_key not in state_dicts:
            return

        proj_sd = {k: v.float() for k, v in state_dicts[proj_key].items()}

        # Rebuild projector if shape doesn't match (e.g. LLM hidden ≠ repr_proj_out_dim)
        weight_keys = sorted(k for k in proj_sd if k.endswith(".weight"))
        actual_out = proj_sd[weight_keys[-1]].shape[0]
        current_out = self.repr_projector[-1].out_features
        if actual_out != current_out:
            proj_in  = proj_sd[weight_keys[0]].shape[1]
            proj_mid = proj_sd[weight_keys[0]].shape[0]
            self.repr_projector = nn.Sequential(
                nn.Linear(proj_in, proj_mid),
                nn.GELU(),
                nn.Linear(proj_mid, actual_out),
            ).to(self.device)
            for p in self.repr_projector.parameters():
                p.requires_grad_(False)

        self.repr_encoder.load_state_dict(
            {k: v.float() for k, v in state_dicts[enc_key].items()}, strict=False
        )
        self.repr_projector.load_state_dict(proj_sd, strict=False)
        self.repr_encoder.eval()
        self.repr_projector.eval()

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
