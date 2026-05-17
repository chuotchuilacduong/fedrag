"""Unified federated round loop for FedCondGraphRAG.

Round 0  — bootstrap: Stage B (client condense) + Stage C (server gradient
           matching) to produce the initial synthetic global graph.

Round >= 1 — FL loop: each sampled client loads aggregated model weights,
             trains locally on its QA partition (Stage D), then sends updated
             GNN + projector weights back.  Server FedAvg-aggregates the
             weights and re-runs Stage C with the updated surrogate to refine
             the synthetic graph.

The shared LLM (frozen) is instantiated once inside FedTrainer and passed by
reference to all clients so that only one copy lives in memory.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from fedcond_grag.client.client import FedCondQAClient
from fedcond_grag.server.server import FedCondQAServer


class FedTrainer:
    """Single-process simulator of the unified FedCondGraphRAG FL loop."""

    def __init__(self, args):
        self.args = args
        self.message_pool: dict = {}
        self.device = torch.device(
            f"cuda:{getattr(args, 'gpuid', 0)}"
            if torch.cuda.is_available() and getattr(args, "use_cuda", False)
            else "cpu"
        )

        self.clients: list[FedCondQAClient] = []
        for client_id in range(args.num_clients):
            data, data_dir = self._load_client_data(client_id)
            self.clients.append(
                FedCondQAClient(args, client_id, data, data_dir, self.message_pool, self.device)
            )

        global_data, global_dir = self._load_global_data()
        self.server = FedCondQAServer(args, global_data, global_dir, self.message_pool, self.device)

        # Stage D setup: load shared LLM + partition QA data (only if num_rounds > 1)
        self.shared_model = None
        if int(getattr(args, "num_rounds", 1)) > 1:
            self._init_stage_d()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        num_rounds = int(getattr(self.args, "num_rounds", 1))
        client_frac = float(getattr(self.args, "client_frac", 1.0))

        for round_id in range(num_rounds):
            sampled = sorted(random.sample(
                range(self.args.num_clients),
                max(1, int(self.args.num_clients * client_frac)),
            ))
            print(f"\n=== round {round_id} | sampled clients: {sampled} ===")

            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled

            # Server broadcasts synthetic graph + aggregated weights (empty in round 0)
            self.server.send_message()

            for cid in sampled:
                t0 = time.perf_counter()

                # Load synthetic graph + aggregated weights from server
                self.clients[cid].receive_message()

                # Stage B: refresh anchor graph when due
                self.clients[cid].execute()

                # Stage D: local training from round 1 onwards
                if round_id >= 1 and self.shared_model is not None:
                    self.clients[cid].local_train()

                self.clients[cid].send_message()
                print(f"    client_{cid} done in {time.perf_counter() - t0:.1f}s")

            t0 = time.perf_counter()
            # Stage C: gradient matching → update synthetic graph; FedAvg model weights
            self.server.execute()
            print(f"    server agg done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage D initialisation
    # ------------------------------------------------------------------

    def _init_stage_d(self) -> None:
        """Load shared DualGraphLLM and distribute QA data to clients."""
        # 1. Load QA dataset
        from fedcond_grag.dataloader import FedCondQADataset
        qa_root = getattr(self.args, "qa_data_root", "dataset/fedcond_qa")
        try:
            qa_dataset = FedCondQADataset(root=qa_root)
        except FileNotFoundError as exc:
            print(f"[FedTrainer] Stage D disabled: QA dataset not found — {exc}")
            return

        # 2. Partition samples to clients by index % num_clients
        n = self.args.num_clients
        for cid, client in enumerate(self.clients):
            client_samples = [qa_dataset[i] for i in range(len(qa_dataset)) if i % n == cid]
            client.set_local_qa_data(client_samples)
            print(f"    client_{cid}: {len(client_samples)} QA samples")

        # 3. Load shared DualGraphLLM (frozen LLM, trainable GNN + projectors)
        try:
            from fedcond_grag.model import load_model, llama_model_path
        except ImportError as exc:
            print(f"[FedTrainer] Stage D disabled: model import failed — {exc}")
            return

        llm_name = getattr(self.args, "llm_model_name", "7b")
        llm_path = getattr(self.args, "llm_model_path", "") or llama_model_path.get(llm_name, "")
        if not llm_path:
            print(f"[FedTrainer] Stage D disabled: llm_model_path not set")
            return

        self.args.llm_model_path = llm_path
        # Ensure Stage D GNN args have defaults if not set
        for attr, default in (
            ("gnn_model_name", "gt"),
            ("gnn_model_name_c", "gat"),
            ("gnn_num_layers", 4),
            ("gnn_num_layers_c", None),
            ("gnn_in_dim", 384),
            ("gnn_in_dim_c", None),
            ("gnn_hidden_dim", 384),
            ("gnn_hidden_dim_c", None),
            ("gnn_num_heads", 4),
            ("gnn_num_heads_c", None),
            ("gnn_dropout", 0.0),
            ("dual_graph_mode", "both"),
            ("max_txt_len", 512),
            ("max_new_tokens", 32),
            ("llm_frozen", "True"),
        ):
            if not hasattr(self.args, attr):
                setattr(self.args, attr, default)

        try:
            model = load_model["dual_graph_llm"](args=self.args)
        except Exception as exc:
            print(f"[FedTrainer] Stage D disabled: failed to load DualGraphLLM — {exc}")
            return

        # Freeze everything except GNN + projectors
        for name, param in model.named_parameters():
            is_gnn = any(
                name.startswith(k)
                for k in ("graph_encoder", "projector", "condensed_encoder", "projector_c")
            )
            param.requires_grad = is_gnn

        model.to(self.device)
        self.shared_model = model

        # Give each client a reference + snapshot of initial weights
        for client in self.clients:
            client.set_shared_model(model)

        print(f"[FedTrainer] Stage D ready — shared DualGraphLLM loaded on {self.device}")

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _processed_root(self) -> Path:
        return Path(getattr(self.args, "data_root", "processed")) / self.args.dataset

    def _load_client_data(self, client_id: int):
        from torch_geometric.data import Data

        client_dir = self._processed_root() / f"client_{client_id}"
        path = client_dir / "trigraph.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trigraph for client {client_id}: {path}. "
                f"Run `python main.py preprocess --dataset {self.args.dataset}` first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        data = Data(
            x=payload["x"],
            edge_index=payload["edge_index"],
            edge_type=payload["edge_type"],
            node_type=payload["node_type"],
            node_text=payload.get("node_text", []),
        )
        data.y = data.node_type.long()
        data.num_global_classes = 3
        return data, str(client_dir)

    def _load_global_data(self):
        global_dir = self._processed_root() / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        data, _ = self._load_client_data(0)
        return data, str(global_dir)
