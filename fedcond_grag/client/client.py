"""fedcond_qa client: Stage B anchor condensation + Stage D local training."""

from __future__ import annotations

import copy
import random
import time
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data

from fedcond_grag.server.stage_c_aggregate.task import CondensationQATask
from fedcond_grag.client.stage_b_condense import ClientCondensationConfig, ClientCondensor, MotifSelectorConfig
from fedcond_grag.client.stage_b_condense.text_bank import TextBank, build_text_bank, load_frozen_encoder
from fedcond_grag.client.stage_d_retrieve.global_graph_retriever import GlobalGraphRetriever
from fedcond_grag.utils.collate import collate_fn

if TYPE_CHECKING:
    from fedcond_grag.model.dual_graph_llm import DualGraphLLM


class FedCondQAClient:
    """Client for FedCondGraphRAG.

    Round 0  — Stage B: condense local Tri-Graph → anchor graph C_m.
    Round >= 1 — Stage D: local DualGraphLLM training with synthetic graph
                 from server; exchange GNN + projector weights via FedAvg.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        self.args = args
        self.client_id = client_id
        self.data_dir = data_dir
        self.message_pool = message_pool
        self.device = device
        self.task = CondensationQATask(args, client_id, data, data_dir, device)
        self.tri_graph = self.task.splitted_data["data"]
        self.condensed_graph = None
        self.text_bank: TextBank | None = None
        self.condensor: ClientCondensor | None = None

        # Stage D fields (populated by FedTrainer after LLM is loaded)
        self.shared_model: DualGraphLLM | None = None
        self.local_qa_samples: list = []
        self.synthetic_graph: Data | None = None
        self._model_weights: dict | None = None   # per-client GNN/proj state dicts
        self._num_local_samples: int = 0

    # ------------------------------------------------------------------
    # Setup helpers (called by FedTrainer)
    # ------------------------------------------------------------------

    def set_local_qa_data(self, samples: list) -> None:
        self.local_qa_samples = samples
        self._num_local_samples = len(samples)

    def set_shared_model(self, model: "DualGraphLLM") -> None:
        """Store reference to the shared LLM and snapshot initial weights."""
        self.shared_model = model
        self._model_weights = {
            "graph_encoder": copy.deepcopy(model.graph_encoder.state_dict()),
            "projector": copy.deepcopy(model.projector.state_dict()),
            "condensed_encoder": copy.deepcopy(model.condensed_encoder.state_dict()),
            "projector_c": copy.deepcopy(model.projector_c.state_dict()),
        }

    # ------------------------------------------------------------------
    # FL round methods
    # ------------------------------------------------------------------

    def receive_message(self) -> None:
        """Load synthetic graph + aggregated model weights from server."""
        msg = self.message_pool.get("server", {})

        synthetic_graph = msg.get("synthetic_graph")
        if synthetic_graph is not None:
            self.synthetic_graph = synthetic_graph

        model_weights = msg.get("model_weights")
        if model_weights and self._model_weights is not None:
            for key in ("graph_encoder", "projector", "condensed_encoder", "projector_c"):
                if key in model_weights:
                    self._model_weights[key] = {
                        k: v.clone() for k, v in model_weights[key].items()
                    }

    def execute(self) -> None:
        """Stage B: refresh anchor graph if due."""
        start = time.perf_counter()
        refresh_every = int(getattr(self.args, "condense_refresh_every", 10))
        round_id = int(self.message_pool.get("round", 0))
        if self.condensed_graph is None or round_id % refresh_every == 0:
            self.condensed_graph = self._condense_anchor_graph(self.tri_graph)
        self.message_pool[f"client_{self.client_id}_extra_compute"] = (
            self.message_pool.get(f"client_{self.client_id}_extra_compute", 0.0)
            + time.perf_counter() - start
        )

    def local_train(self) -> None:
        """Stage D: train GNN encoder + projector on local QA data."""
        if self.shared_model is None or not self.local_qa_samples:
            return

        # Load this client's weights into the shared model
        self._load_weights_into_model()

        retriever = (
            GlobalGraphRetriever(
                self.synthetic_graph,
                top_r=int(getattr(self.args, "retrieval_top_r", 16)),
                max_nodes=getattr(self.args, "retrieval_max_nodes", None),
            )
            if self.synthetic_graph is not None
            else None
        )

        trainable = (
            list(self.shared_model.graph_encoder.parameters())
            + list(self.shared_model.projector.parameters())
            + list(self.shared_model.condensed_encoder.parameters())
            + list(self.shared_model.projector_c.parameters())
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(getattr(self.args, "local_lr", 1e-5)),
            weight_decay=float(getattr(self.args, "local_wd", 0.05)),
        )

        local_epochs = int(getattr(self.args, "local_epochs", 1))
        batch_size = int(getattr(self.args, "local_batch_size", 4))
        grad_clip = float(getattr(self.args, "local_grad_clip", 0.1))

        self.shared_model.train()
        for _ in range(local_epochs):
            samples = list(self.local_qa_samples)
            random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                mini = samples[i : i + batch_size]
                if retriever is not None:
                    mini = self._attach_condensed_graphs(mini, retriever)
                batch = collate_fn(mini)
                optimizer.zero_grad()
                loss = self.shared_model(batch)
                loss.backward()
                clip_grad_norm_(trainable, grad_clip)
                optimizer.step()

        # Snapshot updated weights back into per-client state
        self._model_weights = {
            "graph_encoder": copy.deepcopy(self.shared_model.graph_encoder.state_dict()),
            "projector": copy.deepcopy(self.shared_model.projector.state_dict()),
            "condensed_encoder": copy.deepcopy(self.shared_model.condensed_encoder.state_dict()),
            "projector_c": copy.deepcopy(self.shared_model.projector_c.state_dict()),
        }

    def send_message(self) -> None:
        if self.condensed_graph is None:
            self.condensed_graph = self._condense_anchor_graph(self.tri_graph)
        msg: dict = {
            "anchor_graph": self.condensed_graph,
            "num_anchor_nodes": int(self.condensed_graph.x.size(0)),
        }
        if self._model_weights is not None and self._num_local_samples > 0:
            msg["model_weights"] = self._model_weights
            msg["num_samples"] = self._num_local_samples
        self.message_pool[f"client_{self.client_id}"] = msg

    def upload(self) -> Data:
        if self.condensed_graph is None:
            self.condensed_graph = self._condense_anchor_graph(self.tri_graph)
        return self.condensed_graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_weights_into_model(self) -> None:
        assert self.shared_model is not None and self._model_weights is not None
        self.shared_model.graph_encoder.load_state_dict(self._model_weights["graph_encoder"])
        self.shared_model.projector.load_state_dict(self._model_weights["projector"])
        self.shared_model.condensed_encoder.load_state_dict(self._model_weights["condensed_encoder"])
        self.shared_model.projector_c.load_state_dict(self._model_weights["projector_c"])

    def _attach_condensed_graphs(self, samples: list, retriever: GlobalGraphRetriever) -> list:
        """Retrieve a condensed subgraph for each sample and attach it in-place."""
        out = []
        for sample in samples:
            s = dict(sample)
            graph = s.get("graph") or s.get("evidence_graph")
            if graph is not None and graph.x.numel() > 0:
                query = graph.x.float().mean(0).to(self.device)
                result = retriever.retrieve(query)
                s["condensed_graph"] = result.data.to(self.device)
            out.append(s)
        return out

    def _condense_anchor_graph(self, graph) -> Data:
        if not hasattr(graph, "node_type") and hasattr(graph, "y"):
            graph.node_type = graph.y
        if not hasattr(graph, "node_type"):
            raise ValueError("fedcond_qa anchor graph requires node_type labels")
        graph = graph.to(self.device)
        if self.text_bank is None:
            node_texts = self._node_texts(graph)
            encoder = load_frozen_encoder("all-MiniLM-L6-v2", dim=384)
            self.text_bank = build_text_bank(
                node_texts,
                encoder=encoder,
                encoder_name="all-MiniLM-L6-v2",
                dim=384,
                device=self.device,
            )
        if self.condensor is None:
            self.condensor = ClientCondensor(
                graph_dim=int(graph.x.size(1)),
                text_dim=int(self.text_bank.node_embeddings.size(1)),
                config=self._stage_b_config(),
            ).to(self.device)

        with torch.no_grad():
            condensed = self.condensor(graph, text_bank=self.text_bank).to_pyg_data()
        condensed.y = condensed.node_type.long()
        condensed.num_global_classes = 3
        return condensed

    def _stage_b_config(self) -> ClientCondensationConfig:
        motif = MotifSelectorConfig(
            entity_ratio=float(getattr(self.args, "stage_b_entity_ratio", 0.05)),
            sentence_budget=int(getattr(self.args, "stage_b_sentence_budget", 3)),
            passage_budget=int(getattr(self.args, "stage_b_passage_budget", 3)),
            lambda_idf=float(getattr(self.args, "stage_b_lambda_idf", 1.0)),
            lambda_pr=float(getattr(self.args, "stage_b_lambda_pr", 0.5)),
            lambda_mmr=float(getattr(self.args, "stage_b_lambda_mmr", 0.3)),
        )
        return ClientCondensationConfig(
            motif=motif,
            text_budgets=(
                int(getattr(self.args, "stage_b_budget_0", 1)),
                int(getattr(self.args, "stage_b_budget_1", 3)),
                int(getattr(self.args, "stage_b_budget_2", 2)),
            ),
            chunk_budget=int(getattr(self.args, "stage_b_chunk_budget", 8)),
            topology_method=str(getattr(self.args, "stage_b_topology_method", "knn")),
            knn_k=int(getattr(self.args, "stage_b_knn_k", 8)),
            prior_weight=float(getattr(self.args, "stage_b_prior_weight", 0.0)),
            self_expr_candidate_size=int(getattr(self.args, "stage_b_self_expr_candidate_size", 16)),
            self_expr_iterations=int(getattr(self.args, "condense_iters", 50)),
            preserve_sep_topology=_as_bool(getattr(self.args, "preserve_sep_topology", True)),
        )

    def _node_texts(self, graph) -> list[str]:
        if hasattr(graph, "node_text"):
            node_text = graph.node_text
            if isinstance(node_text, (list, tuple)) and len(node_text) == graph.x.size(0):
                return [str(text) for text in node_text]
        return [f"node_type_{int(t)} node_{i}" for i, t in enumerate(graph.node_type.detach().cpu().tolist())]


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
