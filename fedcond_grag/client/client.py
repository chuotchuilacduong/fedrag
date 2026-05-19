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
        self._local_adj: list | None = None    # CPU adjacency lists for trigraph

    # ------------------------------------------------------------------
    # Setup helpers (called by FedTrainer)
    # ------------------------------------------------------------------

    def set_local_qa_data(self, samples: list) -> None:
        # Pre-attach evidence graphs once — avoids repeated retrieval across epochs.
        self.local_qa_samples = self._attach_evidence_graphs(samples)
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
        """Stage B: build anchor graph once on first round only."""
        start = time.perf_counter()
        if self.condensed_graph is None:
            # Try to load cached condensed graph (built by preprocessing) for round 0 only
            if self.condensed_graph is None:
                cached = self._try_load_condensed_cache()
                if cached is not None:
                    self.condensed_graph = cached
                    self.message_pool[f"client_{self.client_id}_extra_compute"] = (
                        self.message_pool.get(f"client_{self.client_id}_extra_compute", 0.0)
                        + time.perf_counter() - start
                    )
                    return
            self.condensed_graph = self._condense_anchor_graph(self.tri_graph)
        self.message_pool[f"client_{self.client_id}_extra_compute"] = (
            self.message_pool.get(f"client_{self.client_id}_extra_compute", 0.0)
            + time.perf_counter() - start
        )

    def local_train(self) -> float:
        """Stage D: train GNN encoder + projector on local QA data. Returns avg loss."""
        if self.shared_model is None or not self.local_qa_samples:
            return 0.0

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
        total_loss = 0.0
        total_steps = 0
        for _ in range(local_epochs):
            samples = list(self.local_qa_samples)   # graphs already attached
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
                total_loss += loss.item()
                total_steps += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Snapshot updated weights back into per-client state
        self._model_weights = {
            "graph_encoder": copy.deepcopy(self.shared_model.graph_encoder.state_dict()),
            "projector": copy.deepcopy(self.shared_model.projector.state_dict()),
            "condensed_encoder": copy.deepcopy(self.shared_model.condensed_encoder.state_dict()),
            "projector_c": copy.deepcopy(self.shared_model.projector_c.state_dict()),
        }
        return total_loss / max(total_steps, 1)

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

    def _attach_evidence_graphs(self, samples: list) -> list:
        """Batch-retrieve evidence subgraphs for all samples in one GPU matmul.

        Steps:
          1. Stack q_embs → [B, 384], score vs trigraph x_norm on GPU → [N, B]
          2. topk → seed indices per query (GPU)
          3. 1-hop expand + edge extraction via CPU adj list — O(|kept| × deg)
          4. Move resulting subgraphs to self.device

        x_norm is materialised on GPU only during this call, then freed.
        """
        import torch.nn.functional as F
        from torch_geometric.data import Data as _Data

        if not samples:
            return samples

        # Build CPU adj list once per client lifetime
        if self._local_adj is None:
            src_l = self.tri_graph.edge_index[0].tolist()
            dst_l = self.tri_graph.edge_index[1].tolist()
            N = self.tri_graph.x.size(0)
            adj: list = [[] for _ in range(N)]
            for u, v in zip(src_l, dst_l):
                adj[u].append(v)
                adj[v].append(u)
            self._local_adj = adj

        q_embs = [s.get("q_emb") for s in samples]
        has_emb = [q is not None for q in q_embs]
        if not any(has_emb):
            return samples

        # --- GPU batch scoring ---
        valid_qs = torch.stack([q for q in q_embs if q is not None]).float()  # [B, 384]
        x = self.tri_graph.x.float()
        x_norm_gpu = F.normalize(x, dim=-1).to(self.device)          # [N, 384] GPU
        q_norm_gpu = F.normalize(valid_qs, dim=-1).to(self.device)   # [B, 384] GPU

        top_r = int(getattr(self.args, "retrieval_top_r", 16))
        top_r = min(top_r, x_norm_gpu.size(0))

        # Process in blocks to avoid OOM on very large trigraphs
        BLOCK = 512
        topk_parts: list[torch.Tensor] = []
        for start in range(0, q_norm_gpu.size(0), BLOCK):
            q_block = q_norm_gpu[start : start + BLOCK]          # [b, 384]
            scores = x_norm_gpu @ q_block.T                      # [N, b]
            topk_parts.append(torch.topk(scores, top_r, dim=0).indices.T.cpu())  # [b, top_r]
        topk_idx = torch.cat(topk_parts, dim=0)                  # [B, top_r]

        del x_norm_gpu, q_norm_gpu
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # --- CPU subgraph extraction ---
        x_cpu = self.tri_graph.x
        nt_cpu = self.tri_graph.node_type
        adj = self._local_adj
        out = []
        valid_iter = iter(topk_idx)
        for sample, has in zip(samples, has_emb):
            s = dict(sample)
            if not has:
                out.append(s)
                continue
            seeds = next(valid_iter)
            seed_set = set(seeds.tolist())
            nbrs: set[int] = set()
            for seed in seed_set:
                nbrs.update(adj[seed])
            kept_set = seed_set | nbrs
            kept_list = sorted(kept_set)
            if not kept_list:
                out.append(s)
                continue
            local_map = {gid: lid for lid, gid in enumerate(kept_list)}
            kept_t = torch.tensor(kept_list, dtype=torch.long)
            src_e, dst_e = [], []
            for u in kept_list:
                lu = local_map[u]
                for v in adj[u]:
                    if v in kept_set:
                        src_e.append(lu)
                        dst_e.append(local_map[v])
            if src_e:
                sub_ei = torch.tensor([src_e, dst_e], dtype=torch.long)
                sub_ew = torch.ones(len(src_e), dtype=torch.float32)
            else:
                sub_ei = torch.zeros(2, 0, dtype=torch.long)
                sub_ew = torch.zeros(0, dtype=torch.float32)
            graph = _Data(
                x=x_cpu[kept_t],
                edge_index=sub_ei,
                edge_weight=sub_ew,
                node_type=nt_cpu[kept_t],
            )
            s["graph"] = graph          # keep on CPU; model moves it during forward
            s["evidence_graph"] = graph
            out.append(s)
        return out

    def _attach_condensed_graphs(self, samples: list, retriever: GlobalGraphRetriever) -> list:
        """Retrieve a condensed subgraph for each sample and attach it in-place."""
        out = []
        for sample in samples:
            s = dict(sample)
            graph = s.get("graph") or s.get("evidence_graph")
            if graph is not None and graph.x.numel() > 0:
                # Query must be on same device as retriever's synthetic graph
                retriever_device = retriever._graph.x.device
                query = graph.x.float().mean(0).to(retriever_device)
                result = retriever.retrieve(query)
                s["condensed_graph"] = result.data.cpu()
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

    def _try_load_condensed_cache(self) -> "Data | None":
        """Load pre-built condensed graph from preprocessed cache if available."""
        import os
        from pathlib import Path
        cache_path = Path(self.data_dir) / "condensed_graph.pt"
        if not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location=self.device, weights_only=False)
            condensed = Data(
                x=payload["x"].to(self.device),
                edge_index=payload["edge_index"].to(self.device),
                edge_weight=payload.get("edge_weight", torch.ones(payload["edge_index"].size(1))).to(self.device),
                node_type=payload["node_type"].to(self.device),
            )
            condensed.y = condensed.node_type.long()
            condensed.num_global_classes = 3
            print(f"    [client_{self.client_id}] Loaded cached condensed_graph.pt ({condensed.x.size(0)} anchors)")
            return condensed
        except Exception as exc:
            print(f"    [client_{self.client_id}] Failed to load condensed cache: {exc} — rebuilding")
            return None

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
