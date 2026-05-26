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
from fedcond_grag.client.stage_b_condense import ClientCondensationConfig, ClientCondensor, AnchorSelectorConfig
from fedcond_grag.client.stage_b_condense.node_text_embedder import NodeTextBank, build_text_bank, load_frozen_encoder
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
        self.text_bank: NodeTextBank | None = None
        self.condensor: ClientCondensor | None = None

        # Stage D fields (populated by FedTrainer after LLM is loaded)
        self.shared_model: DualGraphLLM | None = None
        self.local_qa_samples: list = []
        self.synthetic_graph: Data | None = None
        self._model_weights: dict | None = None   # per-client GNN/proj state dicts
        self._num_local_samples: int = 0
        self._local_adj: list | None = None    # CPU adjacency lists for trigraph
        # AdamW persisted across rounds — Adam's v_t needs many steps to warm
        # up; rebuilding it every round (the old behaviour) kept it permanently
        # cold and effective LR ≈ 0.
        self._optimizer: torch.optim.Optimizer | None = None
        # Full pool of pre-attached train samples (evidence graphs attached once
        # at startup). Each round we randomly sample max_per from this pool.
        self._train_pool: list = []

    # ------------------------------------------------------------------
    # Setup helpers (called by FedTrainer)
    # ------------------------------------------------------------------

    def set_local_qa_data(self, samples: list) -> None:
        # Pre-attach evidence graphs once — avoids repeated retrieval across epochs.
        self.local_qa_samples = self._attach_evidence_graphs(samples)
        self._num_local_samples = len(samples)

    def set_full_train_pool(self, samples: list, max_per_round: int | None = None) -> None:
        """Pre-attach evidence graphs for all train samples (done once at startup).

        Each round, call sample_train_for_round(n) to pick a fresh random
        subset so all samples are covered over many FL rounds.
        """
        print(f"    client_{self.client_id}: attaching evidence graphs for "
              f"{len(samples)} train samples (one-time startup cost)...", flush=True)
        self._train_pool = self._attach_evidence_graphs(samples)
        n = max_per_round if max_per_round and max_per_round < len(self._train_pool) else len(self._train_pool)
        self._num_local_samples = n
        self.sample_train_for_round(n)
        print(f"    client_{self.client_id}: pool={len(self._train_pool)}, "
              f"per-round budget={n}", flush=True)

    def sample_train_for_round(self, n: int | None = None) -> None:
        """Pick a fresh random subset of n samples from the full train pool."""
        if not self._train_pool:
            return
        pool = self._train_pool
        if n is None or n >= len(pool):
            self.local_qa_samples = list(pool)
        else:
            self.local_qa_samples = random.sample(pool, n)
        self._num_local_samples = len(self.local_qa_samples)

    def set_shared_model(self, model: "DualGraphLLM") -> None:
        """Store reference to the shared LLM and snapshot initial weights."""
        self.shared_model = model
        self._model_weights = {
            "graph_encoder": copy.deepcopy(model.graph_encoder.state_dict()),
            "projector": copy.deepcopy(model.projector.state_dict()),
            **({"condensed_encoder": copy.deepcopy(model.condensed_encoder.state_dict())}
               if model.condensed_encoder is not None else {}),
            **({"projector_c": copy.deepcopy(model.projector_c.state_dict())}
               if model.projector_c is not None else {}),
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

    def local_train(self, log_fn=None, global_step_start: int = 0) -> tuple[float, int]:
        """Stage D: train GNN encoder + projector on local QA data.

        Returns (avg_loss, steps_taken). global_step_start offsets the WandB
        x-axis so steps are monotone across all clients and rounds.
        """
        if self.shared_model is None or not self.local_qa_samples:
            return 0.0, 0

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
            + (list(self.shared_model.condensed_encoder.parameters())
               if self.shared_model.condensed_encoder is not None else [])
            + (list(self.shared_model.projector_c.parameters())
               if self.shared_model.projector_c is not None else [])
        )
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                trainable,
                lr=float(getattr(self.args, "local_lr", 1e-4)),
                weight_decay=float(getattr(self.args, "local_wd", 0.05)),
                betas=(0.9, 0.95),
            )
        optimizer = self._optimizer

        local_epochs = int(getattr(self.args, "local_epochs", 3))
        batch_size = int(getattr(self.args, "local_batch_size", 4))
        grad_clip = float(getattr(self.args, "local_grad_clip", 1.0))

        # Pre-attach condensed graphs ONCE per round — within a round the
        # synthetic graph is fixed and per-sample retrieval results don't
        # change between epochs/batches. Calling it per mini-batch (the old
        # behaviour) ran the same retrieval `epochs * ceil(N/bs)` times.
        if retriever is not None:
            base_samples = self._attach_condensed_graphs(
                list(self.local_qa_samples), retriever
            )
        else:
            base_samples = list(self.local_qa_samples)

        self.shared_model.train()
        total_loss = 0.0
        total_steps = 0
        steps_per_epoch = (len(base_samples) + batch_size - 1) // batch_size
        total_planned = steps_per_epoch * local_epochs
        log_every = max(1, steps_per_epoch // 10)   # ~10 prints per epoch
        t_start = time.perf_counter()
        for epoch in range(local_epochs):
            samples = list(base_samples)
            random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                mini = samples[i : i + batch_size]
                batch = collate_fn(mini)
                optimizer.zero_grad()
                loss = self.shared_model(batch)
                loss.backward()
                clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                step_loss = loss.item()
                total_loss += step_loss
                total_steps += 1
                if log_fn is not None:
                    log_fn(
                        {
                            f"train/client_{self.client_id}_step_loss": step_loss,
                            "train/step_loss": step_loss,
                        },
                        step=global_step_start + total_steps - 1,
                    )
                if total_steps % log_every == 0 or total_steps == total_planned:
                    elapsed = time.perf_counter() - t_start
                    sps = total_steps / elapsed
                    eta = (total_planned - total_steps) / sps if sps > 0 else 0
                    avg = total_loss / total_steps
                    print(
                        f"    [client_{self.client_id}] ep{epoch+1} "
                        f"step {total_steps}/{total_planned} | "
                        f"loss {avg:.4f} | {sps:.2f} s/s | "
                        f"ETA {eta/60:.1f}m",
                        flush=True,
                    )
                # NOTE: torch.cuda.empty_cache() removed — it forces a full
                # CUDA sync after every step. The PyTorch caching allocator
                # already reuses freed blocks; calling empty_cache() just
                # gives memory back to the driver and re-allocates next step.

        # Snapshot updated weights — skip components that don't exist (shared mode)
        self._model_weights = {
            "graph_encoder": copy.deepcopy(self.shared_model.graph_encoder.state_dict()),
            "projector": copy.deepcopy(self.shared_model.projector.state_dict()),
            **({"condensed_encoder": copy.deepcopy(self.shared_model.condensed_encoder.state_dict())}
               if self.shared_model.condensed_encoder is not None else {}),
            **({"projector_c": copy.deepcopy(self.shared_model.projector_c.state_dict())}
               if self.shared_model.projector_c is not None else {}),
        }
        return total_loss / max(total_steps, 1), total_steps

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
        if self.shared_model.condensed_encoder is not None and "condensed_encoder" in self._model_weights:
            self.shared_model.condensed_encoder.load_state_dict(self._model_weights["condensed_encoder"])
        if self.shared_model.projector_c is not None and "projector_c" in self._model_weights:
            self.shared_model.projector_c.load_state_dict(self._model_weights["projector_c"])

    def _attach_evidence_graphs(self, samples: list) -> list:
        """Build per-sample evidence subgraphs anchored on top-r passage nodes.

        Two seed sources (used independently per sample, anchor-first):
          1. Anchor nodes — sample["anchor_passage_nodes"] is a list of
             (client_id, node_id) for the top-r question-relevant passages.
             We keep only those whose client_id == self.client_id (i.e., live
             on this client's trigraph).
          2. Fallback — if no anchor nodes are local, fall back to the legacy
             q_emb-vs-trigraph top-r retrieval so the graph path still works.

        From the seed set we do 1-hop expansion in the local trigraph and
        return a CPU Data with x / edge_index / edge_weight / node_type /
        node_text preserved. node_text is needed downstream for any future
        text injection of retrieved nodes.
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

        N = self.tri_graph.x.size(0)
        x_cpu = self.tri_graph.x.cpu()
        nt_cpu = self.tri_graph.node_type.cpu()
        node_text = getattr(self.tri_graph, "node_text", None)
        adj = self._local_adj

        # Plan per-sample seeds.
        per_sample_seeds: list[set[int]] = []
        fallback_idx: list[int] = []
        fallback_qs: list[torch.Tensor] = []
        for i, s in enumerate(samples):
            anchors = s.get("anchor_passage_nodes") or []
            local_anchor_nodes = [int(nid) for (cid, nid) in anchors
                                  if int(cid) == int(self.client_id) and 0 <= int(nid) < N]
            if local_anchor_nodes:
                per_sample_seeds.append(set(local_anchor_nodes))
            else:
                per_sample_seeds.append(set())
                if s.get("q_emb") is not None:
                    fallback_idx.append(i)
                    fallback_qs.append(s["q_emb"])

        # Fallback retrieval for samples without local anchors.
        # Always computed on CPU — this is a one-time startup cost and the
        # trigraph features are already in CPU memory. Avoids competing with
        # the LLM for scarce VRAM when other processes are running.
        if fallback_qs:
            top_r = int(getattr(self.args, "retrieval_top_r", 16))
            top_r = min(top_r, N)
            x_norm = F.normalize(self.tri_graph.x.float().cpu(), dim=-1)
            q_norm = F.normalize(torch.stack(fallback_qs).float(), dim=-1)
            BLOCK = 256
            topk_parts: list[torch.Tensor] = []
            for start in range(0, q_norm.size(0), BLOCK):
                q_block = q_norm[start : start + BLOCK]
                scores = x_norm @ q_block.T          # [N, block] on CPU
                topk_parts.append(torch.topk(scores, top_r, dim=0).indices.T)
            topk_idx = torch.cat(topk_parts, dim=0)
            for j, i in enumerate(fallback_idx):
                per_sample_seeds[i] = set(topk_idx[j].tolist())

        # CPU subgraph extraction (1-hop expand + edge filter)
        out = []
        for sample, seed_set in zip(samples, per_sample_seeds):
            s = dict(sample)
            if not seed_set:
                out.append(s)
                continue
            kept_set = set(seed_set)
            for seed in seed_set:
                kept_set.update(adj[seed])
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
            if node_text is not None and isinstance(node_text, (list, tuple)) and len(node_text) == N:
                graph.node_text = [node_text[i] for i in kept_list]
            s["graph"] = graph          # keep on CPU; model moves it during forward
            s["evidence_graph"] = graph
            out.append(s)
        return out

    def _attach_condensed_graphs(self, samples: list, retriever: GlobalGraphRetriever) -> list:
        """Retrieve condensed subgraphs for all samples — one batched matmul."""
        retriever_device = retriever._graph.x.device

        # Collect mean-pool queries for samples that have a valid evidence graph
        queries: list[torch.Tensor] = []
        has_graph: list[bool] = []
        for sample in samples:
            graph = sample.get("graph") or sample.get("evidence_graph")
            if graph is not None and graph.x.numel() > 0:
                queries.append(graph.x.float().mean(0))
                has_graph.append(True)
            else:
                has_graph.append(False)

        if not queries:
            return [dict(s) for s in samples]

        # One [K,d]@[d,N] matmul instead of N sequential [K,d]@[d,1] matmuls
        query_tensor = torch.stack(queries).to(retriever_device)     # [M, d]
        results = retriever.retrieve_batch_queries(query_tensor)

        out: list = []
        result_idx = 0
        for sample, has_g in zip(samples, has_graph):
            s = dict(sample)
            if has_g:
                s["condensed_graph"] = results[result_idx].data.cpu()
                result_idx += 1
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
        motif = AnchorSelectorConfig(
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
