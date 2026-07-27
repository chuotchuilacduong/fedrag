"""fedcond_qa client: Stage B anchor condensation + Stage D local training."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data

from fedcond_grag.server.stage_c_aggregate.task import CondensationQATask
from fedcond_grag.server.lora_aggregate import has_lora, lora_state_dict
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
        # Stage E (FedRAG Phase 1) — broadcast Θ_syn state + outgoing delta Δ_m
        self._synthetic_state: dict | None = None
        self._syn_delta: dict | None = None
        # Stage B refinement (paper B.3.5) runs once per client lifetime
        self._condense_refined: bool = False
        # Paper Phase 0: the anchor graph G̃_m is uploaded exactly once
        # (fedrag mode); legacy modes keep re-sending it every round.
        self._anchor_uploaded: bool = False
        # Stage B refine stats for the trainer to log (consumed once)
        self.last_stage_b_refine: dict | None = None
        self._model_weights: dict | None = None   # per-client GNN/proj state dicts
        self._num_local_samples: int = 0
        self._local_adj: list | None = None    # CPU adjacency lists for trigraph
        # Per-client PPR node map: [Q, top_k] int64 — local trigraph node IDs
        # for each question's PPR-selected passages. Loaded from
        # processed/{dataset}/client_{c}/ppr_node_map.pt.
        self._ppr_node_map: "torch.Tensor | None" = self._load_ppr_node_map()
        # AdamW persisted across rounds — Adam's v_t needs many steps to warm
        # up; rebuilding it every round (the old behaviour) kept it permanently
        # cold and effective LR ≈ 0.
        self._optimizer: torch.optim.Optimizer | None = None
        # Raw (un-attached) train sample pool. Evidence graphs are attached
        # lazily in sample_train_for_round so startup cost is O(1) regardless
        # of pool size. Only the per-round subset is ever built at once.
        self._train_pool: list = []
        self._train_pool_max_per_round: int = 0

    def _load_ppr_node_map(self) -> "torch.Tensor | None":
        """Load this client's per-query PPR node map if available."""
        data_root = getattr(self.args, "data_root", "processed")
        dataset = getattr(self.args, "dataset", "")
        if isinstance(dataset, (list, tuple)):  # GFL-style configs pass a list
            dataset = dataset[0] if dataset else ""
        path = str(Path(data_root) / str(dataset))
        map_path = Path(path) / f"client_{self.client_id}" / "ppr_node_map.pt"
        if map_path.exists():
            m = torch.load(map_path, map_location="cpu", weights_only=True)
            print(f"    [client_{self.client_id}] Loaded ppr_node_map.pt {tuple(m.shape)}")
            return m
        return None

    # ------------------------------------------------------------------
    # Setup helpers (called by FedTrainer)
    # ------------------------------------------------------------------

    def set_local_qa_data(self, samples: list) -> None:
        # Pre-attach evidence graphs once — avoids repeated retrieval across epochs.
        self.local_qa_samples = self._attach_evidence_graphs(samples)
        self._num_local_samples = len(samples)

    def set_full_train_pool(self, samples: list, max_per_round: int | None = None) -> None:
        """Store raw samples for lazy per-round evidence graph attachment.

        Evidence graphs are NOT built here — they are built in
        sample_train_for_round so the cost is O(max_per_round) per round,
        not O(len(samples)) at startup.
        """
        self._train_pool = samples  # raw records, no graph attached yet
        n = max_per_round if max_per_round and max_per_round < len(samples) else len(samples)
        self._train_pool_max_per_round = n
        self._num_local_samples = n
        print(f"    client_{self.client_id}: pool={len(self._train_pool)}, "
              f"per-round budget={n} (evidence graphs built per-round)", flush=True)
        self.sample_train_for_round(n)

    def sample_train_for_round(self, n: int | None = None) -> None:
        """Pick a fresh random subset — evidence graphs are built per mini-batch in local_train."""
        if not self._train_pool:
            return
        n_actual = n if n is not None else self._train_pool_max_per_round
        pool = self._train_pool
        subset = list(pool) if (n_actual is None or n_actual >= len(pool)) else random.sample(pool, n_actual)
        self.local_qa_samples = subset  # raw samples; no graph attached yet
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
        if has_lora(model.model):
            self._model_weights["lora"] = lora_state_dict(model.model)

    # ------------------------------------------------------------------
    # FL round methods
    # ------------------------------------------------------------------

    def receive_message(self) -> None:
        """Load synthetic graph + aggregated model weights from server."""
        msg = self.message_pool.get("server", {})

        synthetic_graph = msg.get("synthetic_graph")
        if synthetic_graph is not None:
            self.synthetic_graph = synthetic_graph

        # Θ_syn = {X_syn, θ_PGE} broadcast for query-conditioned adaptation
        self._synthetic_state = msg.get("synthetic_state", self._synthetic_state)
        self._syn_delta = None

        model_weights = msg.get("model_weights")
        if model_weights and self._model_weights is not None:
            for key in ("graph_encoder", "projector", "condensed_encoder", "projector_c", "lora"):
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
                    self.condensed_graph = self._maybe_refine_condensed(cached)
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

        # Paper B.5.2: during training the synthetic context is retrieved
        # softly/differentiably from Θ_syn^(r,0); hard top-k retrieval over
        # the exported synthetic graph is inference/eval-only.
        soft_syn = (
            self._prepare_soft_syn_context()
            if str(getattr(self.args, "server_stage_c_mode", "")) == "fedrag"
            else None
        )
        retriever = (
            GlobalGraphRetriever(
                self.synthetic_graph,
                top_r=int(getattr(self.args, "retrieval_top_r", 16)),
                max_nodes=getattr(self.args, "retrieval_max_nodes", None),
            )
            if (self.synthetic_graph is not None and soft_syn is None)
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
        lora_enabled = has_lora(self.shared_model.model)
        if lora_enabled:
            # RoLoRA alternates which half is trainable by round parity; every
            # other strategy just trains both halves every round. Both halves
            # stay registered with the (persistent) optimizer regardless --
            # AdamW skips params whose .grad is None, so toggling
            # requires_grad here is enough without rebuilding the optimizer.
            agg_method = str(getattr(self.args, "lora_agg_method", "fedit")).lower()
            round_id = int(self.message_pool.get("round", 0))
            active_half = "lora_B" if round_id % 2 == 1 else "lora_A"
            lora_params = []
            for name, param in self.shared_model.model.named_parameters():
                if ".lora_A." in name or ".lora_B." in name:
                    if agg_method == "rolora":
                        param.requires_grad_(f".{active_half}." in name)
                    else:
                        param.requires_grad_(True)
                    lora_params.append(param)
            trainable += lora_params
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

        base_samples = list(self.local_qa_samples)  # raw samples, no graph attached

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
                # Evidence graph built per query — PPR anchors → 1-hop subgraph + desc
                mini = self._attach_evidence_graphs(mini)
                # Condensed graph retrieved per batch — synthetic graph is fixed within a round
                if retriever is not None:
                    mini = self._attach_condensed_graphs(mini, retriever)
                batch = collate_fn(mini)
                if soft_syn is not None:
                    batch["z_c_soft"] = self._soft_syn_batch_context(batch, soft_syn)
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
        if lora_enabled:
            self._model_weights["lora"] = lora_state_dict(self.shared_model.model)
        return total_loss / max(total_steps, 1), total_steps

    def adapt_synthetic_memory(self) -> float | None:
        """FedRAG Phase 1 (paper §3.5 / B.7): K_mem local steps on a copy of Θ_syn.

        Minimizes L_mem = L_QA + λ_gm·L_GM + λ_align·L_align + λ_reg·L_reg on
        private mini-batches, then stores Δ_m = Θ_syn,m − Θ_syn for upload.
        The prompt module W stays fixed here — only the local synthetic-memory
        copy is optimized. Returns the mean QA loss under synthetic context,
        or None if adaptation is disabled or prerequisites are missing.
        """
        from fedcond_grag.client.stage_e_memory import (
            LocalSyntheticMemory,
            gradient_matching_loss,
        )
        from fedcond_grag.server.stage_c_aggregate.repr_align import (
            encode_nodes,
            encode_nodes_with_edge_weight,
        )

        k_mem = int(getattr(self.args, "syn_mem_steps", 5))
        if (
            k_mem <= 0
            or self._synthetic_state is None
            or self.shared_model is None
            or not self.local_qa_samples
        ):
            return None

        model = self.shared_model
        if model.condensed_encoder is not None:
            ctx_enc, ctx_proj = model.condensed_encoder, model.projector_c
        else:
            ctx_enc, ctx_proj = model.graph_encoder, model.projector
        _param = next(ctx_proj.parameters())
        device = _param.device

        lambda_gm = float(getattr(self.args, "lambda_gm", 0.1))
        lambda_align = float(getattr(self.args, "lambda_align_mem", 0.1))
        lambda_reg = float(getattr(self.args, "lambda_reg_mem", 0.01))
        lambda_syn_div = float(getattr(self.args, "lambda_div", 0.1))
        lambda_deg = float(getattr(self.args, "lambda_deg", 0.05))
        tau = float(getattr(self.args, "syn_soft_tau", 0.1))
        lr = float(getattr(self.args, "syn_mem_lr", 1e-3))
        batch_size = int(getattr(self.args, "syn_mem_batch_size", 0)) or int(
            getattr(self.args, "local_batch_size", 4)
        )

        mem = LocalSyntheticMemory.from_broadcast(self._synthetic_state, device)
        optimizer = torch.optim.Adam(mem.parameters(), lr=lr)
        gm_params = [
            p
            for p in (*ctx_enc.parameters(), *ctx_proj.parameters())
            if p.requires_grad
        ]

        # Client-condensed projections Z_m for L_align — W is fixed during
        # adaptation, so compute once and detach (never uploaded).
        z_client = None
        if self.condensed_graph is not None and lambda_align > 0:
            cg = self.condensed_graph.to(device)
            with torch.no_grad():
                ew = getattr(cg, "edge_weight", None)
                if ew is not None and cg.edge_index.numel() > 0:
                    z_client = encode_nodes_with_edge_weight(
                        cg.x.float(), cg.edge_index, ew.float(), ctx_enc, ctx_proj
                    ).detach()
                else:
                    z_client = encode_nodes(
                        cg.x.float(), cg.edge_index, ctx_enc, ctx_proj
                    ).detach()

        qa_total = 0.0
        for _ in range(k_mem):
            mini = random.sample(
                self.local_qa_samples, min(batch_size, len(self.local_qa_samples))
            )
            mini = self._attach_evidence_graphs(mini)
            batch = collate_fn(mini)

            # Retrieval query z̄_e — pooled evidence prompt rep (paper B.5.2)
            with torch.no_grad():
                z_query = model._encode_one_graph(
                    batch["graph"], model.graph_encoder, model.projector
                ).float()

            # Differentiable soft synthetic retrieval → context token z_c
            z_syn, adj_soft = mem.encode(ctx_enc, ctx_proj)
            z_c = mem.soft_context(z_query, z_syn, tau=tau)  # [B, H] fp32, grad→Θ_syn

            batch_syn = dict(batch)
            batch_syn["z_c_soft"] = z_c
            loss_qa = model(batch_syn)
            qa_total += float(loss_qa.detach().cpu())

            # Exact ∇_Θ L_QA via the context-token surrogate: Θ enters the loss
            # only through z_c, so re-contracting the detached token gradient
            # with z_c reproduces the chain rule without retaining the LLM graph.
            grad_zc = torch.autograd.grad(loss_qa, z_c, retain_graph=False)[0]
            qa_term = (grad_zc.detach() * z_c).sum()

            loss_gm = z_c.new_zeros(())
            if lambda_gm > 0:
                # g_syn: context-branch prompt-module gradient under synthetic
                # context, kept differentiable w.r.t. Θ_syn (first-order approx:
                # the LLM-side token gradient is detached).
                g_syn = torch.autograd.grad(
                    qa_term, gm_params, create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                # g_loc: same gradient under private local evidence context.
                z_c_loc = model._encode_one_graph(batch["graph"], ctx_enc, ctx_proj)
                batch_loc = dict(batch)
                batch_loc["z_c_soft"] = z_c_loc
                loss_loc = model(batch_loc)
                grad_zc_loc = torch.autograd.grad(loss_loc, z_c_loc, retain_graph=False)[0]
                loc_term = (grad_zc_loc.detach() * z_c_loc).sum()
                g_loc = torch.autograd.grad(loc_term, gm_params, allow_unused=True)
                loss_gm = gradient_matching_loss(list(g_loc), list(g_syn)).to(z_c.device)

            loss_align = (
                mem.alignment_loss(z_client, z_syn)
                if z_client is not None
                else z_c.new_zeros(())
            )
            loss_reg = mem.regularization(
                z_syn, adj_soft, lambda_syn_div=lambda_syn_div, lambda_deg=lambda_deg
            )

            total = (
                qa_term
                + lambda_gm * loss_gm
                + lambda_align * loss_align
                + lambda_reg * loss_reg
            )
            optimizer.zero_grad()
            total.backward()
            clip_grad_norm_(list(mem.parameters()), 1.0)
            optimizer.step()

        # Adaptation deposits grads on prompt-module params — clear them so
        # the next local_train round starts clean.
        model.zero_grad(set_to_none=True)

        self._syn_delta = mem.delta(self._synthetic_state)
        return qa_total / max(k_mem, 1)

    def _prepare_soft_syn_context(self) -> "dict | None":
        """Fixed Θ_syn pieces for soft synthetic retrieval during prompt tuning.

        Per Algorithm 1 (line 19) the synthetic memory stays frozen at its
        broadcast value Θ_syn^(r,0) while W is tuned — only the prompt module
        receives gradient through z_c. Edges come from the (fixed) PGE, so
        they are computed once per round.
        """
        if self._synthetic_state is None or self.shared_model is None:
            return None
        from fedcond_grag.client.stage_e_memory import LocalSyntheticMemory

        proj = (
            self.shared_model.projector_c
            if self.shared_model.projector_c is not None
            else self.shared_model.projector
        )
        device = next(proj.parameters()).device
        mem = LocalSyntheticMemory.from_broadcast(self._synthetic_state, device)
        with torch.no_grad():
            _, edge_index, edge_weight = mem.soft_edges()
        return {
            "x": mem.x.detach(),
            "edge_index": edge_index.detach(),
            "edge_weight": edge_weight.detach(),
        }

    def _soft_syn_batch_context(self, batch: dict, soft_syn: dict) -> torch.Tensor:
        """Differentiable soft synthetic retrieval (paper B.5.2).

        z_c = Σ_i α_i(q)·z_i^syn with α = softmax(cos(z̄_e, z^syn)/τ), where
        z̄_e is the pooled evidence prompt representation. z^syn is encoded by
        the trainable context branch, so W is optimized through the synthetic
        context exactly as in Eq. (13).
        """
        import torch.nn.functional as F

        from fedcond_grag.server.stage_c_aggregate.repr_align import (
            encode_nodes,
            encode_nodes_with_edge_weight,
        )

        model = self.shared_model
        if model.condensed_encoder is not None:
            ctx_enc, ctx_proj = model.condensed_encoder, model.projector_c
        else:
            ctx_enc, ctx_proj = model.graph_encoder, model.projector
        with torch.no_grad():
            z_query = model._encode_one_graph(
                batch["graph"], model.graph_encoder, model.projector
            ).float()
        if soft_syn["edge_index"].numel() > 0:
            z_syn = encode_nodes_with_edge_weight(
                soft_syn["x"], soft_syn["edge_index"], soft_syn["edge_weight"],
                ctx_enc, ctx_proj,
            )
        else:
            z_syn = encode_nodes(
                soft_syn["x"], soft_syn["edge_index"], ctx_enc, ctx_proj
            )
        tau = float(getattr(self.args, "syn_soft_tau", 0.1))
        q = F.normalize(z_query, dim=-1)
        z = F.normalize(z_syn.float(), dim=-1)
        alpha = torch.softmax(q @ z.T / max(tau, 1e-6), dim=-1)
        return alpha @ z_syn.float()

    def send_message(self) -> None:
        mode = str(getattr(self.args, "server_stage_c_mode", ""))
        # Paper Phase 0: G̃_m is uploaded exactly once during initialization
        # (fedrag). Legacy server-side modes still consume anchors every round.
        send_anchor = mode != "fedrag" or not self._anchor_uploaded
        msg: dict = {}
        if send_anchor:
            if self.condensed_graph is None:
                self.condensed_graph = self._condense_anchor_graph(self.tri_graph)
            msg["anchor_graph"] = self.condensed_graph
            msg["num_anchor_nodes"] = int(self.condensed_graph.x.size(0))
            if mode == "fedrag":
                self._anchor_uploaded = True
        if self._model_weights is not None and self._num_local_samples > 0:
            msg["model_weights"] = self._model_weights
            msg["num_samples"] = self._num_local_samples
        if self._syn_delta is not None:
            msg["syn_delta"] = self._syn_delta
            msg.setdefault("num_samples", self._num_local_samples)
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
        if "lora" in self._model_weights:
            self.shared_model.model.load_state_dict(self._model_weights["lora"], strict=False)

    def _attach_evidence_graphs(self, samples: list) -> list:
        """Build per-sample evidence subgraphs from per-query PPR anchor nodes.

        anchor_passage_nodes (from passage_node_map.pt) are the PPR-selected
        passage trigraph nodes for this query. We filter to the local client's
        nodes, use them as 1-hop expansion seeds for the evidence graph, and
        cosine-rerank them to produce the LLM text desc.

        Raises RuntimeError if passage_node_map.pt was not generated first.
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
        top_k_desc = max(1, int(getattr(self.args, "top_r_anchor", None) or 5))

        if self._ppr_node_map is None:
            raise RuntimeError(
                f"[client_{self.client_id}] ppr_node_map.pt not found. "
                "Run scripts/preprocess_fedcond_qa.py --dataset <dataset> first."
            )

        # Resolve per-sample PPR anchor nodes from this client's map.
        per_sample_local_anchors: list[list[int]] = []
        for s in samples:
            idx = s.get("idx")
            if idx is None or idx >= self._ppr_node_map.shape[0]:
                raise RuntimeError(
                    f"[client_{self.client_id}] Sample '{s.get('id')}' has no "
                    f"valid dataset index (idx={idx})."
                )
            row = self._ppr_node_map[idx]                          # [top_k]
            local = [int(n) for n in row.tolist() if n >= 0 and n < N]
            if not local:
                raise RuntimeError(
                    f"[client_{self.client_id}] No PPR anchor nodes for sample "
                    f"idx={idx} ('{s.get('id')}'). "
                    "Run scripts/preprocess_fedcond_qa.py again."
                )
            per_sample_local_anchors.append(local)

        # CPU subgraph extraction + cosine-reranked desc
        out = []
        for s, local_anchors in zip(samples, per_sample_local_anchors):
            s = dict(s)
            seed_set = set(local_anchors)

            # 1-hop expansion
            kept_set = set(seed_set)
            for seed in seed_set:
                kept_set.update(adj[seed])
            kept_list = sorted(kept_set)

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
            s["graph"] = graph
            s["evidence_graph"] = graph

            # desc = text of this client's own PPR-retrieved passage nodes, in
            # PPR rank order. local_anchors are trigraph node ids for
            # node_type==2 (Passage) nodes selected by EvidenceLinearRAG at
            # preprocess time (scripts/preprocess_fedcond_qa.py), so this is
            # the actual local retrieval result — not gold/oracle evidence.
            if node_text is not None:
                desc_texts = [
                    node_text[i] for i in local_anchors[:top_k_desc] if i < len(node_text)
                ]
                if desc_texts:
                    s["desc"] = "\n\n".join(desc_texts)

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
        condensed = self._maybe_refine_condensed(condensed, tri_graph=graph)
        condensed.y = condensed.node_type.long()
        condensed.num_global_classes = 3
        return condensed

    def _maybe_refine_condensed(self, condensed: Data, tri_graph=None) -> Data:
        """Stage B refinement (paper B.3.5): minimize the condensation objective

            L_cond = L_ret + λ_rep·L_rep + λ_div·L_div

        over the condensed node features before uploading G̃_m. L_ret is the
        KL divergence between the full-graph and (soft-coverage-lifted)
        condensed passage-retrieval distributions under the client's own local
        training questions; without local questions L_ret is skipped and only
        L_rep + L_div refine the features. Runs once per client lifetime.
        """
        iters = int(getattr(self.args, "condense_refine_iters", 100))
        if iters <= 0 or self._condense_refined:
            return condensed
        from fedcond_grag.client.stage_b_condense import (
            RetrievalRefineConfig,
            refine_condensed_graph,
        )

        cap = int(getattr(self.args, "stage_b_max_queries", 256))
        q_emb = self._local_query_embeddings(cap)
        cfg = RetrievalRefineConfig(
            iterations=iters,
            lr=float(getattr(self.args, "condense_refine_lr", 1e-2)),
            lambda_rep=float(getattr(self.args, "stage_b_lambda_rep", 1.0)),
            lambda_div=float(getattr(self.args, "stage_b_lambda_div", 0.1)),
            delta_margin=float(getattr(self.args, "stage_b_div_margin", 0.5)),
            tau_ret=float(getattr(self.args, "stage_b_tau_ret", 0.1)),
            tau_cov=float(getattr(self.args, "stage_b_tau_cov", 0.1)),
            max_queries=cap,
            knn_k=int(getattr(self.args, "stage_b_knn_k", 8)),
            preserve_sep_topology=_as_bool(
                getattr(self.args, "preserve_sep_topology", True)
            ),
        )
        refined, hist = refine_condensed_graph(
            tri_graph if tri_graph is not None else self.tri_graph,
            condensed,
            query_embeddings=q_emb,
            config=cfg,
        )
        refined.y = refined.node_type.long()
        refined.num_global_classes = 3
        self._condense_refined = True
        self.last_stage_b_refine = {
            "l_cond_start": hist["total"][0],
            "l_cond_end": hist["total"][-1],
            "l_ret_start": hist["ret"][0],
            "l_ret_end": hist["ret"][-1],
        }
        n_q = 0 if q_emb is None else min(int(q_emb.size(0)), cap)
        ret_note = "" if n_q > 0 else ", L_ret skipped (no local queries)"
        print(
            f"    [client_{self.client_id}] Stage B refine (B.3.5): "
            f"L_cond {hist['total'][0]:.4f} → {hist['total'][-1]:.4f} "
            f"({iters} iters, {n_q} queries{ret_note})",
            flush=True,
        )
        return refined

    def _local_query_embeddings(self, cap: int) -> "torch.Tensor | None":
        """Embed local training questions with the frozen sentence encoder.

        Same encoder/space as the node features, so cosine retrieval
        distributions in L_ret are well-defined. Queries never leave the
        client — they only shape the refinement of G̃_m locally.
        """
        samples = self._train_pool or self.local_qa_samples
        questions = [str(s.get("question", "")).strip() for s in samples]
        questions = [q for q in questions if q][: max(cap, 0)]
        if not questions:
            return None
        from fedcond_grag.client.stage_b_condense.node_text_embedder import (
            encode_texts,
            load_frozen_encoder,
        )

        encoder = load_frozen_encoder("all-MiniLM-L6-v2", dim=384)
        return encode_texts(encoder, questions, device=self.device)

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
