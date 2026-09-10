"""Unified federated round loop for FedCondGraphRAG.

Round 0  — bootstrap: Stage B (client condense) + Stage C (server gradient
           matching) to produce the initial synthetic global graph.

Round >= 1 — FL loop: each sampled client loads aggregated model weights,
             trains locally on its QA partition (Stage D), then sends updated
             GNN + projector weights back.  Server FedAvg-aggregates the
             weights and re-runs Stage C with the updated surrogate to refine
             the synthetic graph.

Per-round metrics logged: avg_loss, per-client losses, train_acc, val_acc,
test_acc. Written to /tmp/fl_metrics.jsonl for offline analysis.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch

from fedcond_grag.client.client import FedCondQAClient
from fedcond_grag.constants import ENCODER_DIM
from fedcond_grag.server.lora_aggregate import has_lora, lora_state_dict
from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.utils.comm_cost import message_bytes
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1


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

        # Stage D fields
        self.shared_model = None
        self._train_eval_samples: list = []
        self._val_samples: list = []
        self._test_samples: list = []
        self._eval_only = bool(getattr(args, "eval_only", False))
        if int(getattr(args, "num_rounds", 1)) > 1 or self._eval_only:
            self._init_stage_d()

        load_ckpt = getattr(args, "load_checkpoint", None)
        if load_ckpt:
            self._load_checkpoint(load_ckpt)

        self._metrics_path = Path(getattr(args, "metrics_path", "/tmp/fl_metrics.jsonl"))
        self._metrics_path.write_text("")   # reset on new run

        self._save_best_path = getattr(args, "save_best_path", None)
        self._best_val_acc = float("-inf")

        self._load_dotenv()
        self._wandb = None
        # Always attempt to log — don't gate on WANDB_API_KEY being explicitly
        # set, since a prior `wandb login` (netrc/config-based auth) never
        # sets that env var. Set WANDB_MODE=disabled/offline to opt out.
        try:
            import wandb
            # Auto-generate a descriptive run name from dual_graph_mode when
            # no explicit name is given, so ablations are easy to filter on WandB.
            _mode = getattr(args, "dual_graph_mode", "shared")
            _auto_names = {
                "both": "dual-encoder",
                "dual": "dual-encoder",
                "shared": "ablation-shared-encoder",
                "no_synthetic": "ablation-no-synthetic",
                "evidence_only": "ablation-evidence-only",
                "condensed_only": "ablation-condensed-only",
                "none": "ablation-text-only",
                "text_only": "ablation-text-only",
            }
            _run_name = (getattr(args, "wandb_run_name", None)
                         or _auto_names.get(_mode, f"mode-{_mode}"))
            _tags = list(getattr(args, "wandb_tags", None) or [])
            if _mode not in _tags:
                _tags.append(_mode)
            _tags.append("fl-train")
            self._wandb = wandb.init(
                project=os.environ.get("WANDB_PROJECT", getattr(args, "wandb_project", "fedcond-graphrag")),
                name=_run_name,
                group=getattr(args, "wandb_group", None),
                job_type=getattr(args, "wandb_job_type", None),
                tags=_tags,
                config=vars(args) if hasattr(args, "__dict__") else {},
                resume="allow",
            )
            # Define two independent x-axes so round-level and step-level
            # charts never share the same counter and WandB never drops data.
            wandb.define_metric("comm_round")
            wandb.define_metric("round/*", step_metric="comm_round")
            wandb.define_metric("global_step")
            wandb.define_metric("step/*", step_metric="global_step")
            print(f"[wandb] run: {self._wandb.url}", flush=True)
        except Exception as exc:
            print(f"[wandb] init failed, continuing without: {exc}", flush=True)

        self._init_experiment_dir()

    # ------------------------------------------------------------------
    # Regularization-diversity ablation instrumentation (opt-in via
    # --experiment-dir). See fedcond_grag/analysis/diversity.py for the
    # metrics and docs on "Effect of Regularization on Node Diversity".
    # ------------------------------------------------------------------

    def _init_experiment_dir(self) -> None:
        self._exp_paths: dict | None = None
        self._ckpt_index = 0
        self._last_diversity: dict | None = None
        exp_dir = getattr(self.args, "experiment_dir", None)
        if not exp_dir:
            return

        variant = getattr(self.args, "variant_name", None) or "run"
        base = Path(exp_dir) / variant
        ckpt_dir = base / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        diversity_jsonl = base / "diversity_checkpoints.jsonl"
        diversity_jsonl.write_text("")
        self._exp_paths = {
            "base": base,
            "checkpoints": ckpt_dir,
            "diversity_jsonl": diversity_jsonl,
            "qa_summary": base / "qa_summary.json",
        }
        self._syn_snapshot_every = int(getattr(self.args, "syn_snapshot_every", 10))
        self.server.phase1_callback = self._on_phase1_step

        if self._wandb is not None:
            import wandb
            wandb.define_metric("phase1/server_update")
            wandb.define_metric("phase1/*", step_metric="phase1/server_update")
            wandb.define_metric("phase2/global_round")
            wandb.define_metric("phase2/*", step_metric="phase2/global_round")
            wandb.define_metric("checkpoint/index")
            wandb.define_metric("checkpoint/*", step_metric="checkpoint/index")
            wandb.define_metric("diversity/*", step_metric="checkpoint/index")
            wandb.define_metric("qa/*", step_metric="checkpoint/index")
        print(f"[experiment] regularization-diversity instrumentation active -> {base}", flush=True)

    def _on_phase1_step(self, server: "FedCondQAServer", step_idx: int, num_steps: int) -> None:
        """Called by FedCondQAServer.execute() after every Phase-I (Phase-0
        bootstrap) server update, step_idx==0 meaning 'before update 1'."""
        c = server.last_phase1_components
        if self._wandb is not None and c is not None:
            self._wandb.log({
                "phase1/server_update": step_idx,
                "phase1/loss_total": c["total"],
                "phase1/loss_align": c["align"],
                "phase1/loss_reg": c["div"] + c["deg"],
                "phase1/loss_div": c["div"],
                "phase1/loss_degree": c["deg"],
            })
        every = getattr(self, "_syn_snapshot_every", 10)
        is_snapshot = step_idx == 0 or step_idx == num_steps or (every > 0 and step_idx % every == 0)
        if is_snapshot and self._exp_paths is not None:
            self._save_synthetic_checkpoint(server, phase="phase1", server_update=step_idx, round_id=0)

    def _gnn_c_config(self) -> dict:
        args = self.args
        return {
            "gnn_model_name_c": getattr(args, "gnn_model_name_c", getattr(args, "gnn_model_name", "gcn")),
            "gnn_num_layers_c": getattr(args, "gnn_num_layers_c", None) or getattr(args, "gnn_num_layers", 2),
            "gnn_num_heads_c": getattr(args, "gnn_num_heads_c", None) or getattr(args, "gnn_num_heads", 4),
            "gnn_hidden_dim_c": getattr(args, "gnn_hidden_dim_c", None) or getattr(args, "gnn_hidden_dim", 1024),
            "gnn_in_dim_c": getattr(args, "gnn_in_dim_c", None) or getattr(args, "gnn_in_dim", 1024),
            "gnn_dropout": float(getattr(args, "gnn_dropout", 0.0)),
            "repr_proj_out_dim": int(getattr(args, "repr_proj_out_dim", 4096)),
        }

    def _save_synthetic_checkpoint(
        self, server: "FedCondQAServer", phase: str, server_update: int | None, round_id: int,
    ) -> None:
        """Dump {X_syn, A_syn, node_type, matching repr_encoder/repr_projector
        state} plus X_syn/H_syn diversity diagnostics -- local .pt + JSONL,
        and WandB under the checkpoint/index step axis. No subsampling."""
        from fedcond_grag.analysis.diversity import compute_diversity_metrics
        from fedcond_grag.server.stage_c_aggregate.repr_align import encode_nodes_with_edge_weight

        if server.synthetic_x is None or server.pge is None:
            return

        with torch.no_grad():
            x_syn = server.synthetic_x.detach().clone()
            node_type = server.synthetic_node_type.detach().clone()
            adj = server.pge.inference(server.synthetic_x, server.synthetic_node_type)
            rows, cols = (adj > 0).nonzero(as_tuple=True)
            edge_index = torch.stack([rows, cols], dim=0).long()
            edge_weight = adj[rows, cols].detach().clone()
            server.repr_encoder.eval()
            server.repr_projector.eval()
            h_syn = encode_nodes_with_edge_weight(
                x_syn, edge_index, edge_weight, server.repr_encoder, server.repr_projector,
            )

        metrics_x = compute_diversity_metrics(x_syn)
        metrics_h = compute_diversity_metrics(h_syn)
        idx = self._ckpt_index
        self._ckpt_index += 1

        payload = {
            "checkpoint_index": idx,
            "phase": phase,
            "round": round_id,
            "server_update": server_update,
            "x": x_syn.cpu(),
            "edge_index": edge_index.cpu(),
            "edge_weight": edge_weight.cpu(),
            "node_type": node_type.cpu(),
            "target_degree": float(server._target_anchor_degree),
            "gnn_config": self._gnn_c_config(),
            "encoder_state": {
                "repr_encoder": {k: v.detach().cpu().clone() for k, v in server.repr_encoder.state_dict().items()},
                "repr_projector": {k: v.detach().cpu().clone() for k, v in server.repr_projector.state_dict().items()},
            },
            # Full Theta_syn={X_syn, theta_PGE} in the exact schema
            # load_synthetic_memory_state_dict() expects, so this checkpoint
            # can be spliced into a --load-checkpoint payload and broadcast
            # bit-for-bit (not just diagnosed) -- used to evaluate QA F1 on a
            # held-out split with THIS arm's evolved synthetic memory.
            "synthetic_memory_state": server.synthetic_memory_state_dict(),
            "diversity": {"X_syn": metrics_x, "H_syn": metrics_h},
        }
        ckpt_path = self._exp_paths["checkpoints"] / f"ckpt_{idx:04d}.pt"
        torch.save(payload, ckpt_path)

        variant = getattr(self.args, "variant_name", None) or "run"
        seed = int(getattr(self.args, "seed", 0))
        dataset = getattr(self.args, "dataset", None)
        with self._exp_paths["diversity_jsonl"].open("a") as f:
            for rep_name, m in (("X_syn", metrics_x), ("H_syn", metrics_h)):
                f.write(json.dumps({
                    "variant": variant, "seed": seed, "dataset": dataset,
                    "checkpoint_index": idx, "phase": phase, "round": round_id,
                    "server_update": server_update if server_update is not None else -1,
                    "representation": rep_name, **m,
                }) + "\n")

        self._last_diversity = {
            "checkpoint_index": idx, "phase": phase, "round": round_id,
            "server_update": server_update, "X_syn": metrics_x, "H_syn": metrics_h,
        }

        if self._wandb is not None:
            self._wandb.log({
                "checkpoint/index": idx,
                "checkpoint/phase": phase,
                "checkpoint/server_update": server_update if server_update is not None else -1,
                "checkpoint/global_round": round_id if phase == "phase2" else -1,
                "diversity/X_syn/mean_offdiag_cosine": metrics_x["mean_offdiag_cosine"],
                "diversity/X_syn/effective_rank": metrics_x["effective_rank"],
                "diversity/X_syn/num_nodes": metrics_x["num_nodes"],
                "diversity/X_syn/num_valid_rows": metrics_x["num_valid_rows"],
                "diversity/X_syn/num_near_zero_rows": metrics_x["num_near_zero_rows"],
                "diversity/H_syn/mean_offdiag_cosine": metrics_h["mean_offdiag_cosine"],
                "diversity/H_syn/effective_rank": metrics_h["effective_rank"],
                "diversity/H_syn/num_nodes": metrics_h["num_nodes"],
                "diversity/H_syn/num_valid_rows": metrics_h["num_valid_rows"],
                "diversity/H_syn/num_near_zero_rows": metrics_h["num_near_zero_rows"],
            })
        print(f"    [checkpoint {idx}] {phase} round={round_id} update={server_update} "
              f"X_syn(cos={metrics_x['mean_offdiag_cosine']:.4f}, r_eff={metrics_x['effective_rank']:.4f}) "
              f"H_syn(cos={metrics_h['mean_offdiag_cosine']:.4f}, r_eff={metrics_h['effective_rank']:.4f})",
              flush=True)

    def _log_phase2_server_reg(self, round_id: int) -> None:
        c = self.server.last_phase2_components
        if self._wandb is None or c is None:
            return
        self._wandb.log({
            "phase2/global_round": round_id,
            "phase2/server_reg_loss": c["total"],
            "phase2/server_div_loss": c["div"],
            "phase2/server_degree_loss": c["deg"],
            "phase2/server_reg_steps": c["reg_steps"],
            "phase2/server_reg_enabled": float(c["enabled"]),
        })

    def _write_qa_summary(self, test_metrics: dict | None, n_eval: int) -> None:
        if self._exp_paths is None or test_metrics is None:
            return
        payload = {
            "variant": getattr(self.args, "variant_name", None) or "run",
            "seed": int(getattr(self.args, "seed", 0)),
            "dataset": getattr(self.args, "dataset", None),
            "f1": test_metrics.get("f1"),
            "em": test_metrics.get("em"),
            "hit": test_metrics.get("hit"),
            "num_eval_questions": n_eval,
        }
        self._exp_paths["qa_summary"].write_text(json.dumps(payload, indent=2))
        if self._wandb is not None:
            self._wandb.log({
                "qa/cross_client_f1": payload["f1"],
                "qa/cross_client_em": payload["em"],
                "qa/cross_client_accuracy": payload["hit"],
                "qa/num_eval_questions": payload["num_eval_questions"],
            })
            self._wandb.summary["final/qa_f1"] = payload["f1"]

    def _finalize_experiment_summary(self) -> None:
        if self._wandb is None or self._last_diversity is None:
            return
        d = self._last_diversity
        self._wandb.summary["final/X_syn_mean_offdiag_cosine"] = d["X_syn"]["mean_offdiag_cosine"]
        self._wandb.summary["final/X_syn_effective_rank"] = d["X_syn"]["effective_rank"]
        self._wandb.summary["final/X_syn_near_zero_rows"] = d["X_syn"]["num_near_zero_rows"]
        self._wandb.summary["final/H_syn_mean_offdiag_cosine"] = d["H_syn"]["mean_offdiag_cosine"]
        self._wandb.summary["final/H_syn_effective_rank"] = d["H_syn"]["effective_rank"]
        self._wandb.summary["final/H_syn_near_zero_rows"] = d["H_syn"]["num_near_zero_rows"]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self._eval_only:
            self._run_eval_only()
            return

        num_rounds = int(getattr(self.args, "num_rounds", 1))
        client_frac = float(getattr(self.args, "client_frac", 1.0))
        round_metrics: list[dict] = []

        # Global step counter — monotone across all clients and rounds so the
        # WandB x-axis is continuous and steps never go backwards.
        global_step = 0

        def _step_log(kv: dict, step: int) -> None:
            if self._wandb is not None:
                # Remap keys to step/* namespace and attach global_step x-axis
                self._wandb.log(
                    {"global_step": step, **{k.replace("train/", "step/"): v for k, v in kv.items()}},
                )

        for round_id in range(num_rounds):
            t_round_start = time.perf_counter()
            sampled = sorted(random.sample(
                range(self.args.num_clients),
                max(1, int(self.args.num_clients * client_frac)),
            ))
            print(f"\n=== round {round_id} | sampled clients: {sampled} ===", flush=True)

            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled

            self.server.send_message()
            mem_bytes, tot_bytes = message_bytes(self.message_pool.get("server", {}))
            comm_memory_bytes = mem_bytes
            comm_total_bytes = tot_bytes

            client_losses: dict[int, float] = {}
            client_times: dict[int, float] = {}
            client_syn_mem: dict[int, float] = {}
            stage_b_refine: dict[int, dict] = {}
            for cid in sampled:
                t0 = time.perf_counter()
                self.clients[cid].receive_message()
                self.clients[cid].execute()

                loss = 0.0
                if round_id >= 1 and self.shared_model is not None:
                    # Resample each round so the full pool is covered over many rounds
                    if self._max_train_per_client > 0:
                        self.clients[cid].sample_train_for_round(self._max_train_per_client)
                    loss, steps = self.clients[cid].local_train(
                        log_fn=_step_log, global_step_start=global_step
                    )
                    global_step += steps
                    client_losses[cid] = loss

                    # FedRAG Phase 1: query-conditioned synthetic-memory
                    # adaptation — client refines a local copy of Θ_syn and
                    # uploads only the delta.
                    if str(getattr(self.args, "server_stage_c_mode", "")) == "fedrag":
                        mem_loss = self.clients[cid].adapt_synthetic_memory()
                        if mem_loss is not None:
                            client_syn_mem[cid] = mem_loss
                            print(
                                f"    [client_{cid}] syn-mem adapted | "
                                f"qa(syn) {mem_loss:.4f}",
                                flush=True,
                            )

                # Stage B refinement stats (Phase 0) — consumed once per client
                refine_stats = getattr(self.clients[cid], "last_stage_b_refine", None)
                if refine_stats:
                    stage_b_refine[cid] = refine_stats
                    self.clients[cid].last_stage_b_refine = None

                self.clients[cid].send_message()
                mem_bytes, tot_bytes = message_bytes(self.message_pool.get(f"client_{cid}", {}))
                comm_memory_bytes += mem_bytes
                comm_total_bytes += tot_bytes
                client_times[cid] = time.perf_counter() - t0
                loss_str = f" | loss: {loss:.4f}" if round_id >= 1 else ""
                print(f"    client_{cid}{loss_str} | {client_times[cid]:.2f}s", flush=True)

            t0 = time.perf_counter()
            self.server.execute()
            agg_time = time.perf_counter() - t0
            print(f"    server agg done in {agg_time:.2f}s", flush=True)

            if round_id == 0:
                self._dump_synthetic_snapshot("synthetic_init.pt")
            if round_id == num_rounds - 1:
                self._dump_synthetic_snapshot("synthetic_final.pt")

            # Regularization-diversity ablation (paper Phase II = rounds >= 1):
            # snapshot X_syn/H_syn after this round's aggregation + (optional)
            # server L_reg refinement, opt-in via --experiment-dir.
            if self._exp_paths is not None and round_id >= 1:
                self._log_phase2_server_reg(round_id)
                self._save_synthetic_checkpoint(self.server, phase="phase2", server_update=None, round_id=round_id)
            elif self._exp_paths is not None and round_id == 0:
                # Reference point: X_syn is still exactly Phase-I's final value
                # (no Phase-II adaptation has run yet), but repr_encoder has
                # just been FedAvg-swapped from random to the real trained
                # condensed_encoder inside this same server.execute() call --
                # isolates "what the encoder swap alone does to H_syn" from
                # "what Phase-II synthetic-memory adaptation does on top of it".
                self._save_synthetic_checkpoint(self.server, phase="phase2", server_update=None, round_id=0)

            # self.shared_model still holds whichever sampled client's LOCAL
            # (non-aggregated) weights local_train() last wrote into it --
            # server.execute() only updated self.server.global_model_state (a
            # CPU dict), it never touches shared_model. Without this, do_eval
            # below (and any --save-best decision it drives) would silently
            # score one arbitrary client's local model instead of the actual
            # federated aggregate -- the same aggregate --load-checkpoint
            # restores later, so a checkpoint's saved metrics would otherwise
            # not describe the weights actually saved in it.
            self._load_aggregated_weights_into_model()

            train_acc = val_acc = test_acc = None
            val_metrics = test_metrics = None
            eval_time = None
            eval_every = int(getattr(self.args, "eval_every", 1))
            # Round 0 eval gives the Phase-0 baseline point (untrained prompt
            # module) so wandb charts include the initialization round.
            do_eval = self.shared_model is not None and (round_id % eval_every == 0)
            if do_eval:
                t_eval = time.perf_counter()
                if self._val_samples:
                    val_metrics = self._eval_split_acc(self._val_samples)
                    val_acc = val_metrics["hit"]
                    print(f"    val   : hit {val_metrics['hit']:.2f}% | "
                          f"EM {val_metrics['em']:.2f}% | F1 {val_metrics['f1']:.2f}", flush=True)
                if self._test_samples:
                    test_metrics = self._eval_split_acc(self._test_samples)
                    test_acc = test_metrics["hit"]
                    print(f"    test  : hit {test_metrics['hit']:.2f}% | "
                          f"EM {test_metrics['em']:.2f}% | F1 {test_metrics['f1']:.2f}", flush=True)
                eval_time = time.perf_counter() - t_eval

                if (self._save_best_path and val_acc is not None
                        and val_acc > self._best_val_acc):
                    self._best_val_acc = val_acc
                    self._save_checkpoint(round_id, val_metrics, test_metrics)
                print(f"    eval done in {eval_time:.2f}s", flush=True)

            round_time = time.perf_counter() - t_round_start
            avg_loss = (
                sum(client_losses.values()) / len(client_losses) if client_losses else None
            )
            server_syn_loss = getattr(self.server, "train_loss_match", None)
            metrics = {
                "round": round_id,
                "avg_loss": avg_loss,
                "client_losses": dict(client_losses),
                "client_syn_mem": dict(client_syn_mem),
                "stage_b_refine": dict(stage_b_refine),
                "server_syn_loss": server_syn_loss,
                "client_times": dict(client_times),
                "round_time": round_time,
                "agg_time": agg_time,
                "eval_time": eval_time,
                "comm_memory_mb": comm_memory_bytes / (1024 * 1024),
                "comm_total_mb": comm_total_bytes / (1024 * 1024),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "val_em": val_metrics["em"] if val_metrics else None,
                "val_f1": val_metrics["f1"] if val_metrics else None,
                "test_em": test_metrics["em"] if test_metrics else None,
                "test_f1": test_metrics["f1"] if test_metrics else None,
            }
            round_metrics.append(metrics)
            self._log_metrics(metrics, global_step=global_step)
            if self._exp_paths is not None and test_metrics is not None:
                self._write_qa_summary(test_metrics, len(self._test_samples))

        self._finalize_experiment_summary()
        self._print_metrics_table(round_metrics)

    def _load_aggregated_weights_into_model(self) -> None:
        """Load the server's current FedAvg aggregate into self.shared_model.

        Mirrors FedCondQAClient._load_weights_into_model() exactly, but reads
        directly from self.server.global_model_state instead of a client's
        broadcast copy of it. self.shared_model is one object shared across
        every client in this single-process simulator, so after the per-
        client local_train() loop it still holds whichever sampled client
        trained last -- this call is what makes it hold the actual federated
        aggregate instead, immediately before evaluation.
        """
        state = getattr(self.server, "global_model_state", None)
        if not state or self.shared_model is None:
            return
        if "graph_encoder" in state:
            self.shared_model.graph_encoder.load_state_dict(state["graph_encoder"])
        if "projector" in state:
            self.shared_model.projector.load_state_dict(state["projector"])
        if self.shared_model.condensed_encoder is not None and "condensed_encoder" in state:
            self.shared_model.condensed_encoder.load_state_dict(state["condensed_encoder"])
        if self.shared_model.projector_c is not None and "projector_c" in state:
            self.shared_model.projector_c.load_state_dict(state["projector_c"])
        if "lora" in state:
            self.shared_model.model.load_state_dict(state["lora"], strict=False)

    def _run_eval_only(self) -> None:
        """--eval-only: no training rounds, no FedAvg of trained weights --
        just run the already-loaded (optionally checkpoint-restored)
        shared_model against --dataset's test split once. Intended for
        evaluating a checkpoint trained on a separate <dataset>_train
        pseudo-dataset against this (untouched) dataset's full question set
        -- pair with --qa-test-only at preprocess time so test_idx covers
        all of it, and --max-eval-samples set above the question count so
        nothing gets capped.

        If --load-checkpoint restored a trained synthetic memory (a
        checkpoint saved after this fix, with a "synthetic_memory" key), that
        Theta_syn is broadcast as-is -- the SAME X_syn/theta_PGE the loaded
        condensed_encoder/projector_c were trained/evaluated against.

        Otherwise (older checkpoints with no saved synthetic memory) this
        falls back to a round-0-equivalent Stage B/C bootstrap (client
        condense + server synthetic-graph construction) built from THIS
        dataset's own corpus -- note this is NOT the same Theta_syn the
        checkpoint was originally trained/evaluated against, only a freshly
        re-initialized stand-in (skipping it entirely would leave
        condensed_encoder/projector_c operating on an empty/never-built
        synthetic graph, corrupting the dual-graph prompt for any
        --dual-graph-mode that uses it -- the default, "both").
        """
        if self.shared_model is None:
            raise RuntimeError(
                "--eval-only requires Stage D to have initialized a model "
                "(check the QA cache at --qa-data-root exists and matches --dataset)."
            )
        if not self._test_samples:
            raise RuntimeError(
                "--eval-only: no test samples loaded. Build the QA cache with "
                "--qa-test-only (or check --qa-data-root points at the right cache)."
            )
        self.message_pool["round"] = 0
        sampled = list(range(self.args.num_clients))
        self.message_pool["sampled_clients"] = sampled

        if getattr(self.server, "synthetic_x", None) is not None:
            # --load-checkpoint already restored a trained Theta_syn (X_syn +
            # theta_PGE) -- broadcast it as-is. Running Phase-0 reconstruction
            # here would silently REPLACE it with a freshly re-initialized
            # synthetic memory, i.e. evaluate a different artifact than the
            # one condensed_encoder/projector_c were actually trained/saved
            # against. client.execute() still runs (builds/loads each
            # client's local condensed_graph cache, used only for L_align
            # diagnostics) but nothing is uploaded or server-aggregated.
            print("[eval-only] checkpoint contains a trained synthetic memory -- "
                  "broadcasting it directly, skipping Phase-0 reconstruction", flush=True)
            self.server.send_message()
            for cid in sampled:
                self.clients[cid].receive_message()
                self.clients[cid].execute()
        else:
            print("[eval-only] no synthetic memory in checkpoint (or none loaded) -- "
                  "running Stage B/C bootstrap for this dataset's own synthetic graph "
                  "before evaluating...", flush=True)
            self.server.send_message()
            for cid in sampled:
                self.clients[cid].receive_message()
                self.clients[cid].execute()
                self.clients[cid].send_message()
            self.server.execute()
            # server.execute() just (re)built Theta_syn and broadcast it via its
            # own trailing send_message() -- clients must receive it, or every
            # --eval-only run below silently zeroes the condensed/synthetic
            # channel (client.synthetic_graph stays None -> encode_graphs'
            # `condensed_graph is None` fallback -> z_c = 0 for the whole eval).
            for cid in sampled:
                self.clients[cid].receive_message()

        print(f"[eval-only] evaluating on {len(self._test_samples)} test samples "
              f"(--max-eval-samples={getattr(self.args, 'max_eval_samples', 200)} "
              "-- raise it if this is less than your test set size)", flush=True)
        dump_path = getattr(self.args, "dump_predictions", None)
        if dump_path:
            Path(dump_path).write_text("")   # reset on new run, like _metrics_path
        test_metrics = self._eval_split_acc(self._test_samples, dump_predictions_path=dump_path)
        print(f"    test  : hit {test_metrics['hit']:.2f}% | "
              f"EM {test_metrics['em']:.2f}% | F1 {test_metrics['f1']:.2f}", flush=True)
        self._log_metrics({
            "round": 0, "avg_loss": None, "client_losses": {}, "client_syn_mem": {},
            "stage_b_refine": {}, "server_syn_loss": None, "client_times": {},
            "round_time": None, "agg_time": None, "eval_time": None,
            "train_acc": None, "val_acc": None, "test_acc": test_metrics["hit"],
            "val_em": None, "val_f1": None,
            "test_em": test_metrics["em"], "test_f1": test_metrics["f1"],
        }, global_step=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    # Graph-side submodules FedAvg'd server-side each round (see
    # FedCondQAServer._fedavg_model_weights's own _WEIGHT_KEYS) -- saved/
    # loaded here whenever the run actually trained them (--dual-graph-mode
    # != none) regardless of whether the LLM itself is frozen.
    _GRAPH_WEIGHT_KEYS = ("graph_encoder", "projector", "condensed_encoder", "projector_c")

    def _dump_synthetic_snapshot(self, filename: str) -> None:
        """Opt-in (--synthetic-snapshot-dir) raw Theta_syn = {X_syn, theta_PGE}
        dump, independent of --save-best's val-improvement gating.

        Used to capture the synthetic memory immediately after Phase-0
        bootstrap (round 0) and immediately after the final round, for
        offline empirical privacy-leakage evaluation (scripts/eval_privacy.py)
        -- neither snapshot is otherwise ever persisted: --save-best only
        writes a checkpoint on a new best val score, which need not be round
        0 or the last round.
        """
        snap_dir = getattr(self.args, "synthetic_snapshot_dir", None)
        if not snap_dir:
            return
        try:
            state = self.server.synthetic_memory_state_dict()
        except RuntimeError:
            return
        path = Path(snap_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(state, path / filename)
        print(f"    [synthetic-snapshot] saved {filename} -> {path / filename}", flush=True)

    def _save_checkpoint(self, round_id: int, val_metrics: dict, test_metrics: dict | None) -> None:
        """Persist whatever this run actually trained: the LoRA adapter (LLM
        fine-tune, --llm-frozen False) and/or the FedAvg'd graph_encoder/
        projector/condensed_encoder/projector_c (--llm-frozen True is the
        common case -- those are the only thing training then).
        """
        model = self.shared_model
        graph_state = {
            key: state for key, state in (getattr(self.server, "global_model_state", None) or {}).items()
            if key in self._GRAPH_WEIGHT_KEYS
        }
        model_has_lora = has_lora(model.model)
        # Theta_syn = {X_syn, theta_PGE} -- the learned synthetic memory itself,
        # not just the encoder that projects it. Without this, --eval-only can
        # only ever rebuild a FRESH Phase-0-only Theta_syn on reload, which is
        # a different (less-trained) artifact than what condensed_encoder/
        # projector_c were actually evaluated against here (see the round-1
        # em/f1 gap this was added to close).
        synthetic_state = None
        if getattr(self.server, "synthetic_x", None) is not None:
            synthetic_state = self.server.synthetic_memory_state_dict()
        if not model_has_lora and not graph_state and synthetic_state is None:
            print("    [checkpoint] skipped -- no LoRA adapter, no trained graph "
                  "weights, and no synthetic memory (--dual-graph-mode none with "
                  "--llm-frozen True has nothing to save)", flush=True)
            return
        payload = {
            "round": round_id,
            "val_metrics": dict(val_metrics) if val_metrics else None,
            "test_metrics": dict(test_metrics) if test_metrics else None,
            "dataset": getattr(self.args, "dataset", None),
            "lora_agg_method": getattr(self.args, "lora_agg_method", None),
            **({"lora": lora_state_dict(model.model)} if model_has_lora else {}),
            **graph_state,
            **({"synthetic_memory": synthetic_state} if synthetic_state is not None else {}),
        }

        path = Path(self._save_best_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        saved = [k for k in ("lora", *self._GRAPH_WEIGHT_KEYS, "synthetic_memory") if k in payload]
        print(f"    [checkpoint] new best val hit {self._best_val_acc:.2f}% "
              f"(round {round_id}, saved: {saved}) -> {path}", flush=True)

    def _load_checkpoint(self, path: str) -> None:
        """Load a --save-best checkpoint (LoRA adapter and/or graph_encoder/
        projector/condensed_encoder/projector_c, whichever it contains) into
        self.shared_model.

        The model must have been built with the *same* config (LoRA rank/
        alpha/target-modules if --llm-frozen False, --dual-graph-mode/
        --gnn-* dims for the graph submodules) as the run that produced the
        checkpoint, or load_state_dict(strict=False) will silently attach
        nothing (mismatched key names/shapes).
        """
        if self.shared_model is None:
            raise RuntimeError(
                f"--load-checkpoint {path} given but Stage D never initialized "
                "(needs --num-rounds > 1 or --eval-only)."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        loaded: list[str] = []

        if "lora" in payload:
            if not has_lora(self.shared_model.model):
                raise RuntimeError(
                    f"--load-checkpoint {path} has a LoRA adapter but the current model "
                    "has none -- pass --llm-frozen False with the same --lora-rank/"
                    "--lora-alpha/--lora-target-modules used to produce this checkpoint."
                )
            missing, _ = self.shared_model.model.load_state_dict(payload["lora"], strict=False)
            lora_missing = [k for k in missing if ".lora_A." in k or ".lora_B." in k]
            if lora_missing:
                raise RuntimeError(
                    f"--load-checkpoint {path}: {len(lora_missing)} LoRA keys not found in "
                    f"the current model (config mismatch?) -- first few: {lora_missing[:5]}"
                )
            loaded.append("lora")

        for key in self._GRAPH_WEIGHT_KEYS:
            if key not in payload:
                continue
            submodule = getattr(self.shared_model, key, None)
            if submodule is None:
                raise RuntimeError(
                    f"--load-checkpoint {path} has '{key}' weights but the current model "
                    f"has no '{key}' submodule -- check --dual-graph-mode matches the run "
                    "that produced this checkpoint."
                )
            submodule.load_state_dict(payload[key])
            loaded.append(key)

        # server.repr_encoder/repr_projector are normally kept in sync with
        # condensed_encoder/projector_c via _load_repr_align_weights(), called
        # each round from FedAvg'd weights -- but --load-checkpoint restores
        # straight into self.shared_model, bypassing that path entirely, so
        # without this the server's Phase-0 alignment step (which shapes
        # X_syn) runs against a randomly-initialized encoder instead of the
        # checkpoint's trained one. Sync explicitly whenever we have both.
        if "condensed_encoder" in payload and "projector_c" in payload:
            self.server._load_repr_align_weights({
                "condensed_encoder": payload["condensed_encoder"],
                "projector_c": payload["projector_c"],
            })
            loaded.append("repr_encoder(synced)")

        if "synthetic_memory" in payload:
            self.server.load_synthetic_memory_state_dict(payload["synthetic_memory"])
            loaded.append("synthetic_memory")

        print(f"    [checkpoint] loaded {loaded} from {path} "
              f"(round {payload.get('round')}, val hit {(payload.get('val_metrics') or {}).get('hit')})",
              flush=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _route_samples(self, samples: list, n_clients: int) -> list:
        """Pick, per question, the client whose own evidence best matches it.

        The previous rule was `i % n_clients` -- a question was answered by
        whichever client its array position landed on, with no relevance test
        at all. Since the clients' corpora are disjoint shards, that is close
        to picking one at random: measured on hotpotqa_merged's 1000 test
        questions, the assigned client's top-5 evidence contained the answer
        43.8% of the time, while SOME client's did 82.5% of the time.

        Routing by max cosine(q_emb, anchor embedding) over each client's own
        top-5 PPR anchors lifts that to 55.3% -- and, unlike simply retrieving
        more passages, it does not enlarge the prompt, so the reader still
        sees exactly 5 passages and its conversion rate should hold. (Deeper
        retrieval was measured to *cost* 2 EM: recall rose to 51.5% but the
        extra 15 passages were distractors.)

        Each client exposes only a scalar score per question, not its
        passages, so this stays within the federated boundary.

        Falls back to the old round-robin whenever the signal is unavailable
        (no ppr map, no idx on the sample, missing embeddings).
        """
        import torch.nn.functional as _F

        if not getattr(self.args, "route_eval_by_relevance", False) or n_clients <= 1:
            return [i % n_clients for i in range(len(samples))]

        # Hybrid score: embedding cosine + lexical overlap. Cosine alone
        # recovers only ~53% of the oracle because these questions turn on
        # entity names, which match lexically but blur in a 384-d sentence
        # embedding. Measured answer-in-evidence after routing:
        #     hotpotqa      cos 55.3  lex 60.5  hybrid 60.0  (oracle 82.5)
        #     2wikimultihop cos 32.8  lex 34.8  hybrid 36.3  (oracle 61.7)
        #     musique       cos 13.3  lex 12.4  hybrid 13.4  (oracle 24.6)
        _STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
                 "is", "are", "was", "were", "be", "been", "by", "with", "from", "as",
                 "that", "this", "it", "its", "what", "which", "who", "whom", "whose",
                 "when", "where", "how", "why", "did", "do", "does"}
        use_lexical = str(getattr(self.args, "route_score_mode", "hybrid")).lower() != "cosine"
        scores = torch.full((len(samples), n_clients), float("-inf"))
        q_toks = [
            {w for w in normalize(s.get("question", "")).split()
             if w not in _STOP and len(w) > 2}
            for s in samples
        ]
        for cid, client in enumerate(self.clients):
            pmap = getattr(client, "_ppr_node_map", None)
            tg = getattr(client, "tri_graph", None)
            if pmap is None or tg is None:
                continue
            X = tg.x.detach().float().cpu()
            node_text = getattr(tg, "node_text", None)
            N = X.size(0)
            for i, s in enumerate(samples):
                idx = s.get("idx")
                q = s.get("q_emb")
                if idx is None or q is None or idx >= pmap.shape[0]:
                    continue
                anchors = [int(a) for a in pmap[idx].tolist() if 0 <= a < N][:5]
                if not anchors:
                    continue
                sims = _F.cosine_similarity(
                    q.detach().float().cpu().view(1, -1), X[anchors], dim=1)
                score = float(sims.max())
                if use_lexical and node_text is not None and len(node_text) == N and q_toks[i]:
                    d = normalize(" ".join(node_text[a] for a in anchors if a < len(node_text)))
                    score += len(q_toks[i] & set(d.split())) / len(q_toks[i])
                scores[i, cid] = score

        assign = []
        n_routed = 0
        for i in range(len(samples)):
            row = scores[i]
            if bool(torch.isinf(row).all()):
                assign.append(i % n_clients)      # no signal -- old behaviour
            else:
                assign.append(int(row.argmax()))
                n_routed += 1
        if n_routed:
            print(f"    [eval] relevance-routed {n_routed}/{len(samples)} questions "
                  f"(fell back to round-robin for {len(samples) - n_routed})", flush=True)

        # Cross-client evidence merge: build `desc` from the top-scoring clients
        # instead of only the routed one. Routing picks ONE client, which cannot
        # serve a question whose hops live on different shards -- and 73.4% of
        # hotpotqa's gold evidence spans 2+ clients. Measured answer-in-evidence
        # at an unchanged 5-passage budget (2+2+1 across three clients):
        #     2wikimultihop 36.3% -> 46.0%   hotpotqa 60.0% -> 61.0%
        # musique gains nothing from redistribution (13.4% -> 12.7%); it needs
        # a larger budget (10 passages -> 20.8%, 15 -> 24.6%).
        qx_on = int(getattr(self.args, "query_expand_topk", 0) or 0) > 0
        per_client = int(getattr(self.args, "merge_evidence_topk", 0) or 0)
        if per_client <= 0 and qx_on:
            per_client = 5   # query-expand-only: still need a base k for the routed client
        if per_client > 0:
            # --merge-evidence-clients controls scope for BOTH mechanisms: pass 1
            # for "routed client + query-expansion only", or >1 to also merge
            # across clients. Its argparse default is 3 (all clients).
            n_use = min(int(getattr(self.args, "merge_evidence_clients", n_clients)), n_clients)
            merged = 0
            for i, s in enumerate(samples):
                row = scores[i]
                if bool(torch.isinf(row).all()):
                    continue
                order = sorted(range(n_clients), key=lambda c: -float(row[c]))[:n_use]
                texts: list[str] = []
                for c in order:
                    texts.extend(self._client_top_passages(c, s, per_client))
                if texts:
                    # NOT written to "desc" here: _attach_evidence_graphs rebuilds
                    # desc from the serving client's own anchors and would clobber
                    # it. It copies unknown keys through, so stash it and restore
                    # after attaching.
                    s["_merged_desc_text"] = "\n\n".join(texts)
                    merged += 1
            if merged:
                print(f"    [eval] merged evidence for {merged}/{len(samples)} questions "
                      f"({per_client} passages x {n_use} clients)", flush=True)
        return assign

    def _client_passage_index(self, cid: int):
        """Cache (passage_node_ids, passage_embeds) for one client, built once."""
        cache = getattr(self, "_passage_idx_cache", None)
        if cache is None:
            cache = self._passage_idx_cache = {}
        if cid in cache:
            return cache[cid]
        tg = getattr(self.clients[cid], "tri_graph", None)
        if tg is None:
            cache[cid] = (None, None)
            return cache[cid]
        ntype = tg.node_type.tolist()
        pass_idx = [k for k, t in enumerate(ntype) if t == 2]
        Xp = tg.x[pass_idx].detach().float().cpu() if pass_idx else None
        cache[cid] = (pass_idx, Xp)
        return cache[cid]

    def _client_top_passages(self, cid: int, sample: dict, k: int) -> list:
        """This client's own top-k PPR passage texts for one question, optionally
        extended with query-expansion passages (see below)."""
        client = self.clients[cid]
        pmap = getattr(client, "_ppr_node_map", None)
        tg = getattr(client, "tri_graph", None)
        if pmap is None or tg is None:
            return []
        node_text = getattr(tg, "node_text", None)
        idx = sample.get("idx")
        if node_text is None or idx is None or idx >= pmap.shape[0]:
            return []
        N = tg.x.size(0)
        anchor_ids = [int(a) for a in pmap[idx].tolist() if 0 <= a < N]
        anchors = anchor_ids[:k]

        qx = int(getattr(self.args, "query_expand_topk", 0) or 0)
        if qx > 0 and anchor_ids and sample.get("q_emb") is not None:
            anchors = anchors + self._query_expand_passages(
                cid, sample["q_emb"], anchor_ids[:5], qx, exclude=set(anchor_ids))
        return [node_text[a] for a in anchors if a < len(node_text)]

    def _query_expand_passages(self, cid: int, q_emb, seed_ids: list, k: int, exclude: set) -> list:
        """Rocchio-style query expansion, one round: re-retrieve with
        0.5*question + 0.5*mean(seed passage embeddings) over ALL of this
        client's passages (not just the PPR-reachable ones).

        Why: PPR only propagates through the graph from entities seeded by the
        ORIGINAL question. On multi-hop benchmarks the hop-2+ entity is often
        absent from the question text entirely (it is only revealed by
        resolving hop 1), so single-shot PPR structurally cannot seed toward
        it -- graph expansion from the retrieved passages' own neighbors was
        measured to recover NOTHING (musique oracle stayed at 24.6%). Dense
        retrieval with a query vector that has absorbed the hop-1 evidence can
        reach passages no graph walk from the question would ever find.
        Measured oracle gain (top-5 -> +expansion, answer-in-evidence):
            musique 24.6% -> 38.3%   2wikimultihop 61.7% -> 66.6%   hotpotqa 82.5% -> 85.6%
        """
        import torch.nn.functional as _F
        pass_idx, Xp = self._client_passage_index(cid)
        if not pass_idx or Xp is None:
            return []
        tg = self.clients[cid].tri_graph
        q0 = q_emb.detach().float().cpu()
        pmean = tg.x[seed_ids].detach().float().cpu().mean(0)
        qnew = _F.normalize((0.5 * q0 + 0.5 * pmean).unsqueeze(0), dim=1)
        sims = _F.cosine_similarity(qnew, Xp, dim=1)
        order = sims.argsort(descending=True).tolist()
        out = []
        for j in order:
            node_id = pass_idx[j]
            if node_id in exclude:
                continue
            out.append(node_id)
            if len(out) >= k:
                break
        return out

    def _eval_split_acc(self, samples: list, dump_predictions_path: str | None = None) -> dict:
        """Distribute samples across clients, run inference with on-the-fly retrieval.

        Returns {"hit", "em", "f1"} in percent — hit is the legacy normalized
        substring containment; em/f1 are SQuAD/MuSiQue-style exact match and
        token-level F1.

        If `dump_predictions_path` is given, appends one JSONL row per
        question ({"id", "client_id", "pred", "label"}) there -- used to
        recompute F1 split by question subgroup (e.g. local-vs-cross-client
        gold evidence) after the run, since that split isn't known here.
        """
        from fedcond_grag.client.stage_d_retrieve.global_graph_retriever import GlobalGraphRetriever
        from fedcond_grag.utils.collate import collate_fn

        batch_size = int(getattr(self.args, "eval_batch_size",
                                    getattr(self.args, "local_batch_size", 4)))
        top_r = int(getattr(self.args, "retrieval_top_r", 16))
        n_clients = len(self.clients)
        hits = 0
        em_total = 0.0
        f1_total = 0.0
        self.shared_model.eval()

        assign = self._route_samples(samples, n_clients)

        with torch.no_grad():
            for cid, client in enumerate(self.clients):
                shard = [s for i, s in enumerate(samples) if assign[i] == cid]
                if not shard:
                    continue
                shard = client._attach_evidence_graphs(shard)
                for s in shard:                       # restore the cross-client desc
                    if s.get("_merged_desc_text"):
                        s["desc"] = s["_merged_desc_text"]
                syn_retriever = (
                    GlobalGraphRetriever(client.synthetic_graph, top_r=top_r)
                    if client.synthetic_graph is not None else None
                )
                if syn_retriever is not None:
                    shard = client._attach_condensed_graphs(shard, syn_retriever)
                dump_f = open(dump_predictions_path, "a") if dump_predictions_path else None
                try:
                    for i in range(0, len(shard), batch_size):
                        mini = shard[i : i + batch_size]
                        batch = collate_fn(mini)
                        out = self.shared_model.inference(batch)
                        ids = batch.get("id", [None] * len(mini))
                        for qid, pred, label in zip(ids, out["pred"], out["label"]):
                            if normalize(label) in normalize(pred):
                                hits += 1
                            if exact_match(pred, label):
                                em_total += 1.0
                            f1_total += token_f1(pred, label)
                            if dump_f is not None:
                                dump_f.write(json.dumps({
                                    "id": qid, "client_id": cid, "pred": pred, "label": label,
                                }) + "\n")
                finally:
                    if dump_f is not None:
                        dump_f.close()

        n = max(len(samples), 1)
        return {
            "hit": 100.0 * hits / n,
            "em": 100.0 * em_total / n,
            "f1": 100.0 * f1_total / n,
        }

    def _log_metrics(self, metrics: dict, global_step: int = 0) -> None:
        with self._metrics_path.open("a") as f:
            f.write(json.dumps({k: v for k, v in metrics.items()
                                 if k not in ("client_times",)}) + "\n")
        if self._wandb is None:
            return
        r = metrics["round"]
        log: dict = {"comm_round": r}
        # Loss per round
        if metrics["avg_loss"] is not None:
            log["round/avg_loss"] = metrics["avg_loss"]
            for cid, loss in metrics["client_losses"].items():
                log[f"round/client_{cid}_loss"] = loss
        # Phase 0 / synthetic-memory diagnostics
        if metrics.get("server_syn_loss") is not None:
            log["round/server_syn_loss"] = metrics["server_syn_loss"]
        for cid, loss in metrics.get("client_syn_mem", {}).items():
            log[f"round/client_{cid}_syn_mem_qa"] = loss
        for cid, h in metrics.get("stage_b_refine", {}).items():
            log[f"round/client_{cid}_Lcond_start"] = h["l_cond_start"]
            log[f"round/client_{cid}_Lcond_end"] = h["l_cond_end"]
        # Accuracy per round
        if metrics["train_acc"] is not None:
            log["round/train_acc"] = metrics["train_acc"]
        if metrics["val_acc"] is not None:
            log["round/val_acc"] = metrics["val_acc"]
        if metrics["test_acc"] is not None:
            log["round/test_acc"] = metrics["test_acc"]
        for key in ("val_em", "val_f1", "test_em", "test_f1"):
            if metrics.get(key) is not None:
                log[f"round/{key}"] = metrics[key]
        # Timing
        for cid, t in metrics.get("client_times", {}).items():
            log[f"round/client_{cid}_time_s"] = t
        if metrics.get("round_time") is not None:
            log["round/total_time_s"] = metrics["round_time"]
        if metrics.get("eval_time") is not None:
            log["round/eval_time_s"] = metrics["eval_time"]
        if metrics.get("comm_memory_mb") is not None:
            log["round/comm_memory_mb"] = metrics["comm_memory_mb"]
        if metrics.get("comm_total_mb") is not None:
            log["round/comm_total_mb"] = metrics["comm_total_mb"]
        self._wandb.log(log)

    def _print_metrics_table(self, round_metrics: list[dict]) -> None:
        W = 95
        n = self.args.num_clients
        print("\n" + "=" * W)
        print("FEDERATED TRAINING SUMMARY")
        print("=" * W)
        hdr = f"{'Rnd':>4} | {'AvgLoss':>8} |"
        for cid in range(n):
            hdr += f" {'C'+str(cid)+'Loss':>8} |"
        hdr += f" {'Train%':>7} | {'Val%':>7} | {'Test%':>7}"
        print(hdr)
        print("-" * W)
        def _f(v):
            return f"{v:>6.2f}%" if v is not None else f"{'N/A':>7}"

        for m in round_metrics:
            if m["avg_loss"] is None:
                row = f"{m['round']:>4} | {'(boot)':>8} |" + f" {'—':>8} |" * n
            else:
                row = f"{m['round']:>4} | {m['avg_loss']:>8.4f} |"
                for cid in range(n):
                    v = m["client_losses"].get(cid, float("nan"))
                    row += f" {v:>8.4f} |"
            row += f" {_f(m['train_acc'])} | {_f(m['val_acc'])} | {_f(m['test_acc'])}"
            print(row)
        print("=" * W)
        print(f"Full metrics → {self._metrics_path}")

        if self.shared_model is None or not self._test_samples:
            return
        from fedcond_grag.utils.collate import collate_fn
        from fedcond_grag.client.stage_d_retrieve.global_graph_retriever import GlobalGraphRetriever
        top_r = int(getattr(self.args, "retrieval_top_r", 16))
        n_show = min(10, len(self._test_samples))
        mini = list(self._test_samples[:n_show])
        client = self.clients[0]
        mini = client._attach_evidence_graphs(mini)
        if client.synthetic_graph is not None:
            r = GlobalGraphRetriever(client.synthetic_graph, top_r=top_r)
            mini = client._attach_condensed_graphs(mini, r)
        batch = collate_fn(mini)
        self.shared_model.eval()
        with torch.no_grad():
            out = self.shared_model.inference(batch)
        print("\nSAMPLE PREDICTIONS (test, first 10)")
        print("-" * W)
        for i in range(n_show):
            q  = out["question"][i][:80].replace("\n", " ")
            gt = out["label"][i]
            pr = out["pred"][i][:80].replace("\n", " ")
            hit = "✓" if gt.strip().lower() in pr.strip().lower() else "✗"
            print(f"[{i+1:>2}] {hit} Q : {q}")
            print(f"       GT: {gt}  |  PR: {pr}")
        print()

    # ------------------------------------------------------------------
    # Stage D initialisation
    # ------------------------------------------------------------------

    def _init_stage_d(self) -> None:
        from fedcond_grag.dataloader import FedCondQADataset
        qa_root = getattr(self.args, "qa_data_root", "dataset/fedcond_qa")

        # qa_root is a shared path across datasets by default (dataset/fedcond_qa),
        # rebuilt in place by scripts/build_fedcond_qa_dataset.py / main.py
        # preprocess. If a run for a *different* --dataset rebuilt it last and
        # this run's num_rounds<=1 skipped calling preprocess again, loading it
        # here would silently train/eval against the wrong dataset's questions
        # while trigraph/ppr_node_map are still this dataset's. Refuse instead.
        meta_path = Path(qa_root) / "_meta.json"
        if meta_path.exists():
            try:
                cached_dataset = json.loads(meta_path.read_text()).get("dataset")
            except Exception:
                cached_dataset = None
            if cached_dataset and cached_dataset != self.args.dataset:
                raise RuntimeError(
                    f"QA cache at '{qa_root}' belongs to dataset '{cached_dataset}', "
                    f"not '{self.args.dataset}'. Rebuild it first: "
                    f"python scripts/build_fedcond_qa_dataset.py --dataset {self.args.dataset} "
                    f"--out-root {qa_root} (or re-run 'python main.py preprocess --dataset "
                    f"{self.args.dataset}'). Refusing to silently train/eval on mismatched data."
                )

        top_r = int(getattr(self.args, "top_r_passages", 0))
        top_r_anchor = getattr(self.args, "top_r_anchor", None)
        if top_r_anchor is not None:
            top_r_anchor = int(top_r_anchor)
        try:
            qa_dataset = FedCondQADataset(
                root=qa_root,
                top_r_passages=top_r,
                top_r_anchor=top_r_anchor,
            )
        except FileNotFoundError as exc:
            print(f"[FedTrainer] Stage D disabled: QA dataset not found — {exc}")
            return
        if top_r > 0:
            if qa_dataset.top_r_passages == 0:
                print(f"[FedTrainer] WARNING: --top-r-passages={top_r} requested but "
                      f"passage_embs.pt / passage_node_map.pt not found in {qa_root} — "
                      f"falling back to legacy desc.", flush=True)
            else:
                eff_anchor = qa_dataset.top_r_anchor
                print(f"[FedTrainer] re-ranked desc enabled: top-{top_r} passages per sample, "
                      f"graph anchored on top-{eff_anchor} of those.", flush=True)

        split_dir = Path(qa_root) / "split"

        def _load_idx(fname):
            p = split_dir / fname
            return [int(l.strip()) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

        train_idx = sorted(set(_load_idx("train_indices.txt")))
        val_idx   = _load_idx("val_indices.txt")
        test_idx  = _load_idx("test_indices.txt")

        max_eval = int(getattr(self.args, "max_eval_samples", 200))
        self._train_eval_samples = [qa_dataset[i] for i in train_idx[:max_eval] if i < len(qa_dataset)]
        self._val_samples        = [qa_dataset[i] for i in val_idx[:max_eval]   if i < len(qa_dataset)]
        self._test_samples       = [qa_dataset[i] for i in test_idx[:max_eval]  if i < len(qa_dataset)]
        print(f"    eval sets — train: {len(self._train_eval_samples)}, "
              f"val: {len(self._val_samples)}, test: {len(self._test_samples)} (capped {max_eval})")

        n = self.args.num_clients
        max_per = int(getattr(self.args, "max_train_per_client", 0))
        self._max_train_per_client = max_per
        for cid, client in enumerate(self.clients):
            cid_indices = [i for i in train_idx if i % n == cid]
            all_samples = [qa_dataset[i] for i in cid_indices]
            if max_per > 0 and max_per < len(all_samples):
                # Pre-attach evidence graphs for ALL samples; each round will
                # randomly pick max_per of them so the full dataset is covered.
                client.set_full_train_pool(all_samples, max_per_round=max_per)
            else:
                client.set_local_qa_data(all_samples)
                print(f"    client_{cid}: {len(all_samples)} QA train samples")


        try:
            from fedcond_grag.model import load_model, llama_model_path
        except ImportError as exc:
            print(f"[FedTrainer] Stage D disabled: model import failed — {exc}")
            return

        llm_name = getattr(self.args, "llm_model_name", "7b")
        llm_path = getattr(self.args, "llm_model_path", "") or llama_model_path.get(llm_name, "")
        if not llm_path:
            print("[FedTrainer] Stage D disabled: llm_model_path not set")
            return

        self.args.llm_model_path = llm_path
        for attr, default in (
            ("gnn_model_name",   "gt"),   ("gnn_model_name_c",  "gcn"),
            ("gnn_num_layers",   4),      ("gnn_num_layers_c",  None),
            ("gnn_in_dim",       384),    ("gnn_in_dim_c",      None),
            ("gnn_hidden_dim",   384),    ("gnn_hidden_dim_c",  None),
            ("gnn_num_heads",    4),      ("gnn_num_heads_c",   None),
            ("gnn_dropout",      0.0),    ("dual_graph_mode",   "shared"),
            ("max_txt_len",      512),    ("max_new_tokens",    32),
            ("llm_frozen",       "True"),
            ("lora_rank",        8),      ("lora_alpha",        16),
            ("lora_dropout",     0.05),   ("lora_target_modules", None),
            ("lora_agg_method",  "fedit"), ("lora_agg_scale",   2.0),
            ("llm_gpu_max_memory_gib", None), ("llm_cpu_max_memory_gib", None),
        ):
            if not hasattr(self.args, attr):
                setattr(self.args, attr, default)

        try:
            model = load_model["dual_graph_llm"](args=self.args)
        except Exception as exc:
            print(f"[FedTrainer] Stage D disabled: failed to load DualGraphLLM — {exc}")
            return

        for name, param in model.named_parameters():
            param.requires_grad = (
                any(name.startswith(k) for k in
                    ("graph_encoder", "projector", "condensed_encoder", "projector_c"))
                or ".lora_A." in name or ".lora_B." in name
            )

        if not hasattr(model, "hf_device_map"):
            model.to(self.device)
        self.shared_model = model

        for client in self.clients:
            client.set_shared_model(model)
        self.server.set_shared_llm_model(model.model)

        print(f"[FedTrainer] Stage D ready — shared DualGraphLLM on {self.device}")

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_dotenv(self) -> None:
        """Load .env from project root into os.environ without overwriting existing vars."""
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key and val and key not in os.environ:
                os.environ[key] = val

    def _processed_root(self) -> Path:
        return Path(getattr(self.args, "data_root", "processed")) / self.args.dataset

    def _load_client_data(self, client_id: int):
        from torch_geometric.data import Data
        client_dir = self._processed_root() / f"client_{client_id}"
        path = client_dir / "trigraph.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trigraph for client {client_id}: {path}. "
                f"Run `fedcond_grag preprocess --dataset {self.args.dataset}` first."
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
        """The server only needs the node-feature dimensionality — hand it a
        1-node stub instead of duplicating a client's raw trigraph (multi-GB,
        and the server must never hold private client graphs)."""
        from torch_geometric.data import Data

        global_dir = self._processed_root() / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        feat_dim = int(self.clients[0].tri_graph.x.size(1)) if self.clients else ENCODER_DIM
        stub = Data(
            x=torch.zeros(1, feat_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            node_type=torch.zeros(1, dtype=torch.long),
        )
        stub.y = stub.node_type.long()
        stub.num_global_classes = 3
        return stub, str(global_dir)
