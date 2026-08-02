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
                client_times[cid] = time.perf_counter() - t0
                loss_str = f" | loss: {loss:.4f}" if round_id >= 1 else ""
                print(f"    client_{cid}{loss_str} | {client_times[cid]:.2f}s", flush=True)

            t0 = time.perf_counter()
            self.server.execute()
            agg_time = time.perf_counter() - t0
            print(f"    server agg done in {agg_time:.2f}s", flush=True)

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

        self._print_metrics_table(round_metrics)

    def _run_eval_only(self) -> None:
        """--eval-only: no training rounds, no FedAvg of trained weights --
        just run the already-loaded (optionally checkpoint-restored)
        shared_model against --dataset's test split once. Intended for
        evaluating a checkpoint trained on a separate <dataset>_train
        pseudo-dataset against this (untouched) dataset's full question set
        -- pair with --qa-test-only at preprocess time so test_idx covers
        all of it, and --max-eval-samples set above the question count so
        nothing gets capped.

        Still runs the round-0-equivalent Stage B/C bootstrap (client
        condense + server synthetic-graph construction) first: Θ_syn is
        built from THIS dataset's own corpus, not the training run's --
        skipping it would leave condensed_encoder/projector_c (loaded from
        the checkpoint) operating on an empty/never-built synthetic graph,
        corrupting the dual-graph prompt for any --dual-graph-mode that uses
        it (the default, "both").
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
        print("[eval-only] running Stage B/C bootstrap for this dataset's own "
              "synthetic graph before evaluating...", flush=True)
        self.message_pool["round"] = 0
        sampled = list(range(self.args.num_clients))
        self.message_pool["sampled_clients"] = sampled
        self.server.send_message()
        for cid in sampled:
            self.clients[cid].receive_message()
            self.clients[cid].execute()
            self.clients[cid].send_message()
        self.server.execute()

        print(f"[eval-only] evaluating on {len(self._test_samples)} test samples "
              f"(--max-eval-samples={getattr(self.args, 'max_eval_samples', 200)} "
              "-- raise it if this is less than your test set size)", flush=True)
        test_metrics = self._eval_split_acc(self._test_samples)
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
        if not model_has_lora and not graph_state:
            print("    [checkpoint] skipped -- no LoRA adapter and no trained graph "
                  "weights (--dual-graph-mode none with --llm-frozen True has nothing "
                  "to save)", flush=True)
            return
        payload = {
            "round": round_id,
            "val_metrics": dict(val_metrics) if val_metrics else None,
            "test_metrics": dict(test_metrics) if test_metrics else None,
            "dataset": getattr(self.args, "dataset", None),
            "lora_agg_method": getattr(self.args, "lora_agg_method", None),
            **({"lora": lora_state_dict(model.model)} if model_has_lora else {}),
            **graph_state,
        }

        path = Path(self._save_best_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        saved = [k for k in ("lora", *self._GRAPH_WEIGHT_KEYS) if k in payload]
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

        print(f"    [checkpoint] loaded {loaded} from {path} "
              f"(round {payload.get('round')}, val hit {(payload.get('val_metrics') or {}).get('hit')})",
              flush=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_split_acc(self, samples: list) -> dict:
        """Distribute samples across clients, run inference with on-the-fly retrieval.

        Returns {"hit", "em", "f1"} in percent — hit is the legacy normalized
        substring containment; em/f1 are SQuAD/MuSiQue-style exact match and
        token-level F1.
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

        with torch.no_grad():
            for cid, client in enumerate(self.clients):
                shard = [s for i, s in enumerate(samples) if i % n_clients == cid]
                if not shard:
                    continue
                shard = client._attach_evidence_graphs(shard)
                syn_retriever = (
                    GlobalGraphRetriever(client.synthetic_graph, top_r=top_r)
                    if client.synthetic_graph is not None else None
                )
                if syn_retriever is not None:
                    shard = client._attach_condensed_graphs(shard, syn_retriever)
                for i in range(0, len(shard), batch_size):
                    mini = shard[i : i + batch_size]
                    batch = collate_fn(mini)
                    out = self.shared_model.inference(batch)
                    for pred, label in zip(out["pred"], out["label"]):
                        if normalize(label) in normalize(pred):
                            hits += 1
                        if exact_match(pred, label):
                            em_total += 1.0
                        f1_total += token_f1(pred, label)

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
