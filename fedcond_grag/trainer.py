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
from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.utils.evaluate import (
    exact_match,
    norm_passage_title,
    normalize,
    retrieval_recall_at_k,
    token_f1,
)


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
        if int(getattr(args, "num_rounds", 1)) > 1 or bool(getattr(args, "eval_only", False)):
            self._init_stage_d()

        self._metrics_path = Path(getattr(args, "metrics_path", "/tmp/fl_metrics.jsonl"))
        self._metrics_path.write_text("")   # reset on new run

        self._load_dotenv()
        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
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
        if bool(getattr(self.args, "eval_only", False)):
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

                self.clients[cid].send_message()
                client_times[cid] = time.perf_counter() - t0
                loss_str = f" | loss: {loss:.4f}" if round_id >= 1 else ""
                print(f"    client_{cid}{loss_str} | {client_times[cid]:.1f}s", flush=True)

            t0 = time.perf_counter()
            self.server.execute()
            agg_time = time.perf_counter() - t0
            print(f"    server agg done in {agg_time:.1f}s", flush=True)

            train_acc = val_acc = test_acc = None
            val_metrics = test_metrics = None
            eval_time = None
            eval_every = int(getattr(self.args, "eval_every", 1))
            do_eval = round_id >= 1 and self.shared_model is not None and (round_id % eval_every == 0)
            if do_eval:
                t_eval = time.perf_counter()
                if self._val_samples:
                    val_metrics = self._eval_split_acc(self._val_samples)
                    val_acc = val_metrics["hit"]
                    print(f"    val   : hit {val_metrics['hit']:.1f}% | "
                          f"EM {val_metrics['em']:.1f}% | F1 {val_metrics['f1']:.1f} | "
                          f"R@1 {val_metrics['r@1']:.1f}% | R@2 {val_metrics['r@2']:.1f}% | "
                          f"R@5 {val_metrics['r@5']:.1f}%", flush=True)
                if self._test_samples:
                    test_metrics = self._eval_split_acc(self._test_samples)
                    test_acc = test_metrics["hit"]
                    print(f"    test  : hit {test_metrics['hit']:.1f}% | "
                          f"EM {test_metrics['em']:.1f}% | F1 {test_metrics['f1']:.1f} | "
                          f"R@1 {test_metrics['r@1']:.1f}% | R@2 {test_metrics['r@2']:.1f}% | "
                          f"R@5 {test_metrics['r@5']:.1f}%", flush=True)
                eval_time = time.perf_counter() - t_eval
                print(f"    eval done in {eval_time:.1f}s", flush=True)

                # Keep the best-val-round weights in RAM (the final round is
                # often not the best one); written to disk once at the end.
                if val_metrics is not None and val_metrics["hit"] >= getattr(self, "_best_val_hit", -1.0):
                    self._best_val_hit = val_metrics["hit"]
                    self._best_val_round = round_id
                    self._best_val_state = self._collect_trainable_ckpt()
                    print(f"    [checkpoint] best-val snapshot updated (hit {val_metrics['hit']:.1f}%, round {round_id})",
                          flush=True)

            round_time = time.perf_counter() - t_round_start
            avg_loss = (
                sum(client_losses.values()) / len(client_losses) if client_losses else None
            )
            metrics = {
                "round": round_id,
                "avg_loss": avg_loss,
                "client_losses": dict(client_losses),
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
                **{f"val_r@{k}": val_metrics[f"r@{k}"] if val_metrics else None
                   for k in (1, 2, 5)},
                **{f"test_r@{k}": test_metrics[f"r@{k}"] if test_metrics else None
                   for k in (1, 2, 5)},
            }
            round_metrics.append(metrics)
            self._log_metrics(metrics, global_step=global_step)

        self._print_metrics_table(round_metrics)

        try:
            self._save_final_checkpoint()
            if getattr(self, "_best_val_state", None) is not None:
                self._save_final_checkpoint(suffix="-best", upload=False,
                                            ckpt=self._best_val_state)
                print(f"[checkpoint] best-val round was {self._best_val_round} "
                      f"(hit {self._best_val_hit:.1f}%)", flush=True)
        except Exception as exc:  # never let checkpointing kill a finished run
            print(f"[checkpoint] save failed: {exc}", flush=True)

    def _collect_trainable_ckpt(self) -> dict | None:
        """Snapshot the FedAvg-exchanged parameter groups to CPU tensors."""
        model = self.shared_model
        if model is None:
            return None
        prefixes = ("graph_encoder", "projector", "condensed_encoder", "projector_c")
        ckpt = {"module_weights": {}, "lora": {}, "args": dict(vars(self.args))}
        for source in (model.named_parameters(), model.named_buffers()):
            for n, t in source:
                if any(n.startswith(k) for k in prefixes):
                    ckpt["module_weights"][n] = t.detach().cpu().clone()
                elif "lora_" in n:
                    ckpt["lora"][n] = t.detach().cpu().clone()
        return ckpt

    def _save_final_checkpoint(self, suffix: str = "", upload: bool = True,
                               ckpt: dict | None = None) -> None:
        """Save trainable weights (graph modules + LoRA) and upload to WandB.

        The checkpoint holds exactly the FedAvg-exchanged parameter groups, so
        an eval-only run on another dataset can load it with strict=False.
        Download later with:  wandb artifact get <entity>/<project>/model-<run>:latest

        `suffix` names checkpoint variants (e.g. "-best" for the best-val-round
        snapshot); pass `ckpt` to save an in-memory snapshot instead of the
        current model state. WandB upload is reserved for the final checkpoint.
        """
        model = self.shared_model
        if model is None:
            return
        prefixes = ("graph_encoder", "projector", "condensed_encoder", "projector_c")
        if ckpt is None:
            ckpt = {"module_weights": {}, "lora": {}, "args": dict(vars(self.args))}
            for source in (model.named_parameters(), model.named_buffers()):
                for n, t in source:
                    if any(n.startswith(k) for k in prefixes):
                        ckpt["module_weights"][n] = t.detach().cpu()
                    elif "lora_" in n:
                        ckpt["lora"][n] = t.detach().cpu()

        run_name = None
        if self._wandb is not None:
            run_name = getattr(self._wandb, "name", None)
        tag = (run_name or f"{self.args.dataset}-{getattr(self.args, 'dual_graph_mode', 'run')}") + suffix
        out_dir = Path("checkpoints")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{tag}.pt"
        torch.save(ckpt, path)
        n_lora = len(ckpt["lora"])
        n_mod = len(ckpt["module_weights"])
        print(f"[checkpoint] saved {path} ({n_mod} module tensors, {n_lora} LoRA tensors)",
              flush=True)

        if upload and self._wandb is not None:
            import wandb
            artifact = wandb.Artifact(f"model-{tag}", type="model",
                                      metadata={"dataset": self.args.dataset,
                                                "dual_graph_mode": getattr(self.args, "dual_graph_mode", None),
                                                "llm_frozen": getattr(self.args, "llm_frozen", None)})
            artifact.add_file(str(path))
            self._wandb.log_artifact(artifact)
            print(f"[checkpoint] uploaded WandB artifact model-{tag}", flush=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _run_eval_only(self) -> None:
        """Load a saved checkpoint and evaluate without any local training.

        Runs one artifact-exchange round so clients/server hold the same state
        as at the start of a normal round, then evaluates val + test.
        """
        self.message_pool["round"] = 0
        self.message_pool["sampled_clients"] = list(range(self.args.num_clients))
        self.server.send_message()
        for client in self.clients:
            client.receive_message()
            client.execute()
            client.send_message()
        self.server.execute()

        if self.shared_model is None:
            print("[eval-only] Stage D model unavailable — nothing to evaluate")
            return

        ckpt_path = str(getattr(self.args, "load_checkpoint", "") or "")
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = {}
            state.update(ckpt.get("module_weights", {}))
            state.update(ckpt.get("lora", {}))
            missing, unexpected = self.shared_model.load_state_dict(state, strict=False)
            print(f"[eval-only] loaded {len(state)} tensors from {ckpt_path} "
                  f"(unexpected {len(unexpected)})", flush=True)

        for name, samples in (("val", self._val_samples), ("test", self._test_samples)):
            if not samples:
                continue
            t0 = time.perf_counter()
            m = self._eval_split_acc(samples)
            print(f"    {name:5s} : hit {m['hit']:.1f}% | EM {m['em']:.1f}% | "
                  f"F1 {m['f1']:.1f} | R@1 {m['r@1']:.1f}% | R@2 {m['r@2']:.1f}% | "
                  f"R@5 {m['r@5']:.1f}% | {time.perf_counter() - t0:.0f}s", flush=True)
            if self._wandb is not None:
                self._wandb.log({f"eval_only/{name}_{k}": v for k, v in m.items()})

    def _eval_split_acc(self, samples: list) -> dict:
        """Distribute samples across clients, run inference with on-the-fly retrieval.

        Returns {"hit", "em", "f1", "r@1", "r@2", "r@5"} in percent — hit is
        the legacy normalized substring containment; em/f1 are SQuAD/MuSiQue-style
        exact match and token-level F1; r@k is the passage recall of the PPR
        retrieval that feeds the LLM (gold evidence passages found in the
        top-k retrieved passages).
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
        recall_ks = (1, 2, 5)
        recall_totals = {k: 0.0 for k in recall_ks}
        recall_n = 0
        self.shared_model.eval()

        broadcast = bool(getattr(self.args, "eval_broadcast", False))
        with torch.no_grad():
            for cid, client in enumerate(self.clients):
                if broadcast:
                    shard = list(samples)
                else:
                    # Shard by the sample's true dataset index (idx % C), matching
                    # both training ownership and the per-client PPR maps. Position
                    # in the list only coincides with this when the split block
                    # starts at a multiple of C (true for 2wiki, false for
                    # musique/hotpotqa whose split boundaries aren't %3-aligned).
                    shard = [s for i, s in enumerate(samples)
                             if int(s.get("idx", i)) % n_clients == cid]
                if not shard:
                    continue
                c_hits = c_em = c_f1 = 0.0
                for s in shard:
                    rec = self._sample_retrieval_recall(client, s, recall_ks)
                    if rec:
                        recall_n += 1
                        for k in recall_ks:
                            recall_totals[k] += rec[k]
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
                            c_hits += 1
                        if exact_match(pred, label):
                            c_em += 1.0
                        c_f1 += token_f1(pred, label)
                hits += c_hits
                em_total += c_em
                f1_total += c_f1
                if broadcast:
                    c_n = max(len(shard), 1)
                    print(f"      [broadcast] client_{cid}: hit {100.0 * c_hits / c_n:.1f}% | "
                          f"EM {100.0 * c_em / c_n:.1f}% | F1 {100.0 * c_f1 / c_n:.1f} "
                          f"(n={len(shard)})", flush=True)

        n = max(len(samples) * (n_clients if broadcast else 1), 1)
        metrics = {
            "hit": 100.0 * hits / n,
            "em": 100.0 * em_total / n,
            "f1": 100.0 * f1_total / n,
        }
        for k in recall_ks:
            metrics[f"r@{k}"] = 100.0 * recall_totals[k] / max(recall_n, 1)
        return metrics

    @staticmethod
    def _sample_retrieval_recall(client, sample: dict, ks=(1, 2, 5)) -> dict:
        """Recall@k of the client's PPR passage retrieval for one sample.

        Gold = titles of the sample's evidence passages (record
        "retrieved_passages"); retrieved = rank-ordered PPR passage nodes from
        ppr_node_map.pt. Slots that map to no local node count as misses.
        Returns {} when recall cannot be computed (no map / no gold).
        """
        ppr_map = getattr(client, "_ppr_node_map", None)
        idx = sample.get("idx")
        if ppr_map is None or idx is None or idx >= ppr_map.shape[0]:
            return {}
        gold_titles = {
            norm_passage_title(p) for p in sample.get("retrieved_passages", []) or []
        }
        node_text = getattr(client.tri_graph, "node_text", None)
        if not gold_titles or not node_text:
            return {}
        N = len(node_text)
        retrieved_titles = [
            norm_passage_title(node_text[nid]) if 0 <= nid < N else None
            for nid in ppr_map[idx].tolist()
        ]
        return retrieval_recall_at_k(gold_titles, retrieved_titles, ks)

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
        # Accuracy per round
        if metrics["train_acc"] is not None:
            log["round/train_acc"] = metrics["train_acc"]
        if metrics["val_acc"] is not None:
            log["round/val_acc"] = metrics["val_acc"]
        if metrics["test_acc"] is not None:
            log["round/test_acc"] = metrics["test_acc"]
        for key in ("val_em", "val_f1", "test_em", "test_f1",
                    "val_r@1", "val_r@2", "val_r@5",
                    "test_r@1", "test_r@2", "test_r@5"):
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
        for m in round_metrics:
            if m["avg_loss"] is None:
                print(f"{m['round']:>4} | {'(boot)':>8} |" + f" {'—':>8} |" * n +
                      f" {'—':>7} | {'—':>7} | {'—':>7}")
                continue
            row = f"{m['round']:>4} | {m['avg_loss']:>8.4f} |"
            for cid in range(n):
                v = m["client_losses"].get(cid, float("nan"))
                row += f" {v:>8.4f} |"
            def _f(v):
                return f"{v:>6.1f}%" if v is not None else f"{'N/A':>7}"
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
        client = self.clients[0]
        n_clients = len(self.clients)
        # Only use samples that belong to client_0 in the FL split (idx % n_clients == 0)
        # to avoid RuntimeError from missing PPR anchors for other clients' samples.
        mini = [s for s in self._test_samples if s.get("idx", -1) % n_clients == 0][:n_show]
        if not mini:
            return
        mini = client._attach_evidence_graphs(mini)
        if client.synthetic_graph is not None:
            r = GlobalGraphRetriever(client.synthetic_graph, top_r=top_r)
            mini = client._attach_condensed_graphs(mini, r)
        batch = collate_fn(mini)
        self.shared_model.eval()
        with torch.no_grad():
            out = self.shared_model.inference(batch)
        n_show = len(mini)
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
            ("gnn_model_name",   "gt"),   ("gnn_model_name_c",  "gat"),
            ("gnn_num_layers",   4),      ("gnn_num_layers_c",  None),
            ("gnn_in_dim",       384),    ("gnn_in_dim_c",      None),
            ("gnn_hidden_dim",   384),    ("gnn_hidden_dim_c",  None),
            ("gnn_num_heads",    4),      ("gnn_num_heads_c",   None),
            ("gnn_dropout",      0.0),    ("dual_graph_mode",   "shared"),
            ("max_txt_len",      512),    ("max_new_tokens",    32),
            ("llm_frozen",       "True"),
        ):
            if not hasattr(self.args, attr):
                setattr(self.args, attr, default)

        try:
            model = load_model["dual_graph_llm"](args=self.args)
        except Exception as exc:
            print(f"[FedTrainer] Stage D disabled: failed to load DualGraphLLM — {exc}")
            return

        # LoRA adapters (created when llm_frozen == "False") must stay trainable;
        # everything else in the LLM stays frozen.
        for name, param in model.named_parameters():
            param.requires_grad = any(
                name.startswith(k) for k in
                ("graph_encoder", "projector", "condensed_encoder", "projector_c")
            ) or "lora_" in name

        if not hasattr(model, "hf_device_map"):
            model.to(self.device)
        self.shared_model = model

        for client in self.clients:
            client.set_shared_model(model)

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
        global_dir = self._processed_root() / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        data, _ = self._load_client_data(0)
        return data, str(global_dir)
