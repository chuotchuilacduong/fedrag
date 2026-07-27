"""FedCondGraphRAG unified entry point.

Subcommands:
    preprocess   Build per-client Stage A→B→C artifacts (chunks, trigraph, …).
    fl-train     Run the federated round loop (Stage C aggregation).
    train        Centralized Stage D fit (dual-prompting DualGraphLLM).
    infer        Run Stage D inference and compute eval metrics.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedcond_grag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preprocess", help="Build Stage A→C artifacts per client").set_defaults(func=_run_preprocess)
    sub.add_parser("fl-train",   help="Run the federated round loop").set_defaults(func=_run_fl_train)
    sub.add_parser("train",      help="Stage D centralized fit").set_defaults(func=_run_train)
    sub.add_parser("infer",      help="Stage D inference + metrics").set_defaults(func=_run_infer)

    # First parse the subcommand, then hand the rest to the dispatcher so each
    # subcommand can layer on its own argparse without colliding.
    parsed, rest = parser.parse_known_args(argv)
    return parsed.func(rest) or 0


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def _run_preprocess(argv: list[str]) -> int:
    import runpy

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--num-clients", dest="num_clients", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--topology-method", dest="topology_method", default="knn")
    p.add_argument("--entity-ratio", dest="entity_ratio", type=float, default=0.05)
    known, _ = p.parse_known_args(argv)

    # Step 0: split raw LinearRAG data into per-client chunks (idempotent)
    import scripts.preprocess_data as _pd  # noqa: F401  ensure on sys.path
    sys.argv = [
        "preprocess_data.py",
        "--dataset", known.dataset,
        "--num_clients", str(known.num_clients),
    ]
    runpy.run_module("scripts.preprocess_data", run_name="__main__")

    # Step 1-3: Stage A (trigraph) → B (condense) → C (synthetic) per client
    import scripts.build_client_pipeline as _bp  # noqa: F401
    build_argv = ["build_client_pipeline.py", "--dataset", known.dataset]
    if known.force:
        build_argv.append("--force")
    if known.topology_method != "knn":
        build_argv += ["--topology-method", known.topology_method]
    if known.entity_ratio != 0.05:
        build_argv += ["--entity-ratio", str(known.entity_ratio)]
    sys.argv = build_argv
    runpy.run_module("scripts.build_client_pipeline", run_name="__main__")
    return 0


# ---------------------------------------------------------------------------
# fl-train
# ---------------------------------------------------------------------------

def _run_fl_train(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fedcond_grag fl-train")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--num-clients", dest="num_clients", type=int, default=2)
    p.add_argument("--num-rounds", dest="num_rounds", type=int, default=1)
    p.add_argument("--client-frac", dest="client_frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--data-root", dest="data_root", default="processed")
    p.add_argument("--num-global-syn-nodes", dest="num_global_syn_nodes", type=int, default=128)
    p.add_argument("--server-condense-iters", dest="server_condense_iters", type=int, default=50)
    p.add_argument("--server-stage-c-mode", dest="server_stage_c_mode", default="fedrag",
                   choices=["fedrag", "gradient_match", "repr_align", "both"],
                   help="Synthetic-graph refinement algorithm. 'fedrag' (default) = paper "
                        "Algorithm 1: Phase-0 repr-align init, then client-side "
                        "query-conditioned synthetic-memory adaptation with server delta "
                        "aggregation. Legacy modes refine the synthetic graph server-side "
                        "against client anchors every round.")
    p.add_argument("--syn-mem-steps", dest="syn_mem_steps", type=int, default=5,
                   help="K_mem: client-side synthetic-memory adaptation steps per round "
                        "(fedrag mode; 0 disables adaptation → server only re-broadcasts)")
    p.add_argument("--syn-mem-lr", dest="syn_mem_lr", type=float, default=1e-3,
                   help="η_syn: client-side synthetic-memory Adam LR (fedrag mode)")
    p.add_argument("--syn-mem-batch-size", dest="syn_mem_batch_size", type=int, default=0,
                   help="Mini-batch size for memory adaptation (0 = --local-batch-size)")
    p.add_argument("--syn-soft-tau", dest="syn_soft_tau", type=float, default=0.1,
                   help="τ: temperature for differentiable soft synthetic retrieval")
    p.add_argument("--lambda-gm", dest="lambda_gm", type=float, default=0.1,
                   help="λ_gm: gradient-matching weight in L_mem (0 skips the extra "
                        "local-evidence forward pass per adaptation step)")
    p.add_argument("--lambda-align-mem", dest="lambda_align_mem", type=float, default=0.1,
                   help="λ_align: client-side condensed↔synthetic alignment weight in L_mem")
    p.add_argument("--lambda-reg-mem", dest="lambda_reg_mem", type=float, default=0.01,
                   help="λ_reg: client-side synthetic-graph regularization weight in L_mem")
    p.add_argument("--eta-agg", dest="eta_agg", type=float, default=1.0,
                   help="η_agg: server step size applied to the aggregated memory delta")
    p.add_argument("--eta-reg", dest="eta_reg", type=float, default=1e-2,
                   help="η_reg: server L_reg gradient step size after delta aggregation")
    p.add_argument("--server-reg-steps", dest="server_reg_steps", type=int, default=1,
                   help="Server-side L_reg gradient steps per round (fedrag mode)")
    p.add_argument("--condense-refine-iters", dest="condense_refine_iters", type=int, default=100,
                   help="Stage B refinement steps minimizing L_cond = L_ret(KL) + "
                        "λ_rep·L_rep + λ_div·L_div before uploading the anchor graph "
                        "(paper B.3.5; 0 disables → constructive init only)")
    p.add_argument("--condense-refine-lr", dest="condense_refine_lr", type=float, default=1e-2,
                   help="Adam LR for Stage B condensed-feature refinement")
    p.add_argument("--stage-b-lambda-rep", dest="stage_b_lambda_rep", type=float, default=1.0,
                   help="λ_rep: representation-preservation weight in L_cond")
    p.add_argument("--stage-b-lambda-div", dest="stage_b_lambda_div", type=float, default=0.1,
                   help="λ_div: condensed-node diversity weight in L_cond")
    p.add_argument("--stage-b-div-margin", dest="stage_b_div_margin", type=float, default=0.5,
                   help="δ: cosine margin of the diversity hinge in L_div")
    p.add_argument("--stage-b-tau-ret", dest="stage_b_tau_ret", type=float, default=0.1,
                   help="Retrieval softmax temperature for L_ret distributions")
    p.add_argument("--stage-b-tau-cov", dest="stage_b_tau_cov", type=float, default=0.1,
                   help="τ_a: soft coverage map temperature lifting condensed→full passages")
    p.add_argument("--stage-b-max-queries", dest="stage_b_max_queries", type=int, default=256,
                   help="Cap on local training questions used as L_ret queries")
    p.add_argument("--hid-dim", dest="hid_dim", type=int, default=64)
    p.add_argument("--num-layers", dest="num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--qa-data-root", dest="qa_data_root", default="dataset/fedcond_qa")
    p.add_argument("--llm-model-name", dest="llm_model_name", default="7b")
    p.add_argument("--llm-model-path", dest="llm_model_path", default="")
    p.add_argument("--llm-load-in-8bit", dest="llm_load_in_8bit", action="store_true", default=False)
    p.add_argument("--llm-load-in-4bit", dest="llm_load_in_4bit", action="store_true", default=False)
    p.add_argument("--llm-gpu-max-memory-gib", dest="llm_gpu_max_memory_gib", type=float, default=None,
                   help="Cap GPU VRAM device_map='auto' is allowed to use for LLM weights (GiB). "
                        "Default: auto-detected from the GPU minus ~1.5GiB headroom.")
    p.add_argument("--llm-cpu-max-memory-gib", dest="llm_cpu_max_memory_gib", type=float, default=None,
                   help="Enable CPU RAM overflow for LLM weights that don't fit in --llm-gpu-max-memory-gib "
                        "(e.g. a 7B model on an 8GB GPU) -- sets accelerate's max_memory['cpu'] budget and the "
                        "bnb fp32-cpu-offload flag. Off by default (GPU-only; load fails loudly if it doesn't fit). "
                        "CPU-resident layers run in fp32 on the CPU (bnb 4-bit/8-bit kernels are CUDA-only), so "
                        "this trades a lot of speed for headroom -- expect most of a 7B model's layers to land "
                        "here on an 8GB card, making each step dramatically slower than GPU-only.")
    p.add_argument("--llm-frozen", dest="llm_frozen", default="True",
                   help="'True' (default) keeps the LLM frozen -- only the GNN encoder/projector are "
                        "federated. Set to 'False' to also LoRA-fine-tune the LLM across clients, "
                        "aggregated each round via --lora-agg-method.")
    p.add_argument("--lora-rank", dest="lora_rank", type=int, default=8)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", dest="lora_target_modules", default=None,
                   help="Comma-separated module names to wrap with LoRA (default: q_proj,v_proj)")
    p.add_argument("--lora-agg-method", dest="lora_agg_method", default="fedit",
                   choices=["fedit", "flexlora", "rolora", "flora"],
                   help="Server-side LoRA aggregation strategy, only used when --llm-frozen False. "
                        "fedit=plain weighted avg (baseline), flexlora=stack+SVD-compress, "
                        "rolora=alternate A/B by round parity, flora=stack+merge into base weight "
                        "(requires --llm-frozen False and no 4-bit/8-bit quantization).")
    p.add_argument("--lora-agg-scale", dest="lora_agg_scale", type=float, default=2.0,
                   help="FlexLoRA's delta_W redistribution scale factor (paper's 's')")
    p.add_argument("--llm-gradient-checkpointing", dest="llm_gradient_checkpointing",
                   action="store_true", default=False,
                   help="Enable gradient checkpointing (~4x less activation memory, allows larger batch)")
    p.add_argument("--eval-max-new-tokens", dest="eval_max_new_tokens", type=int, default=16,
                   help="Max tokens to generate during eval/inference (default 16; hotpotqa answers are short)")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=1, help="Run accuracy eval every N rounds (default: every round)")
    p.add_argument("--gnn-model-name", dest="gnn_model_name", default="gt")
    p.add_argument("--gnn-model-name-c", dest="gnn_model_name_c", default="gcn")
    p.add_argument("--gnn-in-dim", dest="gnn_in_dim", type=int, default=384)
    p.add_argument("--gnn-hidden-dim", dest="gnn_hidden_dim", type=int, default=384)
    p.add_argument("--gnn-num-layers", dest="gnn_num_layers", type=int, default=4)
    p.add_argument("--gnn-num-heads", dest="gnn_num_heads", type=int, default=4)
    p.add_argument("--gnn-dropout", dest="gnn_dropout", type=float, default=0.0)
    p.add_argument("--local-epochs", dest="local_epochs", type=int, default=3)
    p.add_argument("--local-lr", dest="local_lr", type=float, default=1e-4)
    p.add_argument("--local-wd", dest="local_wd", type=float, default=0.05)
    p.add_argument("--local-batch-size", dest="local_batch_size", type=int, default=4)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=16,
                   help="Batch size for eval inference (no backprop, can be much larger than train batch)")
    p.add_argument("--local-grad-clip", dest="local_grad_clip", type=float, default=1.0)
    p.add_argument("--retrieval-top-r", dest="retrieval_top_r", type=int, default=16)
    p.add_argument("--max-txt-len", dest="max_txt_len", type=int, default=512)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64)
    p.add_argument("--max-train-per-client", dest="max_train_per_client", type=int, default=0,
                   help="Cap training samples per client per round (0 = all)")
    p.add_argument("--max-eval-samples", dest="max_eval_samples", type=int, default=200,
                   help="Max eval samples for per-round accuracy (default 200)")
    p.add_argument("--dual-graph-mode", dest="dual_graph_mode", default="both",
                   choices=["both", "dual", "evidence_only", "condensed_only",
                            "random_condensed", "none", "text_only", "shared", "no_synthetic"],
                   help="Graph soft-prompt mode. 'shared'=one shared GNN encoder; 'no_synthetic'=ignore server graph, both slots use evidence graph.")
    p.add_argument("--wandb-run-name", dest="wandb_run_name", default=None,
                   help="WandB run display name. Auto-generated from dual_graph_mode if omitted.")
    p.add_argument("--wandb-tags", dest="wandb_tags", nargs="+", default=None,
                   help="WandB tags for filtering (e.g. --wandb-tags ablation shared).")
    p.add_argument("--wandb-group", dest="wandb_group", default=None,
                   help="WandB group name for grouping related runs.")
    p.add_argument("--top-r-passages", dest="top_r_passages", type=int, default=0,
                   help="If >0, re-rank each record's retrieved_passages by q_emb similarity "
                        "and keep the top-r as 'desc'. Also exposes anchor_passage_nodes for "
                        "graph subgraph anchoring. Requires passage_embs.pt + passage_node_map.pt "
                        "under --qa-data-root (build via scripts/build_passage_anchors.py).")
    p.add_argument("--top-r-anchor", dest="top_r_anchor", type=int, default=None,
                   help="Number of re-ranked passages used as graph subgraph anchors "
                        "(default = --top-r-passages). Useful for using all 10 passages as "
                        "text but anchoring the evidence graph on the top-3 only.")
    args = p.parse_args(argv)

    from fedcond_grag.trainer import FedTrainer
    from fedcond_grag.utils.seed import seed_everything

    if args.seed != 0:
        seed_everything(args.seed)
    FedTrainer(args).train()
    return 0


# ---------------------------------------------------------------------------
# train (Stage D)
# ---------------------------------------------------------------------------

def _run_train(argv: list[str]) -> int:
    from fedcond_grag.config import parse_args_llama
    sys.argv = ["train", *argv]
    args = parse_args_llama()
    _stage_d_train(args)
    return 0


def _run_infer(argv: list[str]) -> int:
    from fedcond_grag.config import parse_args_llama
    sys.argv = ["infer", *argv]
    args = parse_args_llama()
    _stage_d_infer(args)
    return 0


def _stage_d_train(args) -> None:
    """Port of legacy train.py main(); unchanged numerics."""
    import json
    import os

    import pandas as pd
    import torch
    import wandb
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from fedcond_grag.dataloader import load_dataset
    from fedcond_grag.model import llama_model_path, load_model
    from fedcond_grag.utils.ckpt import _reload_best_model, _save_checkpoint
    from fedcond_grag.utils.collate import collate_fn
    from fedcond_grag.utils.evaluate import eval_funcs
    from fedcond_grag.utils.lr_schedule import adjust_learning_rate
    from fedcond_grag.utils.seed import seed_everything

    seed = args.seed
    wandb.init(project=f"{args.project}", name=f"{args.dataset}_{args.model_name}_seed{seed}", config=args)
    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in idx_split["train"]]
    val_dataset   = [dataset[i] for i in idx_split["val"]]
    test_dataset  = [dataset[i] for i in idx_split["test"]]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,      drop_last=True,  pin_memory=True, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,      drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    args.llm_model_path = getattr(args, "llm_model_path", "") or llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": params, "lr": args.lr, "weight_decay": args.wd}], betas=(0.9, 0.95))
    trainable, total = model.print_trainable_params()
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total}")

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0.0, 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)
            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
            optimizer.step()
            epoch_loss += loss.item()
            accum_loss += loss.item()
            if (step + 1) % args.grad_steps == 0:
                wandb.log({"Lr": optimizer.param_groups[0]["lr"]})
                wandb.log({"Accum Loss": accum_loss / args.grad_steps})
                accum_loss = 0.0
            progress_bar.update(1)
        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({"Train Loss (Epoch Mean)": epoch_loss / len(train_loader)})

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model(batch).item()
            val_loss /= len(val_loader)
        print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
        wandb.log({"Val Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch
        print(f"Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    path = (
        f"{args.output_dir}/{args.dataset}/"
        f"model_name_{args.model_name}_llm_model_name_{args.llm_model_name}"
        f"_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}"
        f"_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}"
        f"_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv"
    )
    print(f"path: {path}")

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for batch in test_loader:
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    acc = eval_funcs[args.dataset](path)
    print(f"Test Acc {acc}")
    wandb.log({"Test Acc": acc})


def _stage_d_infer(args) -> None:
    """Port of legacy inference.py main(); unchanged numerics."""
    import json
    import os

    import pandas as pd
    import torch
    import wandb
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from fedcond_grag.dataloader import load_dataset
    from fedcond_grag.model import llama_model_path, load_model
    from fedcond_grag.utils.collate import collate_fn
    from fedcond_grag.utils.evaluate import eval_funcs
    from fedcond_grag.utils.seed import seed_everything

    seed = args.seed
    wandb.init(project=f"{args.project}", name=f"{args.dataset}_{args.model_name}_seed{seed}", config=args)
    seed_everything(seed=seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    test_dataset = [dataset[i] for i in idx_split["test"]]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    args.llm_model_path = getattr(args, "llm_model_path", "") or llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph=dataset.graph, graph_type=dataset.graph_type, args=args)

    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    path = (
        f"{args.output_dir}/{args.dataset}/"
        f"model_name_{args.model_name}_llm_model_name_{args.llm_model_name}"
        f"_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}"
        f"_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}"
        f"_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv"
    )
    print(f"path: {path}")

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for batch in test_loader:
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    acc = eval_funcs[args.dataset](path)
    print(f"Test Acc {acc}")
    wandb.log({"Test Acc": acc})


if __name__ == "__main__":
    rc = main()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    gc.collect()
    sys.exit(rc)
