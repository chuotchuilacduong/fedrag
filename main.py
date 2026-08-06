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
    """One-shot data pipeline: everything between raw benchmark download and
    `fl-train`. Idempotent — each step skips work whose output already exists
    (pass --force to rebuild).

        step 1  setup_datasets          → dataset/linearrag/<ds>/{chunks,questions}.json
        step 2  preprocess_data         → processed/<ds>/client_<m>/chunks.json
        step 3  build_client_pipeline   → trigraph.pt + condensed_graph.pt   (Stage A→B)
        step 4  build_fedcond_qa_dataset→ <qa-out-root>/{records.jsonl,split,q_embs.pt}
        step 5  preprocess_fedcond_qa   → processed/<ds>/client_<m>/ppr_node_map.pt
        step 6  build_passage_anchors   → passage_embs.pt (optional, --with-passage-anchors)
    """
    import json
    import runpy

    p = argparse.ArgumentParser(prog="fedcond_grag preprocess")
    p.add_argument("--dataset", default="hotpotqa",
                   choices=["hotpotqa", "2wikimultihop", "musique", "medical",
                            "hotpotqa_train", "2wikimultihop_train", "musique_train"])
    p.add_argument("--num-clients", dest="num_clients", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="Rebuild all artifacts even if they exist")
    p.add_argument("--skip-download", dest="skip_download", action="store_true",
                   help="Skip step 1 (use existing dataset/linearrag/<ds> files)")
    p.add_argument("--qa-out-root", dest="qa_out_root", default="dataset/fedcond_qa",
                   help="Output root for the QA cache (step 4). Use a per-dataset "
                        "dir (e.g. dataset/fedcond_qa_musique) when working with "
                        "multiple datasets, and pass the same path to fl-train "
                        "via --qa-data-root.")
    p.add_argument("--with-passage-anchors", dest="with_passage_anchors",
                   action="store_true",
                   help="Also build passage_embs.pt/passage_node_map.pt "
                        "(needed only for fl-train --top-r-passages > 0)")
    p.add_argument("--topology-method", dest="topology_method", default="knn")
    p.add_argument("--entity-ratio", dest="entity_ratio", type=float, default=0.05)
    p.add_argument("--top-k-passages", dest="top_k_passages", type=int, default=5,
                   help="PPR passages mapped per question per client (step 5)")
    p.add_argument("--qa-test-only", dest="qa_test_only", action="store_true",
                   help="Step 4: put all of --dataset's questions in the test split "
                        "(train/val empty) instead of the default 80/10/10. Use this "
                        "for a dataset held out purely for final eval -- e.g. after "
                        "training on a separate <dataset>_train pseudo-dataset -- so "
                        "none of these questions are ever used for training.")
    known = p.parse_args(argv)
    root = Path(__file__).resolve().parent

    def _step(n, title):
        print(f"\n{'='*70}\n[preprocess {n}/6] {title}\n{'='*70}", flush=True)

    # 1. Download + convert benchmark to LinearRAG format
    _step(1, f"setup_datasets — dataset/linearrag/{known.dataset}")
    linearrag_dir = root / "dataset" / "linearrag" / known.dataset
    if known.skip_download:
        print("  --skip-download: skipped")
    elif known.dataset == "medical":
        print("  'medical' is a private corpus — place chunks.json/questions.json "
              f"under {linearrag_dir} manually; skipping download")
    elif known.dataset.endswith("_train"):
        if not linearrag_dir.joinpath("chunks.json").exists():
            raise FileNotFoundError(
                f"{linearrag_dir} not found. Build it first: "
                f"python scripts/download_train_split.py --dataset {known.dataset[:-len('_train')]}"
            )
        print(f"  '{known.dataset}' built by scripts/download_train_split.py — skipping download")
    elif linearrag_dir.joinpath("chunks.json").exists() and not known.force:
        print("  chunks.json exists — skipped (use --force to re-download)")
    else:
        sys.argv = ["setup_datasets.py", "--dataset", known.dataset] + (
            ["--force"] if known.force else [])
        runpy.run_module("scripts.setup_datasets", run_name="__main__")

    # 2. Partition corpus into per-client shards
    _step(2, f"preprocess_data — {known.num_clients} client shards")
    sys.argv = ["preprocess_data.py", "--dataset", known.dataset,
                "--num_clients", str(known.num_clients)]
    runpy.run_module("scripts.preprocess_data", run_name="__main__")

    # 3. Stage A (trigraph) → Stage B (condensed anchor) per client
    _step(3, "build_client_pipeline — Stage A→B per client")
    build_argv = ["build_client_pipeline.py", "--dataset", known.dataset]
    if known.force:
        build_argv.append("--force")
    if known.topology_method != "knn":
        build_argv += ["--topology-method", known.topology_method]
    if known.entity_ratio != 0.05:
        build_argv += ["--entity-ratio", str(known.entity_ratio)]
    sys.argv = build_argv
    runpy.run_module("scripts.build_client_pipeline", run_name="__main__")

    # 4. QA records + splits + question embeddings
    _step(4, f"build_fedcond_qa_dataset — {known.qa_out_root}")
    qa_root = root / known.qa_out_root
    cached_dataset = None
    meta_path = qa_root / "_meta.json"
    if meta_path.exists():
        try:
            cached_dataset = json.loads(meta_path.read_text()).get("dataset")
        except Exception:
            cached_dataset = None
    if qa_root.joinpath("records.jsonl").exists() and not known.force and cached_dataset == known.dataset:
        print("  records.jsonl exists (matches --dataset) — skipped (use --force to rebuild)")
    else:
        if qa_root.joinpath("records.jsonl").exists() and cached_dataset != known.dataset:
            print(f"  cached QA data at {known.qa_out_root} belongs to dataset "
                  f"'{cached_dataset}', not '{known.dataset}' — rebuilding "
                  "(pass --qa-out-root to keep multiple datasets' QA caches around at once)")
        sys.argv = ["build_fedcond_qa_dataset.py", "--dataset", known.dataset,
                    "--out-root", known.qa_out_root] + (
            ["--test-only"] if known.qa_test_only else [])
        runpy.run_module("scripts.build_fedcond_qa_dataset", run_name="__main__")

    # 5. Per-client PPR passage node maps (evidence retrieval at train time)
    _step(5, "preprocess_fedcond_qa — per-client ppr_node_map.pt")
    processed = root / "processed" / known.dataset
    have_maps = all(
        (processed / f"client_{c}" / "ppr_node_map.pt").exists()
        for c in range(known.num_clients)
    )
    if have_maps and not known.force:
        print("  all ppr_node_map.pt exist — skipped (use --force to rebuild)")
    else:
        sys.argv = ["preprocess_fedcond_qa.py", "--dataset", known.dataset,
                    "--top_k_passages", str(known.top_k_passages)]
        runpy.run_module("scripts.preprocess_fedcond_qa", run_name="__main__")

    # 6. Optional: passage anchors for --top-r-passages re-ranked desc
    _step(6, "build_passage_anchors (optional)")
    if not known.with_passage_anchors:
        print("  skipped (enable with --with-passage-anchors)")
    else:
        sys.argv = ["build_passage_anchors.py",
                    "--qa-root", known.qa_out_root,
                    "--processed-root", f"processed/{known.dataset}",
                    "--num-clients", str(known.num_clients)]
        runpy.run_module("scripts.build_passage_anchors", run_name="__main__")

    print(f"\npreprocess done — ready for:\n"
          f"  python main.py fl-train --dataset {known.dataset} "
          f"--num-clients {known.num_clients} --qa-data-root {known.qa_out_root} ...")
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
    p.add_argument("--server-condense-iters", dest="server_condense_iters", type=int, default=500)
    p.add_argument("--server-stage-c-mode", dest="server_stage_c_mode", default="fedrag",
                   choices=["fedrag", "gradient_match", "repr_align", "both"],
                   help="Synthetic-graph refinement algorithm. 'fedrag' (default) = paper "
                        "Algorithm 1: Phase-0 repr-align init, then client-side "
                        "query-conditioned synthetic-memory adaptation with server delta "
                        "aggregation. Legacy modes refine the synthetic graph server-side "
                        "against client anchors every round.")
    p.add_argument("--syn-mem-steps", dest="syn_mem_steps", type=int, default=200,
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
    p.add_argument("--condense-refine-iters", dest="condense_refine_iters", type=int, default=1000,
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
    p.add_argument("--save-best", dest="save_best", action="store_true",
                   help="Save whatever this run actually trained -- the LoRA adapter "
                        "(--llm-frozen False) and/or the FedAvg'd graph_encoder/projector/"
                        "condensed_encoder/projector_c (the common --llm-frozen True case) "
                        "-- whenever val hit%% improves. Off by default -- pass this to opt "
                        "in, e.g. so the trained model can be reused as the base for "
                        "--eval-only against a held-out test set, or as a frozen backbone "
                        "for other baseline RAG methods.")
    p.add_argument("--save-best-path", dest="save_best_path", default=None,
                   help="Where to write the checkpoint from --save-best. Defaults to "
                        "checkpoints/<dataset>/<lora-agg-method>/best.pt when --save-best "
                        "is set without an explicit path.")
    p.add_argument("--load-checkpoint", dest="load_checkpoint", default=None,
                   help="Load a --save-best checkpoint into the model before training/eval "
                        "-- whichever of LoRA / graph_encoder / projector / condensed_encoder"
                        " / projector_c it contains. Must be paired with the same config "
                        "used to produce it (--llm-frozen False + --lora-rank/--lora-alpha/"
                        "--lora-target-modules for the LoRA part, --dual-graph-mode/--gnn-* "
                        "dims for the graph part), or load_state_dict will silently attach "
                        "nothing (mismatched key names/shapes).")
    p.add_argument("--eval-only", dest="eval_only", action="store_true",
                   help="Skip the training rounds and server aggregation entirely -- just "
                        "run the (optionally --load-checkpoint-restored) model once against "
                        "--dataset's test split. Pair with a --dataset preprocessed via "
                        "'main.py preprocess --qa-test-only' so its whole question set is "
                        "the test split, and --max-eval-samples set above that count.")
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

    if args.save_best:
        if not args.save_best_path:
            args.save_best_path = f"checkpoints/{args.dataset}/{args.lora_agg_method}/best.pt"
    else:
        args.save_best_path = None

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
