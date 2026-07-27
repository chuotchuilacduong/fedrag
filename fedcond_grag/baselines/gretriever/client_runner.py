"""Run G-Retriever independently on each federated client's own passage shard,
and evaluate every client against the *same* global benchmark question set
used by fedcond_grag's own Stage D eval.

Why this isn't a `_vendor/` package like `fedcond_grag/baselines/hipporag/`:
`fedcond_grag/model/graph_llm.py` and `gnn.py` already *are* a fork of
G-Retriever's own `src/model/graph_llm.py` / `gnn.py`
(https://github.com/XiaoxinHe/G-Retriever) — same GraphLLM class, same GCN/
GraphTransformer/GAT encoders, same BOS/EOS_USER/EOS prompt-splicing scheme,
`main.py`'s `_stage_d_train`/`_stage_d_infer` are an explicit port of its
`train.py`/`inference.py`. There is nothing left to vendor; this module just
*drives* that existing model in single-graph, non-federated mode:

  - One client only ever sees its own Tri-Graph (built by
    `scripts/build_client_pipeline.py`, Stage A) and its own local evidence
    retrieval (`FedCondQAClient._attach_evidence_graphs`, PPR over that
    client's own trigraph). No Stage B/C, no condensed/global graph, no
    cross-client weight sharing — each client gets a fresh, independent
    `GraphLLM` and trains/evaluates in total isolation.
  - This is the "traditional single-node graph-RAG under corpus
    fragmentation" baseline, exactly like `baselines/hipporag`: multi-hop
    questions whose evidence spans more than one client's shard are expected
    to fail for most clients, and that gap is what FedCondGraphRAG's
    federated retrieval (Stage C anchor-graph fusion) is meant to close.

Prerequisites (same as the main federated pipeline):
    python main.py preprocess --dataset <dataset> --num-clients <N>
    python scripts/build_fedcond_qa_dataset.py --dataset <dataset>
    python scripts/preprocess_fedcond_qa.py --dataset <dataset>
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARGS: dict[str, Any] = dict(
    data_root="processed",
    qa_data_root="dataset/fedcond_qa",
    llm_model_name="qwen2.5-1.5b",
    llm_model_path="",
    llm_frozen="True",
    llm_load_in_8bit=False,
    llm_load_in_4bit=False,
    gnn_model_name="gt",
    gnn_num_layers=4,
    gnn_in_dim=384,
    gnn_hidden_dim=384,
    gnn_num_heads=4,
    gnn_dropout=0.0,
    max_txt_len=512,
    max_new_tokens=32,
    eval_max_new_tokens=16,
    local_epochs=3,
    local_lr=1e-4,
    local_wd=0.05,
    local_batch_size=4,
    local_grad_clip=1.0,
    eval_batch_size=16,
    max_eval_samples=200,
    max_train_samples=0,   # 0 = use this client's full train shard
    top_r_passages=0,
    top_r_anchor=None,
)


def _build_args(dataset: str, num_clients: int, overrides: dict[str, Any]) -> Namespace:
    merged = {**DEFAULT_ARGS, **overrides, "dataset": dataset, "num_clients": num_clients}
    return Namespace(**merged)


def _load_client_trigraph(processed_root: Path, dataset: str, client_id: int) -> tuple[Data, str]:
    client_dir = processed_root / dataset / f"client_{client_id}"
    path = client_dir / "trigraph.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trigraph for client {client_id}: {path}. "
            f"Run `python main.py preprocess --dataset {dataset} --num-clients {client_id + 1}` first."
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


def _load_model(args: Namespace, device: torch.device):
    from fedcond_grag.model import load_model, llama_model_path

    llm_path = args.llm_model_path or llama_model_path.get(args.llm_model_name, "")
    if not llm_path:
        raise ValueError(f"Unknown llm_model_name={args.llm_model_name!r} and no llm_model_path set")
    args.llm_model_path = llm_path

    model = load_model["graph_llm"](args=args)
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(k) for k in ("graph_encoder", "projector"))
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    return model


def _eval_client(client, model, samples: list, args: Namespace) -> dict[str, float]:
    from fedcond_grag.utils.collate import collate_fn
    from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1

    if not samples:
        return {"hit": 0.0, "em": 0.0, "f1": 0.0}

    batch_size = int(getattr(args, "eval_batch_size", 16))
    hits = em_total = 0.0
    f1_total = 0.0
    model.eval()
    with torch.no_grad():
        shard = client._attach_evidence_graphs(list(samples))
        for i in range(0, len(shard), batch_size):
            mini = shard[i : i + batch_size]
            batch = collate_fn(mini)
            out = model.inference(batch)
            for pred, label in zip(out["pred"], out["label"]):
                if normalize(label) in normalize(pred):
                    hits += 1
                if exact_match(pred, label):
                    em_total += 1.0
                f1_total += token_f1(pred, label)

    n = len(samples)
    return {"hit": 100.0 * hits / n, "em": 100.0 * em_total / n, "f1": 100.0 * f1_total / n}


def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    **arg_overrides: Any,
) -> dict[str, Any]:
    """Train + evaluate a fresh, independent GraphLLM on one client's own
    Tri-Graph shard, evaluated against the full shared benchmark question
    set. No cross-client fusion of any kind."""
    from fedcond_grag.client.client import FedCondQAClient
    from fedcond_grag.dataloader import FedCondQADataset

    device = torch.device(device)
    args = _build_args(dataset, num_clients, arg_overrides)
    processed_root = _ROOT / args.data_root

    data, data_dir = _load_client_trigraph(processed_root, dataset, client_id)
    client = FedCondQAClient(args, client_id, data, data_dir, message_pool={}, device=device)
    num_docs = int(getattr(data, "num_nodes", data.x.size(0)))

    qa_dataset = FedCondQADataset(root=args.qa_data_root, top_r_passages=args.top_r_passages,
                                   top_r_anchor=args.top_r_anchor)
    idx_split = qa_dataset.get_idx_split()
    train_idx = [i for i in idx_split["train"] if i % num_clients == client_id]
    max_eval = int(args.max_eval_samples)
    val_samples = [qa_dataset[i] for i in idx_split["val"][:max_eval]]
    test_samples = [qa_dataset[i] for i in idx_split["test"][:max_eval]]
    max_train = int(getattr(args, "max_train_samples", 0))
    if max_train > 0:
        train_idx = train_idx[:max_train]
    train_samples = [qa_dataset[i] for i in train_idx]

    model = _load_model(args, device)
    client.set_shared_model(model)
    client.set_local_qa_data(train_samples)

    trainable = list(model.graph_encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.local_lr, weight_decay=args.local_wd, betas=(0.9, 0.95))
    client._optimizer = optimizer

    total_loss, total_steps = 0.0, 0
    for _ in range(int(args.local_epochs)):
        loss, steps = client.local_train()
        total_loss += loss * steps
        total_steps += steps
    avg_train_loss = total_loss / total_steps if total_steps else None

    val_metrics = _eval_client(client, model, val_samples, args)
    test_metrics = _eval_client(client, model, test_samples, args)

    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "num_docs": num_docs,
        "num_train_samples": len(train_samples),
        "num_questions": len(val_samples) + len(test_samples),
        "train_loss": avg_train_loss,
        "val_hit": val_metrics["hit"], "val_em": val_metrics["em"], "val_f1": val_metrics["f1"],
        "test_hit": test_metrics["hit"], "test_em": test_metrics["em"], "test_f1": test_metrics["f1"],
    }


def run_all_clients(
    dataset: str,
    num_clients: int,
    save_root: Path | str = _ROOT / "output" / "baselines" / "gretriever",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run every client's local G-Retriever baseline independently and write
    a summary JSON (per-client metrics + mean across clients)."""
    per_client = [
        run_client_baseline(dataset, client_id, num_clients, **kwargs)
        for client_id in range(num_clients)
    ]

    metric_keys = sorted({k for r in per_client for k, v in r.items() if isinstance(v, (int, float))})
    mean_metrics = {
        k: sum(r.get(k, 0.0) or 0.0 for r in per_client) / len(per_client)
        for k in metric_keys
    }

    summary = {"dataset": dataset, "num_clients": num_clients, "per_client": per_client, "mean": mean_metrics}

    out_path = Path(save_root) / dataset / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
