"""Per-client local GRAG baseline (https://github.com/HuieL/GRAG, MIT
licensed -- vendored under `_vendor/`, see `_vendor/VENDORED.md`).

Like `baselines/hipporag` and `baselines/gretriever`, this drives GRAG in
single-client, non-federated mode: one client only ever sees its own corpus
shard, evaluated against the full global benchmark question set, to measure
what a traditional single-node graph-RAG method loses when its corpus is
fragmented across clients that can't see each other's passages.

GRAG's own method expects a knowledge graph with labeled relation edges
(subject --relation--> object), e.g. from WebQSP/ExplaGraphs -- but this
project's benchmarks (hotpotqa/musique/2wikimultihop) are free text, not a
pre-built KG, and GRAG's own repo has no text-to-KG step. Two sources are
used to build that graph per client, in priority order:

  1. HippoRAG's cached OpenIE triples for this client
     (`baselines/hipporag`'s `openie_results_ner_*.json`, if that baseline
     has already been run for this dataset/client) -- real
     (subject, relation, object) triples extracted by an LLM, exactly what
     GRAG expects. Reused as-is, not recomputed: the OpenIE step is
     expensive (LLM call per passage) and HippoRAG needs the exact same
     thing, so there's no reason to pay for it twice.
  2. Otherwise, falls back to this project's own Tri-Graph
     (`processed/<dataset>/client_<m>/trigraph.pt`, same one `gretriever`
     uses) with generic edge labels ("mentions" for S-E edges, "contains"
     for P-E edges) standing in for real relations -- coarser, but free and
     always available once Stage A has run.

Either way, node/edge text is embedded with this project's own Stage A
encoder (`all-MiniLM-L6-v2`, 384-dim) rather than GRAG's own sbert wrapper
(1024-dim, `sentence-transformers/all-roberta-large-v1`) -- consistency
with the rest of the repo, and it's the embedding space the Tri-Graph
fallback is already in. `gnn_in_dim`/`gnn_hidden_dim` are set to 384 to
match, same as `baselines/gretriever` already does for the same reason.

Train/test split: by default (`qa_train_root` unset) this trains and
evaluates off a single `qa_data_root`'s own 80/10/10 split, same as before.
Pass `qa_train_root` to instead train on a SEPARATE qa cache's own train
split (built from a `<dataset>_train` pseudo-dataset via
`scripts/download_train_split.py` + `main.py preprocess --dataset
<dataset>_train`) while val/test still come from `qa_data_root` (built with
`main.py preprocess --qa-test-only`, so none of its questions were ever
seen during training) -- mirrors the same train/test separation
`main.py fl-train --eval-only --load-checkpoint` uses for fedrag's own
training, for a fair baseline comparison.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from src.model.graph_llm import GraphLLM  # noqa: E402  (vendored GRAG package)
from src.utils.collate import collate_fn as grag_collate_fn  # noqa: E402
from src.utils.graph_retrieval import retrive_on_graphs  # noqa: E402

from fedcond_grag.client.stage_a_trigraph.node_encoder import DEFAULT_MODEL, load_encoder  # noqa: E402
from fedcond_grag.model import llama_model_path  # noqa: E402
from fedcond_grag.utils.evaluate import exact_match, normalize, token_f1  # noqa: E402

# Must match baselines/hipporag/client_runner.py::DEFAULT_SAVE_ROOT.
HIPPO_OPENIE_ROOT = Path(os.environ.get("LOCALAPPDATA", "C:/")) / "fedrag_baselines" / "hipporag"

DEFAULT_SAVE_ROOT = _ROOT / "output" / "baselines" / "grag"

DEFAULT_ARGS: dict[str, Any] = dict(
    llm_model_name="qwen2.5-1.5b",
    llm_model_path="",
    llm_frozen="True",
    max_txt_len=512,
    max_new_tokens=32,
    gnn_model_name="gat",     # 'gat' | 'trans' | 'gcn' -- see _vendor/VENDORED.md, only 'gat' is dimension-safe here
    gnn_num_layers=4,
    gnn_in_dim=384,           # matches the Tri-Graph's all-MiniLM-L6-v2 embeddings, not GRAG's own 1024-dim default
    gnn_hidden_dim=384,
    gnn_num_heads=4,
    gnn_dropout=0.0,
    alignment_mlp_layers=3,
    distance_operator="euclidean",
    local_epochs=3,
    local_lr=1e-4,
    local_wd=0.05,
    local_batch_size=2,
    eval_batch_size=4,
    max_eval_samples=200,
    max_train_samples=0,      # 0 = use this client's full train shard
    retrieval_topk=10,        # top-k seed nodes (GRAG's own `topk` default)
    retrieval_k=2,            # ego-subgraph hop radius (GRAG's own `k` default)
    retrieval_topk_entity=5,  # nodes/edges kept per seed subgraph (GRAG's own `topk_entity` default)
    qa_data_root="dataset/fedcond_qa",
    # If set, train on this SEPARATE root's own train split instead of
    # qa_data_root's -- see module docstring on qa_train_root below.
    qa_train_root="",
    # If set, load a fedrag `--save-best` LoRA checkpoint as a frozen
    # backbone before training -- see fedcond_grag/baselines/checkpoint_utils.py.
    # Requires llm_frozen=False (adds the LoRA structure to load into).
    load_checkpoint="",
    # Which HippoRAG OpenIE cache to build the graph from; see _find_hippo_openie.
    openie_llm_name="",
    # Trade compute for GPU activation memory during local training.
    llm_gradient_checkpointing=False,
    # Append per-question {id, client_id, pred, label} JSONL rows (test split
    # only) here -- so F1 can be recomputed post-hoc split by subgroup (e.g.
    # local-vs-cross-client gold evidence).
    dump_predictions_path="",
)


def _build_args(overrides: dict[str, Any]) -> Namespace:
    return Namespace(**{**DEFAULT_ARGS, **overrides})


def _find_hippo_openie(dataset: str, client_id: int, openie_llm_name: str = "") -> dict | None:
    """Load the HippoRAG OpenIE cache for this client.

    `openie_llm_name` picks the exact file. It matters: that directory now
    holds one cache per model that has been run (plus .pre_proprag backups),
    and the original `glob(...)[0]` took whichever the filesystem listed first.
    That silently loaded a different model's extraction -- in one run it picked
    a triple-free cache while the intended one held 207k triples, and the only
    visible symptom was num_nodes coming out identical to a previous run.
    """
    client_dir = HIPPO_OPENIE_ROOT / dataset / f"client_{client_id}"
    if openie_llm_name:
        exact = client_dir / f"openie_results_ner_{openie_llm_name}.json"
        if not exact.exists():
            raise FileNotFoundError(
                f"No OpenIE cache for openie_llm_name={openie_llm_name!r} at {exact}. "
                f"Available: {sorted(p.name for p in client_dir.glob('openie_results_ner_*.json'))}"
            )
        return json.loads(exact.read_text(encoding="utf-8"))

    # No name given: stay deterministic and never pick a .pre_proprag backup.
    matches = sorted(p for p in client_dir.glob("openie_results_ner_*.json")
                     if not p.name.endswith(".pre_proprag.json"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"    [grag] WARNING: {len(matches)} OpenIE caches in {client_dir}; "
              f"using {matches[0].name}. Pass --openie_llm_name to choose explicitly.", flush=True)
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _build_graph_from_openie(openie: dict, encoder) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    """Build GRAG's expected (Data, textual_nodes, textual_edges) from
    HippoRAG's cached OpenIE triples: entities and passages become nodes,
    (subject, relation, object) triples and (passage, "mentions", entity)
    links become labeled edges."""
    node_ids: dict[str, int] = {}
    node_texts: list[str] = []

    def _node_id(text: str) -> int:
        if text not in node_ids:
            node_ids[text] = len(node_texts)
            node_texts.append(text)
        return node_ids[text]

    src_list, dst_list, rel_list = [], [], []
    for doc in openie["docs"]:
        passage_node = _node_id(doc["passage"])
        for entity in doc.get("extracted_entities", []):
            entity_node = _node_id(entity)
            src_list.append(passage_node); dst_list.append(entity_node); rel_list.append("mentions")
            src_list.append(entity_node); dst_list.append(passage_node); rel_list.append("mentions")
        for subj, rel, obj in doc.get("extracted_triples", []):
            subj_node, obj_node = _node_id(subj), _node_id(obj)
            src_list.append(subj_node); dst_list.append(obj_node); rel_list.append(rel)
            src_list.append(obj_node); dst_list.append(subj_node); rel_list.append(rel)

    return _finalize_graph(node_texts, src_list, dst_list, rel_list, encoder)


def _build_graph_from_trigraph(trigraph_payload: dict, encoder) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    """Fallback: build GRAG's expected format from this project's own
    Tri-Graph, with generic per-edge-type labels standing in for real
    relations (see module docstring)."""
    node_texts = list(trigraph_payload.get("node_text") or [])
    edge_index = trigraph_payload["edge_index"]
    edge_type = trigraph_payload["edge_type"]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    rel_list = ["mentions" if t == 0 else "contains" for t in edge_type.tolist()]

    return _finalize_graph(node_texts, src_list, dst_list, rel_list, encoder, node_embeds=trigraph_payload.get("x"))


def _finalize_graph(
    node_texts: list[str],
    src_list: list[int],
    dst_list: list[int],
    rel_list: list[str],
    encoder,
    node_embeds=None,
) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    x = node_embeds if node_embeds is not None else encoder.encode(
        node_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
    ).float().cpu()

    unique_rels = sorted(set(rel_list))
    rel_embeds = encoder.encode(
        unique_rels, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
    ).float().cpu()
    rel_to_idx = {r: i for i, r in enumerate(unique_rels)}
    edge_attr = rel_embeds[[rel_to_idx[r] for r in rel_list]]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

    nodes_df = pd.DataFrame({"node_id": range(len(node_texts)), "node_attr": node_texts})
    edges_df = pd.DataFrame({"src": src_list, "dst": dst_list, "edge_attr": rel_list})
    return graph, nodes_df, edges_df


def _load_client_graph(dataset: str, client_id: int, num_clients: int, encoder, openie_llm_name: str = "") -> tuple[Data, pd.DataFrame, pd.DataFrame, str]:
    openie = _find_hippo_openie(dataset, client_id, openie_llm_name)
    if openie is not None:
        graph, nodes_df, edges_df = _build_graph_from_openie(openie, encoder)
        return graph, nodes_df, edges_df, "hipporag_openie"

    trigraph_path = _ROOT / "processed" / dataset / f"client_{client_id}" / "trigraph.pt"
    if not trigraph_path.exists():
        raise FileNotFoundError(
            f"No HippoRAG OpenIE cache and no trigraph for client {client_id}: {trigraph_path}. "
            f"Run `python main.py preprocess --dataset {dataset} --num-clients {client_id + 1}` first."
        )
    payload = torch.load(trigraph_path, map_location="cpu", weights_only=False)
    graph, nodes_df, edges_df = _build_graph_from_trigraph(payload, encoder)
    return graph, nodes_df, edges_df, "trigraph_fallback"


def _retrieve_sample(graph: Data, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, q_emb: torch.Tensor, args: Namespace):
    # retrive_on_graphs's sims=None path scores every candidate seed node by
    # extracting its k-hop subgraph and re-embedding it (GRAG's own
    # query-conditioned relevance) -- workable for GRAG's original small
    # task KGs (WebQSP: dozens/hundreds of nodes) but O(graph.num_nodes)
    # subgraph extractions per question here, where the Tri-Graph fallback
    # has tens of thousands of nodes. Precompute a cheap node-embedding
    # cosine similarity and pass it as `sims` instead, which sends this
    # down the intended fast path (`find_topk_subgraph`) that only ever
    # extracts a subgraph for the `retrieval_topk` seed nodes.
    sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), graph.x)
    subgraph, desc = retrive_on_graphs(
        graph, q_emb, nodes_df, edges_df,
        topk=args.retrieval_topk, k=args.retrieval_k, topk_entity=args.retrieval_topk_entity,
        augment="none", sims=sims,
    )
    return subgraph, desc


def _eval_split(model: GraphLLM, batches, n: int, args: Namespace,
                 client_id: int = 0, dump_predictions_path: str | None = None) -> dict[str, float]:
    if not n:
        return {"hit": 0.0, "em": 0.0, "f1": 0.0}
    hits = em_total = f1_total = 0.0
    model.eval()
    dump_f = open(dump_predictions_path, "a") if dump_predictions_path else None
    try:
        with torch.no_grad():
            for batch in batches:
                out = model.inference(batch)
                ids = batch.get("id", [None] * len(out["pred"]))
                for qid, pred, label in zip(ids, out["pred"], out["label"]):
                    if normalize(label) in normalize(pred):
                        hits += 1
                    if exact_match(pred, label):
                        em_total += 1.0
                    f1_total += token_f1(pred, label)
                    if dump_f is not None:
                        dump_f.write(json.dumps({
                            "id": qid, "client_id": client_id, "pred": pred, "label": label,
                        }) + "\n")
    finally:
        if dump_f is not None:
            dump_f.close()
    return {"hit": 100.0 * hits / n, "em": 100.0 * em_total / n, "f1": 100.0 * f1_total / n}


def run_client_baseline(
    dataset: str,
    client_id: int,
    num_clients: int,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    **arg_overrides: Any,
) -> dict[str, Any]:
    """Build client `client_id`'s own graph (HippoRAG OpenIE triples if
    cached, else the Tri-Graph), train a fresh independent GRAG model on it,
    and evaluate against the full global question set for `dataset`."""
    from fedcond_grag.dataloader import FedCondQADataset

    device = torch.device(device)
    args = _build_args(arg_overrides)

    encoder = load_encoder(DEFAULT_MODEL)
    graph, nodes_df, edges_df, graph_source = _load_client_graph(
        dataset, client_id, num_clients, encoder, getattr(args, "openie_llm_name", ""))

    qa_dataset = FedCondQADataset(root=args.qa_data_root)
    idx_split = qa_dataset.get_idx_split()
    max_eval = int(args.max_eval_samples)
    max_train = int(args.max_train_samples)

    if args.qa_train_root:
        # Separate train/test qa caches -- see module docstring. Train
        # samples come from qa_train_root's own train split; val/test come
        # from qa_data_root's test split (val is empty when qa_data_root was
        # built with --qa-test-only, so test doubles as val here too).
        train_qa_dataset = FedCondQADataset(root=args.qa_train_root)
        train_idx = [i for i in train_qa_dataset.get_idx_split()["train"] if i % num_clients == client_id]
        if max_train > 0:
            train_idx = train_idx[:max_train]
        eval_idx = idx_split["test"][:max_eval]
        val_idx = test_idx = eval_idx
    else:
        train_qa_dataset = qa_dataset
        train_idx = [i for i in idx_split["train"] if i % num_clients == client_id]
        if max_train > 0:
            train_idx = train_idx[:max_train]
        val_idx = idx_split["val"][:max_eval]
        test_idx = idx_split["test"][:max_eval]

    def _prepare_one(i, source_dataset) -> dict:
        row = source_dataset[i]
        # Prefer the dataset's own precomputed embedding (raw question
        # text, all-MiniLM-L6-v2, built by scripts/build_fedcond_qa_dataset.py)
        # over re-embedding row["question"], which is already wrapped in
        # the "Question: ...\nAnswer: " prompt template.
        if "q_emb" in row:
            q_emb = row["q_emb"].float()
        else:
            q_emb = encoder.encode(
                [row["question"]], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
            ).float().cpu()[0]
        subgraph, desc = _retrieve_sample(graph, nodes_df, edges_df, q_emb, args)
        return {"id": row["id"], "question": row["question"], "label": row["label"],
                "desc": desc, "graph": subgraph}

    def _iter_batches(idx_list, source_dataset, batch_size: int):
        """Retrieve and collate one mini-batch at a time.

        Materialising every sample up front (the previous behaviour) is not
        survivable on this graph. A 2-hop ego subgraph over the
        PropRAG-derived KG averages ~2.4k nodes / ~16.7k edges, i.e. ~29MB
        of x/edge_attr per question -- so holding 1267 of them wants ~37GB
        of host RAM on a 15GB box, and the process gets to ~13GB and takes
        WSL down with it. Retrieval is only ~183ms/sample, so re-running it
        each epoch costs ~7min/client, which is noise next to 7B generation
        over 1000 questions. Peak memory becomes batch_size x 29MB instead.
        """
        for i in range(0, len(idx_list), batch_size):
            chunk = idx_list[i : i + batch_size]
            if chunk:
                yield grag_collate_fn([_prepare_one(j, source_dataset) for j in chunk])

    llm_path = args.llm_model_path or llama_model_path.get(args.llm_model_name, "")
    if not llm_path:
        raise ValueError(f"Unknown llm_model_name={args.llm_model_name!r} and no llm_model_path set")
    args.llm_model_path = llm_path

    model = GraphLLM(args)
    if args.load_checkpoint:
        from fedcond_grag.baselines.checkpoint_utils import load_lora_checkpoint
        load_lora_checkpoint(model.model, args.load_checkpoint)
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(k) for k in ("graph_encoder", "projector"))
    if not hasattr(model, "hf_device_map"):
        model.to(device)

    trainable = list(model.graph_encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.local_lr, weight_decay=args.local_wd, betas=(0.9, 0.95))

    batch_size = int(args.local_batch_size)
    eval_bs = int(args.eval_batch_size)
    total_loss, total_steps = 0.0, 0
    for epoch in range(int(args.local_epochs)):
        model.train()
        for batch in _iter_batches(train_idx, train_qa_dataset, batch_size):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
        print(f"    [grag] client {client_id} epoch {epoch + 1}/{args.local_epochs} "
              f"loss={total_loss / max(total_steps, 1):.4f}", flush=True)
    avg_train_loss = total_loss / total_steps if total_steps else None

    print(f"    [grag] client {client_id} evaluating {len(test_idx)} test questions", flush=True)
    test_metrics = _eval_split(model, _iter_batches(test_idx, qa_dataset, eval_bs), len(test_idx), args,
                                client_id=client_id, dump_predictions_path=(args.dump_predictions_path or None))
    if val_idx == test_idx:
        # The qa_train_root path sets val_idx = test_idx = the same held-out
        # test split, so evaluating both ran 1000 identical 7B generations
        # twice per client for byte-identical numbers. Reuse the result.
        val_metrics = test_metrics
    else:
        print(f"    [grag] client {client_id} evaluating {len(val_idx)} val questions", flush=True)
        val_metrics = _eval_split(model, _iter_batches(val_idx, qa_dataset, eval_bs), len(val_idx), args)

    return {
        "dataset": dataset,
        "client_id": client_id,
        "num_clients": num_clients,
        "graph_source": graph_source,
        "num_nodes": int(graph.num_nodes),
        "num_train_samples": len(train_idx),
        "num_questions": len(val_idx) + len(test_idx),
        "train_loss": avg_train_loss,
        "val_hit": val_metrics["hit"], "val_em": val_metrics["em"], "val_f1": val_metrics["f1"],
        "test_hit": test_metrics["hit"], "test_em": test_metrics["em"], "test_f1": test_metrics["f1"],
    }


def run_all_clients(dataset: str, num_clients: int, save_root: Path | str = DEFAULT_SAVE_ROOT, **kwargs: Any) -> dict[str, Any]:
    """Run every client's local GRAG baseline independently and write a
    summary JSON (per-client metrics + mean across clients)."""
    dump_path = kwargs.get("dump_predictions_path")
    if dump_path:
        Path(dump_path).write_text("")   # reset on new run
    per_client = []
    for client_id in range(num_clients):
        per_client.append(run_client_baseline(dataset, client_id, num_clients, **kwargs))
        # Hand the finished client's VRAM back before the next one builds its
        # own 7B. Dropping the last reference is not enough: the caching
        # allocator keeps those blocks reserved, so client N+1 allocates fresh
        # and the two briefly coexist. On a 16GB card that transient measured
        # 15975MiB where steady-state training sat at 15072, and musique died
        # there with `CUDA driver error: out of memory` inside the GAT conv.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
