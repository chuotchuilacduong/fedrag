"""Build trigraph + condensed graph + synthetic graph for all clients.

Runs Stage A → B → C for every client that is missing any of:
  processed/{dataset}/client_{m}/trigraph.pt
  processed/{dataset}/client_{m}/condensed_graph.pt
  processed/{dataset}/client_{m}/synthetic_graph.pt

Usage (from project root, fedcond env):
    python scripts/build_client_pipeline.py --dataset hotpotqa
    python scripts/build_client_pipeline.py --dataset hotpotqa --clients 1 2 3 4
    python scripts/build_client_pipeline.py --dataset hotpotqa --force

Skips any stage whose output already exists (unless --force).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_a_trigraph import build_trigraph_for_client, save_trigraph
from fedcond_grag.client.stage_a_trigraph.node_encoder import load_encoder
from fedcond_grag.client.stage_b_condense import (
    ClientCondensationConfig,
    build_text_bank,
    condense_client_graph,
    load_text_bank,
    save_text_bank,
)
from fedcond_grag.client.stage_b_condense.text_bank import load_frozen_encoder
from fedcond_grag.server.server import FedCondQAServer

PROCESSED_ROOT = _ROOT / "processed"
ENCODER_MODEL = "all-MiniLM-L6-v2"
ENCODER_DIM = 384


# ---------------------------------------------------------------------------
# Stage A — Tri-Graph
# ---------------------------------------------------------------------------

def build_trigraph(client_dir: Path, dataset: str, force: bool) -> Data:
    out = client_dir / "trigraph.pt"
    if out.exists() and not force:
        print(f"    [A] trigraph.pt exists — skipping")
        payload = torch.load(out, map_location="cpu", weights_only=False)
        return Data(
            x=payload["x"],
            edge_index=payload["edge_index"],
            edge_type=payload["edge_type"],
            node_type=payload["node_type"],
            node_text=payload.get("node_text", []),
        )

    chunks_path = client_dir / "chunks.json"
    with chunks_path.open() as f:
        chunks = json.load(f)
    print(f"    [A] Indexing {len(chunks)} chunks with LinearRAG...")
    t = time.time()
    encoder = load_encoder(ENCODER_MODEL)
    graph = build_trigraph_for_client(
        chunks,
        working_dir=client_dir / "linearrag_cache",
        dataset_name=dataset,
        encoder=encoder,
    )
    save_trigraph(graph, out)
    ne = (graph.node_type == 0).sum().item()
    ns = (graph.node_type == 1).sum().item()
    np_ = (graph.node_type == 2).sum().item()
    print(f"    [A] Done in {time.time()-t:.0f}s — {graph.x.shape[0]} nodes "
          f"(ent={ne}, sen={ns}, pas={np_}), {graph.edge_index.shape[1]} edges")
    return graph


# ---------------------------------------------------------------------------
# Stage B — Client condensation
# ---------------------------------------------------------------------------

def build_condensed(client_dir: Path, graph: Data, force: bool, topology_method: str = "knn", entity_ratio: float = 0.05) -> dict:
    out = client_dir / "condensed_graph.pt"
    if out.exists() and not force:
        print(f"    [B] condensed_graph.pt exists — skipping")
        return torch.load(out, map_location="cpu", weights_only=False)

    bank_path = client_dir / "text_bank.pt"
    if bank_path.exists() and not force:
        print(f"    [B] Loading cached text_bank.pt...")
        bank = load_text_bank(bank_path)
    else:
        print(f"    [B] Building text bank ({len(graph.node_text)} nodes)...")
        t = time.time()
        encoder = load_frozen_encoder(ENCODER_MODEL, dim=ENCODER_DIM)
        bank = build_text_bank(
            graph.node_text,
            encoder=encoder,
            encoder_name=ENCODER_MODEL,
            dim=ENCODER_DIM,
        )
        save_text_bank(bank, bank_path)
        print(f"    [B] Text bank built in {time.time()-t:.0f}s")

    from fedcond_grag.client.stage_b_condense.motif_core_selector import MotifSelectorConfig
    n_ent = int((graph.node_type == 0).sum().item())
    k_ent = max(1, int(entity_ratio * n_ent))
    print(f"    [B] Running client condensation ({topology_method} topology, entity_ratio={entity_ratio:.3f}, core_entities={k_ent}/{n_ent})...")
    t = time.time()
    motif_cfg = MotifSelectorConfig(entity_ratio=entity_ratio)
    cfg = ClientCondensationConfig(topology_method=topology_method, knn_k=8, motif=motif_cfg)
    condensed, _ = condense_client_graph(
        graph,
        text_bank=bank,
        graph_embeddings=graph.x,
        config=cfg,
        return_artifacts=True,
    )
    payload = {
        "x": condensed.x.detach().cpu(),
        "edge_index": condensed.edge_index.detach().cpu(),
        "edge_weight": condensed.edge_weight.detach().cpu(),
        "node_type": condensed.node_type.detach().cpu(),
    }
    torch.save(payload, out)
    print(f"    [B] Done in {time.time()-t:.0f}s — K={condensed.x.shape[0]} anchors")
    return payload


# ---------------------------------------------------------------------------
# Stage C — Server condensation
# ---------------------------------------------------------------------------

def _make_server_args(log_dir: str, num_syn_nodes: int = 128) -> Namespace:
    return Namespace(
        fl_algorithm="fedcond_qa",
        task="condensation_qa",
        dataset=["hotpotqa"],
        model=["gcn"],
        hid_dim=64,
        num_layers=2,
        dropout=0.0,
        lr=0.01,
        weight_decay=0.0,
        optim="adam",
        metrics=["accuracy"],
        num_clients=1,
        num_global_syn_nodes=num_syn_nodes,
        server_condense_iters=50,
        condense_iters=10,
        local_epochs=0,
        lr_feat=1e-2,
        lr_adj=1e-2,
        pge_hidden=64,
        pge_topk=8,
        type_emb_dim=8,
        surrogate_type_weight=1.0,
        surrogate_link_weight=0.5,
        surrogate_align_weight=0.1,
        match_norm_weight=0.0,
        condense_refresh_every=10,
        preserve_sep_topology=True,
        use_cuda=False,
        gpuid=0,
        dp_mech="no_dp",
        dp_epsilon=0.0,
        dp_delta=1e-5,
        dp_clip=1.0,
        train_val_test="default_split",
        processing="raw",
        processing_percentage=0.1,
        feature_mask_prob=0.1,
        homo_injection_ratio=0.0,
        hete_injection_ratio=0.0,
        debug=False,
        wandb_name="build_pipeline",
        log_root=log_dir,
        data_root=log_dir,
    )


def build_synthetic(client_dir: Path, condensed_payload: dict, force: bool) -> None:
    out = client_dir / "synthetic_graph.pt"
    if out.exists() and not force:
        print(f"    [C] synthetic_graph.pt exists — skipping")
        return

    anchor = Data(
        x=condensed_payload["x"].float(),
        edge_index=condensed_payload["edge_index"].long(),
        edge_weight=condensed_payload["edge_weight"].float(),
        node_type=condensed_payload["node_type"].long(),
    )
    anchor.y = anchor.node_type.clone()
    anchor.num_global_classes = 3

    message_pool = {
        "sampled_clients": [0],
        "client_0": {"anchor_graph": anchor},
        "round": 0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        args = _make_server_args(tmp)
        device = torch.device("cpu")
        torch.manual_seed(42)
        server = FedCondQAServer(args, anchor, tmp, message_pool, device)

        print(f"    [C] Running server condensation ({args.server_condense_iters} iters)...")
        t = time.time()
        server.execute()
        synthetic = server.export_synthetic_graph()
        print(f"    [C] Done in {time.time()-t:.0f}s — {synthetic.x.shape[0]} syn nodes")

    torch.save({
        "x": synthetic.x.detach().cpu(),
        "edge_index": synthetic.edge_index.detach().cpu(),
        "edge_weight": synthetic.edge_weight.detach().cpu() if hasattr(synthetic, "edge_weight") else None,
        "node_type": synthetic.node_type.detach().cpu(),
    }, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_client(dataset: str, client_id: int, force: bool, topology_method: str = "knn", entity_ratio: float = 0.05) -> None:
    client_dir = PROCESSED_ROOT / dataset / f"client_{client_id}"
    if not (client_dir / "chunks.json").exists():
        print(f"  [client_{client_id}] No chunks.json — skipping")
        return

    print(f"\n=== client_{client_id} ===")
    graph = build_trigraph(client_dir, dataset, force)
    condensed = build_condensed(client_dir, graph, force, topology_method=topology_method, entity_ratio=entity_ratio)
    build_synthetic(client_dir, condensed, force)
    print(f"  client_{client_id} done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trigraph + condensed + synthetic for all clients.")
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "2wikimultihop", "musique", "medical"])
    parser.add_argument("--clients", type=int, nargs="+",
                        help="Client IDs to process (default: all with chunks.json)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if outputs already exist")
    parser.add_argument("--topology-method", default="knn",
                        choices=["knn", "self_expression"],
                        help="Stage B topology reconstruction method")
    parser.add_argument("--entity-ratio", type=float, default=0.05,
                        help="Fraction of entity nodes to use as Stage B core (default: 0.05). "
                             "Lower = fewer core nodes = faster Stage B (e.g. 0.01 is ~5x faster).")
    args = parser.parse_args()

    dataset_dir = PROCESSED_ROOT / args.dataset
    if args.clients:
        client_ids = args.clients
    else:
        client_ids = sorted(
            int(p.name.split("_")[1])
            for p in dataset_dir.glob("client_*")
            if (p / "chunks.json").exists()
        )

    print(f"Dataset: {args.dataset}")
    print(f"Clients to process: {client_ids}")
    print(f"Force rebuild: {args.force}")
    print(f"Topology method: {args.topology_method}")

    t0 = time.time()
    for cid in client_ids:
        process_client(args.dataset, cid, args.force, topology_method=args.topology_method, entity_ratio=args.entity_ratio)

    print(f"\nAll done in {time.time()-t0:.0f}s")
    print(f"Outputs in: {dataset_dir}")


if __name__ == "__main__":
    main()
