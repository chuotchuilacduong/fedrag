"""Stage C smoke test: server global condensation on real client_0 C_m."""

from __future__ import annotations
import sys, time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Data

from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.server.stage_c_aggregate.surrogate import edge_index_to_dense

CLIENT_DIR = Path("/home/admin1/G-Retriever/processed/hotpotqa/client_0")
OUT_PATH = CLIENT_DIR / "synthetic_graph.pt"

print("=== Stage C Smoke Test: server global condensation ===")
t0 = time.time()

# 1. Load condensed anchor graph C_m
print("\n[1] Loading C_m (condensed_graph.pt)...", end=" ", flush=True)
payload = torch.load(CLIENT_DIR / "condensed_graph.pt", map_location="cpu", weights_only=False)
anchor = Data(
    x=payload["x"].float(),
    edge_index=payload["edge_index"].long(),
    edge_weight=payload["edge_weight"].float(),
    node_type=payload["node_type"].long(),
)
anchor.y = anchor.node_type.clone()
anchor.num_global_classes = 3
K = anchor.x.shape[0]
d = anchor.x.shape[1]
print(f"OK — K={K} anchors, d={d}")
n_ent = (anchor.node_type == 0).sum().item()
n_sen = (anchor.node_type == 1).sum().item()
n_pas = (anchor.node_type == 2).sum().item()
print(f"   entity={n_ent}, sentence={n_sen}, passage={n_pas}")

# 2. Build args (same pattern as unit tests, scaled for real data)
def make_args(tmp_path, num_syn_nodes: int = 128) -> Namespace:
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
        server_condense_iters=10,
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
        wandb_name="smoke",
        log_root=str(tmp_path),
        data_root=str(tmp_path),
    )

import tempfile
tmp_dir = tempfile.mkdtemp()
NUM_SYN = 128
args = make_args(tmp_dir, num_syn_nodes=NUM_SYN)
print(f"\n[2] Args: num_global_syn_nodes={NUM_SYN}, server_condense_iters={args.server_condense_iters}")

# 3. Set up message pool with real anchor graph
message_pool = {
    "sampled_clients": [0],
    "client_0": {"anchor_graph": anchor},
    "round": 0,
}

# 4. Instantiate server
print("\n[3] Instantiating FedCondQAServer...", end=" ", flush=True)
device = torch.device("cpu")
torch.manual_seed(42)
server = FedCondQAServer(args, anchor, tmp_dir, message_pool, device)
print("OK")

# 5. Run server condensation
print(f"\n[4] Running server.execute() ({args.server_condense_iters} gradient-match steps)...", flush=True)
t_exec = time.time()
server.execute()
print(f"   Done in {time.time() - t_exec:.1f}s")

# 6. Export synthetic global graph
print("\n[5] Exporting synthetic graph...", end=" ", flush=True)
synthetic = server.export_synthetic_graph()
print(f"OK — {synthetic.x.shape[0]} nodes, {synthetic.edge_index.shape[1]} edges")

# 7. Verify invariants
print("\n[6] Invariant checks:")

ok_size = synthetic.x.shape[0] == NUM_SYN
print(f"   syn node count = {NUM_SYN}:    {'PASS' if ok_size else 'FAIL'} (got {synthetic.x.shape[0]})")

ok_x = not torch.isnan(synthetic.x).any() and not torch.isinf(synthetic.x).any()
print(f"   x has no NaN/Inf:          {'PASS' if ok_x else 'FAIL'}")

ok_types = set(synthetic.node_type.unique().tolist()) <= {0, 1, 2}
all_three = synthetic.node_type.unique().numel() == 3
print(f"   node_type in {{0,1,2}}:       {'PASS' if ok_types else 'FAIL'}")
print(f"   all 3 types present:       {'PASS' if all_three else 'FAIL'} {synthetic.node_type.unique().tolist()}")

ok_no_sp = True
if synthetic.edge_index.numel() > 0:
    src_types = synthetic.node_type[synthetic.edge_index[0]]
    dst_types = synthetic.node_type[synthetic.edge_index[1]]
    sp = ((src_types == 1) & (dst_types == 2)) | ((src_types == 2) & (dst_types == 1))
    ok_no_sp = not sp.any().item()
print(f"   No S-P edges:              {'PASS' if ok_no_sp else 'FAIL'}")

ok_loss = hasattr(server, "train_loss_match") and torch.isfinite(torch.tensor(server.train_loss_match))
print(f"   train_loss_match finite:   {'PASS' if ok_loss else 'FAIL'} (loss={getattr(server, 'train_loss_match', 'N/A'):.4f})")

ok_msg = "synthetic_x" in message_pool.get("server", {})
print(f"   message_pool has syn_x:    {'PASS' if ok_msg else 'FAIL'}")

# 8. Save synthetic graph
print(f"\n[7] Saving synthetic graph to {OUT_PATH}...")
torch.save({
    "x": synthetic.x.detach().cpu(),
    "edge_index": synthetic.edge_index.detach().cpu(),
    "edge_weight": synthetic.edge_weight.detach().cpu() if hasattr(synthetic, "edge_weight") else None,
    "node_type": synthetic.node_type.detach().cpu(),
}, OUT_PATH)
print(f"   Saved — {OUT_PATH.stat().st_size // 1024} KB")

total = time.time() - t0
all_pass = ok_size and ok_x and ok_types and all_three and ok_no_sp and ok_loss and ok_msg
print(f"\n=== {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'} in {total:.1f}s ===")
