"""Stage B smoke test: condense client_0 Tri-Graph → C_m.pt"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fedcond_grag.client.stage_a_trigraph import load_trigraph
from fedcond_grag.client.stage_b_condense import (
    ClientCondensationConfig,
    condense_client_graph,
    build_text_bank,
    save_text_bank,
    load_text_bank,
)
from fedcond_grag.client.stage_b_condense.text_bank import load_frozen_encoder

CLIENT_DIR = Path("/home/admin1/G-Retriever/processed/hotpotqa/client_0")
TEXT_BANK_PATH = CLIENT_DIR / "text_bank.pt"
OUT_PATH = CLIENT_DIR / "condensed_graph.pt"

print("=== Stage B Smoke Test: client_0 ===")
t0 = time.time()

# 1. Load Tri-Graph
print("\n[1] Loading Tri-Graph...", end=" ", flush=True)
graph = load_trigraph(CLIENT_DIR / "trigraph.pt")
print(f"OK — {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges, dim={graph.x.shape[1]}")

n_entities = (graph.node_type == 0).sum().item()
n_sentences = (graph.node_type == 1).sum().item()
n_passages = (graph.node_type == 2).sum().item()
print(f"   entity={n_entities}, sentence={n_sentences}, passage={n_passages}")

# 2. Text bank (build or load from cache)
if TEXT_BANK_PATH.exists():
    print(f"\n[2] Loading cached text bank from {TEXT_BANK_PATH}...", end=" ", flush=True)
    bank = load_text_bank(TEXT_BANK_PATH)
    print(f"OK — {bank.num_nodes} nodes, dim={bank.dim}")
else:
    print(f"\n[2] Building text bank for {len(graph.node_text)} nodes (this may take ~2-3 min)...")
    encoder = load_frozen_encoder("all-MiniLM-L6-v2", dim=384)
    bank = build_text_bank(
        graph.node_text,
        encoder=encoder,
        encoder_name="all-MiniLM-L6-v2",
        dim=384,
    )
    save_text_bank(bank, TEXT_BANK_PATH)
    print(f"   Saved to {TEXT_BANK_PATH}")
    print(f"   {bank.num_nodes} nodes, dim={bank.dim}")

# 3. Run condensation
print(f"\n[3] Running Stage B condensation (knn topology)...", flush=True)
cfg = ClientCondensationConfig(
    topology_method="knn",
    knn_k=8,
)
t_cond = time.time()
condensed, artifacts = condense_client_graph(
    graph,
    text_bank=bank,
    graph_embeddings=graph.x,
    config=cfg,
    return_artifacts=True,
)
elapsed = time.time() - t_cond
print(f"   Done in {elapsed:.1f}s")

# 4. Inspect outputs
print(f"\n[4] Condensed graph (C_m):")
print(f"   x shape:          {condensed.x.shape}")
print(f"   edge_index shape: {condensed.edge_index.shape}")
print(f"   edge_weight shape:{condensed.edge_weight.shape}")
print(f"   node_type unique: {condensed.node_type.unique().tolist()}")
K = condensed.x.shape[0]
n_ent_c = (condensed.node_type == 0).sum().item()
n_sen_c = (condensed.node_type == 1).sum().item()
n_pas_c = (condensed.node_type == 2).sum().item()
print(f"   K={K} anchors: entity={n_ent_c}, sentence={n_sen_c}, passage={n_pas_c}")

# 5. Verify invariants
print(f"\n[5] Invariant checks:")

# No NaN/Inf in embeddings
ok_x = not torch.isnan(condensed.x).any() and not torch.isinf(condensed.x).any()
print(f"   x has no NaN/Inf:        {'PASS' if ok_x else 'FAIL'}")

# No raw text fields
has_text = hasattr(condensed, 'node_text') and condensed.node_text is not None
print(f"   No raw text in upload:   {'PASS' if not has_text else 'FAIL'}")

# edge_weight in valid range
ok_w = (condensed.edge_weight >= 0).all() and (condensed.edge_weight <= 1).all()
print(f"   edge_weight in [0,1]:    {'PASS' if ok_w else 'FAIL'}")

# No S-P edges in condensed graph (node_type 1-2 pairs)
ok_no_sp = True
if condensed.edge_index.numel() > 0:
    src_types = condensed.node_type[condensed.edge_index[0]]
    dst_types = condensed.node_type[condensed.edge_index[1]]
    sp_mask = ((src_types == 1) & (dst_types == 2)) | ((src_types == 2) & (dst_types == 1))
    ok_no_sp = not sp_mask.any().item()
print(f"   No S-P edges:            {'PASS' if ok_no_sp else 'FAIL'}")

# fusion gate in [0,1]
gate = artifacts.fusion_gate
ok_gate = (gate >= 0).all() and (gate <= 1).all()
print(f"   Fusion gate in [0,1]:    {'PASS' if ok_gate else 'FAIL'}")

# motif selection has entities
motif = artifacts.motif_selection
ok_motif = motif.core_node_ids.numel() > 0
print(f"   Motif has core nodes:    {'PASS' if ok_motif else 'FAIL'}")

# 6. Save C_m
print(f"\n[6] Saving condensed graph to {OUT_PATH}...")
payload = {
    "x": condensed.x.detach().cpu(),
    "edge_index": condensed.edge_index.detach().cpu(),
    "edge_weight": condensed.edge_weight.detach().cpu(),
    "node_type": condensed.node_type.detach().cpu(),
    "hashed_local_ids": condensed.hashed_local_ids.detach().cpu() if condensed.hashed_local_ids is not None else None,
}
torch.save(payload, OUT_PATH)
print(f"   Saved — {OUT_PATH.stat().st_size // 1024} KB")

total = time.time() - t0
all_pass = ok_x and not has_text and ok_w and ok_no_sp and ok_gate and ok_motif
print(f"\n=== {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'} in {total:.1f}s ===")
