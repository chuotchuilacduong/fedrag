# Diagram vs Code Verification Report

Verification date: 2026-06-02  
Image: architecture diagram (Dance.png)  
Code base: `fedcond_grag/`

Legend: ✅ correct, ⚠️ minor inaccuracy, ❌ wrong/missing

---

## 1. Tri-graph Construction (middle-left panel)

| Diagram | Code | Status |
|---|---|---|
| Passage → Sentence → Entity with S-E and P-E edges | `trigraph_builder.py`: S-E (edge_type=0) and P-E (edge_type=1) only — no S-P, no E-E | ✅ |
| Three node types: P (passage), E (entity), S (sentence) | `node_type ∈ {0=Entity, 1=Sentence, 2=Passage}` | ✅ |

---

## 2. Stage B — Anchor Expansion (top-left)

### 2a. Anchor node selection

| Diagram | Code | Status |
|---|---|---|
| Candidate nodes {e1,e2,e3} → anchor e4 selected | `anchor_node_selector.py`: score(v) = λ_idf·IDF + λ_pr·PageRank + λ_mmr·MMR, then S-E-P motif expansion → `core_node_ids` | ✅ |

### 2b. Hierarchical text condensation

| Diagram | Code | Status |
|---|---|---|
| Hop 0 / hop 1 / hop 2 structure → Aggregate | `neighbor_gating.py` `hierarchical_text_condensation()`: budgets=(1,3,2), hop_weights=[0.4,0.4,0.2] | ✅ |
| "Aggregate" implies weighted attention | **Current code uses uniform mean** (`score_and_select`: weights[:k]=1/k, no W_q/W_k attention). flow.md formula `attn(W_q,W_k)` is outdated. Diagram "Aggregate" label is vague but acceptable. | ⚠️ |
| Output labels: x̃_e4, x̃_e2, **x̃_c1** | Third anchor label is `x̃_c1` — "c" is not a node type in S-E-P. Should be a passage anchor, e.g. x̃_p1. | ⚠️ |

### 2c. Graph-Text Fusion — MISSING from diagram

**❌ Critical missing step.**

Diagram flow: `Text condensation output (x̃) → KNN topology`

Actual code flow (`client_condensor.py` line 123):
```
t_tilde, contexts, traces = hierarchical_text_condensation(...)
x_fused, gate = self.fusion(core_graph_embeddings, t_tilde)   ← MISSING
topology = knn_topology(x_fused, ...)                         ← uses x_fused, not t̃
```

`GraphTextFusion.forward()` (`graph_text_fusion.py`):
```python
fused = LayerNorm(W_g(x_graph) + 0.5 * W_t(t_tilde))
```
- gate is a **fixed scalar buffer = 0.5** (not a learned sigmoid gate)
- flow.md still shows `gate = σ(W_gate([x;t̃]))` — this is wrong in flow.md too

What the diagram should show: `x̃ (text) + x (graph) → Fusion (fixed gate=0.5) → x_fused → KNN`

The vector going into KNN topology is `x_fused` (graph+text fused), not the raw text condensation output `t̃`.

### 2d. Topology reconstruction

| Diagram | Code | Status |
|---|---|---|
| KNN shown | `knn_topology()` — default `topology_method="knn"` | ✅ |
| Output: G̃_m | `ClientCondensedGraph(x=x_fused, edge_index, edge_weight, node_type)` | ✅ |

---

## 3. Upload: G̃_m + ω_m → Server

| Diagram | Code | Status |
|---|---|---|
| G̃_m = (Ṽ_m, Ẽ_m, X̃_m) + ω_m | `client.send_message()`: `anchor_graph` + `model_weights` + `num_samples` | ✅ concept |
| G̃_m and ω_m always uploaded together | **ω_m is only uploaded when `_model_weights is not None and _num_local_samples > 0`** — in Round 0 (condensation only), there is no ω_m. The diagram doesn't distinguish this round-dependency. | ⚠️ |

---

## 4. Stage D — Retrieval (bottom-left)

### 4a. Evidence graph G^e

| Diagram | Code | Status |
|---|---|---|
| Question → Entity Activation → **Personalized PageRank** → top-k passages → G^e | **❌ PPR does NOT run at inference time.** Actual code (`_attach_evidence_graphs`): loads precomputed `anchor_passage_nodes` (1 per sample, filled during preprocessing by LinearRAG), falls back to cosine top-R search if missing, then 1-hop expand in local trigraph. PPR was done by LinearRAG during offline indexing — not at per-round inference time. | ❌ |
| 1-hop expansion from anchor nodes | `_attach_evidence_graphs()`: `kept_set = seed_set ∪ {1-hop neighbors}` | ✅ |
| Output: evidence subgraph G^e | `graph = Data(x, edge_index, edge_weight, node_type)` stored in `sample["evidence_graph"]` | ✅ |

**Correct diagram description for retrieval:**
```
anchor_passage_nodes (precomputed offline by LinearRAG: NER → BFS entity activation → PPR → top-k)
  → 1-hop expand in local Tri-graph G_m
  → G^e (evidence subgraph)
```

### 4b. Condensed graph G^c retrieval

| Diagram | Code | Status |
|---|---|---|
| G_global → top-k retrieval → G^c | `_attach_condensed_graphs()` + `GlobalGraphRetriever`: `query = mean_pool(G_ev.x)`, cosine search in G_global, top-R + 1-hop expand | ✅ |

---

## 5. Stage D — DualGraphLLM (right panel, "Graph augmented generation")

| Diagram | Code | Status |
|---|---|---|
| G^e + G^c feed into LLM via GNN encoder + projection | `encode_graphs()`: `z_e = projector(graph_encoder(G_ev))`, `z_c = projector(graph_encoder(G_cn))` (shared mode) | ✅ |
| One GNN encoder + one projection block | Default `dual_graph_mode="shared"` → single `graph_encoder` + `projector` for both graphs | ✅ |
| Two soft tokens → frozen LLM | `graph_tokens = torch.stack([z_e, z_c])` then `[BOS, z_e, z_c, text_tokens]` as input | ✅ |
| LLM frozen (❄️ icon) | Base `GraphLLM` freezes the LLM — only GNN + projector trainable | ✅ |
| ω_m on GNN encoder (trainable) | Trained components: `graph_encoder`, `projector` (+ `condensed_encoder`, `projector_c` in dual mode) | ✅ |
| passages + query shown as text input | `samples["desc"]` (passages) + `samples["question"]` tokenized and embedded | ✅ |

---

## 6. Server (a1) — Global Aggregation (FedAvg)

| Diagram | Code | Status |
|---|---|---|
| ω_1, ω_2, …, ω_m → (1/m)Σω_i → ω_Global | **❌ Wrong formula.** Code (`_fedavg_model_weights`): **weighted** FedAvg `Σ(n_m / N_total) * w_m` where n_m = number of local samples. Simple mean 1/m is incorrect. | ❌ |
| All client GNN weights aggregated | `graph_encoder`, `projector`, `condensed_encoder`, `projector_c` (if present) | ✅ |

**Correct formula:** ω_Global = Σ_m (n_m / Σ n_m) · ω_m

---

## 7. Server (a2) — Global Graph Condensation

| Diagram | Code | Status |
|---|---|---|
| G^c → GNN(ω_Global) → H_syn | `encode_nodes_with_edge_weight(synthetic_x, edge_index, edge_weight, repr_encoder, repr_projector)` — repr_encoder loaded from FedAvg'd weights | ✅ |
| G̃_m (M graph) → H_m | `precompute_anchor_reprs(anchor_graphs, repr_encoder, repr_projector, device)` — one H_m per client anchor graph | ✅ |
| Repr. Alignment → Update | `representation_alignment_loss(h_syn, anchor_h_list)` + `loss.backward()` + `optimizer.step()` updating `synthetic_x` and `PGE` | ✅ |
| Shows only repr_align mode | Runtime default is `repr_align` (set in main.py). Code also supports `gradient_match` and `both` modes, not shown. Acceptable simplification. | ⚠️ |
| PGE (parametric graph edge generator) not shown | `TypeAwarePGE(feature_dim, hidden_dim, type_emb_dim, topk)` generates synthetic graph topology. PGE parameters are also updated each step but not visible in diagram. | ⚠️ |

---

## 8. Server → Client broadcast

| Diagram | Code | Status |
|---|---|---|
| Server sends G^c back to client | `server.send_message()`: `synthetic_graph = export_synthetic_graph()` in message | ✅ |
| Server sends ω_Global back | `msg["model_weights"] = self.global_model_state` | ✅ |

---

## Summary of Issues

| # | Location | Issue | Severity |
|---|---|---|---|
| 1 | Stage D Retrieval | **PPR shown as runtime step** — PPR runs offline during LinearRAG indexing, not per-inference. Actual runtime: load precomputed anchor nodes + 1-hop expand | ❌ |
| 2 | Server (a1) | **FedAvg formula wrong** — diagram shows `(1/m)Σω_i` but code uses weighted average `Σ(n_m/N_total)·ω_m` | ❌ |
| 3 | Stage B | Graph-Text Fusion not explicitly shown — diagram treats `x̃` as the fused embedding going into KNN. Acceptable simplification: fusion is just linear projection + scaled addition + LayerNorm, no conceptual flow change. | ⚠️ |
| 4 | Stage B labels | `x̃_c1` label is incorrect — "c" is not a node type; should be a passage node label (x̃_p1) | ⚠️ |
| 5 | Upload arrows | ω_m only uploaded in Round ≥ 1 (after local training), not Round 0 | ⚠️ |
| 6 | Server (a2) | PGE (TypeAwarePGE) that generates synthetic graph edges not shown | ⚠️ |
| 7 | Server (a2) | Only repr_align mode shown; gradient_match and both modes not indicated | ⚠️ |
| 8 | flow.md | Stage B formula still shows learned gate `σ(W_gate([x;t̃]))` and attention `attn(W_q,W_k)` — both removed; code uses fixed gate=0.5 and uniform mean | ⚠️ (doc) |

---

## What is Correct

- Tri-graph node types (E/S/P) and edge types (S-E, P-E)
- Anchor selection flow: candidate scoring → anchor → motif expansion → core_ids
- Hop 0/1/2 structure in text condensation
- KNN topology reconstruction
- G̃_m = (V, E, X) upload structure
- 1-hop expansion for evidence graph G^e
- Cosine search in G_global for condensed graph G^c
- DualGraphLLM: two soft tokens [z_e, z_c] → frozen LLM
- LLM frozen, GNN trainable
- FedAvg aggregates GNN weights from all clients
- Repr alignment: H_syn vs H_m, update synthetic graph
- Server broadcasts G^c + ω_Global to clients
