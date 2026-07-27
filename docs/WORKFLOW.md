# FedCondGraphRAG — Project Workflow

End-to-end reference for how FedCondGraphRAG works, including the inputs and
outputs of every stage, where they live on disk, and which CLI subcommands
drive them. Companion to the design notes in `docs/plan/`; this file describes
the **as-built** system after the refactor.

---

## 1. What the system does

FedCondGraphRAG is a **federated** retrieval-augmented QA system over multi-hop
QA corpora (HotpotQA, 2WikiMultihop, MuSiQue, Medical). Each *client* holds a
private slice of passages; the *server* never sees the passages.

There are four stages. Stages A, B, D run on the client; Stage C runs on the
server.

```
   ┌─────────────────────────────  CLIENT m  ─────────────────────────────┐
   │                                                                       │
   │  passages           Stage A          Stage B        anchor graph C_m │
   │  (private)  ───►  Tri-Graph  ───►  Condensor  ───►  (numeric only)   │
   │                  E/S/P nodes     budget K ≪ N                        │
   │                                                                       │
   │                              ┌──────── upload ──────────┐            │
   └──────────────────────────────│──────────────────────────│────────────┘
                                  ▼                          ▼
                          ┌───────────────────  SERVER  ──────────────────┐
                          │                                                │
                          │   {C_1, …, C_M}        Stage C                │
                          │   anchor graphs   ──►  Gradient-matching      │
                          │                        condensation           │
                          │                                                │
                          │   synthetic global graph G_global  (K_g nodes)│
                          └────────────────────────────────────────────────┘
                                              │
                                              ▼  broadcast
   ┌─────────────────────────────  CLIENT m  ─────────────────────────────┐
   │                                                                       │
   │  question q                                                          │
   │       │                                                              │
   │       ▼      Stage D                                                  │
   │  EvidenceLinearRAG ──► E_q (evidence subgraph from local Tri-Graph)  │
   │  GlobalGraphRetriever ──► G_q (subgraph of G_global)                  │
   │                                                                       │
   │  ┌─ GraphTransformer(E_q) ──► z_e ──┐                                │
   │  │                                  ├──► projector ──► soft-prompt   │
   │  └─ GAT(G_q)            ──► z_c ──┘     tokens │                     │
   │                                                ▼                     │
   │                          DualGraphLLM   ──►  answer string           │
   └───────────────────────────────────────────────────────────────────────┘
```

Key idea: clients never share raw passages or model weights. They share a tiny
**numeric anchor graph** C_m (a few hundred nodes of fused embeddings). The
server fuses anchors into a global synthetic graph via **gradient matching**,
not weight averaging. At query time clients combine local *text* evidence (via
LinearRAG) with the global *embedding* graph in a dual-prompt LLM.

---

## 2. Package layout (after refactor)

```
fedcond_grag/
├── cli.py                          # subcommands: preprocess | fl-train | train | infer
├── trainer.py                      # FedTrainer round loop
├── config.py                       # Stage D argparse (+ FL knobs)
├── __init__.py                     # load_client / load_server / load_task helpers
│
├── baselines/                      # third-party RAG methods compared against FedCondGraphRAG
│   ├── linearrag/                  # LinearRAG engine — used by Stage A/D *and* as a standalone baseline
│   └── hipporag/                   # per-client local HippoRAG baseline (see §11a)
│
├── dataloader/                     # corpus loaders, partition, FedCondQADataset
│   ├── fedcond_qa_dataset.py
│   ├── hotpot_loader.py
│   ├── linearrag_loader.py
│   ├── federated_partition.py
│   └── corpus_index.py
│
├── client/
│   ├── client.py                   # FedCondQAClient
│   ├── stage_a_trigraph/           # Stage A — Tri-Graph builder
│   ├── stage_b_condense/           # Stage B — ClientCondensor
│   └── stage_d_retrieve/           # Stage D — query-time retrieval + E_q
│
├── server/
│   ├── server.py                   # FedCondQAServer
│   └── stage_c_aggregate/          # Stage C — PGE, SurrogateGNN, gradient matching
│
├── model/                          # Stage D model
│   ├── dual_graph_llm.py
│   ├── graph_llm.py
│   └── gnn.py
│
└── utils/                          # ckpt, collate, evaluate, seed, …
```

---

## 3. End-to-end data flow

```
   dataset/linearrag/<ds>/{chunks.json, questions.json}
                  │
                  │  scripts/preprocess_data.py
                  ▼
   processed/<ds>/client_<m>/chunks.json
                  │
                  │  scripts/build_client_pipeline.py — Stage A
                  ▼
   processed/<ds>/client_<m>/trigraph.pt           (PyG Data)
                  │
                  │  Stage B (ClientCondensor)
                  ▼
   processed/<ds>/client_<m>/condensed_graph.pt    (anchor C_m)
                  │
                  │  Stage B.3.5 refine at fl-train round 0 (cached)
                  ▼
   processed/<ds>/client_<m>/condensed_graph_refined.pt
                  │
                  │  Stage C / FedRAG Phase 0 — at fl-train time only:
                  │  server fuses ALL client anchors into Θ_syn in-memory
                  ▼
   message_pool["server"]  (G_syn broadcast; no per-client disk artifact)
                  │
                  │  preprocess_fedcond_qa.py — Stage D cache build
                  ▼
   dataset/fedcond_qa/
     records.jsonl, split/{train,val,test}_indices.txt,
     cached_graphs/<id>.pt           (E_q  per question)
     cached_condensed_graphs/<id>.pt (G_q  per question)
     cached_desc/<id>.txt            (passage text)
                  │
                  │  main.py train  (Stage D fit)
                  ▼
   output/fedcond_qa/<model_args>.csv   (predicted answers + labels)
                  │
                  │  eval_funcs['fedcond_qa']
                  ▼
   Hit / F1 / Accuracy
```

---

## 4. Stage A — Tri-Graph building (client)

**Purpose:** turn a client's raw passages into a heterogeneous graph with three
node types (Entity, Sentence, Passage), where embeddings come from a frozen
sentence transformer and edges come from co-occurrence + LinearRAG's NER pass.

**Code:**
- `fedcond_grag/client/stage_a_trigraph/trigraph_builder.py` →
  `build_trigraph_for_client(passages, working_dir, dataset_name, encoder)`
- Backed by `fedcond_grag/baselines/linearrag/LinearRAG.index(passages)`.

**Input:**
| Field | Shape / Type | Notes |
|---|---|---|
| `passages` | `Sequence[str]` | LinearRAG format: `"N:title. text…"` where `N` is a sequential index. |
| `working_dir` | path | LinearRAG cache (parquet embedding stores + NER cache). |
| `dataset_name` | str | Sub-dir under working_dir (e.g. `"hotpotqa_client_0"`). |
| `encoder` | `SentenceTransformer` | Default `all-MiniLM-L6-v2`, dim 384. |

**Output:** a `torch_geometric.data.Data` with

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `x` | `[N, d]` | `float32` | L2-normalised node embeddings (d=384). |
| `edge_index` | `[2, E]` | `int64` | Undirected; both directions stored. |
| `edge_type` | `[E]` | `int64` | `0` = Sentence–Entity, `1` = Passage–Entity. |
| `node_type` | `[N]` | `int64` | `0` = Entity, `1` = Sentence, `2` = Passage. |
| `node_text` | `list[str]` | — | Raw text per node (local-only, never uploaded). |

**Invariants** (see `docs/plan/02_DATA_AND_TRIGRAPH.md`):
- Only S–E and P–E edges; no S–P (the "S-E-P invariant").
- P–P sequential edges produced by `LinearRAG.add_adjacent_passage_edges()` are
  filtered out — they're not part of the Tri-Graph topology.

**On disk:** `processed/<dataset>/client_<m>/trigraph.pt`
(plus a sibling `linearrag_cache/` containing the parquet embedding stores).

**CLI:** invoked indirectly via
```bash
python main.py preprocess --dataset hotpotqa
```
which calls `scripts/build_client_pipeline.py` for every client folder that
has a `chunks.json`.

---

## 5. Stage B — Client-side condensation

**Purpose:** compress the local Tri-Graph (N nodes, often 10k–100k) down to a
small anchor graph C_m (K ≈ 100–500 nodes) that preserves the type and
semantic structure but contains **no text** — only embeddings. The anchor is
what the client uploads to the server.

**Code:**
- `fedcond_grag/client/stage_b_condense/client_condensor.py` → `ClientCondensor` (nn.Module orchestrator)
- Helpers in `motif_core_selector.py`, `text_bank.py`, `chunk_selection.py`,
  `neighbor_gating.py`, `topology_reconstruction.py`, `graph_text_fusion.py`,
  `evidence_prior.py`.

**Input:**
| Field | Type | Notes |
|---|---|---|
| `tri_graph` | PyG `Data` from Stage A | Must carry `node_type` and `x`. |
| `text_bank` | `TextBank` | Frozen encoder snapshot of node_text — local-only; built with `build_text_bank(graph.node_text, encoder, …)`. |
| `config` | `ClientCondensationConfig` | See below. |

**`ClientCondensationConfig` knobs (defaults shown):**
```python
motif:                   MotifSelectorConfig()       # entity ratio, sentence/passage budgets, IDF/PR/MMR mix
text_budgets:            (1, 3, 2)                    # per-type chunk budgets used by hierarchical condensation
chunk_budget:            8
hop_weights:             (0.4, 0.4, 0.2)              # neighbor gating contribution per hop
topology_method:         "knn"                        # or "self_expressive"
knn_k:                   8
prior_weight:            0.0
self_expr_iterations:    50
preserve_sep_topology:   True                         # keep S–E + P–E pattern in C_m's edges
```

**Output:** a `ClientCondensedGraph` (numeric upload object). After
`.to_pyg_data()`:

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `x` | `[K, d]` | `float32` | Fused embedding of each selected motif node. |
| `edge_index` | `[2, E']` | `int64` | Reconstructed kNN or self-expressive topology. |
| `edge_weight` | `[E']` | `float32` | Similarity/weight per edge. |
| `node_type` | `[K]` | `int64` | Preserves the 0/1/2 type labels. |

No `node_text` is kept — anchor graphs are numeric-only by construction.

**Retrieval-preserving refinement (paper B.3.5).** After the constructive
initialization above, the condensed features X̃ are refined by minimizing

```
L_cond = L_ret + λ_rep·L_rep + λ_div·L_div
```

- `L_ret` — KL divergence between the full-graph passage-retrieval
  distribution and the condensed one lifted back through the soft coverage
  map `A_m(p, p̃) = softmax_p̃(cos(x_p, x̃_p̃)/τ_a)` (paper Eq. 2). Queries are
  the client's own local training questions embedded with the same frozen
  sentence encoder as the node features; without local questions L_ret is
  skipped. The lifted scores are renormalized into a proper distribution so
  the KL stays non-negative.
- `L_rep` — soft-assignment reconstruction of sampled full-graph node
  representations under a frozen random GNN θ⁰ shared by both graphs.
- `L_div` — hinge on pairwise cosine similarity of condensed features
  (margin δ), averaged over |Ṽ|² pairs.

Only X̃ is optimized; the condensed topology is then rebuilt from the refined
features via similarity kNN (paper B.3.4). Runs once per client (round 0),
on both the fresh-condense and cached-`condensed_graph.pt` paths.

**Code:** `fedcond_grag/client/stage_b_condense/condensation_refine.py`
(`refine_condensed_graph`, `RetrievalRefineConfig`), invoked from
`FedCondQAClient._maybe_refine_condensed`. Knobs: `--condense-refine-iters`
(default 100; 0 disables), `--condense-refine-lr`, `--stage-b-lambda-rep`,
`--stage-b-lambda-div`, `--stage-b-div-margin`, `--stage-b-tau-ret`,
`--stage-b-tau-cov`, `--stage-b-max-queries`.

**On disk:** `processed/<dataset>/client_<m>/condensed_graph.pt` (dict with
`x`, `edge_index`, `edge_weight`, `node_type`).

**Smoke test:** `python scripts/stage_b_smoke.py`;
refinement: `python -m pytest tests/test_stage_b_condensation_refine.py`

---

## 6. Stage C — Server-side aggregation

**Purpose:** fuse the per-client anchor graphs `{C_1, …, C_M}` into a single
**global synthetic graph** `G_global` with `K_g` learnable nodes. The fusion
objective is **gradient matching**: a small `SurrogateGNN` trained on the
synthetic graph should yield parameter gradients close to the weighted average
of the gradients you'd see if you trained on each client's anchor separately.
This is the only step the server runs.

**Code:**
- `fedcond_grag/server/server.py` → `FedCondQAServer`
- `fedcond_grag/server/stage_c_aggregate/pge.py` → `TypeAwarePGE` (parameterises
  the synthetic edge probabilities given node features + types).
- `fedcond_grag/server/stage_c_aggregate/surrogate.py` → `SurrogateGNN`,
  `surrogate_loss`, `gradient_match_loss`, `parameter_gradients`,
  `edge_index_to_dense`.
- `fedcond_grag/server/stage_c_aggregate/task.py` → minimal `CondensationQATask`.

**Input (per round):** `message_pool["client_<m>"]["anchor_graph"]` for every
sampled client m. Each is a PyG `Data` matching the Stage B schema above.

**Outputs (in `message_pool["server"]` after `send_message()`):**
| Key | Shape | Meaning |
|---|---|---|
| `synthetic_x` | `[K_g, d]` | Learnable node features (`nn.Parameter`). |
| `synthetic_adj` | `[K_g, K_g]` | Dense soft adjacency produced by `pge.inference(x, node_type)`. |
| `synthetic_node_type` | `[K_g]` | 0/1/2 labels per synthetic node, sampled to match aggregate per-type ratios across clients. |

**`export_synthetic_graph()`** returns a PyG `Data`:
| Field | Shape | dtype |
|---|---|---|
| `x` | `[K_g, d]` | `float32` |
| `edge_index` | `[2, E_g]` | `int64`, sparsified from `adj > 0` |
| `edge_weight` | `[E_g]` | `float32` |
| `node_type` | `[K_g]` | `int64` |

**Knobs (from `fedcond_grag.server.stage_c_aggregate.config`):**
```
num_global_syn_nodes:    1024          # K_g
server_condense_iters:   50            # gradient-matching steps per round
hid_dim, num_layers:     SurrogateGNN dimensions
pge_hidden, pge_topk:    TypeAwarePGE dimensions and topology cap
preserve_sep_topology:   True          # G_global stays S–E + P–E only
surrogate_type_weight:   1.0           # node-type loss weight
surrogate_link_weight:   0.5           # link-prediction loss weight
match_norm_weight:       0.0
```

**On disk:** nothing — Stage C / FedRAG Phase 0 runs at fl-train time only and
the synthetic memory lives in `message_pool["server"]`. (`main.py preprocess`
used to also write a per-client `synthetic_graph.pt`; that offline Stage C was
removed — it fused only one client's anchor with the legacy gradient-match
mode and nothing consumed it. Preprocess now stops at Stage A→B.)

**Smoke test:** `python scripts/stage_c_smoke.py`

---

## 7. Stage D — Dual-graph prompting (training + inference)

**Purpose:** answer a question by feeding the LLM two graph-derived
soft-prompt tokens alongside the question text:
- `z_e`: encoding of the **evidence subgraph** E_q (built from local Tri-Graph
  using LinearRAG's PPR retrieval).
- `z_c`: encoding of the **condensed subgraph** G_q (extracted from G_global by
  cosine + 1-hop expansion around the query embedding).

**Inference-time pipeline:**

```
question q
   │
   ├─► EvidenceLinearRAG (subclass of LinearRAG)
   │      ├─ retrieve top-k passages (PPR over E/S/P graph)
   │      └─ capture actived_entities + sorted_passage_hash_ids
   │
   ├─► build_evidence_graph(trigraph, captured_state) ──► E_q
   │
   ├─► GlobalGraphRetriever(synthetic_graph).retrieve(q_embedding)
   │      ├─ cosine sim → top-R seed nodes
   │      └─ 1-hop expansion ──► G_q
   │
   └─► DualGraphLLM(question, desc=passages, evidence_graph=E_q, condensed_graph=G_q)
              │
              ├─ GraphTransformer encoder on E_q ──► z_e
              ├─ GAT encoder           on G_q  ──► z_c
              ├─ projector → soft-prompt tokens
              └─ LLM forward → answer string
```

**Code:**
- `fedcond_grag/client/stage_d_retrieve/`:
  - `linearrag_retriever.py` (`LinearRAGRetriever`)
  - `evidence_linearrag.py` (`EvidenceLinearRAG`, `EvidenceRetrievalResult`,
    `_CaptureLinearRAG` subclass)
  - `evidence_graph_builder.py` (`build_evidence_graph` → `EvidenceGraph`)
  - `global_graph_retriever.py` (`GlobalGraphRetriever`,
    `retrieve_global_subgraph`)
- `fedcond_grag/model/dual_graph_llm.py` (`DualGraphLLM` extends `GraphLLM`).
- `fedcond_grag/dataloader/fedcond_qa_dataset.py` (`FedCondQADataset`).

**Cached dataset layout** (offline cache built by
`scripts/preprocess_fedcond_qa.py`):
```
dataset/fedcond_qa/                # or $FEDCOND_QA_PATH
├── records.jsonl                  # one record per question
├── split/
│   ├── train_indices.txt          # 0-based row indices, one per line
│   ├── val_indices.txt
│   └── test_indices.txt
├── cached_graphs/<id>.pt          # E_q  (evidence subgraph), PyG Data
├── cached_condensed_graphs/<id>.pt# G_q  (synth subgraph), PyG Data
└── cached_desc/<id>.txt           # plain-text retrieved passages (optional)
```

`records.jsonl` row schema (subset used by the dataset):
```json
{
  "id": "5ae0…",
  "question": "Who is the mother of …?",
  "answer": "Alice|Alicia",
  "retrieved_passages": ["1:title. text…", "2:title. text…"]
}
```

**`FedCondQADataset.__getitem__(i)` returns:**
| Key | Type / shape | Meaning |
|---|---|---|
| `id` | str | record id |
| `question` | str | `"Question: <q>\nAnswer: "` |
| `label` | str | lowercased gold answer (pipe-joined if multi-answer) |
| `graph` | PyG `Data` | the evidence subgraph E_q |
| `evidence_graph` | alias of `graph` | for compatibility |
| `condensed_graph` | PyG `Data` | the global synth subgraph G_q |
| `desc` | str | retrieved passages concatenated |
| `retrieved_passages` | `list[str]` | raw passage texts |

**`DualGraphLLM` knobs (set via `fedcond_grag/config.py`):**
```
--model_name dual_graph_llm
--gnn_model_name      gt        # evidence encoder (graph transformer)
--gnn_model_name_c    gat       # condensed encoder
--gnn_in_dim          384       # = sentence-transformer dim
--gnn_hidden_dim      384
--gnn_in_dim_c        384
--gnn_hidden_dim_c    384
--dual_graph_mode     both      # both | evidence_only | condensed_only |
                                #   random_condensed | text_only
```

**Output of `model.inference(batch)`:** a dict-of-lists with at least
`{"pred": [...], "label": [...], "id": [...]}` — flushed line-by-line to a CSV
under `output/<dataset>/<args-hash>.csv`.

**Evaluation:** `fedcond_grag.utils.evaluate.eval_funcs['fedcond_qa']`
computes Accuracy / Hit / F1 / Precision / Recall via the legacy
`get_accuracy_fedcond_qa(path)` function.

---

## 7a. Stage E — Query-conditioned synthetic memory adaptation (FedRAG)

**Purpose:** the FedRAG algorithm's Phase 1 (paper §3.5/§3.6, Algorithm 1).
Instead of the server refining the synthetic graph against client anchors
every round, each client receives the full synthetic-memory parameters
`Θ_syn = {X_syn, θ_PGE}`, adapts a **local copy** with private QA feedback,
and uploads only the parameter delta `Δ_m`. The server aggregates deltas
(sample-weighted) and applies one server-side regularization step:

```
Θ_syn^(r+1) = Θ_syn^(r) + η_agg · Σ_m (n_m/Σn) Δ_m − η_reg · ∇ L_reg
```

**Client-side objective** (K_mem steps per round, prompt module W frozen):

```
L_mem = L_QA + λ_gm·L_GM + λ_align·L_align + λ_reg·L_reg
```

- `L_QA` — QA loss where the condensed soft-prompt slot `z_c` comes from
  **differentiable soft retrieval** (paper B.5.2): the pooled evidence prompt
  rep `z̄_e` attends over projected synthetic nodes, `z_c = Σ α_i z_i^syn`,
  so the frozen-LLM loss backpropagates into `X_syn` and `θ_PGE`. Gradients
  reach Θ_syn without retaining the LLM graph via a context-token surrogate
  (`(∂L/∂z_c).detach() · z_c` — exact, since Θ_syn enters only through z_c).
- `L_GM` — first-order gradient matching (paper B.7): cosine distance between
  the context-branch prompt-module gradients induced by synthetic vs. private
  local evidence context.
- `L_align` — client-local RowSoftmax alignment of the condensed anchor C_m
  against the synthetic projections (server never sees Z_m).
- `L_reg` — synthetic diversity + degree regularization (paper B.4.2).

**Code:**
- `fedcond_grag/client/stage_e_memory/synthetic_memory.py` —
  `LocalSyntheticMemory`, `gradient_matching_loss`, `aggregate_syn_deltas`.
- `fedcond_grag/client/client.py::adapt_synthetic_memory` — the K_mem loop.
- `fedcond_grag/server/server.py::_apply_syn_deltas / _server_reg_step` —
  Eq. (18)–(19); Phase 0 initialization reuses the repr-align path.
- `fedcond_grag/model/dual_graph_llm.py` — `samples["z_c_soft"]` overrides the
  condensed slot with a precomputed differentiable tensor.

**Enable:** `--server-stage-c-mode fedrag` (the fl-train default). Round 0
runs Phase 0 (repr-align init from anchors); rounds ≥ 1 run prompt tuning,
then memory adaptation, then delta aggregation. Knobs: `--syn-mem-steps`
(K_mem), `--syn-mem-lr`, `--syn-soft-tau`, `--lambda-gm`, `--lambda-align-mem`,
`--lambda-reg-mem`, `--eta-agg`, `--eta-reg`, `--server-reg-steps`. Legacy
server-side refinement remains available via
`--server-stage-c-mode gradient_match|repr_align|both`.

**Retrieval schedule (paper B.5.2):** during prompt tuning the synthetic
context slot z_c is produced by *differentiable soft retrieval* over the
frozen broadcast Θ_syn^(r,0) (pooled evidence prompt rep z̄_e attends over
projected synthetic nodes) — W is optimized through the synthetic branch.
Hard top-k retrieval over the exported synthetic graph is used only at
eval/inference. The condensed-branch encoder defaults to **GCN** so the soft
edge weights from the PGE carry gradient (GAT ignores scalar edge weights).

**Communication schedule:** the anchor C_m is uploaded exactly once (Phase 0,
first round the client participates); every later round carries only
`{W_m, Δ_m}`. The server treats anchor-less rounds as normal in fedrag mode.

**Privacy scope:** the upload is `{W_m, Δ_m}` plus the one-time anchor C_m —
no queries, answers, evidence graphs, or retrieval traces leave the client.

**Test:** `python -m pytest tests/test_stage_e_synthetic_memory.py`

---

## 8. Federated round loop

`fedcond_grag/trainer.py::FedTrainer` is the round-loop driver:

```python
for round_id in range(num_rounds):
    sampled = random.sample(range(num_clients), int(num_clients * client_frac))
    message_pool["round"]            = round_id
    message_pool["sampled_clients"]  = sampled

    server.send_message()                # round 0: empty; later: G_global
    for cid in sampled:
        client[cid].execute()            # re-condense if stale
        client[cid].send_message()       # upload anchor C_m
    server.execute()                     # Stage C gradient matching
```

The trainer assumes each client's Tri-Graph is already on disk at
`processed/<dataset>/client_<id>/trigraph.pt` (Stage A output). Stage B happens
inside `client.execute()` using the configured `ClientCondensationConfig`.

The trainer is intentionally lean (~110 lines) — no algorithm dispatch, no
gfl-style task registry, no wandb-table communication accounting. Add those
back if needed; they're not load-bearing for correctness.

### 8a. Federated LoRA aggregation baselines (`fedcond_grag/server/lora_aggregate/`)

By default (`--llm-frozen True`) Stage D only federates the GNN encoder +
projector — the LLM backbone never trains, so there's nothing to aggregate
for it. Passing `--llm-frozen False` turns on local LoRA fine-tuning of the
LLM (`fedcond_grag/model/graph_llm.py`'s existing peft `LoraConfig` branch)
on top of the usual GNN federation, and `--lora-agg-method` picks how the
server combines every sampled client's LoRA adapter each round. Ported from
FedLLM-Factory (https://github.com/boyi-liu/FedLLM-Factory, `alg/*.py`):

| `--lora-agg-method` | Idea |
|---|---|
| `fedit` (default) | Plain weighted average of `lora_A`/`lora_B` independently — the naive baseline every federated-LoRA paper compares against; averaging A and B separately only approximates averaging the true update `B@A`. |
| `flexlora` | Stack each client's (data-weighted) A/B along the rank axis, form the *exact* summed update `delta_W = B_stack @ A_stack`, SVD-truncate back to `--lora-rank` so the adapter shape is unchanged. |
| `rolora` | Alternates which half is trainable/aggregated by round parity (odd rounds: `lora_B`, even: `lora_A`) instead of touching both every round — avoids compounding FedIT's averaging error. |
| `flora` | Same exact-sum reconstruction as `flexlora`, but merges it straight into the frozen base weight and resets every client to the same fixed initial adapter next round, so successive rounds compose into a full-rank backbone change. **Requires `--llm-frozen False` and no `--llm-load-in-4bit`/`--llm-load-in-8bit`** — merging into a quantized `Params4bit`/`Int8Params` base weight isn't supported (raises `RuntimeError`). |

```bash
python main.py fl-train --dataset hotpotqa --num-clients 5 --num-rounds 10 \
    --llm-frozen False --lora-agg-method flexlora --lora-rank 8
```

Related flags: `--lora-rank`, `--lora-alpha`, `--lora-dropout`,
`--lora-target-modules` (comma-separated, default `q_proj,v_proj`) configure
the LoRA adapter itself; `--lora-agg-scale` is FlexLoRA's redistribution
factor (paper's `s`). See `fedcond_grag/server/lora_aggregate/__init__.py`.

---

## 9. CLI reference

All commands go through the root `main.py` shim, which dispatches to
`fedcond_grag.cli.main()`.

```bash
# Full data pipeline in ONE command — download → partition → Stage A/B →
# QA records/splits → PPR node maps (each step skips existing outputs).
# See docs/SETUP.md for the fresh-machine runbook.
python main.py preprocess --dataset hotpotqa --num-clients 3
python main.py preprocess --dataset hotpotqa --num-clients 3 --force
python main.py preprocess --dataset musique --num-clients 3 \
    --qa-out-root dataset/fedcond_qa_musique   # multi-dataset safe

# Federated round loop (Stage C aggregation; Stage B happens inside the client)
python main.py fl-train --dataset hotpotqa --num-clients 5 --num-rounds 1

# ...with federated LoRA fine-tuning of the LLM too (see §8a)
python main.py fl-train --dataset hotpotqa --num-clients 5 --num-rounds 10 \
    --llm-frozen False --lora-agg-method flexlora

# Stage D — centralized fit on the cached FedCondQA dataset
python main.py train \
    --dataset fedcond_qa --model_name dual_graph_llm --llm_frozen True \
    --gnn_in_dim 384 --gnn_hidden_dim 384 --gnn_in_dim_c 384 --gnn_hidden_dim_c 384 \
    --gnn_model_name gt --gnn_model_name_c gat --seed 0

# Stage D — inference + metrics (re-uses checkpoint from `train`)
python main.py infer \
    --dataset fedcond_qa --model_name dual_graph_llm --seed 0
```

Subcommand `--help` is shallow because each subcommand re-parses its remaining
argv inside the dispatched function; pass `-h` to a subcommand only for the
top-level (`python main.py train -h` shows just `-h`, not the full Stage D
flags). The full Stage D argparse lives in `fedcond_grag/config.py`.

`run.sh` at the repo root chains a few Stage D configurations across seeds —
edit it for your own sweep.

---

## 10. Where things live on disk

```
G-Retriever/
├── dataset/linearrag/<ds>/        # raw LinearRAG inputs
│     chunks.json                  # list[str] passages (LinearRAG format)
│     questions.json               # list of {id, question, answer, …}
│
├── processed/<ds>/                # all federated artifacts
│     questions.json
│     client_<m>/
│         chunks.json              # this client's slice
│         trigraph.pt              # Stage A output
│         text_bank.pt             # Stage B intermediate
│         condensed_graph.pt       # Stage B output (anchor C_m)
│         synthetic_graph.pt       # Stage C output (G_global broadcast)
│         linearrag_cache/         # parquet embedding stores
│
├── dataset/fedcond_qa/            # Stage D cache (built by preprocess_fedcond_qa.py)
│     records.jsonl
│     split/{train,val,test}_indices.txt
│     cached_graphs/<id>.pt
│     cached_condensed_graphs/<id>.pt
│     cached_desc/<id>.txt
│
└── output/<ds>/<args>.csv         # Stage D inference output → eval input
```

**Supported datasets** (`scripts/preprocess_data.py --dataset …`):
`hotpotqa`, `2wikimultihop`, `musique`, `medical`.

`hotpotqa`, `musique`, `2wikimultihop` are our primary evaluation benchmarks,
populated by `scripts/setup_datasets.py`. It pulls the standard 1000-question
dev split + full retrieval corpus for each (sourced from the copies checked
into [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)'s
`reproduce/dataset/`, not a fresh HuggingFace re-sample) so results stay
comparable across systems evaluated on the same benchmark. Raw files are
cached under `dataset/raw/`.

---

## 11. End-to-end run, 5 clients, hotpotqa

```bash
# 1-4. Everything data-side in one command (download → partition → Stage A/B
#      → QA records → PPR maps). Individual scripts remain runnable standalone;
#      see docs/SETUP.md §6 for the step ↔ script mapping.
python main.py preprocess --dataset hotpotqa --num-clients 5

# 5. Train DualGraphLLM (Stage D).
python main.py train \
    --dataset fedcond_qa --model_name dual_graph_llm --llm_frozen True \
    --gnn_in_dim 384 --gnn_hidden_dim 384 --gnn_in_dim_c 384 --gnn_hidden_dim_c 384 \
    --gnn_model_name gt --gnn_model_name_c gat --seed 0

# 6. Inspect the metric printed at the end of train, or re-run inference:
python main.py infer \
    --dataset fedcond_qa --model_name dual_graph_llm --seed 0
```

---

## 11a. Baselines (`fedcond_grag/baselines/`)

Third-party RAG methods evaluated against FedCondGraphRAG for comparison
live here, one subpackage per method. Each subpackage vendors or
pip-installs the method's own code unmodified and adds only a thin
per-client runner -- the point is a fair, reused-not-reimplemented baseline,
not a reimplementation of the method.

**`linearrag/`** -- the LinearRAG engine (NER, embedding store, PPR retrieval).
Dual-purpose: Stage A (`trigraph_builder.py`) and Stage D
(`linearrag_retriever.py`, `evidence_linearrag.py`) import it as the tri-graph
backend, and it also stands on its own as a baseline (plain LinearRAG
retrieval, no federated condensation) -- there's no dedicated per-client
runner script for it yet (unlike `hipporag/` below).

**`hipporag/`** -- per-client local HippoRAG (https://github.com/OSU-NLP-Group/HippoRAG):
```
fedcond_grag/baselines/hipporag/
└── client_runner.py     # builds each client's local corpus shard from dataset/raw/<name>_corpus.json
                          # (same idx % num_clients rule as preprocess_data.py), runs HippoRAG.index()
                          # + rag_qa() on that shard alone, evaluated against the *global* question set
```

A real pip dependency (installed straight from upstream's GitHub `main`,
pinned to a commit SHA -- see `requirements.txt`), not vendored.

This measures what happens to a traditional single-node graph-RAG method
when its corpus is fragmented across clients that can't see each other's
passages: each client answers the full benchmark test set using only its
own shard, so multi-hop questions whose evidence spans multiple clients are
expected to fail for most clients. That gap is the comparison point against
FedCondGraphRAG's federated retrieval.

Requires an OpenAI-compatible LLM endpoint (default: local Ollama at
`http://localhost:11434/v1`) for OpenIE + QA, and an embedding model
(default: `Transformers/sentence-transformers/all-MiniLM-L6-v2`, matching
the encoder used everywhere else in this repo).

```bash
# one-time: start a local OpenAI-compatible LLM server
ollama serve &
ollama pull qwen2.5:7b-instruct

# run every client, all 3 datasets
python scripts/run_hipporag_baseline.py --dataset all --num_clients 5

# just one client (e.g. to sanity-check before a full run)
python scripts/run_hipporag_baseline.py --dataset hotpotqa --client 0 --num_clients 5
```

Output: `%LOCALAPPDATA%/fedrag_baselines/hipporag/<dataset>/client_<m>/`
(HippoRAG's own index + OpenIE cache) and
`.../hipporag/<dataset>/summary.json` (per-client + mean recall@k / EM / F1
across clients). This is deliberately *not* under the repo -- HippoRAG's
OpenAI response cache uses `filelock.FileLock`, which fails on Windows over
a UNC/network path (this repo may live at `\\wsl.localhost\...`); override
with `--save_root` to change it.

**`gretriever/`** -- per-client local G-Retriever
(https://github.com/XiaoxinHe/G-Retriever). Not vendored: this project's own
`fedcond_grag/model/graph_llm.py` / `gnn.py` already *are* a fork of
G-Retriever's model (same GraphLLM class, same GCN/GraphTransformer/GAT
encoders, same BOS/EOS_USER/EOS prompt scheme). `client_runner.py` just
drives that existing model in single-client, non-federated mode: one client
trains + evaluates a fresh `GraphLLM` on its own Tri-Graph shard (built by
Stage A) with its own local PPR evidence retrieval, no Stage B/C, no
cross-client sharing.

```bash
python scripts/run_gretriever_baseline.py --dataset hotpotqa --num_clients 5
python scripts/run_gretriever_baseline.py --dataset hotpotqa --client 0 --num_clients 5
```

**`grag/`** -- per-client local GRAG (https://github.com/HuieL/GRAG). GRAG is
built on G-Retriever but its GNN is genuinely different (query-conditioned
node/edge features: `graph_encoder(x, edge_index, question_node, edge_attr,
question_edge)`), so unlike `gretriever/` it can't reuse this project's
existing model -- `model/gnn.py` + `model/graph_llm.py` and the retrieval
algorithm (`utils/graph_retrieval.py`, `utils/text_graph.py`) are vendored
under `_vendor/` (MIT licensed, see `_vendor/VENDORED.md` for the handful of
hardcoded-dimension / eager-import deviations needed to run it here).

GRAG's own method expects a real knowledge graph (subject --relation-->
object triples, e.g. from WebQSP), which this project's text-corpus
benchmarks don't have. `client_runner.py` builds one per client, preferring
`baselines/hipporag`'s cached OpenIE triples when that baseline has already
been run for the same dataset/client (reused as-is -- no reason to pay for
the LLM extraction twice), and otherwise falling back to this project's own
Tri-Graph with generic per-edge-type labels ("mentions"/"contains") standing
in for real relations.

```bash
python scripts/run_grag_baseline.py --dataset hotpotqa --num_clients 5
python scripts/run_grag_baseline.py --dataset hotpotqa --client 0 --num_clients 5
```

**`flare/`** -- per-client local FLARE (https://github.com/jzbjyb/FLARE).
Training-free, unlike the other three: FLARE iteratively generates a
temporary look-ahead continuation of the answer, decides whether to
retrieve by checking for low-confidence tokens (masking them to form the
query), and regenerates conditioned on what's retrieved -- repeated
sentence-by-sentence. Only the confidence-masking logic itself
(`ApiReturn`) is vendored under `_vendor/`; FLARE's own orchestration loop
is tightly coupled to the legacy OpenAI completions API (the only path in
their code that returns per-token logprobs -- verified empirically that
Ollama's `/v1/completions` doesn't return them, but `/v1/chat/completions`
does), an Elasticsearch+Wikipedia retriever, and dataset templates that
don't cover HotpotQA/MuSiQue, so `client_runner.py` reimplements the loop
against this project's own per-client corpus and local Ollama chat
completions (see `_vendor/VENDORED.md` for the full reasoning).

```bash
python scripts/run_flare_baseline.py --dataset hotpotqa --num_clients 5
python scripts/run_flare_baseline.py --dataset hotpotqa --client 0 --num_clients 5
```

**`comorag/`** -- per-client local ComoRAG
(https://github.com/EternityJune25/ComoRAG). Itself a fork of HippoRAG
(many files byte-identical to what `baselines/hipporag` uses) that adds a
"veridical / semantic / episodic" memory-pool retrieval loop on top --
iterative reasoning cycles that probe a shared memory pool and consolidate
newly retrieved evidence with what's already known, aimed at long-narrative
reasoning. Not pip-installable upstream, vendored under `_vendor/` (see
`_vendor/VENDORED.md` for the HippoRAG lineage and five packaging/bugfix
deviations, including a genuine upstream `EmbeddingStore` initialization
bug found and fixed via smoke testing).

```bash
python scripts/run_comorag_baseline.py --dataset hotpotqa --num_clients 5
python scripts/run_comorag_baseline.py --dataset hotpotqa --client 0 --num_clients 5
```

---

## 12. Testing surface

| Test file | Stage covered |
|---|---|
| `tests/test_data_pipeline.py` | A (data loading + partition) |
| `tests/test_linearrag_loader.py` | A (LinearRAG input loaders) |
| `tests/test_stage_b_graph_condensation.py` | B (motif, text bank, topology) |
| `tests/test_stage_c_fedcond_qa.py` | C (server gradient matching, `load_client`/`load_server`) |
| `tests/test_evidence_retrieval.py` | D (EvidenceLinearRAG + E_q) |
| `tests/test_global_graph_retriever.py` | D (G_q from G_global) |
| `tests/test_stage_d_dual_prompting.py` | D (DualGraphLLM encoding + collate) |

Run `python -m pytest tests/ -q` to verify the whole stack.

---

## 13. References

- `docs/plan/01_OVERVIEW.md` — original 4-stage design.
- `docs/plan/02_DATA_AND_TRIGRAPH.md` — Tri-Graph invariants.
- `docs/plan/03_CLIENT_CONDENSATION.md` — Stage B motif selection.
- `docs/plan/04_SERVER_CONDENSATION.md` — Stage C gradient matching.
- `docs/plan/05_INFERENCE_PROMPTING.md` — Stage D dual prompting.
- `docs/plan/06_TRAINING_EVAL.md` — Stage D training + eval.
- `docs/plan/08_APPENDIX_HYPERPARAMS.md` — hyperparameter reference.
