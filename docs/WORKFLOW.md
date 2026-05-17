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
├── linearrag/                      # LinearRAG engine — used by Stage A and Stage D
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
                  │  Stage C (FedCondQAServer.execute)
                  ▼
   processed/<ds>/client_<m>/synthetic_graph.pt    (G_global, broadcast)
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
- Backed by `fedcond_grag/linearrag/LinearRAG.index(passages)`.

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

**On disk:** `processed/<dataset>/client_<m>/condensed_graph.pt` (dict with
`x`, `edge_index`, `edge_weight`, `node_type`).

**Smoke test:** `python scripts/stage_b_smoke.py`

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

**On disk:** in the offline smoke pipeline, `processed/<dataset>/client_<m>/synthetic_graph.pt` is written per client (one copy per directory for caching). In a real federated run this artifact is broadcast in `message_pool["server"]`.

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

---

## 9. CLI reference

All commands go through the root `main.py` shim, which dispatches to
`fedcond_grag.cli.main()`.

```bash
# Stage A→B→C orchestrator (uses scripts/build_client_pipeline.py)
python main.py preprocess --dataset hotpotqa --num_clients 5
python main.py preprocess --dataset hotpotqa --clients 0 1 2 --force

# Federated round loop (Stage C aggregation; Stage B happens inside the client)
python main.py fl-train --dataset hotpotqa --num-clients 5 --num-rounds 1

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

---

## 11. End-to-end run, 5 clients, hotpotqa

```bash
# 0. Place raw LinearRAG inputs at dataset/linearrag/hotpotqa/{chunks,questions}.json

# 1. Partition the corpus into 5 client slices.
python scripts/preprocess_data.py --dataset hotpotqa --num_clients 5

# 2. Build Stage A→B→C artifacts for every client.
python main.py preprocess --dataset hotpotqa

# 3. (Optional) run the federated round loop end-to-end.
python main.py fl-train --dataset hotpotqa --num-clients 5 --num-rounds 1

# 4. Build the Stage D dual-graph cache (one PyG Data per question).
python scripts/preprocess_fedcond_qa.py

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
