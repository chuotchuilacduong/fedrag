# FedCondGraphRAG — Technical Flow (derived from code)

Last updated: 2026-06-11 (re-verified line-by-line against source). All details read
directly from code — no speculation.

---

## Overview

FedCondGraphRAG is a federated learning framework for multi-hop QA.
**M clients** each hold a private passage corpus. The server never sees raw text.
Four stages, run in order:

```
Stage A  (offline, per client)    Tri-Graph construction
Stage B  (Round 0, per client)    Anchor Graph condensation
Stage C  (every round, server)    Synthetic Graph optimization
Stage D  (Round ≥ 1, per client)  DualGraphLLM local training
```

Entry point: `main.py`. Subcommands:
- `preprocess` — Stage A + B (+ offline Stage C check) artifacts
- `fl-train`   — Stage C + D federated training loop
- `train` / `infer` — Stage D centralized fit / inference (legacy path, no FL)

---

## Entry Point: `main.py`

```
main.py
├── preprocess  → _run_preprocess()                          (main.py:38)
│   ├── runpy: scripts/preprocess_data.py                    # chunk partition
│   └── runpy: scripts/build_client_pipeline.py              # Stage A + B (+C) artifacts
│
├── fl-train    → _run_fl_train() → FedTrainer(args).train() (main.py:76)
├── train       → _run_train()   (Stage D centralized; wandb + early stopping)
└── infer       → _run_infer()   (Stage D inference + legacy csv metrics)
```

`--seed 0` (the default) **skips seeding entirely** (`main.py:153`); only a
non-zero seed calls `seed_everything`.

---

## Data Processing Pipeline (offline)

Four steps. Steps 0–1 are wrapped by `main.py preprocess`; steps 2–3 are
**separate scripts run manually** — fl-train hard-fails without their outputs.

### Step 0 — Chunk partition (`scripts/preprocess_data.py`)

```
INPUT:  dataset/linearrag/{dataset}/   (LinearRAG chunks + questions)

PROCESS (partition_linearrag_chunks, dataloader/data_preprocess.py:358):
  client[chunk.index % num_clients] ← chunk       # round-robin by chunk index
  (negative index falls back to client 0)
  Asserts: no overlap, full coverage, every client non-empty

OUTPUT: processed/{dataset}/client_{m}/chunks.json
        processed/{dataset}/questions.json        # shared, written once
```

### Step 1 — Per-client graph artifacts (`scripts/build_client_pipeline.py`)

For every client with `chunks.json`, runs Stage A → B → C with disk caching
(skip if output exists, `--force` to rebuild). Stage B/C run **on CPU**.

```
[A] trigraph.pt          — Tri-Graph from LinearRAG indexing (see Stage A)
[B] text_bank.pt         — MiniLM chunk/node embeddings for condensation
[B] condensed_graph.pt   — anchor graph cache (see Stage B)
[C] synthetic_graph.pt   — offline single-anchor Stage C run (128 syn nodes,
                           seed 42, CPU). Sanity/check artifact only —
                           fl-train's server builds its own synthetic graph
                           from scratch and never loads this file.
```

CLI: `--dataset --clients --force --topology-method {knn,self_expression} --entity-ratio` (default 0.05).

> There is **no** `processed/{dataset}/global/trigraph.pt` artifact. At
> runtime `FedTrainer._load_global_data()` just creates the `global/` dir and
> hands the server **a copy of client_0's trigraph** (`trainer.py:504`).

### Step 2 — PPR node maps (`scripts/preprocess_fedcond_qa.py`)

Precomputes, per client, which local trigraph passage nodes are relevant to
each question. Purely local — client c uses only its own chunks + trigraph.

```
PROCESS per client:
  - title→node map: passage nodes (node_type==2), title normalized by
    stripping a leading "N:" prefix, taking text before the first ":",
    lowercasing (first node wins on duplicate titles)
  - Index client chunks with EvidenceLinearRAG (all-MiniLM-L6-v2)
  - For each question (batches of 500): entity activation → BFS → PPR
    → top_k_passages → map each passage title to a local node ID (or -1)

OUTPUT: processed/{dataset}/client_{m}/ppr_node_map.pt
        [Q, top_k] int64; default --top_k_passages 5; -1 = no hit
```

> Missing `ppr_node_map.pt`, or a question whose row has no valid node ID on
> some client, → `RuntimeError` during training (`client.py:341,358`). No fallback.

### Step 3 — QA records (`scripts/build_fedcond_qa_dataset.py`)

```
PROCESS:
  1. Encode all questions with all-MiniLM-L6-v2 (normalized) → q_embs.pt [Q, 384]
  2. Per-question passage texts = "title: first-3-evidence-sentences" from
     q["evidence"]; encoded once, deduplicated across questions
  3. desc:
       - 0 passages           → the question text itself
       - ≤ TOP_K_DESC (5)     → all passages, ORIGINAL order (no ranking)
       - > 5                  → cosine-rank vs q_emb, top-5, joined by "\n\n"
  4. Sequential 80/10/10 split: indices [0,.8n) / [.8n,.9n) / [.9n,n)

OUTPUT: dataset/fedcond_qa/records.jsonl   {id, question, answer, desc, retrieved_passages}
        dataset/fedcond_qa/split/{train,val,test}_indices.txt
        dataset/fedcond_qa/q_embs.pt
```

> Output path is a fixed `dataset/fedcond_qa/` (no per-dataset subfolder) —
> rebuilding for another dataset overwrites it unless `--qa-data-root` points elsewhere.

### Runtime loader — `FedCondQADataset` (`dataloader/fedcond_qa_dataset.py`)

Each `dataset[i]` returns:

```python
{
  "idx": i,                                  # row into ppr_node_map
  "id": record_id,
  "question": f"Question: {q}\nAnswer: ",    # prompt-formatted here
  "label": str(answer).lower(),              # list answers joined with "|"
  "desc": cached_desc/{id}.txt  >  record["desc"]  >  retrieved_passages dump,
  "retrieved_passages": [...],
  "q_emb": q_embs[i],                        # [384] — attached but NOT consumed
}                                            # in the fl-train path (see below)
```

`top_r_passages`/`top_r_anchor` (re-ranked desc + anchor override) are plumbed
through CLI → dataset, but the required `passage_embs.pt`/`passage_node_map.pt`
loading is not implemented in the dataset — the path is currently inert
(`--top-r-passages` defaults to 0).

---

## Stage A: Tri-Graph Construction

**Code:** `fedcond_grag/client/stage_a_trigraph/`

```
INPUT:  client chunks.json

PROCESS:
  - Extract entities (E), sentences (S), passages (P)
  - Build edges — exactly 2 edge types:
      edge_type=0: S–E  (sentence contains entity)
      edge_type=1: P–E  (passage contains entity)
      (no S-P, no E-E, no P-P edges)
  - Encode all nodes with all-MiniLM-L6-v2 → 384-dim

OUTPUT: trigraph.pt
  x [N,384], edge_index [2,E], edge_type [E],
  node_type [N] (0=Entity, 1=Sentence, 2=Passage), node_text [N]
```

---

## Stage B: Client-Side Anchor Graph Condensation

**Code:** `client/client.py` → `_condense_anchor_graph()` (`client.py:445`)
**Core:** `client/stage_b_condense/client_condensor.py`

Runs on Round 0 only; `execute()` first tries the `condensed_graph.pt` cache.
The whole condensation runs under `torch.no_grad()` (`client.py:468`) — nothing
in Stage B is trained.

### Step 1: Anchor Node Selection (`anchor_node_selector.py`)

```python
score(v) = λ_idf·IDF(v) + λ_pr·PageRank(v) + λ_mmr·MMR(v)
# λ_idf=1.0, λ_pr=0.5, λ_mmr=0.3  (client.py:479-481)
```

- Entity budget: `entity_ratio · N_entities` (default 0.05)
- Sentence budget: 3, Passage budget: 3
- S-E-P motif expansion → `core_node_ids`

### Step 2: Hierarchical Text Condensation (`neighbor_gating.py:96`)

For each core node v, two things are computed:

```
(a) Hop context c_v (LOCAL-ONLY, goes to audit artifacts, NOT used in fusion):
    hops 0/1/2 with budgets (1, 3, 2)
    hop-2 candidates prefetched by degree difficulty 1/(1+deg), 4× budget
    per hop (score_and_select): cosine(g_v, neighbor text emb) →
      top-budget candidates, softmax weights over their scores
    c_v = Σ_h hop_weights[h] · (weighted sum of selected text embeddings)
    hop_weights = [0.4, 0.4, 0.2]

(b) t_tilde_v (THIS feeds the fusion step):
    collect the selected neighbor nodes from all hops
    flatten their text-chunk embeddings (select_chunks, chunk_selection.py:53)
    score chunks vs g_v (scaled dot, score_chunks) →
      top chunk_budget = 8 chunks, softmax weights
    t_tilde_v = weighted sum of those chunk embeddings
```

> Selection is relevance-ranked against the core node embedding g_v (cosine
> for neighbors, scaled dot for chunks) with softmax weights over the top-k —
> no learned parameters anywhere in Stage B.

### Step 3: Graph-Text Fusion (`graph_text_fusion.py`)

```python
# Identity additive fusion — no projections, gate is a fixed buffer (0.5):
x_fused = LayerNorm( x_core  +  0.5 · t_tilde )
# requires graph_dim == text_dim (both MiniLM 384); raises otherwise
```

The fused features stay in the **MiniLM embedding space** — this is what makes
the Stage D cosine retrieval (`mean(G_ev.x)` vs synthetic graph features)
geometrically meaningful. LayerNorm's affine params are never trained (γ=1,
β=0), so it acts as deterministic normalization.

### Step 4: Topology Reconstruction (`topology_reconstruction.py`)

Default `knn_topology`: cosine KNN on `x_fused`, k=8, S-E-P mask
(`preserve_sep_topology=True` — only E-S / E-P edges allowed).

Alternative `self_expressive_topology`:
```
min_C α‖CX − X‖²_F + β‖C‖₁ + ‖C ⊙ (1−S)‖²_F
α=8.0, β=5.0, candidate_size=16, 50 ISTA iterations, step 1e-2
```

### Output

```
ClientCondensedGraph → condensed_graph.pt:
  x [K, 384] fused embeddings, edge_index, edge_weight, node_type
  K ≪ N
```

Uploaded to server as numeric tensors only — no raw text.

---

## Stage C: Server-Side Synthetic Graph Optimization

**Code:** `server/server.py` → `execute()` (`server.py:96`)

### execute() — actual order of operations

```
1. Collect anchor graphs C_m from sampled clients (early-return if none)
2. First call: init_synthetic_graph(anchor_graphs)
3. If global_model_state exists (i.e. round ≥ 2) and mode uses repr_align:
     load PREVIOUS round's FedAvg weights into repr_encoder       (server.py:110)
4. Run optimization loop (server_condense_iters = 50 steps)
5. _fedavg_model_weights()  — aggregate THIS round's client uploads (server.py:137)
     → updates global_model_state AND refreshes repr_encoder
6. send_message()  — broadcast synthetic graph + new w_global      (server.py:138)
```

> **Timing nuance:** the Stage C optimization at round *t* uses the encoder
> weights FedAvg'd at round *t−1*. The round-*t* FedAvg happens AFTER the
> optimization loop, and is what gets broadcast to clients.
> On round 1, repr_align runs with the randomly-initialized repr_encoder
> (no FedAvg weights exist yet).

### Initialization (`init_synthetic_graph`, server.py:140)

```python
# N_syn = num_global_syn_nodes (fl-train CLI default: 128)
# Per-type node counts proportional to type ratio across all anchor graphs.
# Init per type: SAMPLE REAL ANCHOR NODE FEATURES (with replacement)
#   init = anchor_features[rand_idx] + 0.01·randn
# Fallback randn·0.02 only if a type has zero anchor nodes.

synthetic_x = nn.Parameter(...)                     # learnable
pge = TypeAwarePGE(hidden=256, type_emb=16, topk=8, preserve_sep=True)
optimizer = Adam([synthetic_x, *pge.parameters()], lr=lr_feat=1e-2)
```

### Optimization Modes (mode = `server_stage_c_mode`, CLI default `repr_align`)

**`gradient_match`** (`server.py:113`):
```
For each C_m:
  loss_m = surrogate_loss(SurrogateGNN(C_m, dense_adj_m))   # type cls + link, weights 1.0/0.5
  g_m    = ∂loss_m/∂θ_surrogate
g_anchor = Σ coeff_m · g_m     where coeff_m = num_nodes(C_m) / total_anchor_nodes
                               # weighted by NODE COUNT, not sample count

adj_syn = PGE(synthetic_x, node_type)                       # sparsified
g_syn   = ∂surrogate_loss(synthetic_x, adj_syn)/∂θ          # create_graph=True (2nd order)
L_gm    = gradient_match_loss(g_syn, g_anchor)
→ backward → update synthetic_x + PGE
```

**`repr_align`** (`server.py:280`, `stage_c_aggregate/repr_align.py`):
```
target_degree = compute_target_degree(anchor_graphs)
H_m = repr_projector(repr_encoder(C_m, edge_weight_m))    # PER-NODE [N_m, H], detached

adj_soft    = PGE(synthetic_x, node_type, sparsify=False)  # soft → grad flows to PGE
edge_weight = adj_soft[rows, cols]
H_syn       = repr_projector(repr_encoder(synthetic_x, edge_index, edge_weight))  # [K_g, H]

# Attention-reconstruction alignment — node-level, NOT graph-pooled:
A_m  = softmax(H_m @ H_synᵀ / √H, dim=1)                  # [N_m, K_g]
L_ra = Σ_m (1/Σ_j N_j) · ‖H_m − A_m·H_syn‖²_F             # each anchor node must be
                                                           # reconstructible from syn nodes
     + λ_div·diversity_loss(H_syn)            # mean |cos| off-diag, λ_div = 0.1
     + λ_deg·degree_regularization(adj_soft)  # MSE to anchor avg degree, λ_deg = 0.05
```

**`both`** (`server.py:312`): `L = w_gm·L_gm + w_ra·L_ra`, single backward.

A best-loss snapshot (`best_state`) is tracked across steps, but
`export_synthetic_graph()` exports the **current** state, not the best one.

### FedAvg (`_fedavg_model_weights`, server.py:366)

```
w_global[key] = Σ (n_m / N_total) · w_m[key]      # weighted by num_samples
keys: graph_encoder, projector, condensed_encoder, projector_c
      (condensed_* present only in dual mode)

repr_encoder refresh (_load_repr_align_weights, server.py:446):
  prefers condensed_encoder + projector_c (dual mode);
  FALLS BACK to graph_encoder + projector (shared mode).
  repr_projector is rebuilt if output dim mismatches the LLM hidden size.
```

### Broadcast (`send_message`, server.py:80)

```
Server → message_pool["server"]:
  synthetic_x / synthetic_adj / synthetic_node_type   (raw tensors)
  synthetic_graph  (Data: x, edge_index, edge_weight, node_type
                    — via PGE.inference + threshold)
  model_weights    (w_global, once it exists)
```

---

## Stage D: DualGraphLLM Local Training

**Code:** `client/client.py` → `local_train()` (`client.py:169`)
**Model:** `model/dual_graph_llm.py` (extends `model/graph_llm.py`)

### Training data paths (two modes, `trainer.py:402`)

```
max_train_per_client = 0 (default):
  set_local_qa_data(samples)   — stores RAW samples only (client.py:85)
max_train_per_client > 0:
  set_full_train_pool(samples) — stores raw pool; each round
  sample_train_for_round() draws a fresh random subset of size max_per_round
```

**Neither path pre-attaches graphs.** Evidence + condensed graphs are built
**per mini-batch** inside the training loop (`client.py:225-230`), so startup
cost is O(1) in pool size. (A code comment at `trainer.py:406` still says
"Pre-attach evidence graphs for ALL samples" — stale; the code is lazy.)

Train samples are assigned round-robin: dataset index i → client `i % num_clients`.

### Evidence Graph — `_attach_evidence_graphs()` (`client.py:306`)

```python
# CPU adjacency list of the local trigraph is built once per client lifetime.
for sample in batch:
    row = ppr_node_map[sample["idx"]]          # full row, width = top_k (5)
    seeds = [n for n in row if 0 <= n < N]     # ALL valid entries used
    # RuntimeError if seeds empty or ppr_node_map.pt missing — no fallback

    kept = seeds ∪ {1-hop neighbors in local trigraph}
    sample["graph"] = Data(x=trigraph.x[kept], edge_index=induced_edges,
                           edge_weight=ones, node_type=..., node_text=[...])
```

`desc` is deliberately NOT overwritten with PPR passage text — the record's
cosine-ranked gold evidence stays as the LLM text context; the local federated
knowledge enters only through the graph soft-prompt token (`client.py:403`).
(A leftover `top_k_desc` variable at `client.py:338` is computed but unused.)

### Condensed Graph — `_attach_condensed_graphs()` (`client.py:413`) + `GlobalGraphRetriever`

```python
# One batched matmul for the whole mini-batch:
queries     = stack([mean(sample.graph.x) for each sample])    # [B, 384]
scores_all  = normalize(G_global.x) @ normalize(queries).T     # cosine, [K, B]

# Per sample:
seeds   = top-r nodes by cosine (retrieval_top_r = 16)
kept    = seeds ∪ 1-hop neighbors along G_global edges          # NOT just top-r
          (optionally clamped to max_nodes by score)
sample["condensed_graph"] = induced subgraph of G_global over kept
```

The query is the **mean-pooled evidence-graph features**, not the question
embedding — `item["q_emb"]` is carried through collate but never consumed in
the fl-train path.

### DualGraphLLM Forward (`dual_graph_llm.py:119`)

```python
z_e = projector(mean_pool(graph_encoder(G_ev)))      # [B, H]
z_c = ...                                            # depends on dual_graph_mode:

#   "shared" (DEFAULT) → condensed graph through the SAME graph_encoder+projector;
#                        condensed_encoder/projector_c are NOT created (None)
#   "no_synthetic"     → also no separate encoders; BOTH slots encode G_ev
#   "both"/"dual"      → separate condensed_encoder + projector_c created
#   "evidence_only"    → z_c ·= 0     "condensed_only" → z_e ·= 0
#   "random_condensed" → z_c = random noise
#   "none"/"text_only" → both ·= 0
# (zeroing is multiply-by-0 to preserve grad_fn under the frozen LLM)

# Input sequence per sample:
[BOS] [z_e] [z_c] [desc≤max_txt_len] [question] [EOS_USER] [label≤max_new_tokens] [EOS]
# - one batched word_embedding call, LEFT-padding, labels = IGNORE_INDEX (-100)
#   everywhere except the label tokens
# - BOS/EOS_USER/EOS resolved from the tokenizer (resolve_prompt_template,
#   graph_llm.py:23):
#     Qwen Instruct (eos <|im_end|>)  → ChatML markers
#     Qwen base (eos <|endoftext|>)   → plain format: "" / "\nAnswer:" / <|endoftext|>
#       (ChatML on base models makes them echo the input instead of answering)
#     Llama (bos <s>)                 → <s>[INST] / [/INST] / </s>
#     anything else                   → tokenizer bos / "\nAnswer:" / tokenizer eos
# - bos/pad embeddings cached once in __init__; an empty bos_text ("") is
#   handled with an explicit empty LONG tensor (tokenizer("") returns a float
#   tensor that would crash nn.Embedding) — graph_llm.py:183-189

loss = LLM(inputs_embeds, attention_mask, labels)    # LLM frozen; CE loss on answer
```

`inference()` is the same prompt without the label, then `model.generate()`
(max `eval_max_new_tokens`, default 16). Labels end with the tokenizer's real
EOS, so generation stops at the native `eos_token_id`; decoding uses
`skip_special_tokens=True` and strips whitespace — no string-level truncation.

### LLM backbone (`graph_llm.py`)

- `llm_model_path` resolved via the `llama_model_path` registry
  (`model/__init__.py`): Llama-2 7b/13b (±chat), Qwen2-1.5B,
  Qwen2.5-1.5B/7B(-Instruct). fl-train default name: `"7b"`.
- Loaded fp16 (or 4-bit nf4 / 8-bit via flags), SDPA attention,
  `device_map="auto"`. `llm_frozen="True"` (fl-train default) freezes all LLM
  params; otherwise LoRA (r=8, α=16, q_proj/v_proj) — the LoRA path is not
  used by fl-train.
- FedTrainer additionally sets `requires_grad` only for
  `graph_encoder/projector/condensed_encoder/projector_c` (`trainer.py:446`).

### Encoders (effective fl-train defaults)

- `graph_encoder`: `gnn_model_name` = **gcn** (main.py CLI default), 4 layers,
  in/hidden 384, edge_attr passed to conv layers, bfloat16
- `projector`: Linear(384→2048) → GELU → Linear(2048→LLM hidden)
- `condensed_encoder`/`projector_c`: only created in `both`/`dual`/ablation
  modes other than shared/no_synthetic
- `_init_stage_d` fallback defaults (`gt`/`gat`, `trainer.py:428`) apply only
  when the attrs are missing — the fl-train CLI always sets gcn/gcn.

### Optimization (`client.py:191`)

```python
# Trainable: graph_encoder, projector (+ condensed_encoder, projector_c if present)
# Frozen:    LLM backbone
optimizer = AdamW(lr=local_lr=1e-4, weight_decay=local_wd=0.05, betas=(0.9, 0.95))
# Created once per client, PERSISTED across rounds (Adam moments stay warm)
clip_grad_norm_(trainable, local_grad_clip=1.0)
local_epochs = 3, local_batch_size = 4  (defaults); samples shuffled per epoch
# per-step loss logged to WandB step/* with a monotone global_step
```

After training, the client snapshots the present state dicts and uploads
`{anchor_graph, num_anchor_nodes, model_weights, num_samples}` (`client.py:276`).

---

## FL Round Loop (`trainer.py` → `FedTrainer.train()`)

```
FedTrainer.__init__():
  ├── FedCondQAClient × num_clients  (trigraph.pt + ppr_node_map.pt per client)
  ├── FedCondQAServer                (global data = a copy of client_0's trigraph)
  ├── _init_stage_d() only if num_rounds > 1:
  │     FedCondQADataset (records.jsonl) → train/val/test indices
  │     train samples round-robin: index i → client (i % num_clients)
  │     eval sets capped at max_eval_samples (200) — FIRST max_eval indices
  │     DualGraphLLM loaded; only graph_encoder/projector(±_c) require grad
  │     (any failure → "Stage D disabled", loop still runs Stage B/C only)
  ├── /tmp/fl_metrics.jsonl reset
  └── .env loaded; WandB init only if WANDB_API_KEY present
        run name auto-derived from dual_graph_mode; two x-axes:
        round/* vs comm_round, step/* vs global_step

train(), each round r ∈ [0, num_rounds):
  sampled = sorted(random.sample(clients, max(1, int(num_clients · client_frac))))
            # fraction-based; applies to EVERY round including round 0

  server.send_message()              # round 0: empty dict (nothing exists yet)

  for cid in sampled:
    client.receive_message()         # synthetic graph + w_global (if present)
    client.execute()                 # Stage B: round 0 build/load cache; later no-op
    if r ≥ 1 and Stage D ready:
      client.sample_train_for_round()  (only if max_train_per_client > 0)
      client.local_train()           # Stage D
    client.send_message()            # anchor graph (+ weights, num_samples)

  server.execute()                   # Stage C optimize → FedAvg → broadcast

  if r ≥ 1 and Stage D ready and r % eval_every == 0:
    val/test metrics via _eval_split_acc()

  round metrics → /tmp/fl_metrics.jsonl + WandB round/*

After all rounds: summary table + sample predictions (first 10 test samples,
graphs attached via client_0, hit mark = plain lowercase substring).
```

---

## Evaluation

### fl-train per-round eval (`_eval_split_acc`, trainer.py:216)

```
- Eval samples sharded round-robin: sample i → client (i % num_clients)
- Each client re-attaches evidence + condensed graphs (same retrieval code
  path as training; rebuilt on every eval call)
- DualGraphLLM.inference() in batches of eval_batch_size (16), torch.no_grad
- THREE METRICS (utils/evaluate.py), all in percent over len(samples):

    normalize(s): lowercase → strip punctuation → drop articles (a/an/the)
                  → drop "<pad>" → collapse whitespace

    hit = normalize(label) in normalize(pred)   # substring containment
    em  = normalize(pred) == normalize(label)   # SQuAD-style exact match
    f1  = token-level F1 on normalized tokens   # SQuAD/MuSiQue-style
          (Counter intersection → precision/recall → harmonic mean;
           empty pred or gold → 1.0 iff both empty, else 0.0)
```

**Backward compatibility:** `val_acc`/`test_acc` in the summary table, the
jsonl, and WandB (`round/val_acc`, `round/test_acc`) **are the hit metric** —
same formula and keys as historical runs, so old/new runs overlay directly.
EM/F1 are additive: `round/val_em`, `round/val_f1`, `round/test_em`,
`round/test_f1`. `train_acc` is declared but never computed (always None/N/A).

### Centralized `train`/`infer` subcommands (legacy path)

Predictions are written to a csv/jsonl file, then scored by
`get_accuracy_fedcond_qa(path)` (`utils/evaluate.py:73`): splits multi-answer
labels on `"|"`, prediction on newlines, prints acc/hit/precision/recall/F1,
and **returns hit** as "Test Acc".

---

## Key Data Structures at Runtime

| Object | Shape | Description |
|---|---|---|
| `trigraph.pt` | dict → Data | x [N,384], edge_index, edge_type, node_type, node_text |
| `ppr_node_map.pt` | [Q, 5] int64 | local passage node IDs per question; -1 = miss |
| `condensed_graph.pt` | Data | x [K,384], edges, edge_weight, node_type — Stage B output |
| synthetic graph | Parameter + PGE | x [128,384] learnable; adjacency from TypeAwarePGE |
| `G_ev` (per sample) | Data | PPR seeds + 1-hop subgraph of local trigraph |
| `G_cn` (per sample) | Data | top-16 cosine seeds + 1-hop subgraph of G_global |
| `records.jsonl` | JSONL | {id, question, answer, desc (top-5 ranked), retrieved_passages} |

`collate_fn` (`utils/collate.py`): dict-of-lists; `graph`/`evidence_graph`/
`condensed_graph` become PyG `Batch` only if present on **all** samples of the
mini-batch — mixed None drops the key (model then zero-fills that slot).

---

## Configuration Reference (effective `fl-train` defaults from `main.py`)

| Arg | Default | Description |
|---|---|---|
| `--dataset` | hotpotqa | Dataset name |
| `--num-clients` | 2 | FL clients |
| `--num-rounds` | 1 | Total rounds (>1 needed to enable Stage D) |
| `--client-frac` | 1.0 | Fraction of clients sampled per round |
| `--seed` | 0 | **seed=0 skips seeding entirely** (`main.py:153`) |
| `--num-global-syn-nodes` | 128 | Synthetic graph size (CLI overrides config.py's 1024) |
| `--server-condense-iters` | 50 | Stage C steps per round |
| `--server-stage-c-mode` | repr_align | CLI overrides config.py's gradient_match |
| `--lambda-div` / `--lambda-deg` | 0.1 / 0.05 | repr_align regularizers |
| `--repr-align-weight` / `--grad-match-weight` | 1.0 / 1.0 | mode="both" loss weights |
| `--hid-dim` / `--num-layers` | 64 / 2 | SurrogateGNN (gradient_match) |
| `--gnn-model-name` / `-c` | gcn / gcn | Stage D encoder architectures |
| `--gnn-num-layers` | 4 | layers; in/hidden dim 384, heads 4 |
| `--local-epochs` | 3 | Epochs per client per round |
| `--local-lr` / `--local-wd` | 1e-4 / 0.05 | Client AdamW |
| `--local-batch-size` | 4 | Train batch size |
| `--eval-batch-size` | 16 | Inference batch size |
| `--local-grad-clip` | 1.0 | Gradient clipping |
| `--retrieval-top-r` | 16 | Cosine seeds from G_global per sample |
| `--max-txt-len` / `--max-new-tokens` | 512 / 64 | Token truncation (train) |
| `--eval-max-new-tokens` | 16 | Generation length at eval |
| `--eval-every` | 1 | Eval every N rounds |
| `--max-train-per-client` | 0 | 0 = use all; >0 = per-round random subset |
| `--max-eval-samples` | 200 | Eval set cap |
| `--dual-graph-mode` | shared | See DualGraphLLM modes |
| `--llm-model-name` / `--llm-model-path` | "7b" / "" | Resolved via `llama_model_path` registry |
| `--llm-load-in-4bit` / `--llm-load-in-8bit` | off | Quantization flags |
| `--llm-gradient-checkpointing` | off | ~4× less activation memory |
| `--qa-data-root` | dataset/fedcond_qa | records.jsonl location |
| `--data-root` | processed | trigraph/ppr artifacts root |
| `--wandb-run-name/-tags/-group` | auto | WandB metadata |
| `--top-r-passages` | 0 | Re-ranked desc path (currently inert — see Step 3 loader) |
| `--top-r-anchor` | None | Anchor count override for the re-ranked path |

**`server/stage_c_aggregate/config.py`** is applied only for attrs *missing*
on args — since the fl-train CLI always sets `num_global_syn_nodes` (128) and
`server_stage_c_mode` (repr_align), the config.py values 1024 / gradient_match
never take effect in `fl-train`. Values that do apply: `lr_feat=1e-2`,
`pge_hidden=256`, `pge_topk=8`, `type_emb_dim=16`, `surrogate_type_weight=1.0`,
`surrogate_link_weight=0.5`, `repr_proj_out_dim=4096`.

---

## Design Decisions (what the code actually does)

1. **No raw text leaves clients.** Anchor graphs are numeric tensors only
   (x, edge_index, edge_weight, node_type).

2. **Stage B is no-grad.** `ClientCondensor` runs under `torch.no_grad()`;
   nothing in Stage B is trained.

3. **Fusion is identity-additive with a fixed gate, in MiniLM space.**
   `x_fused = LayerNorm(x + 0.5·t̃)` — no projections, no learned gate.
   Keeping anchor features in the original embedding space is what makes the
   Stage D cosine retrieval against the synthetic graph meaningful.

4. **`t̃` comes from chunk embeddings, relevance-ranked.** The hop-weighted
   context is computed but stored only in local audit artifacts; the fused text
   vector is a softmax-weighted sum of the top-8 chunks scored against the core
   node embedding (scaled dot). Neighbor selection per hop uses cosine top-k.
   No learned parameters anywhere in Stage B.

5. **PPR anchors are mandatory.** Missing `ppr_node_map.pt` or an all-invalid
   row → `RuntimeError`. No fallback.

6. **Graph attachment is fully lazy.** Both `set_local_qa_data` and
   `set_full_train_pool` store raw samples; evidence + condensed graphs are
   built per mini-batch inside `local_train()` (and per eval call). Startup
   cost is O(1) in dataset size — required for the full 33k-question runs.

7. **Stage C lags FedAvg by one round.** `server.execute()` optimizes the
   synthetic graph with the *previous* round's FedAvg'd repr_encoder, then
   FedAvg-aggregates the current round's uploads, then broadcasts. Round 1's
   repr_align therefore runs with a randomly-initialized encoder.

8. **Anchor gradients are node-count weighted; FedAvg is sample-count weighted.**
   `compute_anchor_gradients` weights by `num_nodes(C_m)/total_nodes`;
   `_fedavg_model_weights` weights by `num_samples`.

9. **Synthetic nodes are initialized from real anchor features** (sampled with
   replacement + 0.01 noise), not pure random; pure random only for node types
   absent from all anchor graphs.

10. **Shared encoder by default.** `dual_graph_mode="shared"` reuses one GNN +
    projector for both graph tokens; `condensed_encoder`/`projector_c` are not
    even instantiated (also true for `no_synthetic`). On the server,
    repr_encoder then falls back to the FedAvg'd `graph_encoder` weights.

11. **Condensed retrieval is seeds + 1-hop, batched.** Top-16 cosine seeds in
    G_global, expanded 1 hop along synthetic edges, induced subgraph; all
    queries of a mini-batch scored in a single matmul. Query = mean-pooled
    evidence-graph features (MiniLM space), not the question embedding.

12. **Eval reports hit / EM / F1; hit IS the legacy acc.** hit = normalized
    substring containment, kept under the historical `val_acc`/`test_acc` keys
    (WandB + jsonl) so old and new runs are directly comparable; EM/F1 are
    additive. Prompt markers are tokenizer-native (`resolve_prompt_template`,
    with a base-vs-instruct Qwen distinction), so generation stops at the real
    `eos_token_id` — no string-level truncation.

13. **Client optimizer persists across rounds** (AdamW, betas (0.9, 0.95)) so
    Adam's second moment stays warm.

14. **`--seed 0` (the default) skips seeding** — runs are non-deterministic
    unless a non-zero seed is passed.
