# FedCondGraphRAG — Implementation Plan

> **Federated Graph Condensation for Retrieval-Augmented Generation over Textual Graphs**
>
> Một framework kết hợp:
> - **LinearRAG** (ICLR'26) — relation-free Tri-Graph + two-stage retrieval
> - **DANCE** (ICML 2025-style) — federated graph condensation cho TAG
> - **G-Retriever** (NeurIPS 2024) — graph soft-prompt cho LLM
>
> Mục tiêu: học một **global condensed graph** từ nhiều client local QA corpora, sử dụng nó như một "structural memory" bổ sung cho LinearRAG evidence retrieval ở thời điểm inference, qua một LLM frozen với **dual graph prompting**.

---

## Part I — Research Narrative

### 1. Motivation

Hiện trạng của ba dòng nghiên cứu liên quan:

1. **GraphRAG cổ điển** (LightRAG, HippoRAG, GraphRAG của Microsoft) dựa nặng vào **relation extraction** để xây knowledge graph. Phân tích của LinearRAG cho thấy hai loại lỗi cơ bản:
   - **Local inaccuracy**: triple sai về nghĩa (ví dụ "Einstein won Nobel Prize for theory of relativity" — vốn sai về mặt sự thật).
   - **Global inconsistency**: thiếu cơ chế hợp nhất các quan hệ giữa các đoạn văn, tạo ra subgraph mâu thuẫn.
   Hậu quả: nhiều GraphRAG **không tốt hơn vanilla RAG** trên các benchmark thực tế.

2. **LinearRAG** giải bằng cách bỏ relation extraction, dùng Tri-Graph `Entity / Sentence / Passage` với hai loại cạnh `Sentence–Entity` và `Passage–Entity`. Retrieval gồm hai stage: (i) **entity activation** qua semantic bridging trên subgraph entity-sentence, và (ii) **passage retrieval** qua Personalized PageRank trên subgraph entity-passage. Nhưng LinearRAG là một hệ **centralized** và **không có cơ chế cross-corpus** — mỗi câu hỏi chỉ retrieve trên một corpus.

3. **DANCE** giải bài toán **federated learning trên TAG**: mỗi client có một text-attributed graph riêng, dùng node condensation theo class + hierarchical text condensation theo budget + self-expressive topology reconstruction. Nhưng DANCE giả định **node classification (có nhãn)** và không liên quan đến RAG/QA.

4. **G-Retriever** chỉ ra rằng nếu graph quá lớn để textualize hoàn toàn, có thể dùng **graph soft prompt** (GNN encoder → projection → concat với text embedding) để cho LLM "nhìn" graph. Nhưng G-Retriever giả định graph là **cố định và centralized**.

**Khoảng trống**: chưa có công trình nào kết hợp được:
- Tri-Graph chất lượng cao (không phụ thuộc relation extraction),
- học liên cơ sở (federated) mà không upload raw text,
- và đưa "common structural memory" này vào LLM ở dạng soft prompt cùng với retrieval truyền thống.

### 2. Research Questions

- **RQ1**: Có thể condense client-side Tri-Graph thành một anchor graph nhỏ mà vẫn giữ được **cấu trúc Sentence–Entity–Passage (S-E-P)** không?
- **RQ2**: Một **global synthetic graph** học từ nhiều anchor graphs (qua gradient matching + PGE) có thể đóng vai trò "structural prior" cross-corpus cho QA không?
- **RQ3**: Khi LLM đã có LinearRAG passages, thêm graph token `z_e` (evidence) và `z_c` (condensed global) có cải thiện QA không, và đóng góp của từng loại là bao nhiêu?

### 3. Key Idea (one-liner)

> **Học một global "skeleton graph" embedding-only từ các Tri-Graph được condense ở mỗi client; ở inference, evidence graph cụ thể theo query (LinearRAG) và global skeleton subgraph được mã hoá song song bởi hai GNN và đưa vào LLM dưới dạng dual soft prompt.**

### 4. Framework Overview

```text
[ Phase 1: Client offline ]
  Local QA Corpus
    -> LinearRAG Tri-Graph (Entity, Sentence, Passage)
    -> Query-agnostic Structure-Semantic Motif Selection (S-E-P core)
    -> DANCE-style budgeted text condensation per core node
    -> Self-expressive (hoặc KNN) topology reconstruction
    -> Client Condensed Graph C_m (embedding-only)
    -> Upload C_m (no raw text)

[ Phase 2: Server offline ]
  Receive {C_1, ..., C_N}
    -> Anchor gradients qua surrogate task
    -> Init global synthetic graph (X_global, A_global=PGE(X_global))
    -> Gradient matching để học X_global + PGE params

[ Phase 3: Inference per query q ]
  q -> LinearRAG retrieval -> (P_q passages, E_q evidence graph)
  q -> Global graph retrieval -> G_global(q)  (top-r nodes + 1-hop)
  E_q -> Evidence GNN encoder -> z_e
  G_global(q) -> Condensed GNN encoder -> z_c
  LLM input = [z_e ; z_c ; embedded(textualize(E_q) ⊕ P_q ⊕ q)]
  -> Answer Y
```

Điểm thiết kế quan trọng:
- **Condensed graph KHÔNG textualize**. Nó là vector graph qua GNN, đóng vai trò *structural memory* bổ sung. Grounding sự thật vẫn dựa vào passages của LinearRAG.
- **Topology S-E-P phải được bảo toàn** trong client condensation — nếu không, core graph rơi rạc và mất ý nghĩa.

### 5. Expected Contributions

1. **FedCondGraphRAG** — framework đầu tiên kết hợp federated graph condensation với graph RAG cho QA.
2. **Query-agnostic Structure-Semantic Motif Selection** — thay cho label-aware node condensation của DANCE, phù hợp với QA setting nơi không có nhãn class.
3. **Server-side global condensed graph với PGE + gradient matching** trên Tri-Graph với 3 loại node.
4. **Dual graph prompting** (z_e + z_c) với LLM frozen, bám sát kiến trúc G-Retriever.
5. Phân tích thực nghiệm: ảnh hưởng của (i) cách chọn core, (ii) topology reconstruction, (iii) đóng góp của z_e vs z_c, và (iv) sensitivity với compression ratio.

---

## Part II — Scope & Minimum Viable Version

### 6. Scope của bản đầu tiên

**Use first**

```text
Dataset:               HotpotQA (multi-hop, có supporting facts)
Clients:               5 (partition theo hash(document_title))
Graph construction:    LinearRAG Tri-Graph
Client core selection: Query-agnostic S-E-P Motif Selection
Client text cond.:     DANCE-style budgeted neighbor gating + chunk selection
                       (top-k softmax đầu tiên, entmax để sau)
Client topology:       KNN baseline trước → self-expressive sau
Server:                Synthetic X_global + PGE adjacency
Server training:       Gradient matching (trajectory matching defer)
Prompting:             G-Retriever style với 2 graph tokens (concat late fusion)
LLM:                   Llama-2-7B frozen (như G-Retriever gốc)
Graph encoder:         Graph Transformer / GAT (mỗi side một bản)
```

**Defer to later**

```text
Trajectory matching
Differential privacy / secure aggregation
Cross-client multi-hop partition (supporting facts spread across clients)
Community-aware core selection
Full entmax với straight-through estimator
LoRA fine-tuning trên LLM
2WikiMultiHopQA, MuSiQue datasets
```

### 7. Codebase Strategy

- Bắt đầu từ **G-Retriever** repo (`XiaoxinHe/G-Retriever`) làm khung. Giữ nguyên thư mục `src/model/`, `src/dataset/`, scripts huấn luyện.
- Thêm package mới `fedcond_grag/` ngang cấp với `src/`.
- Tái dùng tối đa: `GraphTransformer`, projection layer, LLM wrapper, `textualize_graph()` utility.
- Tái implement (vì LinearRAG/DANCE không có common code): Tri-Graph builder, motif selector, client condensor, PGE, gradient matching.

### 8. Code Structure (file-level)

```text
fedcond_grag/
  data/
    hotpot_loader.py            # Load HotpotQA → corpus/questions/SP-facts
    twowiki_loader.py           # (later)
    musique_loader.py           # (later)
    corpus_index.py             # passage_id, sentence_id ↔ text
    federated_partition.py      # split corpus thành N client buckets

  graph_building/
    entity_extractor.py         # spaCy NER + normalization
    trigraph_builder.py         # build Tri-Graph(Entity, Sentence, Passage)
    node_encoder.py             # SentenceBERT / all-MiniLM / E5-small
    graph_store.py              # save/load .pt files

  graph_condensation/
    motif_core_selector.py      # S-E-P motif selection (entity-anchor first)
    text_bank.py                # frozen text encoder + chunk cache
    neighbor_gating.py          # cross-modal attention, hop budgets
    chunk_selection.py          # token budget, entmax later
    graph_text_fusion.py        # gated fusion → x_fused
    evidence_prior.py           # S_ij prior matrix
    topology_reconstruction.py  # KNN baseline + self-expression
    client_condensor.py         # orchestrator per client

  server_condensation/
    anchor_gradient.py          # compute target gradients từ C_m
    synthetic_graph.py          # learnable X_global, init strategies
    pge.py                      # Parameterized Graph Estimator MLP
    gradient_matching.py        # main server optimizer
    trajectory_matching.py      # (later)

  retrieval/
    linearrag_retriever.py      # entity activation + PPR passage retrieval
    evidence_graph_builder.py   # build E_q từ activated entities & passages
    global_graph_retriever.py   # top-r + 1-hop trên (X_global, A_global)

  models/
    evidence_gnn_encoder.py     # GNN on E_q → z_e
    condensed_gnn_encoder.py    # GNN on G_global(q) → z_c
    graph_token_fusion.py       # concat / gated / attention fusion
    dual_graph_prompt_model.py  # full pipeline: 2 GNN + LLM forward
    evidence_relevance_head.py  # surrogate task head (optional)

  training/
    train_client_condensation.py
    train_server_global_graph.py
    train_dual_graph_prompt.py

  evaluation/
    qa_metrics.py               # EM, F1
    retrieval_metrics.py        # SP recall/F1, recall@k
    graph_metrics.py            # condensed ratio, motif coverage, etc.
    ablation_runner.py

  configs/
    hotpot_base.yaml
    motif_selection.yaml
    client_cond.yaml
    server_pge.yaml
    dual_prompt.yaml

  scripts/
    01_prepare_hotpot.sh
    02_build_trigraphs.sh
    03_run_client_condensation.sh
    04_train_server_global.sh
    05_train_dual_prompt.sh
    06_eval_qa.sh
```

---

## Part III — Component Specs

### 9. Data Preparation

#### 9.1 HotpotQA normalization

Schema mỗi sample:

```text
question_id
question
answer
supporting_facts          # list of (title, sent_idx)
candidate_documents       # list of {title, sentences[]}
```

Schema corpus:

```text
passage_id  ::= sha1(title) hoặc int incremental
title
passage_text              # concat các câu
sentence_list             # list of {sentence_id, text, offset}
```

Processed files:

```text
processed/hotpot/corpus.jsonl
processed/hotpot/train_questions.jsonl
processed/hotpot/dev_questions.jsonl
processed/hotpot/supporting_facts.jsonl
```

#### 9.2 Federated partition

Phiên bản đầu (đơn giản, deterministic):

```text
client_id = hash(document_title) mod num_clients
```

Lưu per-client:

```text
processed/hotpot/client_{m}/corpus.jsonl
processed/hotpot/client_{m}/passages.jsonl
processed/hotpot/client_{m}/sentences.jsonl
```

Phiên bản sau: topic-based, entity-overlap based, hoặc cố tình **split supporting facts** giữa các client để buộc cross-client knowledge transfer.

#### 9.3 Checkpoints (data)

- Dataset load đúng số mẫu (HotpotQA distractor dev ≈ 7405).
- Mỗi passage có `sentence_list` không rỗng.
- `supporting_facts` map được về `(passage_id, sentence_id)`.
- Tổng passage trên các client = tổng passage gốc (không trùng, không mất).

### 10. LinearRAG Tri-Graph Construction

#### 10.1 Node & Edge

- **Nodes**: `V_e` (entity), `V_s` (sentence), `V_p` (passage).
- **Edges (main)**: `S–E` (sentence mentions entity), `P–E` (passage contains entity).
- **KHÔNG** thêm cạnh `S–P` trực tiếp vào main graph (chỉ giữ làm metadata để debug). Đây là điều kiện sống còn để giữ topology S-E-P; thiếu nó, motif selection sẽ thoái hoá.

#### 10.2 Entity extraction

- spaCy `en_core_web_sm` hoặc `en_core_web_trf` cho NER.
- Normalize: lowercase, strip dấu câu, merge whitespace, optional alias merging (Wikipedia redirects/Wikidata QID lookup ở phiên bản sau).

#### 10.3 Node embeddings

- Frozen encoder dùng chung qua tất cả clients: `all-MiniLM-L6-v2` (d=384) hoặc `intfloat/e5-small-v2` (d=384). Chọn 1 và cố định.
- Embedding rules:
  - Entity node: encode(entity string).
  - Sentence node: encode(sentence text).
  - Passage node: encode("Title: <t>. " + first 1–2 sentences hoặc whole passage cắt 256 tokens).
- L2 normalize, optional LayerNorm.

#### 10.4 Storage format (per client)

```text
client_{m}/trigraph.pt = {
  x: [N, d] node embeddings,
  edge_index: [2, E],
  edge_type:  [E]  (0: S-E, 1: P-E),
  node_type:  [N]  (0: entity, 1: sentence, 2: passage),
  node_source: [N] (entity_id / sentence_id / passage_id local),
  node_text: [N]   (local-only, không upload)
}
```

#### 10.5 Checkpoints (tri-graph)

- NER cho ra tập entity không rỗng cho mỗi passage có nội dung.
- Tồn tại đủ 2 loại cạnh `S–E` và `P–E`.
- **Không có** cạnh `S–P` trong `edge_index` của main graph.
- Sentence/passage không bị cô lập (mỗi node có ≥ 1 entity neighbor).
- Số entity duplicate được kiểm soát (rate ≤ 5%).

### 11. Query-Agnostic Structure-Semantic Motif Selection

#### 11.1 Vì sao thay label-aware của DANCE

- DANCE chọn core theo pseudo-label class, áp cho node classification. Trong QA setting, không có nhãn class cho node.
- Hơn nữa, client condensation xảy ra **offline trước query**; không thể dùng query để chọn.
- Vì topology là `S–E–P` không có `S–P` trực tiếp, nếu chọn sentence và passage độc lập, condensed graph sẽ **đứt** ở các entity bridge.

Đơn vị chọn phải là **motif S–E–P**: một entity anchor cùng với các sentence và passage neighbor.

#### 11.2 Entity anchor scoring

Score mỗi entity:

```text
entity_score(e) =
    log(1 + deg_S(e))
  + log(1 + deg_P(e))
  + λ_idf · log(|V_p| / (1 + deg_P(e)))
  + λ_pr  · PageRank(e)            # PR trên bipartite E↔S↔E∪E↔P↔E
```

Sau đó chọn anchor với MMR (Maximal Marginal Relevance):

```text
final_score(e) = entity_score(e) - λ_mmr · max_{e' ∈ Selected} cos(x_e, x_{e'})
```

Chọn `K_e = ⌈r_e · |V_e|⌉` entity anchor.

#### 11.3 Sentence/passage neighbor selection (per anchor)

Mỗi anchor `e`:

- Chọn `B_s` sentence neighbors trong `N_S(e)`, score = `α · #entity_mentions + β · local_centrality + γ · cos(x_s, mean_{e' selected entity neighbor})`.
- Chọn `B_p` passage neighbors trong `N_P(e)`, score tương tự.

Kết quả là tập motif `{ (e, S_e, P_e) }`. Lấy union → core node set; giữ tất cả cạnh `S–E`, `P–E` trong cảm ứng subgraph (induced subgraph).

#### 11.4 Output

```text
core_node_ids
core_edge_index    # induced S–E, P–E edges trên core
selected_motifs    # list of (entity_id, sentence_ids, passage_ids)
```

#### 11.5 Checkpoints (motif)

- Core chứa đủ 3 loại node.
- Mọi sentence/passage được chọn đều kết nối ≥ 1 entity được chọn (no orphans).
- Số valid S-E-P motif > random baseline (sanity test).
- Core không degenerate thành "chỉ hub entities" (Gini coefficient trên degree không quá lệch).

### 12. DANCE-style Hierarchical Text Condensation

(Áp lên các node trong core. Mục tiêu: thay vì giữ node embedding raw, tổng hợp evidence từ multi-hop dưới ngân sách.)

#### 12.1 Frozen text bank

Mỗi node `u`:
- Chunk hoá `s_u` thành `{s_{u,r}}`.
- Mỗi chunk: `e_{u,r} = Enc(s_{u,r})`.
- Node-level: `t_u = Pool({e_{u,r}})` (attention-pool theo node embedding).
- Cache vào `client_{m}/text_bank.pt` — chỉ local, không upload.

#### 12.2 Graph-side embedding

`g_v = GNN_current(local Tri-Graph)[v]`. Round đầu dùng init embedding (đầu vào `x` của Tri-Graph).

#### 12.3 Budgeted neighbor gating

Với mỗi core node `v`, định nghĩa `N^{(ℓ)}_v` (ℓ = 0, 1, 2). Vì cạnh là S-E-P, 2-hop là **bắt buộc** để một sentence node nhìn thấy passage node qua entity bridge.

Score:

```text
s_{v,u} = (W_q g_v)^T (W_k t_u) / sqrt(d)
```

Top-k softmax cho phiên bản đầu, với budget **theo Table 5 của paper DANCE**:

```text
B_0 = 1   # chính bản thân v  (N^(0)_v = {v} luôn cố định, B_0=1 implicit)
B_1 = 3   # 1-hop neighbors
B_2 = 2   # 2-hop neighbors   (Bảng 5: 2-hop budget = 2)
```

Sau đó (phiên bản sau): thay softmax bằng **entmax + straight-through estimator + ΠB truncate** (xem chi tiết ở Part X mục 38).

**Pre-filter 2-hop bằng difficulty score**. Vì `|N^(2)_v|` có thể lên tới `deg^2` (hàng nghìn node), cần pre-filter trước khi tính score:

```text
u_w = H(p^{t-1}(y | w)) = -Σ_c p^{t-1}(c|w) log p^{t-1}(c|w)   # Eq. (21) DANCE
Ñ^(2)_v = TopK_{B_2}({u_w}_{w ∈ N^(2)_v})
```

Trong adaptation của chúng ta (không có class label), entropy thay bằng **GNN embedding norm variance** hoặc **node degree heuristic** — xem Part X §39.

Hierarchical context (Eq. 7):

```text
c_v = Σ_ℓ γ_ℓ · Σ_{u ∈ S^{(ℓ)}_v} α^{(ℓ)}_{v,u} · t_u,    γ_ℓ ≥ 0, Σγ_ℓ = 1
```

Lưu ý: `c_v` được tính nhưng **không trực tiếp vào** feature `x_v` cuối cùng (Eq. 10 chỉ dùng `t̃_v`). `c_v` đóng vai trò **intermediate cho interpretability** và có thể dùng làm input cho alignment loss `o^t_v = Dec_t(c_v)` (xem Part X §38, point 7).

#### 12.4 Chunk selection

Trong các neighbor đã chọn, gom tập chunk `E_v`. Score theo cross-attention với `q_v = W_s g_v`. Top `B_tok = 8` chunks.

```text
ã_{v,(u,r)} = (q_v^T e_{u,r}) / sqrt(d)
π_{v,·} = TopK_{B_tok}(softmax(ã_{v,·}))
t̃_v = Σ_{(u,r)} π_{v,(u,r)} · e_{u,r}
```

#### 12.5 Local evidence traces

Lưu local (không upload):
- core node id
- selected neighbors + weights
- selected chunks + weights + source spans

Dùng cho audit và debug. KHÔNG truyền ra ngoài.

#### 12.6 Checkpoints (text cond.)

- Mỗi core node có ≤ `Σ_ℓ B_ℓ` neighbors trong `S_v`.
- Mỗi core node có ≤ `B_tok` chunks contributing tới `t̃_v`.
- `t̃_v` shape khớp với encoder output.
- Trong upload object: zero string fields, zero spans.

### 13. Graph-Text Fusion

Fuse cho mỗi core node:

```text
α_v = σ(w^T [g_v ; t̃_v])
x_v = LN( W_g g_v + α_v · W_t t̃_v )
X    = stack(x_v) ∈ R^{K × d}
```

Đây là feature của client condensed graph.

#### Checkpoints (fusion)

- Shape `X = [K, d]`, không NaN.
- Gate values phân tán (không gần hết 0 hoặc gần hết 1).
- Norm của `x_v` ổn định trong toàn bộ batch.

### 14. Self-Expressive Topology Reconstruction

#### 14.1 Evidence prior

```text
S_ij = cosine(t̃_i, t̃_j),  scaled to [0, 1]
```

#### 14.2 Candidate support

Với mỗi `i`:

```text
C_x(i) = TopK_q( {x_i^T x_j : j ≠ i} )
C_S(i) = TopK_q( {S_ij    : j ≠ i} )
C(i)   = C_x(i) ∪ C_S(i)
```

#### 14.3 Sparse self-expression

Optimize `Z` (K × K) sao cho `X ≈ XZ`, với:

```text
diag(Z) = 0
Z_ij = 0 nếu j ∉ C(i)
||Z||_1 + λ_3 Σ (1 - S_ij) |Z_ij|     # sparse + prior-aware
```

Iterative shrinkage thresholding:

```text
G = X^T (XZ - X);  mask theo C
Z ← Z - η G
τ_ij = η (λ_1 + λ_3 (1 - S_ij))
Z_ij ← sign(Z_ij) · max(|Z_ij| - τ_ij, 0)
```

#### 14.4 Adjacency

```text
W = |Z| + |Z|^T
A_hat = TopK_per_row_k(W)         # k ≈ 8
symmetrize
```

#### 14.5 Recommended order

1. KNN adjacency baseline trên `x_fused` (cosine top-k).
2. Evidence-prior KNN baseline trên `(x_fused + α · S)`.
3. Self-expression đầy đủ.

#### 14.6 Checkpoints (topology)

- `A_hat` symmetric, diag = 0.
- Mỗi row có ≤ k cạnh.
- Edge weights finite, không có NaN.
- Graph không fully dense, không fully disconnected.
- `num_components(A_hat)` < K/4 (ít nhất một số cluster lớn).

### 15. Client Condensed Graph Output

```text
ClientCondensedGraph C_m = {
  x:           [K, d]      # fused node embeddings
  edge_index:  [2, E_c]
  edge_weight: [E_c]
  node_type:   [K]         # 0/1/2 entity/sentence/passage
  optional pseudo_role:    # later
  optional hashed_local_ids
}
```

Cấm upload: raw text, selected chunks, source spans, evidence traces, summaries.

#### Checkpoints (upload)

- Kích thước upload < ngưỡng (eg K ≤ 0.1 · |V_local|).
- Không string field nào trong object.
- Có đủ `x, edge_index, edge_weight, node_type`.

### 16. Server-side Anchor Gradient

#### 16.1 Surrogate task (query-agnostic)

Vì client condensation không có query, surrogate đầu tiên là task **không cần query label**:

- **Node-type prediction**: phân loại 3 nhãn (entity/sentence/passage).
- **Link prediction**: trên condensed graph.
- **Contrastive learning**: positive pairs từ cạnh thật, negative pairs từ random.

Recommended first: **node-type prediction + link prediction** (cả hai supervised trực tiếp từ `node_type` và `edge_index`).

#### 16.2 Anchor gradient

Với mỗi `C_m`:

```text
ĝ_m = ∇_θ L_surrogate( GNN_θ(C_m) )
ĝ_anchor = Σ_m w_m · ĝ_m       # w_m = |V_m| / Σ |V_m'|
```

#### Checkpoints (anchor)

- Forward được mọi `C_m` mà không OOM.
- Loss finite, gradient finite.
- `ĝ_anchor` có entry cho mọi layer cần khớp.

### 17. Server Global Graph + PGE

#### 17.1 Synthetic node features

```text
X_global ∈ R^{K_g × d}  learnable
```

Init: lấy mean cluster theo node_type từ một subset client (warm start) hoặc Gaussian.

#### 17.2 PGE adjacency

```text
e_ij = MLP_PGE([ |x_i - x_j| ; x_i ⊙ x_j ; type_emb(i) ; type_emb(j) ]) ∈ [0,1]
```

Sparsify:

```text
A_global = TopK_per_row_k(e_·)
symmetrize
```

#### 17.3 Checkpoints (PGE)

- Edge weights không cùng giá trị (variance > 0).
- A_global sparse (avg degree ≈ k).
- Không fully dense, không empty.

### 18. Gradient Matching

Optimize `{ X_global, θ_PGE }`:

```text
ĝ_global = ∇_θ L_surrogate( GNN_θ(X_global, A_global) )
L_match  = 1 - cos(ĝ_global, ĝ_anchor)  + λ_norm · ||ĝ_global - ĝ_anchor||^2
```

Outer loop: cập nhật `X_global, θ_PGE` theo `L_match`.

#### Checkpoints (matching)

- `L_match` giảm trong các epoch đầu.
- `||X_global||` ổn định, không bùng nổ.
- A_global vẫn sparse sau training.

### 19. Query-time Retrieval

#### 19.1 LinearRAG evidence branch

Input: `q`. Output:

- `P_q`: top-K passages qua **(stage 1)** entity activation + **(stage 2)** PPR trên entity-passage subgraph.
- `E_q`: evidence graph chứa entity activated, sentence chứa entity activated, passage trong `P_q`, cùng các cạnh S-E, P-E.

`E_q` là **query-specific và text-grounded**.

#### 19.2 Global graph branch

Input: `z_q = Enc(q)`. Output `G_global(q)`:

```text
seed = TopR_{i}( cos(z_q, X_global[i]) )
expand 1-hop theo A_global
G_global(q) = induced subgraph
```

`G_global(q)` là **embedding-only, no text**.

#### Checkpoints (query-time)

- `|V(E_q)| ≤ budget_E`, `|V(G_global(q))| ≤ budget_C`.
- Retrieval deterministic với seed cố định.
- Full global graph **không bao giờ** được pass thẳng vào LLM.

### 20. Dual Graph Prompting (Option B)

#### 20.1 Hai GNN encoders

```text
z_e = MLP_e( POOL( GNN_e(E_q) ) ) ∈ R^{d_l}
z_c = MLP_c( POOL( GNN_c(G_global(q)) ) ) ∈ R^{d_l}
```

`GNN_e` là Graph Transformer trên Tri-Graph với edge type embedding. `GNN_c` đơn giản hơn (GCN/GAT).

#### 20.2 Late fusion (first version)

```text
soft_prompt = [z_e ; z_c]    # 2 tokens
```

Sau:
- gated: `α · z_e + (1-α) · z_c`,
- query-aware: `α = σ(W [z_q; z_e; z_c])`,
- attention fusion: cho query làm Q, [z_e, z_c] làm K/V.

#### 20.3 LLM input (G-Retriever style)

```text
text_input  = textualize(E_q) ⊕ "\n\nRetrieved passages:\n" ⊕ P_q ⊕ "\n\nQuestion: " ⊕ q
h_t         = TextEmbedder(text_input)
input_seq   = [z_e ; z_c ; h_t]
Y           = LLM(input_seq)
```

Loss: standard causal LM trên token answer.

#### Checkpoints (prompting)

- `z_e.shape == z_c.shape == [d_l]` (= 4096 cho Llama-2-7B).
- Pipeline chạy được khi mask 1 trong 2 token (cho ablation).
- Text input không vượt context length (truncate `P_q` theo ngân sách).

---

## Part IV — Training Schedule

### 21. Stages

**Stage A — Non-FL LinearRAG baseline**
- Build Tri-Graph centralized trên toàn HotpotQA.
- Chạy LinearRAG retrieval thuần.
- Đo retrieval recall + answer EM/F1 với prompt LLM.

**Stage B — Client graph condensation**
- Per client: Tri-Graph → motif core → text cond. → topology recon.
- Output `C_m`.
- Đo condensed ratio, motif coverage, connectivity.

**Stage C — Server global graph**
- Gửi `C_m` lên, học `X_global, θ_PGE` qua gradient matching.
- Đo `L_match`, PGE edge distribution, sparsity.

**Stage D — Dual graph prompting**
- Build inference pipeline. Huấn luyện `GNN_e, GNN_c, MLP_proj` end-to-end với LLM frozen.
- Đo QA metrics.

**Stage E — Full ablations** (xem Part V).

### 22. Milestone Timeline

| # | Milestone | Output |
|---|---|---|
| M1 | Data setup | HotpotQA processed, federated partition |
| M2 | Tri-Graph builder | Per-client `trigraph.pt` |
| M3 | Motif selection | Core graphs, motif coverage report |
| M4 | Text condensation | text_bank, t̃_v, fusion x_fused |
| M5 | Topology recon | KNN + self-expression `A_hat` |
| M6 | Server global graph | `X_global`, `θ_PGE`, gradient matching curves |
| M7 | Dual graph prompting | End-to-end QA pipeline runs |
| M8 | Baselines + ablations | Full evaluation table |

---

## Part V — Evaluation

### 23. Main baselines

```text
B1. Zero-shot LLM (no retrieval, no graph)
B2. LinearRAG passages → LLM
B3. LinearRAG passages + textualized E_q → LLM            (Vanilla G-Retriever style)
B4. B3 + z_e only                                          (evidence graph token alone)
B5. B3 + z_c only                                          (condensed graph token alone)
B6. B3 + z_e + random z_c                                  (sanity: graph token có thật sự hoạt động?)
B7. B3 + z_e + KNN-based global z_c                        (PGE vs KNN)
B8. B3 + z_e + gradient-matched PGE z_c                    (ours, full)
```

### 24. Condensation ablations

```text
A1. random core selection
A2. degree / PageRank core selection
A3. semantic k-medoids selection
A4. S-E-P motif selection (ours)
A5. A4 + self-expression topology
A6. A4 + KNN topology only
```

### 25. Metrics

**QA**:
- Answer Exact Match (EM)
- Answer F1

**Retrieval/evidence**:
- Supporting fact recall, F1
- Passage recall@k

**Graph condensation**:
- Condensed ratio `K / |V|`
- Valid S-E-P motifs retained
- Connectivity: average degree, # connected components
- Gini coefficient on entity degree

**Server graph**:
- Gradient matching loss curve
- Edge sparsity, mean degree
- # connected components, PGE edge weight distribution histogram

**Graph token contribution**:
- (z_e + z_c) − z_e only
- (z_e + z_c) − (z_e + random z_c)
- learned PGE vs KNN

### 26. Reporting

- Mean ± std qua 3 seeds (5 nếu compute cho phép).
- Per-client breakdown của motif coverage.
- Failure case study: 20 sample sai để tay phân tích.

---

## Part VI — Debug Checklist

### 27.1 Data
- supporting facts map đúng `(passage_id, sentence_id)`.
- Không passage nào duplicate giữa clients.
- Tổng nodes/edges/passages cộng dồn khớp với centralized.

### 27.2 Tri-Graph
- Có cạnh S-E, P-E.
- **Không có cạnh S-P** trong main graph.
- Tỉ lệ sentence/passage isolated < 1%.

### 27.3 Motif selection
- Mọi core sentence/passage kết nối ≥ 1 core entity.
- # valid S-E-P motifs > random baseline.

### 27.4 Text condensation
- Neighbor budget tôn trọng (`|S^{(ℓ)}_v| ≤ B_ℓ`).
- Chunk budget tôn trọng.
- Local traces có tồn tại, nhưng KHÔNG xuất hiện trong upload.

### 27.5 Topology
- `A_hat` symmetric, diag = 0.
- Sparse, finite weights.
- Số component hợp lý.

### 27.6 Server
- PGE adjacency không collapse.
- `L_match` giảm.
- Global graph retrieval trả về sub-graph hợp lệ.

### 27.7 LLM
- `z_e.shape == z_c.shape == [d_l]`.
- Text input không vượt context.
- Random graph token baseline kém hơn learned token (proof of useful signal).

---

## Part VII — Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **R1**: Core selection độc lập theo node type → condensed graph đứt | Bắt buộc entity-centered S-E-P motif selection. |
| **R2**: DANCE over-engineering (entmax + difficulty + self-expression cùng lúc) | Triển khai theo thứ tự: top-k softmax → KNN topology → self-expression sau. |
| **R3**: Condensed graph không có text grounding | Vẫn dùng passages của LinearRAG cho factual grounding. Condensed graph chỉ là structural memory qua GNN. |
| **R4**: LLM ignore graph tokens | Chạy `random z_c` và `z_e/z_c only` ablations để xác minh đóng góp. |
| **R5**: PGE tạo adjacency nhiễu | Top-k sparsify, monitor edge weight distribution, so với KNN baseline. |
| **R6**: Cross-client supporting facts chia rời | Phiên bản đầu dùng hash partition. Sau đó cố tình split supporting facts để stress test. |
| **R7**: Gradient matching không hội tụ | Fallback: dùng simple loss matching (MSE trên node embedding) như smoke test trước khi gradient matching. |

---

## Part VIII — Final Implementation Order

```text
 1. Data + HotpotQA loader
 2. Federated partition
 3. LinearRAG Tri-Graph builder
 4. Node embeddings (cached)
 5. Query-agnostic S-E-P motif selector
 6. Client DANCE-style text condensation
 7. Client topology reconstruction (KNN → self-expr)
 8. Client condensed graph upload object + audit
 9. Server anchor gradient computation
10. Server synthetic global graph + PGE
11. Gradient matching loop
12. LinearRAG query-time evidence retrieval
13. Global graph query-time retrieval
14. Evidence GNN encoder
15. Condensed GNN encoder
16. Late fusion + LLM input
17. Baselines + ablations
```

> **Design invariant không được vi phạm**: topology `Sentence — Entity — Passage` phải được bảo toàn xuyên suốt client condensation. Client condensation phải luôn chọn entity-centered motifs, không chọn sentence và passage node độc lập.

---

## Appendix A — Default Hyperparameters (first run)

Bảng dưới đây kết hợp default của G-Retriever, LinearRAG, và **Table 5 của paper DANCE** (cột bên phải ghi nguồn nếu là DANCE).

| Group | Param | Value | Source |
|---|---|---|---|
| Encoding | text encoder | `all-MiniLM-L6-v2` (d=384) | DANCE Table 5 (frozen SBERT) |
| Encoding | passage cap | 256 tokens | ours |
| Federation | # clients | 5 | DANCE Table 5 |
| Federation | communication rounds | 200 | DANCE Table 5 |
| Federation | local epochs / round | 3 | DANCE Table 5 |
| Federation | partition | hash(title) mod 5 | ours |
| Motif | entity ratio r_e | 0.05 | ours |
| Motif | B_s, B_p per anchor | 3, 3 | ours |
| Motif | λ_idf, λ_pr, λ_mmr | 1.0, 0.5, 0.3 | ours |
| Motif | node-cond refresh | every 10 rounds | DANCE §4.2 |
| Text cond. | B_0, B_1, B_2 | **1, 3, 2** | **DANCE Table 5** (B_1=3, B_2=2) |
| Text cond. | B_tok | 8 | ours (paper không nêu) |
| Text cond. | γ_ℓ (hop weights) | [0.4, 0.4, 0.2] | ours (paper không nêu; γ_0 + γ_1 + γ_2 = 1) |
| Text cond. | summary mix ratio | search ∈ {0.4, 0.6, 0.8} | DANCE Table 5 |
| Fusion | λ_align (alignment loss) | 0.1 | ours (paper không nêu rõ) |
| Topology | KNN k (baseline) | 8 | ours |
| Topology | self-expr q (candidate size) | 16 | ours (paper không nêu; ≈ 2k) |
| Topology | α (reconstruction weight) | search ∈ {4, 8, 12} | DANCE Table 5 |
| Topology | β (L1 weight) | search ∈ {3, 5, 10} | DANCE Table 5 |
| Topology | inner iterations L (ISTA) | 50 | ours (paper không nêu) |
| Topology | step size η (ISTA) | 1e-2 | ours (paper không nêu) |
| Topology | final top-k per row | 8 | ours |
| Server | K_g (global nodes) | 1024 | ours |
| Server | PGE hidden | 256 | ours |
| Server | global top-k per row | 8 | ours |
| Retrieval | top-R seed (global graph) | 16 | ours |
| Retrieval | top-K passages | 5 | ours |
| Prompting | LLM | Llama-2-7B (frozen) | G-Retriever |
| Prompting | d_l | 4096 | Llama-2-7B |
| Prompting | GNN_e | Graph Transformer 2L | G-Retriever default |
| Prompting | GNN_c | GAT 2L | ours |
| GNN backbone (DANCE part) | type | 2-layer GCN | DANCE Table 5 |
| GNN backbone (DANCE part) | hidden | ∈ {64, 128} | DANCE Table 5 |
| GNN backbone (DANCE part) | lr | 1e-2 | DANCE Table 5 |
| GNN backbone (DANCE part) | weight decay | 5e-4 | DANCE Table 5 |
| GNN backbone (DANCE part) | dropout | 0.5 | DANCE Table 5 |
| Training (stage D) | optimizer | AdamW | G-Retriever |
| Training (stage D) | lr (proj + GNN) | 1e-4 | G-Retriever |
| Training (stage D) | batch size | 4 | G-Retriever |
| Training (stage D) | grad accum | 4 | G-Retriever |

> **Lưu ý**: paper DANCE Table 5 dùng α và β trong Eq. (13) (reconstruction và L1 weights). Trong Algorithm 4 lại xuất hiện λ_1, λ_3 (soft-thresholding). Hai bộ tên này **không phải cùng tham số**; quan hệ là `λ_1 ∝ β/α` và `λ_3 ∝ 1/α` sau khi chuẩn hoá. Khi implement, dùng α và β theo Eq. (13) làm primary hyperparameter, derive λ_1 và λ_3.

---

## Appendix B — Relationship to source codebases (high-level)

| Source | Reused / adapted | Modified or replaced |
|---|---|---|
| **G-Retriever** | GraphTransformer, projection MLP, LLM wrapper, training loop, `textualize_graph()` | Single-encoder → dual-encoder, PCST retrieval → LinearRAG + global graph retrieval, soft prompt = `[h_g]` → `[z_e; z_c]` |
| **LinearRAG** | Tri-Graph construction, entity activation, PPR passage retrieval | Centralized → federated (per-client subgraph); built thành module trong codebase G-Retriever |
| **DANCE** | text bank caching, neighbor gating, chunk selection, self-expressive topology, federated round structure | Label-aware node condensation → query-agnostic S-E-P motif selection; per-class quotas removed; surrogate task changed từ classification sang node-type prediction + link prediction |

> Chi tiết file-level và workflow "copy and fix" xem **Part IX** bên dưới.

---

# Part IX — Code Integration Strategy ("copy and fix")

> **Nguyên tắc cốt lõi**: hạn chế tối đa việc viết lại từ đầu. Chiến lược là **fork G-Retriever làm host repo**, **vendor LinearRAG vào dưới dạng package con**, và chỉ viết module **DANCE-style + Server-side condensation** từ đầu (vì DANCE không có code chính thức và phần server là đóng góp gốc của project này).

## 28. Repository Layout sau khi tích hợp

Sau khi clone và tích hợp, repo của chúng ta trông như sau:

```text
FedCondGraphRAG/                             # = fork của G-Retriever
├── dataset/                                 # giữ nguyên (G-Retriever)
├── figs/                                    # giữ nguyên
├── src/                                     # giữ NGUYÊN của G-Retriever
│   ├── dataset/
│   │   ├── preprocess/
│   │   │   ├── expla_graphs.py              # giữ
│   │   │   ├── scene_graphs.py              # giữ
│   │   │   ├── webqsp.py                    # giữ
│   │   │   └── hotpot.py                    # MỚI (mục 9)
│   │   ├── utils/
│   │   │   └── retrieval.py                 # giữ (PCST – có thể bị thay cho fedcond pipeline)
│   │   ├── __init__.py                      # FIX: register hotpot_fedcond
│   │   ├── expla_graphs.py / scene_graphs.py / webqsp.py
│   │   └── hotpot_fedcond.py                # MỚI – Dataset class cho HotpotQA dual-graph
│   ├── model/
│   │   ├── __init__.py                      # FIX: register dual_graph_llm
│   │   ├── gnn.py                           # giữ (GraphTransformer/GAT/GCN); thêm GNN_c nhẹ
│   │   ├── graph_llm.py                     # giữ – là parent class
│   │   ├── dual_graph_llm.py                # MỚI – subclass với 2 encoder + late fusion
│   │   ├── pt_llm.py / llm.py / inference_llm.py    # giữ
│   ├── utils/                               # giữ toàn bộ (collate, evaluate, ckpt, seed, lr_schedule)
│   └── config.py                            # FIX nhẹ: thêm CLI args cho dual graph + server
├── train.py                                 # FIX nhẹ: thêm route train_server cho stage C
├── inference.py                             # FIX nhẹ: gọi dual_graph_llm
├── run.sh                                   # giữ; thêm scripts mới
│
├── fedcond_grag/                            # ============ PACKAGE MỚI ============
│   ├── external/
│   │   └── linearrag/                       # === vendor toàn bộ src/ của LinearRAG ===
│   │       ├── LinearRAG.py                 # copy nguyên
│   │       ├── config.py                    # copy nguyên (sẽ rename khi import)
│   │       ├── evaluate.py                  # copy (có thể không dùng)
│   │       └── utils.py                     # copy
│   │       └── __init__.py                  # MỚI: re-export public API
│   │
│   ├── data/
│   │   ├── hotpot_loader.py                 # MỚI (mục 9)
│   │   ├── federated_partition.py           # MỚI
│   │   └── corpus_index.py                  # MỚI
│   │
│   ├── graph_building/
│   │   ├── trigraph_builder.py              # WRAP linearrag.LinearRAG.index()
│   │   ├── entity_extractor.py              # WRAP spaCy NER của LinearRAG
│   │   ├── node_encoder.py                  # MỚI – dùng SentenceTransformer
│   │   └── graph_store.py                   # MỚI – save .pt
│   │
│   ├── graph_condensation/                  # === DANCE re-implementation từ paper ===
│   │   ├── motif_core_selector.py           # MỚI (mục 11)
│   │   ├── text_bank.py                     # MỚI (mục 12.1)
│   │   ├── neighbor_gating.py               # MỚI (mục 12.3)
│   │   ├── chunk_selection.py               # MỚI (mục 12.4)
│   │   ├── graph_text_fusion.py             # MỚI (mục 13)
│   │   ├── evidence_prior.py                # MỚI (mục 14.1)
│   │   ├── topology_reconstruction.py       # MỚI (mục 14)
│   │   └── client_condensor.py              # MỚI – orchestrator
│   │
│   ├── server_condensation/                 # === Đóng góp gốc ===
│   │   ├── anchor_gradient.py               # MỚI (mục 16)
│   │   ├── synthetic_graph.py               # MỚI (mục 17)
│   │   ├── pge.py                           # MỚI (mục 17.2)
│   │   └── gradient_matching.py             # MỚI (mục 18)
│   │
│   ├── retrieval/
│   │   ├── linearrag_retriever.py           # WRAP linearrag.LinearRAG.qa() pipeline
│   │   ├── evidence_graph_builder.py        # MỚI – build E_q từ activated entities
│   │   └── global_graph_retriever.py        # MỚI – top-r + 1-hop trên A_global
│   │
│   ├── training/
│   │   ├── train_client_condensation.py     # MỚI – stage B
│   │   ├── train_server_global_graph.py     # MỚI – stage C
│   │   └── (stage D dùng src/train.py + dual_graph_llm)
│   │
│   ├── configs/                             # MỚI – YAML configs
│   └── scripts/                             # MỚI – wrapper sh scripts
│
└── readme.md
```

## 29. Step-by-step "Copy and Fix" Workflow

### Step 0 — Khởi tạo

```bash
git clone https://github.com/XiaoxinHe/G-Retriever.git FedCondGraphRAG
cd FedCondGraphRAG
git checkout -b fedcond-grag

# Vendor LinearRAG
git clone https://github.com/DEEP-PolyU/LinearRAG.git /tmp/linearrag
mkdir -p fedcond_grag/external/linearrag
cp -r /tmp/linearrag/src/* fedcond_grag/external/linearrag/

# Vendor requirements
cat /tmp/linearrag/requirements.txt >> requirements_linearrag.txt
```

### Step 1 — Sửa import paths trong LinearRAG (search-and-replace)

Trong các file `fedcond_grag/external/linearrag/*.py`, sửa các absolute import:

```text
"from src.config import LinearRAGConfig"        →  "from .config import LinearRAGConfig"
"from src.LinearRAG import LinearRAG"           →  "from .LinearRAG import LinearRAG"
"from src.evaluate import Evaluator"            →  "from .evaluate import Evaluator"
"from src.utils import ..."                     →  "from .utils import ..."
```

Một dòng `sed`:

```bash
cd fedcond_grag/external/linearrag
find . -name "*.py" -exec sed -i \
  -e 's|from src\.|from .|g' \
  -e 's|import src\.|import .|g' \
  {} +
```

Tạo `fedcond_grag/external/linearrag/__init__.py`:

```python
from .config import LinearRAGConfig
from .LinearRAG import LinearRAG
from .utils import LLM_Model, setup_logging

__all__ = ["LinearRAG", "LinearRAGConfig", "LLM_Model", "setup_logging"]
```

### Step 2 — Hợp nhất env

LinearRAG dùng `sentence_transformers + spacy + en_core_web_trf`; G-Retriever dùng `torch + torch_geometric + transformers + peft`. Cài chung:

```bash
conda create -n fedcond python=3.9 -y
conda activate fedcond
# G-Retriever deps
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install peft pandas ogb transformers wandb sentencepiece torch_geometric \
  datasets pcst_fast gensim scipy==1.12 protobuf
# LinearRAG deps
pip install sentence-transformers spacy openai
python -m spacy download en_core_web_trf
```

### Step 3 — Smoke test (verify import works)

Trước khi viết logic, đảm bảo cả hai code base có thể được import song song:

```python
# scripts/smoke_test_imports.py
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig
print("Both codebases importable.")
```

### Step 4 — Wrap LinearRAG behind a thin façade

Trong `fedcond_grag/retrieval/linearrag_retriever.py`, không gọi class `LinearRAG` trực tiếp trong dataset class — bọc lại để dễ mock và test:

```python
# fedcond_grag/retrieval/linearrag_retriever.py
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig

class LinearRAGRetriever:
    """Thin wrapper exposing only what FedCondGraphRAG needs."""
    def __init__(self, passages, embedding_model, spacy_model, llm_model, **kw):
        cfg = LinearRAGConfig(
            embedding_model=embedding_model,
            spacy_model=spacy_model,
            llm_model=llm_model,
            **kw,
        )
        self.rag = LinearRAG(global_config=cfg)
        self.rag.index(passages)            # build Tri-Graph once

    def retrieve(self, question: str) -> dict:
        """Run two-stage retrieval; return passages + activated entities."""
        result = self.rag.qa([{"question": question}])
        return {
            "passages": result[0].get("retrieved_passages", []),
            "activated_entities": result[0].get("activated_entities", []),
            "tri_graph_view": self.rag.get_tri_graph_view(),   # may need to expose
        }
```

Nếu `LinearRAG.qa()` không trả về `activated_entities` ở top level, ta cần một **patch nhỏ** trong `fedcond_grag/external/linearrag/LinearRAG.py`: thêm `return` hoặc expose qua attribute (xem mục 32). Đây là kiểu "fix" duy nhất cần áp lên code đã copy.

### Step 5 — Build Tri-Graph re-export

`fedcond_grag/graph_building/trigraph_builder.py` không build lại Tri-Graph; nó **gọi** `LinearRAG.index()` rồi convert state thành format PyG:

```python
# fedcond_grag/graph_building/trigraph_builder.py
import torch
from torch_geometric.data import Data
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig

NODE_TYPE = {"entity": 0, "sentence": 1, "passage": 2}

def build_trigraph_for_client(passages, cfg) -> Data:
    rag = LinearRAG(global_config=cfg)
    rag.index(passages)
    # Trích entities, sentences, passages từ trạng thái LinearRAG.
    # (chính xác tên attribute lấy được sau khi inspect run.py output)
    entities  = rag.entity_list           # may be rag.graph.entities
    sentences = rag.sentence_list         # may be rag.graph.sentences
    passages_ = rag.passage_list          # rag.passages
    C = rag.contain_matrix                # |P| x |E| sparse  (Eq. 1)
    M = rag.mention_matrix                # |S| x |E| sparse  (Eq. 2)

    x_e = rag.entity_embeddings           # already cached by LinearRAG
    x_s = rag.sentence_embeddings
    x_p = rag.passage_embeddings
    x   = torch.cat([x_e, x_s, x_p], dim=0)

    node_type = torch.cat([
        torch.full((len(entities),),  NODE_TYPE["entity"]),
        torch.full((len(sentences),), NODE_TYPE["sentence"]),
        torch.full((len(passages_),), NODE_TYPE["passage"]),
    ])

    # S-E edges từ M, P-E edges từ C; offset index theo block
    se_edges = _coo_to_edge_index(M, offset_src=len(entities),
                                     offset_dst=0)
    pe_edges = _coo_to_edge_index(C, offset_src=len(entities)+len(sentences),
                                     offset_dst=0)
    edge_index = torch.cat([se_edges, pe_edges], dim=1)
    edge_type  = torch.cat([torch.zeros(se_edges.size(1)),
                            torch.ones (pe_edges.size(1))]).long()

    return Data(x=x, edge_index=edge_index, edge_type=edge_type,
                node_type=node_type)
```

> Lưu ý: nếu tên attribute `rag.entity_list`, `rag.contain_matrix`, ... không khớp với LinearRAG repo thực tế, đây là chỗ duy nhất cần **inspect và rename** sau khi clone. Coi đó là Task 1 sau khi `git clone`.

### Step 6 — Subclass `GraphLLM` thành `DualGraphLLM`

Đây là điểm "fix" sâu nhất trong G-Retriever. Thay vì viết lại `graph_llm.py`, ta **kế thừa**:

```python
# src/model/dual_graph_llm.py
import torch, torch.nn as nn
from src.model.graph_llm import GraphLLM    # giữ nguyên parent

class DualGraphLLM(GraphLLM):
    """G-Retriever với 2 graph encoders (evidence + condensed) + late fusion."""

    def __init__(self, args, **kw):
        super().__init__(args, **kw)
        # Reuse parent.graph_encoder làm evidence encoder (GNN_e)
        # Thêm GNN_c và projection riêng
        self.condensed_encoder = self._build_gnn(
            in_dim=args.gnn_in_dim,
            hidden_dim=args.gnn_hidden_dim,
            out_dim=args.gnn_out_dim,
            num_layers=args.gnn_num_layers_c,
            gnn_type=args.gnn_model_name_c,
        )
        self.projector_c = nn.Sequential(
            nn.Linear(args.gnn_out_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.llm.config.hidden_size),
        )

    def encode_graphs(self, batch):
        # parent.encode_graphs trả về z_e (graph token cho evidence graph E_q)
        z_e = super().encode_graphs(batch["evidence_graph"])
        # encode global condensed subgraph G_global(q)
        z_c_raw = self.condensed_encoder(batch["condensed_graph"])
        z_c = self.projector_c(z_c_raw.mean(0))
        return z_e, z_c

    def forward(self, samples):
        z_e, z_c = self.encode_graphs(samples)
        # build LLM input: [z_e ; z_c ; textualized_E_q ; passages ; question]
        ...   # 90% logic copy từ parent.forward, chỉ đổi prefix tokens
```

`src/model/__init__.py` thêm:

```python
from src.model.dual_graph_llm import DualGraphLLM
load_model["dual_graph_llm"] = DualGraphLLM
```

### Step 7 — Dataset class

`src/dataset/hotpot_fedcond.py` follow đúng pattern của `src/dataset/webqsp.py`:

```python
# src/dataset/hotpot_fedcond.py
import torch
from torch.utils.data import Dataset

from fedcond_grag.retrieval.linearrag_retriever import LinearRAGRetriever
from fedcond_grag.retrieval.global_graph_retriever import GlobalGraphRetriever
from fedcond_grag.retrieval.evidence_graph_builder import build_evidence_graph

class HotpotFedCondDataset(Dataset):
    def __init__(self, split="train"):
        self.questions = load_hotpot_split(split)
        self.lr   = LinearRAGRetriever.load_cached()
        self.ggr  = GlobalGraphRetriever.load_cached()
        self.prompt = "Answer the question using the given graphs and passages."
        self.graph_type = "tri_graph"

    def get_idx_split(self):
        return {"train": [...], "val": [...], "test": [...]}

    def __getitem__(self, idx):
        q = self.questions[idx]
        out = self.lr.retrieve(q["question"])
        E_q = build_evidence_graph(out)
        G_global_q = self.ggr.retrieve(q["question"])
        return dict(
            id=q["question_id"],
            question=q["question"],
            label=q["answer"],
            desc=textualize(E_q),                     # tận dụng utility của G-Retriever
            evidence_graph=E_q,
            condensed_graph=G_global_q,
            retrieved_passages=out["passages"],
        )
```

`src/dataset/__init__.py` thêm `load_dataset["hotpot_fedcond"] = HotpotFedCondDataset`.

`src/utils/collate.py` chỉ cần FIX nhẹ: hỗ trợ batch 2 graphs thay vì 1 (cộng thêm 4 dòng tạo `Batch` cho `condensed_graph`).

### Step 8 — Train script reuse

Stage D (dual graph prompting) chạy thẳng `train.py` đã có sẵn:

```bash
python train.py --dataset hotpot_fedcond --model_name dual_graph_llm \
                --gnn_model_name gt --gnn_model_name_c gat \
                --llm_model_name 7b_chat --llm_frozen True
```

Vì cả dataset và model đều đã được register qua registry pattern của G-Retriever, không cần đụng `train.py` (trừ optional CLI args).

Stage B (client condensation) và Stage C (server global) có script **riêng** vì pipeline khác (không có LLM trong vòng lặp):

```bash
python -m fedcond_grag.training.train_client_condensation --config configs/client_cond.yaml
python -m fedcond_grag.training.train_server_global_graph --config configs/server_pge.yaml
```

## 30. File-by-file Mapping: LinearRAG → FedCondGraphRAG

| LinearRAG file | FedCondGraphRAG target | Action |
|---|---|---|
| `src/LinearRAG.py` | `fedcond_grag/external/linearrag/LinearRAG.py` | **Copy nguyên**, sửa import path. Có thể cần patch nhỏ để expose `activated_entities`, `entity_embeddings`, `contain_matrix`, `mention_matrix` qua attributes hoặc method getter. |
| `src/config.py` | `fedcond_grag/external/linearrag/config.py` | **Copy nguyên**. |
| `src/evaluate.py` | `fedcond_grag/external/linearrag/evaluate.py` | **Copy nguyên** (có thể dùng để so sánh QA metrics nếu format predictions tương thích). |
| `src/utils.py` (LLM_Model, setup_logging) | `fedcond_grag/external/linearrag/utils.py` | **Copy nguyên**. `LLM_Model` chỉ dùng cho stage A baseline; stage D dùng LLM của G-Retriever. |
| `run.py` | (không copy) | Logic của `run.py` chia đôi: indexing → `trigraph_builder.build_trigraph_for_client`; QA → `linearrag_retriever.retrieve`. |
| `requirements.txt` | append vào root `requirements.txt` | Merge deps. |
| `scripts/` | tham khảo | Không copy; viết scripts mới trong `fedcond_grag/scripts/`. |
| `dataset/` (HuggingFace data) | (không cần) | Project dùng HotpotQA raw từ official source, không cần data đã preprocess của LinearRAG. |

## 31. File-by-file Mapping: G-Retriever → FedCondGraphRAG

| G-Retriever file | Action | Lý do |
|---|---|---|
| `src/model/graph_llm.py` | **Giữ nguyên** | Là parent class cho `DualGraphLLM`. |
| `src/model/gnn.py` | **Giữ + thêm import** | Reuse `GraphTransformer`, `GAT`, `GCN`. Thêm helper `_build_gnn(type='gt'/'gat'/'gcn', ...)` nếu chưa có. |
| `src/model/llm.py` | **Giữ nguyên** | LLM wrapper với LoRA support. |
| `src/model/pt_llm.py` | **Giữ nguyên** | Baseline B3 (prompt tuning). |
| `src/model/inference_llm.py` | **Giữ nguyên** | Baseline B1 (zero-shot). |
| `src/model/__init__.py` | **FIX**: register `DualGraphLLM` | 2 dòng `import` + 1 dòng `load_model[...]`. |
| `src/dataset/utils/retrieval.py` (PCST) | **Giữ; KHÔNG dùng cho hotpot_fedcond** | Chỉ giữ cho các dataset gốc của G-Retriever. Pipeline mới dùng LinearRAG retrieval. |
| `src/dataset/webqsp.py` | **Giữ** | Là template cho `hotpot_fedcond.py`. |
| `src/dataset/__init__.py` | **FIX**: register `hotpot_fedcond` | 2 dòng. |
| `src/utils/collate.py` | **FIX nhẹ**: hỗ trợ 2 graphs trong batch | ~4 dòng để batch condensed_graph song song với evidence_graph. |
| `src/utils/evaluate.py` | **FIX**: thêm `eval_funcs["hotpot_fedcond"]` | Implement EM/F1 cho HotpotQA. |
| `src/utils/{ckpt, seed, lr_schedule}.py` | **Giữ nguyên** | Generic. |
| `src/config.py` | **FIX nhẹ**: thêm CLI args (`--gnn_model_name_c`, `--global_top_r`, ...) | ~10 dòng. |
| `train.py`, `inference.py` | **Giữ nguyên** | Vì đã dùng registry pattern. |

## 32. Patches có thể cần áp vào LinearRAG sau khi copy

Đây là các điểm có thể (chưa chắc) cần fix sau khi inspect code thực tế:

1. **Expose internal state**: `LinearRAG.qa()` hiện trả về answer + passages. Ta cần thêm activated entities và evidence graph view. Patch: thêm method `get_activated_entities(question_id)` và `get_evidence_subgraph(question_id)`.

2. **Decouple LLM call**: `LinearRAG.qa()` gọi LLM của LinearRAG để sinh answer. Ta muốn **chỉ retrieve**, không sinh answer. Patch: thêm flag `do_generate=False` để skip phần generation, chỉ trả về retrieved structure.

3. **Per-client indexing**: `LinearRAG.index(passages)` hoạt động trên 1 corpus. Cho federated setting, ta tạo N instance, mỗi instance index riêng. Không patch cần thiết, chỉ wrap.

4. **Embedding cache I/O**: thêm `save_state(path)` / `load_state(path)` để Tri-Graph có thể được build offline (mất ~vài giờ cho HotpotQA), save, rồi load nhanh ở mỗi run training.

5. **Tensor dtype consistency**: LinearRAG dùng numpy/scipy sparse. Khi convert sang PyG `Data`, ép dtype về `float32` + `int64` cho edge_index để tương thích với GPU GNN forward.

## 33. Cái gì THỰC SỰ phải viết từ đầu

Sau khi tận dụng tối đa 2 repo trên, danh sách module phải code mới:

```text
fedcond_grag/data/hotpot_loader.py
fedcond_grag/data/federated_partition.py
fedcond_grag/data/corpus_index.py

fedcond_grag/graph_condensation/   (toàn bộ thư mục - DANCE không có code chính thức)
fedcond_grag/server_condensation/  (toàn bộ - đóng góp gốc)

fedcond_grag/retrieval/evidence_graph_builder.py
fedcond_grag/retrieval/global_graph_retriever.py

fedcond_grag/training/train_client_condensation.py
fedcond_grag/training/train_server_global_graph.py

src/model/dual_graph_llm.py        (~150 dòng subclass)
src/dataset/hotpot_fedcond.py      (~120 dòng theo template webqsp.py)
src/dataset/preprocess/hotpot.py   (~80 dòng)
```

Ước tính tổng dòng code phải viết mới: **~2.5K–3.5K LoC**, so với nếu viết lại cả Tri-Graph + LinearRAG retrieval + LLM wrapper từ đầu sẽ là **8K–10K LoC**.

## 34. Tham chiếu API ngoài cần verify khi clone

Khi clone về máy lần đầu, **chạy 3 verification scripts** trước khi viết logic mới, để biết exact API surface của LinearRAG (vì README không document đầy đủ):

```python
# scripts/verify_linearrag_api.py
import inspect
from fedcond_grag.external.linearrag import LinearRAG

# 1) Liệt kê attributes/methods của LinearRAG sau khi index()
rag = LinearRAG(global_config=...)
rag.index(["Some sample passage about Einstein."])
print([a for a in dir(rag) if not a.startswith("_")])

# 2) Inspect class config
print(inspect.getsource(type(rag.config)))

# 3) Chạy qa() trên 1 question, in toàn bộ return structure
out = rag.qa([{"question": "Who is Einstein?"}])
print(out)
```

Output của 3 scripts trên xác định chính xác tên attribute (`rag.entity_list` vs `rag.graph.entities` vs `rag._entities`) ở `trigraph_builder.py` mục 29 Step 5.

## 35. Migration Checkpoints (theo thứ tự thời gian)

Sau mỗi step ở §29, chạy checkpoint nhỏ trước khi tiếp:

| Step | Checkpoint command | Pass criteria |
|---|---|---|
| Step 0 | `ls fedcond_grag/external/linearrag/*.py` | ≥ 4 files |
| Step 1 | `python -c "from fedcond_grag.external.linearrag import LinearRAG"` | Không ImportError |
| Step 2 | `python -c "import spacy; spacy.load('en_core_web_trf')"` | Load ok |
| Step 3 | `python scripts/smoke_test_imports.py` | "Both codebases importable." |
| Step 4 | `python scripts/test_linearrag_wrapper.py` | retrieve(q) returns dict |
| Step 5 | `python scripts/test_trigraph_to_pyg.py` | PyG Data với 3 node_type values |
| Step 6 | `python -c "from src.model.dual_graph_llm import DualGraphLLM"` + dummy forward | Forward không crash |
| Step 7 | `python -c "from src.dataset import load_dataset; ds=load_dataset['hotpot_fedcond'](); print(ds[0].keys())"` | dict có evidence_graph + condensed_graph |
| Step 8 | `python train.py --dataset hotpot_fedcond --model_name dual_graph_llm --num_epochs 1` | Train 1 epoch không OOM, val loss hữu hạn |

---

# Part X — DANCE Reference Implementation Guide

> **Tại sao có Part này.** DANCE (Chen et al., 2026, arXiv:2601.16519) **không có code công khai** tại thời điểm viết. Tất cả phần `fedcond_grag/graph_condensation/` phải được implement từ paper. Phần này là tài liệu tham chiếu chi tiết cho người implement, gồm: (1) bridge giữa notation gốc và project của chúng ta, (2) pseudo-code Python-style cho 4 algorithm gốc, (3) **mười subtle points dễ implement sai** khi đọc paper, (4) adaptations từ original DANCE sang QA setting (no labels), (5) pre-flight checklist trước khi chạy stage B.
>
> Mọi reference `Eq. (X)` và `Algo Y` ở Part X này trỏ về paper DANCE.

## 36. Notation Bridge: DANCE → FedCondGraphRAG

| DANCE notation | DANCE meaning | Tương ứng trong project |
|---|---|---|
| `G^(m) = (V^(m), A^(m), S^(m))` | local TAG của client m | local **Tri-Graph** với 3 node types (entity/sentence/passage) |
| `V_L^(m)` | labeled subset của client m | **không tồn tại** (QA setting không có node label) |
| `y_v` / `ỹ_v` | true / pseudo class label | thay bằng **node_type** ∈ {0,1,2} |
| `ω^(t-1)` | last-round global model (GNN + classifier) | last-round **GNN_e + projection** của stage D, hoặc init random |
| `z_v = ω^(t-1)(G^(m))[v]` | node embedding từ global model | `g_v` của chúng ta (output GNN trên local Tri-Graph) |
| `π_v` | confidence của predicted label | **không có** (không classify); dùng heuristic thay thế nếu cần (xem §39) |
| `τ` | confidence threshold | bỏ qua trong adaptation |
| `V̂^(m)_t` | condensed core ở round t (size K) | **core node set** sau motif selection |
| `t_u` | node-level text embedding (pooled chunks) | giữ nguyên |
| `e_{u,r}` | chunk-level embedding | giữ nguyên |
| `c_v` | hierarchical context vector (Eq. 7) | intermediate, **chỉ dùng cho interpretability + optional alignment loss** |
| `t̃_v` | condensed evidence embedding (Eq. 9) | đi thẳng vào fusion → `x_v` |
| `x_v` | fused feature (Eq. 10) | node feature của client condensed graph |
| `Â^(m)_t` | reconstructed adjacency (Eq. 14) | `edge_index` + `edge_weight` của `C_m` upload object |
| `Δω^(t)_m` | model update gửi server | **không upload** trong project của chúng ta — thay vào đó upload `C_m` (graph object) làm anchor |

> ⚠️ Khác biệt cấu trúc cơ bản: trong **original DANCE**, mỗi round client (1) condense local graph, (2) train GNN trên condensed graph với CE loss, (3) upload `Δω^(t)_m`. Trong **FedCondGraphRAG**, client chỉ condense và upload **graph object `C_m`** làm anchor; server-side gradient matching xử lý tích hợp toàn cục. Do đó các bước "local training & upload" của DANCE Algo 1 line 9 sẽ được **thay** bằng "compute & upload condensed graph object" (xem §39.4).

## 37. Pseudo-code chi tiết cho 4 Algorithm

Đây là Python-style pseudo-code mà người implement có thể chuyển trực tiếp thành code. Tất cả tensor shape được note inline.

### 37.1 Algorithm 1 — Round-wise loop (adapted)

```python
# === Server side ===
def fedcond_round(t: int, clients: list, global_state) -> dict:
    # global_state chứa GNN_e + projection của stage D, hoặc init random ở round 1
    sampled = sample_clients(clients, fraction=rho)
    broadcast(global_state, to=sampled)
    
    # Mỗi client TRẢ VỀ một condensed graph object (KHÔNG phải Δω như DANCE original)
    anchors = []
    for m in sampled:
        C_m = client_round(m, t, global_state)
        anchors.append(C_m)
    
    # Stage C (Section 16-18) xử lý anchors
    server_state = update_server_global_graph(anchors)
    return server_state

# === Client side ===
def client_round(m, t: int, ω_prev) -> ClientCondensedGraph:
    # 1) Text caching (chỉ lần đầu hoặc khi encoder thay đổi)
    if not has_cache(m):
        for u in V_m:
            chunks_u = chunker(s_u)                    # list[str]
            e_u = [Enc(c) for c in chunks_u]           # list[Tensor d]
            t_u = attention_pool(e_u)                  # Tensor d
        save_cache(m, e=e_u, t=t_u)
    
    # 2) Forward GNN once với last-round model để có g_v (Eq. trước (5))
    g = forward_gnn(ω_prev.gnn, local_trigraph(m))     # Tensor [|V_m|, d]
    
    # 3) Node condensation (mỗi 10 round; còn lại reuse core cũ)
    if t % 10 == 1 or t == 1:
        # Original DANCE: label-aware. Adapted: query-agnostic motif selection.
        core_ids = motif_core_select(local_trigraph(m), g, x_e=cached_node_emb)
        save_core(m, core_ids)
    else:
        core_ids = load_core(m)
    
    # 4) Hierarchical text condensation (mỗi round) — Algo 3
    t_tilde, alpha_per_hop, pi_per_node = hierarchical_text_cond(
        core=core_ids, g=g, t_bank=load_cache(m),
        B0=1, B1=3, B2=2, B_tok=8,
    )
    save_evidence_pack(m, alpha_per_hop, pi_per_node)  # LOCAL ONLY
    
    # 5) Self-expressive topology reconstruction — Algo 4
    A_hat, X_fused = self_expressive_topology(
        core=core_ids, g=g[core_ids], t_tilde=t_tilde,
        alpha=8.0, beta=5.0, q=16, L=50, eta=1e-2, k_final=8,
    )
    
    # 6) Build ClientCondensedGraph object (KHÔNG có local training)
    C_m = ClientCondensedGraph(
        x=X_fused,
        edge_index=edges_from(A_hat),
        edge_weight=weights_from(A_hat),
        node_type=node_type_of(core_ids),    # entity/sentence/passage
    )
    return C_m
```

### 37.2 Algorithm 2 — (REPLACED) Query-Agnostic S-E-P Motif Selection

Thay thế **toàn bộ** Algo 2 của DANCE (label-aware clustering) bằng motif selection (xem Section 11 của plan này).

### 37.3 Algorithm 3 — Hierarchical Text Condensation

```python
def hierarchical_text_cond(core, g, t_bank, B0, B1, B2, B_tok,
                            W_q, W_k, W_s, gamma):
    """
    core: list[int]                  -- core node ids
    g: Tensor [|V|, d]               -- graph-side embeddings
    t_bank: dict                     -- per-node text bank
        t_bank['e'][u]: list[Tensor d]   -- chunk embs for node u
        t_bank['t'][u]: Tensor d          -- pooled node text emb
    W_q, W_k, W_s: nn.Linear(d, d)
    gamma: Tensor [3]                -- hop weights, sum to 1
    """
    d = g.size(1)
    t_tilde = {}; alpha_hops = {}; pi_chunks = {}
    
    for v in core:
        g_v = g[v]                                      # [d]
        
        # ---- Build candidate neighborhoods per hop ----
        N0 = [v]
        N1 = direct_neighbors(v)                        # 1-hop
        N2_full = two_hop_neighbors(v)                  # could be huge
        
        # SUBTLE POINT 3: 2-hop pre-filter by difficulty score (Eq. 21)
        # Original: u_w = entropy of predicted class distribution
        # Adapted: u_w = local heuristic, see §39
        u_scores = [difficulty_score(w) for w in N2_full]
        N2 = topk(N2_full, by=u_scores, k=B2 * 4)       # over-fetch 4x rồi score lại
        
        # ---- Neighbor gating per hop (Eq. 5, 6, 7) ----
        S_per_hop = {}                  # selected neighbors per hop
        alpha_per_hop = {}              # selected weights per hop
        
        for hop, (N_hop, B_hop) in enumerate(zip([N0, N1, N2], [B0, B1, B2])):
            if not N_hop:
                S_per_hop[hop] = []
                alpha_per_hop[hop] = empty([])
                continue
            
            # Eq. (5)
            scores = torch.stack([
                (W_q(g_v) @ W_k(t_bank['t'][u])) / sqrt(d)
                for u in N_hop
            ])                                          # [|N_hop|]
            
            # Eq. (6): entmax → ΠB hard truncation (with STE)
            # First version: top-k softmax (xem §38 point 1)
            alpha = topk_softmax(scores, k=B_hop)       # [|N_hop|], len(nonzero) ≤ B_hop
            
            selected_idx = nonzero(alpha)
            S_per_hop[hop]    = [N_hop[i] for i in selected_idx]
            alpha_per_hop[hop] = alpha
        
        # ---- Hierarchical context c_v (Eq. 7) - chỉ dùng cho audit/alignment ----
        c_v = torch.zeros(d)
        for hop in [0, 1, 2]:
            for i, u in enumerate(S_per_hop[hop]):
                c_v += gamma[hop] * alpha_per_hop[hop][i] * t_bank['t'][u]
        
        # ---- Chunk selection (Eq. 8, 9) ----
        # E_v = chunks từ tất cả selected neighbors (qua mọi hop)
        E_v = []
        for hop in [0, 1, 2]:
            for u in S_per_hop[hop]:
                for r, e_ur in enumerate(t_bank['e'][u]):
                    E_v.append((u, r, e_ur))
        
        if not E_v:
            t_tilde[v] = torch.zeros(d)
            continue
        
        q_v = W_s(g_v)                                  # [d]
        chunk_scores = torch.stack([
            (q_v @ e_ur) / sqrt(d) for (_, _, e_ur) in E_v
        ])                                              # [|E_v|]
        
        pi = topk_softmax(chunk_scores, k=B_tok)        # [|E_v|]
        
        t_tilde_v = sum(
            pi[i] * E_v[i][2] for i in range(len(E_v)) if pi[i] > 0
        )
        t_tilde[v] = t_tilde_v
        pi_chunks[v] = list(zip(E_v, pi.tolist()))      # local audit
        alpha_hops[v] = alpha_per_hop
    
    return t_tilde, alpha_hops, pi_chunks
```

### 37.4 Algorithm 4 — Self-Expressive Topology Reconstruction (ISTA)

```python
def self_expressive_topology(core, g_core, t_tilde, W_g, W_t, w_gate,
                              alpha_recon, beta_l1, q, L, eta, k_final):
    """
    Solve Eq. (13):
        min_Z α||X - XZ||_F^2 + β||Z||_1 + Σ (1 - S_ij) |Z_ij|
        s.t. Z_ij = 0 if j ∉ C(i), diag(Z) = 0
    """
    K = len(core)
    
    # ---- 1) Gated cross-modal fusion (Eq. 10) ----
    X = torch.zeros(K, d)
    for i, v in enumerate(core):
        g_v = g_core[i]
        t_v = t_tilde[v]
        alpha_v = torch.sigmoid(w_gate @ torch.cat([g_v, t_v]))   # scalar
        x_v = layer_norm(W_g(g_v) + alpha_v * W_t(t_v))            # [d]
        X[i] = x_v
    
    # ---- 2) Evidence prior matrix S (cosine of t_tilde) ----
    T = torch.stack([t_tilde[v] for v in core])                    # [K, d]
    S = cosine_similarity_matrix(T, T)                              # [K, K]
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)                 # rescale to [0,1]
    
    # ---- 3) Candidate support C(i) (Eq. 12) ----
    sim_X = X @ X.T                                                 # [K, K]
    sim_X.fill_diagonal_(-inf)                                      # exclude self
    
    C = [None] * K
    for i in range(K):
        Cx_i = topk_indices(sim_X[i], k=q)
        Cs_i = topk_indices(S[i], k=q)
        C[i] = set(Cx_i.tolist()) | set(Cs_i.tolist())
        C[i].discard(i)
    
    # ---- 4) Proximal gradient (ISTA) on masked Z ----
    Z = torch.zeros(K, K)
    mask = torch.zeros(K, K, dtype=torch.bool)
    for i in range(K):
        for j in C[i]:
            mask[i, j] = True
    
    # Derived from Eq. (13): scale L1 + prior by 1/α; below normalizes
    # Algo 4 uses (λ_1, λ_3) where λ_1 ≈ β/α and λ_3 ≈ 1/α (heuristically).
    lam_1 = beta_l1 / alpha_recon
    lam_3 = 1.0 / alpha_recon
    
    for it in range(L):
        # Gradient of α||X - XZ||_F^2 wrt Z: 2α · X^T (XZ - X)
        # (chia 2α để dùng η chuẩn hoá)
        G = X.T @ (X @ Z - X)                                       # [K, K]
        G = G * mask.float()
        G.fill_diagonal_(0)
        
        Z = Z - eta * G
        
        # Soft-thresholding với prior-aware threshold
        # τ_ij = η(λ_1 + λ_3 (1 - S_ij))
        tau = eta * (lam_1 + lam_3 * (1.0 - S))                     # [K, K]
        Z = torch.sign(Z) * torch.relu(torch.abs(Z) - tau)
        
        # Re-mask sau soft-thresh
        Z = Z * mask.float()
        Z.fill_diagonal_(0)
    
    # ---- 5) Symmetrize + top-k per row (Eq. 14) ----
    W = torch.abs(Z) + torch.abs(Z).T
    A_hat = topk_per_row(W, k=k_final)
    A_hat = (A_hat + A_hat.T) / 2.0                                # symmetrize
    
    return A_hat, X
```

## 38. Mười Subtle Points dễ Implement Sai

Đây là những điểm mà nếu đọc paper nhanh sẽ làm sai, dẫn đến accuracy thấp hoặc behavior khác:

### Point 1 — entmax + Straight-Through Estimator ≠ softmax + top-k

DANCE Eq. (6) và Eq. (8) dùng `Π_B(entmax(·))`, **không** softmax. Khác biệt cốt lõi:
- `entmax-α` với α=1.5 cho phép weights = 0 **chính xác** trong forward pass (sparse by design).
- `softmax` luôn cho mọi entry > 0; ép sparsity phải post-hoc bằng top-k.

Implementation:
- Forward: dùng `entmax15` từ thư viện [`entmax`](https://github.com/deep-spin/entmax) (pip install entmax), rồi áp `Π_B`.
- Backward: gradient của hard top-B truncate **không tự nhiên flow back**; cần straight-through estimator.

```python
class HardTopBSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_full, B):
        ctx.save_for_backward(p_full)
        # forward: keep top-B, zero out rest, renormalize
        topk_vals, topk_idx = torch.topk(p_full, B)
        out = torch.zeros_like(p_full)
        out.scatter_(0, topk_idx, topk_vals)
        out = out / (out.sum() + 1e-12)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        # straight-through: gradient passes như identity
        return grad_out, None
```

**Phiên bản đầu** của project có thể dùng `topk_softmax` (softmax → top-k → renormalize) làm approximation; chuyển sang entmax+STE ở Stage E ablation. Cần ghi rõ trong report rằng baseline là approximation.

### Point 2 — `c_v` (Eq. 7) KHÔNG vào trực tiếp `x_v` (Eq. 10)

Đọc paper kỹ:
- Eq. (7): `c_v = Σ γ_ℓ Σ α^(ℓ) t_u` — hierarchical context.
- Eq. (10): `x_v = LN(W_g g_v + α_v · W_t t̃_v)` — **chỉ dùng `t̃_v`** từ Eq. (9), không phải `c_v`.

Vậy `c_v` đóng vai trò gì? Theo paper:
- **Interpretability**: cho phép trace evidence theo hop, hỗ trợ "evidence packs".
- Có thể dùng làm input cho `o^t_v = Dec_t(c_v)` trong alignment loss (Eq. 11), nhưng paper viết `o^t_v = Dec_t(t̃_v)`.

**Khuyến nghị implement**: tính `c_v` chỉ cho audit (lưu locally), không dùng trong forward path của loss. Nếu accuracy thấp, thử `o^t_v = Dec_t(c_v)` thay `Dec_t(t̃_v)` như ablation.

### Point 3 — Difficulty score pre-filter chỉ áp cho 2-hop

Paper rất rõ: `Ñ^(2)_v = TopK_{B_2}({u_w}_{w ∈ N^(2)_v})`. Nhưng `Ñ^(0)_v = {v}` và `Ñ^(1)_v = N^(1)_v` (không pre-filter).

Lý do: `|N^(2)_v|` có thể đến `O(deg^2)` (hàng nghìn); scoring tất cả tốn kém. 1-hop và 0-hop nhỏ hơn nhiều.

Trong adaptation của chúng ta (không có pseudo-label entropy), thay `u_w` bằng:

```python
def difficulty_score_adapted(w, g, x_e):
    """
    Heuristic thay cho entropy (vì không có classifier):
    - Norm variance của graph embedding so với neighborhood mean
    - Hoặc inverse PageRank (rare nodes are 'harder')
    """
    return 1.0 / (1.0 + degree(w))     # easiest first to start; thử variants sau
```

### Point 4 — Frozen text encoder

Eq. trên Sec 4.3: `Enc` is **fixed**; `{t_u}` được **cache across rounds**. Gradients **KHÔNG** backprop vào `Enc`.

Trainable parameters trong stage B:
- `W_q, W_k, W_s` (cross-modal attention)
- `γ_ℓ` (hop weights — nếu coi là learnable; paper không nói rõ)
- `W_g, W_t, w_gate` (fusion)
- `W_decoder` (alignment loss heads, nếu dùng)
- GNN backbone parameters

KHÔNG trainable:
- `Enc` (SBERT/MiniLM)

Code check:

```python
for name, p in text_encoder.named_parameters():
    p.requires_grad_(False)
assert sum(p.requires_grad for p in text_encoder.parameters()) == 0
```

### Point 5 — Node condensation chạy mỗi 10 rounds, không phải mỗi round

Paper §4.2: "We perform node condensation every 10 communication rounds." Lý do: clustering (k-means) đắt; class distribution không thay đổi quá nhanh.

Trong adaptation của chúng ta, vì motif selection cũng đắt (PageRank + MMR), **giữ cùng nhịp**: refresh core mỗi 10 rounds. Text condensation và topology reconstruction vẫn refresh mỗi round.

### Point 6 — Cache invalidation rule

Cache `{e_{u,r}, t_u}` chỉ phụ thuộc vào `Enc` và `s_u`. Nó phải bị invalidate khi:
1. `Enc` thay đổi (e.g., switch encoder hoặc fine-tune).
2. Text `s_u` thay đổi (data update).

Trong project này: `Enc` cố định trong toàn stage B → cache **persistent**, chỉ build 1 lần. Đây là 1 ưu thế lớn so với LLM-based augmentation (LLaTA, LLM4FGL) — không cần re-encode mỗi round.

### Point 7 — `c_v` vs `t̃_v` vs `x_v`: ba vector khác nhau

| Vector | Eq. | Tính từ | Đóng góp vào |
|---|---|---|---|
| `t_u` | trên Eq. (5) | Pool({e_{u,r}}) cho mỗi node u | scoring (Eq. 5) |
| `c_v` | Eq. (7) | Σ γ_ℓ Σ α^(ℓ) t_u | (chỉ audit) |
| `t̃_v` | Eq. (9) | Σ π · e_{u,r} (chunk-level) | fusion → `x_v` |
| `x_v` | Eq. (10) | LN(W_g g_v + α_v · W_t t̃_v) | self-expression → `Â^(m)_t` |

Sai lầm thường gặp: dùng `c_v` thay `t̃_v` trong fusion → mất chunk-level granularity.

### Point 8 — Alignment loss (Eq. 11) trong original yêu cầu labels

```text
L_m = (1/|V̂_L|) Σ_{v ∈ V̂_L} CE(softmax(o_v), y_v)
    + λ_align · (1/|V̂|) Σ_v D(softmax(o^g_v), softmax(o^t_v))
```

- Term 1 (CE) cần `y_v` (class label).
- Term 2 (KL alignment) cần `o^g_v = Dec_g(g_v)` và `o^t_v = Dec_t(t̃_v)`.

**Adaptation trong project**: term 1 thay bằng node-type CE (3 classes: entity/sentence/passage). Term 2 giữ nguyên cấu trúc (KL alignment giữa graph-view logits và text-view logits) — đây vẫn có tác dụng regularize fusion bất kể có class labels hay không. Chi tiết xem §39.3.

### Point 9 — Self-expression Z không trainable qua autograd

Eq. (13) là một bài toán convex (LASSO with prior); paper §4.4 giải bằng "proximal gradient updates". Đây là **inner loop riêng**, không phải SGD trên params của model. Implementation:

```python
# Z không phải nn.Parameter; là tensor tạm, optimized bằng ISTA inside the round
Z = torch.zeros(K, K, requires_grad=False)
for _ in range(L):
    G = X.T @ (X @ Z - X)
    Z = Z - eta * G
    Z = soft_threshold(Z, tau=eta * (lam_1 + lam_3 * (1 - S)))
    # ... mask
```

`X` đến từ fusion (có grad), nhưng grad chỉ flow qua `X` lên `W_g, W_t, w_gate`, **không qua `Z`**.

### Point 10 — Local training của DANCE bị REPLACE trong project chúng ta

Paper Algo 1 line 9: "train on the condensed TAG using standard subgraph-FL, obtain update Δω^(t)_m, and upload Δω^(t)_m only."

Trong **FedCondGraphRAG**, client KHÔNG train trên condensed graph. Lý do:
- Original DANCE: end-task là **node classification** → cần train classifier trên `V̂^(m)_t,L`.
- Chúng ta: end-task là **QA via LLM prompting** → classifier không có; chỉ cần `C_m` làm anchor cho server gradient matching.

Thay vì upload `Δω^(t)_m`, client upload **C_m object** (graph data, không phải model delta). Server side (Section 16-18) tính anchor gradient từ surrogate task áp lên `C_m`. Đây là cấu trúc federated **khác** với DANCE.

## 39. Adaptations: Original DANCE → FedCondGraphRAG

### 39.1 Node condensation: pseudo-labels → node-type

| Original DANCE | Chúng ta |
|---|---|
| `ỹ_v = y_v if v ∈ V_L else ŷ_v` | bỏ; thay bằng `node_type[v]` (entity/sentence/passage) |
| `π_v = max_c p(c|v)` | bỏ; **không có threshold filter** |
| Cluster {z_v} per class | Cluster theo node_type không cần thiết (motif selection lo phần này) |
| Distribution-preserving quota per class | Quota per node_type: `K_e / K_s / K_p` đã được fix qua motif selection |

### 39.2 Difficulty score (Eq. 21): entropy → graph heuristic

| Original | Adaptation |
|---|---|
| `u_v = H(p(y|v))` (entropy of class distribution) | `u_v = 1 / (1 + degree(v))` hoặc `||g_v - mean(g_neighbors)||^2` |

Lý do: không có classifier để compute entropy.

### 39.3 Alignment loss: class CE → node-type CE + KL alignment

Loss của chúng ta:

```text
L_client = λ_type · (1/|V̂|) Σ_v CE(softmax(o_v), node_type[v])
         + λ_align · (1/|V̂|) Σ_v KL(softmax(o^g_v), softmax(o^t_v))
         + λ_link · binary_CE(link_predictor(x_i, x_j), A[i,j])
```

Trong đó:
- `o_v = Dec(x_v)`, `Dec: R^d → R^3` (3 node types).
- `o^g_v = Dec_g(g_v)`, `o^t_v = Dec_t(t̃_v)` cùng output dim.
- `link_predictor(x_i, x_j) = σ(MLP([x_i, x_j, x_i ⊙ x_j]))`.
- Default: `λ_type = 1.0`, `λ_align = 0.1`, `λ_link = 0.5`.

### 39.4 Communication: model delta → graph object

| DANCE | Chúng ta |
|---|---|
| Upload `Δω^(t)_m ∈ R^{|θ|}` (gradient/update) | Upload `C_m = (x, edge_index, edge_weight, node_type)` |
| Server: `Agg({Δω^(t)_m})` (FedAvg) | Server: anchor gradient matching trên `{C_m}` |
| Round structure: train → upload → aggregate | Round structure: condense → upload (1 lần ở stage B) |

Trong project, **stage B là 1-shot** (không phải lặp lại 200 rounds như DANCE). Chúng ta sinh `C_m` ổn định sau khi GNN encoder ban đầu (warm-up) đã đủ tốt; sau đó stage C (server) là pha riêng. Đây là một **đơn giản hóa**.

> **Trade-off**: bỏ round-wise refresh đồng nghĩa mất tính "model-in-the-loop" của DANCE. Nếu accuracy thấp ở stage D, fallback: chạy `C_m` refresh sau mỗi 10 epoch của stage D.

### 39.5 Per-class quota: REPLACED bằng node-type ratio (S-E-P motif quota)

Trong motif selection (Section 11):

```text
K_e = ⌈r_e · |V_e|⌉                     # entity anchors (default r_e = 0.05)
K_s ≤ K_e · B_s                          # sentence nodes (≤ B_s per anchor)
K_p ≤ K_e · B_p                          # passage nodes (≤ B_p per anchor)
K = K_e + K_s + K_p
```

Không có per-class budget (vì không có class). Thay vào đó: per-anchor budget (B_s, B_p).

## 40. Pre-flight Checklist trước khi chạy Stage B

Trước khi gọi `train_client_condensation.py`, verify:

```text
[ ] Text encoder `requires_grad = False` toàn bộ (verified bằng assertion)
[ ] Cache `{e_{u,r}, t_u}` được build và load đúng cho mọi client
[ ] Tri-Graph per client có cả 3 node types và cả 2 edge types
[ ] Motif core size K khớp config (K_e, K_s, K_p)
[ ] entmax library installed (nếu dùng entmax mode), hoặc fallback topk_softmax verified
[ ] Difficulty score adapter chạy được mà không crash trên 0-degree nodes
[ ] Self-expression solver converge (loss giảm trong L iterations)
[ ] Output C_m có `x, edge_index, edge_weight, node_type` shape đúng
[ ] Output C_m KHÔNG chứa string field, source span, raw text
[ ] Save/load C_m round-trip không mất dtype (float32, int64)
[ ] Memory footprint per client < ngưỡng (default: K · d < 1M values)
```

Mỗi mục có 1 unit test trong `tests/test_dance_components.py` (xem §41).

## 41. Suggested Unit Tests Skeleton

```python
# tests/test_dance_components.py
import torch
from fedcond_grag.graph_condensation import (
    neighbor_gating, chunk_selection, graph_text_fusion,
    topology_reconstruction
)

def test_neighbor_gating_respects_budget():
    g_v = torch.randn(384)
    t_neighbors = [torch.randn(384) for _ in range(20)]
    alpha = neighbor_gating.score_and_select(
        g_v, t_neighbors, W_q=torch.eye(384), W_k=torch.eye(384),
        B=3,
    )
    assert (alpha > 0).sum() <= 3, "budget violation"
    assert alpha.shape == (20,)

def test_chunk_selection_respects_token_budget():
    q_v = torch.randn(384)
    chunks = [torch.randn(384) for _ in range(50)]
    pi = chunk_selection.score_and_select(
        q_v, chunks, W_s=torch.eye(384), B_tok=8,
    )
    assert (pi > 0).sum() <= 8

def test_fusion_gate_not_collapsed():
    g = torch.randn(10, 384)
    t = torch.randn(10, 384)
    x, alpha_gate = graph_text_fusion.fuse(g, t, ...)
    # Sanity: gate values not all near 0 or all near 1
    assert 0.05 < alpha_gate.mean() < 0.95

def test_self_expression_sparse_and_symmetric():
    X = torch.randn(50, 384)
    A_hat = topology_reconstruction.self_expressive(
        X, alpha=8.0, beta=5.0, q=16, L=50, eta=1e-2, k_final=8
    )
    # Symmetric
    assert torch.allclose(A_hat, A_hat.T, atol=1e-6)
    # Sparse: avg row sum ≤ k_final
    assert (A_hat > 0).sum(dim=1).float().mean() <= 8.0
    # No self-loops
    assert torch.diag(A_hat).abs().sum() < 1e-6

def test_text_encoder_frozen():
    from fedcond_grag.graph_condensation.text_bank import build_bank
    bank = build_bank(passages=["test"], encoder_name='all-MiniLM-L6-v2')
    n_trainable = sum(p.requires_grad for p in bank.encoder.parameters())
    assert n_trainable == 0
```

## 42. References (DANCE specific)

Các Eq. number và Algo number trong Part X này trỏ về:

> Chen, Z., Lu, H., Li, X., Sun, H., Li, J., Qin, H., Li, R.-H., Wang, G. **"DANCE: Dynamic, Available, Neighbor-gated Condensation for Federated Text-Attributed Graphs."** arXiv:2601.16519, Jan 2026.

Các thuật ngữ map sang section trong paper:
- §4.2 → motif selection module (replaced)
- §4.3 → hierarchical text condensation module
- §4.4 → self-expressive topology reconstruction module
- §A (Appendix A) → difficulty score Eq. (21)
- §B.6 → DP noise injection (defer to v2 of project)
- §D.1–D.3 → theoretical guarantees (informative, not blocking)
- Table 5 → default hyperparameters đã được merge vào Appendix A của plan này.

---

# Part XI — Federated Learning Infrastructure: OpenFGL/gfl Integration

> **Tại sao có Part này.** Repo [`chuotchuilacduong/gfl`](https://github.com/chuotchuilacduong/gfl) thực ra là một fork/extension của [**OpenFGL**](https://github.com/xkLi-Allen/OpenFGL) (Li et al., 2024, arXiv:2408.16288) — một framework benchmark cho federated graph learning với sẵn ≥ 10 thuật toán FL, 34 dataset, 12 GNN model, partitioning (Louvain/Metis), DP infra, và quan trọng nhất: **toàn bộ 4 baseline federated graph condensation** mà DANCE so sánh (FedC4, FedGVD, FedGM, FedGTA). Tích hợp gfl giúp tiết kiệm thêm ~2K LoC cho stage C (server condensation) và **toàn bộ baseline section**, vì các baselines này đã được implement sẵn và đã được sửa lỗi qua nhiều round.
>
> Part này là extension của **Part IX** (Code Integration Strategy). Đọc Part IX trước; Part XI bổ sung một tầng thứ ba sau G-Retriever và LinearRAG.

## 43. Vì sao gfl là right tool

Khi đối chiếu với plan hiện tại:

| Module trong plan | Phải code lại (trước khi biết gfl) | Có sẵn trong gfl? |
|---|---|---|
| Federated partition (Section 9.2) | hash(title) — đơn giản | ✅ **Louvain, Metis, Dirichlet, label-skew** (`openfgl/utils/`) |
| Subgraph FL trainer (Section 21) | Phải code FGLTrainer skeleton | ✅ **`FGLTrainer`** sẵn — `openfgl/flcore/trainer.py` |
| Server-side condensation (Section 16-18) | PGE + gradient matching | ✅ **FedGM** = chính paper Zhang et al. 2025a — có code; reuse logic |
| FedAvg / FedProx / FedSage+ baselines | Phải code 3-5 thuật toán | ✅ Hơn 10 thuật toán FL sẵn — chỉ chọn flag |
| Federated graph condensation baselines | Phải code 4 baselines DANCE Bảng 1 | ✅ **FedC4, FedGVD, FedGM, FedRGD** sẵn |
| GNN backbones (GCN, GAT, GraphSAGE) | Phải copy từ G-Retriever | ✅ 12 GNN models sẵn — gồm cả những cái G-Retriever không có (SGC, MLP, etc.) |
| DP / privacy mechanisms | Defer to v2 | ✅ DP framework sẵn (`--dp_mech`, `--noise_scale`) |
| Communication cost tracking | Phải code | ✅ `--comm_cost` flag sẵn |
| Multi-client simulation | Phải code multiprocessing | ✅ Sẵn |
| Evaluation modes | Phải code | ✅ 4 modes: local-on-local, local-on-global, global-on-local, global-on-global |

→ gfl xử lý **toàn bộ infrastructure layer**. Chúng ta chỉ cần inject Tri-Graph + LinearRAG + DANCE logic.

## 44. Repo identification

Sau khi inspect `main.py`, `config.py` của `chuotchuilacduong/gfl`:

- Path mặc định: `/home/zhanghao/fgl/OpenFGL/openfgl/dataset` → xác nhận đây là OpenFGL fork.
- `from flcore.trainer import FGLTrainer` ↔ OpenFGL upstream `from openfgl.flcore.trainer import FGLTrainer`.
- `supported_fl_algorithm` ở `config.py` chứa: `fedavg`, `fedprox`, `scaffold`, `moon`, `feddc`, `fedproto`, `fedtgp`, `fedpub`, `fedstar`, `fedgta`, `fedtad`, `gcfl_plus`, `fedsage_plus`, `adafgl`, `feddep`, `fggp`, `fgssl`, `fedgl`, `fedhgn3`, **`fedgm`**, **`fedrgd`**, `hyperion`, `fedigl`, `fedlog`, `fedomg`, `fedaux`, `centralized`, **`fedgvd`**, `fedgkg`, **`fedc4`**.
- Có riêng `flcore/fedgm/fedgm_config.py`, `flcore/hyperion/hyperion_config.py` → mỗi algorithm có module độc lập với config riêng.
- Có flag `--method ∈ {GCond, SGDD, DosCond}` → reuse các condensation methods khác làm ablation.
- Có sẵn `--num_global_syn_nodes`, `--server_condense_iters`, `--condense_iters` → **chính xác** match với Section 17 (server synthetic graph) của plan.

> ⚠️ `gfl` không có README chi tiết và không có paper riêng (citation chưa được public). Khi cần documentation chi tiết, **đọc của OpenFGL upstream** (`xkLi-Allen/OpenFGL`) — API tương thích.

## 45. Cấu trúc thư mục gfl (top-level)

```text
gfl/
├── flcore/                 # ← FL algorithms (mỗi sub-folder = 1 thuật toán)
│   ├── trainer.py          # FGLTrainer — orchestrator chung
│   ├── fedavg/
│   ├── fedprox/
│   ├── fedsage_plus/
│   ├── fedgta/
│   ├── fedgm/              # ← FedGM (Zhang 2025a) — REFERENCE cho stage C
│   │   ├── fedgm_config.py
│   │   ├── fedgm_server.py
│   │   ├── fedgm_client.py
│   │   └── ...
│   ├── fedc4/              # ← FedC4 (Chen 2025) — baseline DANCE
│   ├── fedgvd/             # ← FedGVD (Dai 2025) — baseline DANCE
│   ├── fedrgd/             # ← FedRGD — variant của FedGM
│   ├── hyperion/
│   └── ...
├── task/                   # ← Task definitions
│   └── (node_cls, link_pred, node_clust, graph_cls, ...)
├── model/                  # ← GNN backbones (GCN, GAT, SGC, GraphSAGE, MLP, ...)
├── utils/                  # ← Partitioning, seeds, metrics
│   ├── basic_utils.py      # seed_everything
│   └── (partition utils)
├── data/                   # ← Dataset loaders
├── config.py               # ← Giant argparse — toàn bộ args
├── main.py                 # ← Entry point (39 lines)
├── env.yaml                # Conda env
└── requirements.txt
```

API public chính:

```python
from flcore.trainer import FGLTrainer
trainer = FGLTrainer(args)
trainer.train()
```

`args` được parse từ `config.py` với hơn 100 flags. Quan trọng nhất:
- `--fl_algorithm` chọn thuật toán
- `--simulation_mode subgraph_fl_louvain` cho partitioning
- `--num_clients`, `--num_rounds`, `--client_frac`
- `--dataset`, `--task`, `--model`
- `--num_global_syn_nodes`, `--server_condense_iters` cho FedGM family
- `--dp_mech`, `--noise_scale` cho DP

## 46. Layout cập nhật — 3-tier integration

So với Part IX (G-Retriever + LinearRAG), giờ có 3 codebase được merge:

```text
FedCondGraphRAG/                            # fork của G-Retriever (host)
├── src/                                    # G-Retriever core (giữ)
│   ├── dataset/        ...
│   ├── model/          ...
│   └── utils/          ...
├── train.py, inference.py                  # giữ
│
├── fedcond_grag/                           # custom logic (đa số code mới)
│   ├── data/                               # HotpotQA loader, federated split
│   ├── graph_building/                     # Tri-Graph wrap LinearRAG
│   ├── graph_condensation/                 # DANCE re-implementation (Part X)
│   ├── retrieval/                          # LinearRAG retriever wrapper
│   ├── training/                           # stage A/B/D entry scripts
│   ├── external/
│   │   ├── linearrag/                      # vendored from DEEP-PolyU/LinearRAG
│   │   └── gfl/                            # ← NEW: vendored from chuotchuilacduong/gfl
│   │       ├── flcore/                     # FL algorithms (FedAvg, FedGM, FedC4, ...)
│   │       ├── task/                       # task definitions
│   │       ├── model/                      # GNN backbones (extra: SGC, MLP, GraphSAGE)
│   │       ├── utils/                      # partitioning + helpers
│   │       └── config.py                   # giant argparse
│   └── server_condensation/                # CUSTOM — extends fedgm logic
│       ├── fedcond_qa/                     # CUSTOM FL algorithm
│       │   ├── fedcond_qa_config.py
│       │   ├── fedcond_qa_server.py        # extends fedgm_server.py
│       │   ├── fedcond_qa_client.py
│       │   ├── surrogate_tasks.py          # node-type, link-pred (no class label)
│       │   └── __init__.py
│       └── adapters.py                     # bridge gfl ↔ Tri-Graph format
│
├── configs/                                # YAML configs
└── scripts/                                # shell wrappers
```

### 46.1 Server-side condensation: REPLACED bằng "fedcond_qa" trên nền fedgm

Trong Part IX, mục `fedcond_grag/server_condensation/` ban đầu chứa:
- `anchor_gradient.py`, `synthetic_graph.py`, `pge.py`, `gradient_matching.py`

→ **Thay** bằng custom FL algorithm `fedcond_qa/` đăng ký vào registry của gfl. Mỗi file copy từ `flcore/fedgm/` rồi sửa.

## 47. FedGM as reference cho Server Condensation (Stage C)

**FedGM** (Zhang et al. 2025a, "Rethinking Federated Graph Learning: A Data Condensation Perspective", arXiv:2505.02573) là **chính baseline mà DANCE so sánh** trong Bảng 1, và là kiến trúc gần nhất với stage C của chúng ta. Trong gfl, FedGM được implement ở `flcore/fedgm/`. Đọc kỹ trước khi viết stage C.

### 47.1 Mapping FedGM → Stage C của FedCondGraphRAG

| FedGM concept (gfl) | Stage C của chúng ta (Section 16-18) | Action |
|---|---|---|
| Client local subgraph condensation | Client Tri-Graph → C_m (đã làm ở Stage B) | **Skip**: stage B đã xử lý |
| `--num_global_syn_nodes` (server synthetic nodes) | `K_g` (Section 17.1) | Map trực tiếp |
| Server-side condensation iterations (`--server_condense_iters`) | Outer loop của gradient matching | Map trực tiếp |
| Per-client condensation iterations (`--condense_iters`) | Number of ISTA inner iterations (DANCE Algo 4) | Map (đã có ở Section 14) |
| `--method ∈ {GCond, SGDD, DosCond}` | Condensation strategy | Bắt đầu `GCond` (gradient matching cơ bản) |
| FedGM aggregation: weighted sum của client condensed graphs | Anchor gradient aggregation (Section 16.2) | Reuse logic |
| FedGM gradient matching loss | `L_match = 1 - cos(g_global, g_anchor)` (Section 18.1) | Reuse |

### 47.2 Subclass strategy

Tạo `fedcond_qa_server.py` kế thừa từ `FedGMServer`:

```python
# fedcond_grag/server_condensation/fedcond_qa/fedcond_qa_server.py
from fedcond_grag.external.gfl.flcore.fedgm.fedgm_server import FedGMServer

class FedCondQAServer(FedGMServer):
    """Server cho FedCondGraphRAG, kế thừa FedGM."""
    
    def __init__(self, args):
        super().__init__(args)
        # Thêm node_type embedding cho synthetic nodes (Tri-Graph specific)
        self.type_emb = nn.Embedding(3, args.hid_dim)  # 3 = entity/sentence/passage
    
    def init_synthetic_graph(self, anchor_graphs):
        """Override: init X_global theo node_type ratio từ anchors."""
        # Sample types from average anchor distribution
        type_counts = self._aggregate_type_counts(anchor_graphs)
        ratios = type_counts / type_counts.sum()
        n_per_type = (ratios * self.args.num_global_syn_nodes).long()
        # Init mỗi block từ cluster mean của anchor nodes cùng type
        ...
    
    def compute_anchor_gradients(self, anchor_graphs):
        """Override: gradient từ SURROGATE task (no class labels)."""
        # gfl FedGM dùng CE trên class labels.
        # Chúng ta thay bằng node-type prediction + link prediction.
        loss = self._surrogate_task_loss(anchor_graphs)
        return torch.autograd.grad(loss, self.gnn.parameters())
    
    def server_condense_step(self):
        """Override: thêm PGE adjacency learning."""
        # FedGM gốc có A_global cố định hoặc dense.
        # Chúng ta thay bằng PGE (Section 17.2).
        A_global = self.pge(self.X_global)
        # ... gradient matching trên (X_global, A_global)
```

### 47.3 Custom config

Tạo `fedcond_qa_config.py` follow đúng pattern `fedgm_config.py`:

```python
# fedcond_grag/server_condensation/fedcond_qa/fedcond_qa_config.py
config = {
    "method": "GCond",                  # base method
    "op_epoche": 5,                     # outer loop epochs
    "server_condense_iters": 50,        # server condensation iters
    "condense_iters": 50,               # client-side (ISTA inner)
    "local_epochs": 0,                  # KEY DIFFERENCE: no client local training
    "num_global_syn_nodes": 1024,       # K_g
    # PGE-specific
    "pge_hidden": 256,
    "pge_topk": 8,
    # Surrogate task
    "surrogate_type_weight": 1.0,
    "surrogate_link_weight": 0.5,
    "surrogate_align_weight": 0.1,
}
```

### 47.4 Register vào main.py registry

Trong `main.py` của gfl đã có pattern:

```python
fedgm_based_algorithms = ['fedgm', 'fedrgd', 'fedgkg']
if args.fl_algorithm in fedgm_based_algorithms:
    from flcore.fedgm.fedgm_config import config as fedgm_cfg
    ...
```

Mở rộng thành:

```python
fedgm_based_algorithms = ['fedgm', 'fedrgd', 'fedgkg', 'fedcond_qa']  # ADD
if args.fl_algorithm in fedgm_based_algorithms:
    if args.fl_algorithm == 'fedcond_qa':
        from fedcond_grag.server_condensation.fedcond_qa.fedcond_qa_config import config as cfg
    else:
        from flcore.fedgm.fedgm_config import config as cfg
    ...
```

Trong `flcore/__init__.py` hoặc `flcore/trainer.py` (tuỳ vào cách gfl resolve algorithm):

```python
ALGORITHM_REGISTRY = {
    "fedavg": (FedAvgServer, FedAvgClient),
    "fedgm":  (FedGMServer,  FedGMClient),
    "fedcond_qa": (FedCondQAServer, FedCondQAClient),   # ADD
    ...
}
```

## 48. Custom FL Algorithm "fedcond_qa" — what to add

Khi cài đặt `fedcond_qa` đăng ký vào gfl, theo convention OpenFGL mỗi algorithm cần:

```text
fedcond_qa/
├── __init__.py
├── fedcond_qa_config.py          # extra config dict
├── fedcond_qa_server.py          # subclass FedGMServer
├── fedcond_qa_client.py          # subclass FedGMClient
├── fedcond_qa_trainer.py         # optional: nếu cần custom train loop
└── surrogate_tasks.py            # node-type + link-pred loss
```

`fedcond_qa_client.py` quan trọng nhất — đây là nơi inject **Stage B** (DANCE-style condensation):

```python
# fedcond_qa_client.py
from fedcond_grag.external.gfl.flcore.fedgm.fedgm_client import FedGMClient
from fedcond_grag.graph_condensation import client_condensor

class FedCondQAClient(FedGMClient):
    """Client làm Tri-Graph + S-E-P motif + DANCE condensation."""
    
    def __init__(self, args, client_id, local_data):
        super().__init__(args, client_id, local_data)
        # Local data ở đây là Tri-Graph (đã build sẵn từ Stage A)
        self.tri_graph = local_data
    
    def local_train(self):
        """Override: KHÔNG train GNN local. Chỉ run condensation."""
        if self.round % self.args.condense_refresh_every == 0:
            self.condensed_graph = client_condensor.condense(
                tri_graph=self.tri_graph,
                last_round_gnn=self.gnn,
                config=self.args,
            )
        # Return condensed graph (KHÔNG return model delta như FedGM gốc)
        return self.condensed_graph
    
    def upload(self):
        """Override: upload C_m thay vì Δω."""
        return self.condensed_graph     # PyG Data object
```

→ Như vậy, Stage B (Part III §11-15) trở thành **thuần `local_train()` method** của `FedCondQAClient`. FL infrastructure của gfl lo phần round loop, sampling, aggregation gọi.

## 49. Baselines free từ gfl

Quan trọng cho Section 23-24 (Evaluation): các baseline mà DANCE so sánh và mà chúng ta cũng cần so sánh đều có sẵn:

| Baseline | DANCE Bảng 1 | gfl flag | Mục đích |
|---|---|---|---|
| FedAvg | ✅ | `--fl_algorithm fedavg` | Trivial baseline |
| FedSage+ | ✅ | `--fl_algorithm fedsage_plus` | Subgraph-FL canonical |
| FedGTA | ✅ | `--fl_algorithm fedgta` | Topology-aware aggregation |
| GCond | ✅ | `--fl_algorithm fedgm --method GCond --num_clients 1` | Centralized condensation |
| SFGC | ✅ | (gfl chưa list; có thể có ở variant) | Structure-free condensation |
| FedC4 | ✅ | `--fl_algorithm fedc4` | Federated condensation |
| FedGVD | ✅ | `--fl_algorithm fedgvd` | Generative virtual nodes |
| FedGM | ✅ | `--fl_algorithm fedgm` | Our direct comparison |
| DANCE | ✅ | **chưa có** — phải tự code (xem Part X) | Direct comparison |
| **Ours (fedcond_qa)** | — | `--fl_algorithm fedcond_qa` | Đóng góp của chúng ta |

→ Trừ DANCE (cần tự implement theo Part X), tất cả còn lại có thể chạy với 1 dòng lệnh:

```bash
python main.py --fl_algorithm fedavg     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedsage_plus --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgta     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgm      --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedc4      --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgvd     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedcond_qa --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
```

Để các baselines này chạy được trên Tri-Graph, cần một **dataset adapter** thoả mãn interface của `openfgl/task/subgraph_fl/`. Xem §50.4.

## 50. Integration Workflow

### 50.1 Vendor gfl

```bash
# Trong host repo
cd FedCondGraphRAG
git clone https://github.com/chuotchuilacduong/gfl.git /tmp/gfl
mkdir -p fedcond_grag/external/gfl
cp -r /tmp/gfl/{flcore,task,model,utils,config.py,main.py} fedcond_grag/external/gfl/
cp /tmp/gfl/requirements.txt requirements_gfl.txt
```

### 50.2 Rewrite imports

Như đã làm với LinearRAG (§29 Step 1), rewrite import path:

```bash
cd fedcond_grag/external/gfl
find . -name "*.py" -exec sed -i \
  -e 's|from flcore\.|from fedcond_grag.external.gfl.flcore.|g' \
  -e 's|from utils\.|from fedcond_grag.external.gfl.utils.|g' \
  -e 's|from model\.|from fedcond_grag.external.gfl.model.|g' \
  -e 's|from task\.|from fedcond_grag.external.gfl.task.|g' \
  -e 's|from config |from fedcond_grag.external.gfl.config |g' \
  {} +
```

### 50.3 Merge dependencies

`requirements_gfl.txt` của gfl:
```
torch==2.0.1+cu117
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu117
torch-sparse==0.6.18+pt20cu117
scipy==1.14.0
numpy==1.26.4
ogb==1.3.6
```

→ Tương thích với G-Retriever (`torch==2.0.1`, `torch-geometric`). Khác nhỏ: `torch-scatter` và `torch-sparse` version. Chọn version higher; test compatibility.

### 50.4 Dataset adapter: Tri-Graph → gfl format

gfl/OpenFGL kỳ vọng dataset là PyG `Data` với `.x`, `.edge_index`, `.y`. Tri-Graph của chúng ta:

```python
# fedcond_grag/external/gfl/task/subgraph_fl/hotpot_trigraph.py
import torch
from torch_geometric.data import Data
from fedcond_grag.graph_building.trigraph_builder import build_trigraph_for_client

def load_hotpot_trigraph(args):
    """
    Load HotpotQA Tri-Graph và trả về list[Data] (1 per client).
    OpenFGL Louvain partitioning được skip vì đã có partition theo title hash.
    """
    clients_data = []
    for m in range(args.num_clients):
        passages = load_client_passages(m)
        tri_graph_pyg = build_trigraph_for_client(passages, cfg=...)
        # node_type as "y" để gfl không complain
        tri_graph_pyg.y = tri_graph_pyg.node_type
        tri_graph_pyg.num_classes = 3
        clients_data.append(tri_graph_pyg)
    return clients_data
```

Register vào gfl dataset loader (xem `openfgl/data/`).

### 50.5 Custom task: condensation_qa

gfl default tasks: `node_cls, link_pred, node_clust, graph_cls, graph_reg`. Thêm task `condensation_qa`:

```python
# fedcond_grag/external/gfl/task/condensation_qa.py
class CondensationQATask:
    """Task không có class labels; loss = surrogate (node-type + link)."""
    
    def __init__(self, args):
        self.args = args
    
    def loss(self, output, batch):
        # surrogate loss thay cho CE
        from fedcond_grag.server_condensation.fedcond_qa.surrogate_tasks import (
            node_type_ce, link_prediction_bce, kl_alignment
        )
        l = (self.args.surrogate_type_weight * node_type_ce(output, batch)
             + self.args.surrogate_link_weight * link_prediction_bce(output, batch)
             + self.args.surrogate_align_weight * kl_alignment(output, batch))
        return l
    
    def evaluate(self, output, batch):
        # Trả về dict metrics (accuracy của node-type prediction làm proxy)
        return {"type_acc": ..., "link_auc": ...}
```

Register vào `flcore/trainer.py` task selector.

### 50.6 Verification

Kịch bản verification gồm 3 bước:

```bash
# Step 1: gfl import works
python -c "from fedcond_grag.external.gfl.flcore.trainer import FGLTrainer; print('OK')"

# Step 2: FedAvg trên Tri-Graph chạy được (sanity baseline)
python -m fedcond_grag.external.gfl.main \
    --fl_algorithm fedavg \
    --dataset hotpot_trigraph \
    --num_clients 5 \
    --num_rounds 3 \
    --simulation_mode subgraph_fl_louvain \
    --task condensation_qa \
    --model gcn

# Step 3: fedcond_qa custom algorithm chạy được
python -m fedcond_grag.external.gfl.main \
    --fl_algorithm fedcond_qa \
    --dataset hotpot_trigraph \
    --num_clients 5 \
    --num_rounds 3 \
    --num_global_syn_nodes 1024 \
    --server_condense_iters 10 \
    --simulation_mode subgraph_fl_louvain
```

Nếu 3 step trên pass, stage B + C đã chạy được trong khung gfl. Stage D (dual-graph prompting với LLM) vẫn chạy tách rời qua `train.py` của G-Retriever (xem Part IX §29 Step 8).

## 51. Cập nhật bảng "thực sự phải viết mới" (đối chiếu Part IX §33)

Sau khi có gfl, ước tính LoC mới giảm thêm:

| Module | Trước khi có gfl | Sau khi có gfl |
|---|---|---|
| Federated partition | ~150 LoC | 0 (dùng Louvain của gfl) |
| FGL trainer skeleton | ~300 LoC | 0 (`FGLTrainer` sẵn) |
| `server_condensation/anchor_gradient.py` | ~200 LoC | ~50 LoC (subclass FedGM) |
| `server_condensation/synthetic_graph.py` | ~200 LoC | ~50 LoC (extend init logic) |
| `server_condensation/pge.py` | ~150 LoC | ~150 LoC (chưa có sẵn, vẫn phải viết) |
| `server_condensation/gradient_matching.py` | ~300 LoC | ~80 LoC (chỉ override loss function) |
| Baselines | ~2K LoC nếu viết 5 baselines | 0 (free từ gfl) |
| **Tổng** | ~2.5K–3.5K LoC mới | **~1K–1.5K LoC mới** |

→ Tiết kiệm thêm khoảng 60% so với plan Part IX. Tổng tiết kiệm so với viết-từ-đầu **80%**.

## 52. Cập nhật Appendix B — relationship matrix

Thêm gfl vào bảng:

| Source | Reused / adapted | Modified or replaced |
|---|---|---|
| G-Retriever | GNN encoders, projection MLP, LLM wrapper, training loop, `textualize_graph()` | Single-encoder → dual-encoder; PCST retrieval → LinearRAG + global graph |
| LinearRAG | Tri-Graph build, entity activation, PPR retrieval | Centralized → federated; wrapped trong `LinearRAGRetriever` façade |
| DANCE (paper, no code) | text bank caching, neighbor gating, chunk selection, self-expressive topology | Label-aware → query-agnostic; class CE → node-type CE + KL alignment; local training → upload C_m |
| **gfl / OpenFGL** | **FGLTrainer, partition utils, FedGM logic, 10+ baseline algorithms, DP infra, evaluation modes, comm-cost tracking** | **Custom algorithm `fedcond_qa` extends `fedgm`; custom task `condensation_qa`; custom dataset `hotpot_trigraph`** |

## 53. Risks specific to gfl integration

| Risk | Mitigation |
|---|---|
| **R8 — gfl không có "no-label" task** | Tự define `CondensationQATask`; node_type làm proxy label |
| **R9 — FedGM/FedC4 trên Tri-Graph mạnh hơn dự đoán** | Tốt cho positioning: nếu fedcond_qa thắng cả FedGM, kết quả mạnh; nếu thua, ablation cho thấy đóng góp của S-E-P motif + DANCE text condensation |
| **R10 — Louvain partition cho Tri-Graph có thể chia rời S-E-P motifs** | Test partition quality bằng motif coverage trước khi train; fallback: partition theo hash(title) như Section 9.2 |
| **R11 — gfl chưa có DANCE; phải tự implement (Part X) trong khung gfl** | Reuse FedGMClient subclass + custom local_train(); xem §48 |
| **R12 — Version conflict torch-scatter/sparse giữa G-Retriever và gfl** | Lock versions; build wheel cho cu118 từ `data.pyg.org` |
| **R13 — OpenFGL khá ít star (41), chưa nổi tiếng → có thể có bug** | Đọc kỹ commit history + issues; preserve `gfl` snapshot version trong submodule SHA |

## 54. Migration timeline cập nhật

Cập nhật Part IV §22 với gfl integration:

| Milestone | Output | Sử dụng gfl? |
|---|---|---|
| M1 — Data setup | HotpotQA processed | ❌ |
| M2 — Tri-Graph builder | Per-client `trigraph.pt` | wrap LinearRAG |
| M3 — Motif selection | Core graphs | ❌ |
| **M4a — gfl vendor + verify** | `fedavg` chạy được trên Tri-Graph | ✅ FREE BASELINE |
| M4 — Text condensation (DANCE) | `t_tilde_v`, `x_fused` | ❌ (theo Part X) |
| M5 — Topology recon (DANCE) | `A_hat` | ❌ (theo Part X) |
| **M6 — Server condensation (`fedcond_qa`)** | Anchor gradients, PGE, gradient matching | ✅ EXTEND FedGM |
| **M6b — Free baselines** | FedSage+, FedGTA, FedGM, FedC4, FedGVD results | ✅ 1 LINE EACH |
| M7 — Dual graph prompting (Stage D) | E2E QA pipeline | ❌ (G-Retriever) |
| M8 — Ablations + analysis | Full eval table | mix |

→ **M4a và M6b** là milestone mới chỉ tồn tại nhờ gfl, giúp có baselines sớm và **giảm risk** đáng kể (nếu fedcond_qa không thắng baseline đơn giản như fedavg, biết ngay từ đầu chứ không phải đến cuối project).

## 55. Tóm tắt 3-tier stack

```text
┌───────────────────────────────────────────────────────────────┐
│              FedCondGraphRAG (custom logic)                    │
│   - Tri-Graph builder, S-E-P motif, DANCE adaptation,          │
│   - Dual graph prompting, evidence/condensed encoders          │
│   ≈ 1K–1.5K LoC mới                                            │
└────────────┬──────────────────────┬────────────────────────────┘
             │                      │
             ▼                      ▼
┌─────────────────────┐  ┌─────────────────────────┐
│  G-Retriever        │  │  LinearRAG              │  ← Section 47 references both
│  - LLM + GNN soft   │  │  - Tri-Graph build       │
│    prompt           │  │  - 2-stage retrieval     │
│  - PCST (unused)    │  │  - PPR + sem bridging    │
│  - LoRA / freeze    │  │                          │
│  ≈ 5K LoC reused    │  │  ≈ 3K LoC reused         │
└─────────────────────┘  └─────────────────────────┘
             │                      
             ▼                      
┌──────────────────────────────────────────────────────┐
│  gfl / OpenFGL (federated infrastructure)             │
│  - FGLTrainer, round loop, client/server abstraction  │
│  - Partitioning (Louvain, Metis, Dirichlet)           │
│  - FedAvg, FedProx, FedSage+, FedGTA, FedGM, FedC4,   │
│    FedGVD, FedRGD (free baselines)                    │
│  - DP, comm cost, evaluation modes                    │
│  ≈ 10K LoC reused                                     │
└──────────────────────────────────────────────────────┘
```

Tổng: ≈ **18K LoC reused** vs **1K–1.5K LoC viết mới** = **92% reuse**.

---

*End of plan.*
