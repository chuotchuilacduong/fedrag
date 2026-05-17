---
file: 01_OVERVIEW.md
title: Research Narrative & Scope
load_priority: always-load
prerequisites: []
related: [02_DATA_AND_TRIGRAPH.md, 09_INT_HOST_REPO.md]
covers_sections: "Part I (Motivation, RQs, Contributions); Part II (MVV, Codebase Strategy, Code Structure)"
project: FedCondGraphRAG
---

# Research Narrative & Scope

> **How to use this file.** Đây là file **load đầu tiên** trong mọi session. Chứa motivation, research questions, framework overview, scope của bản đầu tiên (MVV), và bố cục code structure 3-tier (G-Retriever + LinearRAG + DANCE). Agent nên hiểu rõ phần này trước khi đi vào chi tiết module.

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

---

## See also
- **Detailed module specs**: `02_DATA_AND_TRIGRAPH.md` → `05_INFERENCE_PROMPTING.md`
- **Integration with host repo (G-Retriever)**: `09_INT_HOST_REPO.md`
- **DANCE methodology deep-dive**: `10_DANCE_REFERENCE.md`
- **Federated infrastructure**: `11_INT_GFL.md`
- **Hyperparameters**: `08_APPENDIX_HYPERPARAMS.md`
