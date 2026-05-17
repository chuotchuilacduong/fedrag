---
file: 02_DATA_AND_TRIGRAPH.md
title: Data Preparation & Tri-Graph Construction
load_priority: task-load
prerequisites: [01_OVERVIEW.md]
related: [03_CLIENT_CONDENSATION.md, 09_INT_HOST_REPO.md, 11_INT_GFL.md]
covers_sections: "Part III §9 (Data Preparation, HotpotQA, Federated Partition); §10 (LinearRAG Tri-Graph)"
project: FedCondGraphRAG
---

# Data Preparation & Tri-Graph Construction

> **How to use this file.** Load khi làm việc với HotpotQA data loading, federated partitioning, hoặc xây Tri-Graph từ corpus. Spec này song hành với `09_INT_HOST_REPO.md` §29 Step 5 (`trigraph_builder.build_trigraph_for_client()`) — file này nói **what**, file 09 nói **how to wrap LinearRAG**.

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

---

## See also
- **Module dùng Tri-Graph làm input**: `03_CLIENT_CONDENSATION.md` (motif selection bắt đầu từ Tri-Graph)
- **Code-level wrap của LinearRAG để build Tri-Graph**: `09_INT_HOST_REPO.md` §29 Step 5
- **API verification của LinearRAG (đọc rag.entity_list, contain_matrix...)**: `09_INT_HOST_REPO.md` §34
- **Federated partition via Louvain (replace hash-based)**: `11_INT_GFL.md` §50.4
- **Default hyperparams (passage cap, num_clients, ...)**: `08_APPENDIX_HYPERPARAMS.md`
