---
file: 08_APPENDIX_HYPERPARAMS.md
title: Hyperparameters & Codebase Relationship (Appendix)
load_priority: reference
prerequisites: []
related: [03_CLIENT_CONDENSATION.md, 04_SERVER_CONDENSATION.md, 10_DANCE_REFERENCE.md, 11_INT_GFL.md]
covers_sections: "Appendix A (all default hyperparams + DANCE Table 5 + ours); Appendix B (high-level codebase relationship matrix)"
project: FedCondGraphRAG
---

# Hyperparameters & Codebase Relationship (Appendix)

> **How to use this file.** Reference file — lookup khi cần default value cho hyperparam. Mỗi entry note rõ source (DANCE Table 5, G-Retriever default, hoặc "ours"). Bảng Appendix B (codebase mapping) đã được mở rộng trong `11_INT_GFL.md` §52 — đọc cả hai.

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

---

## See also
- **Updated codebase matrix (4-row: G-Retriever, LinearRAG, DANCE, gfl)**: `11_INT_GFL.md` §52
- **Per-module hyperparam context**: `03_CLIENT_CONDENSATION.md` (DANCE budgets), `04_SERVER_CONDENSATION.md` (PGE/server)
- **DANCE original Table 5 source**: `10_DANCE_REFERENCE.md` §42 (References)
