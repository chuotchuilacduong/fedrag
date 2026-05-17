---
file: 07_DEBUG_RISKS_ORDER.md
title: Debug Checklist, Risks & Implementation Order
load_priority: task-load
prerequisites: [01_OVERVIEW.md]
related: [10_DANCE_REFERENCE.md, 11_INT_GFL.md]
covers_sections: "Part VI §27 (Debug checklist per module); Part VII (R1-R7 risks); Part VIII (Final 17-step order)"
project: FedCondGraphRAG
---

# Debug Checklist, Risks & Implementation Order

> **How to use this file.** Reference file — load khi gặp bug, hoặc khi cần biết thứ tự implement chuẩn. Risks ở đây là **base risks** (R1-R7); risks bổ sung specific cho gfl integration (R8-R13) ở `11_INT_GFL.md` §53.

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

---

## See also
- **DANCE-specific debug**: `10_DANCE_REFERENCE.md` §40 (pre-flight checklist) + §41 (unit tests)
- **gfl integration risks (R8-R13)**: `11_INT_GFL.md` §53
- **Migration checkpoints sau mỗi step copy-and-fix**: `09_INT_HOST_REPO.md` §35
- **Final implementation order đã được refine bởi M4a/M6b**: `11_INT_GFL.md` §54
