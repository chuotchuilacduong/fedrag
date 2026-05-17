---
file: 03_CLIENT_CONDENSATION.md
title: Client-side Graph Condensation (Stage B)
load_priority: task-load
prerequisites: [01_OVERVIEW.md, 02_DATA_AND_TRIGRAPH.md]
related: [10_DANCE_REFERENCE.md, 04_SERVER_CONDENSATION.md, 11_INT_GFL.md]
covers_sections: "Part III §11 (S-E-P Motif Selection); §12 (DANCE Text Condensation); §13 (Fusion); §14 (Topology); §15 (Output)"
project: FedCondGraphRAG
---

# Client-side Graph Condensation (Stage B)

> **How to use this file.** Đây là spec của stage B — biến local Tri-Graph thành condensed graph C_m. **PHẢI load `10_DANCE_REFERENCE.md` cùng** khi implement §12, §14 vì DANCE không có code chính thức và file này chỉ là spec ngắn; pseudo-code và 10 subtle points nằm ở 10. File này nói **what**, file 10 nói **how to avoid mistakes**.

---

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

---

## See also
- **DANCE pseudo-code đầy đủ + 10 subtle points** (BẮT BUỘC đọc khi code §12, §14): `10_DANCE_REFERENCE.md` §37-§38
- **DANCE adaptations cho QA setting (no labels)**: `10_DANCE_REFERENCE.md` §39
- **Pre-flight checklist + unit tests**: `10_DANCE_REFERENCE.md` §40-§41
- **Hyperparams cho stage B (B_0, B_1, B_2, B_tok, α, β, ...)**: `08_APPENDIX_HYPERPARAMS.md`
- **Module tiếp theo (server uses C_m)**: `04_SERVER_CONDENSATION.md`
- **Tích hợp như "local_train()" trong gfl FGLTrainer**: `11_INT_GFL.md` §48
- **Debug checklist riêng cho stage B**: `07_DEBUG_RISKS_ORDER.md` §27.3-§27.5
