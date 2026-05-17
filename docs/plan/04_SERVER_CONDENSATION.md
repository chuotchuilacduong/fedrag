---
file: 04_SERVER_CONDENSATION.md
title: Server-side Global Graph & PGE (Stage C)
load_priority: task-load
prerequisites: [01_OVERVIEW.md, 03_CLIENT_CONDENSATION.md]
related: [11_INT_GFL.md, 05_INFERENCE_PROMPTING.md]
covers_sections: "Part III §16 (Anchor Gradient); §17 (Global Graph + PGE); §18 (Gradient Matching)"
project: FedCondGraphRAG
---

# Server-side Global Graph & PGE (Stage C)

> **How to use this file.** Spec của stage C — học global synthetic graph từ {C_m}. **PHẢI load `11_INT_GFL.md` cùng** khi implement vì stage C được build trên nền **FedGM** của OpenFGL/gfl (subclass `FedGMServer`). File 11 cung cấp infrastructure (FGLTrainer, round loop, partitioning) và mapping FedGM→Stage C.

---

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

---

## See also
- **gfl/FedGM infrastructure & FedCondQAServer skeleton** (BẮT BUỘC khi implement): `11_INT_GFL.md` §47-§48
- **Server training entry point + free baselines**: `11_INT_GFL.md` §49-§50
- **Surrogate task definitions (node-type + link-pred + KL align)**: `10_DANCE_REFERENCE.md` §39.3
- **Hyperparams cho server (K_g, server_condense_iters, ...)**: `08_APPENDIX_HYPERPARAMS.md`
- **Output đi đâu (query-time retrieval)**: `05_INFERENCE_PROMPTING.md`
- **Risks specific to gfl integration**: `11_INT_GFL.md` §53
