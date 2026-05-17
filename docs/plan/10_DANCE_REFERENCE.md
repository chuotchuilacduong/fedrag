---
file: 10_DANCE_REFERENCE.md
title: "DANCE Reference Implementation Guide (CRITICAL: no official code)"
load_priority: task-load
prerequisites: [01_OVERVIEW.md, 03_CLIENT_CONDENSATION.md]
related: [03_CLIENT_CONDENSATION.md, 11_INT_GFL.md, 07_DEBUG_RISKS_ORDER.md]
covers_sections: "Part X §36-42: notation bridge, 4 algorithms full pseudo-code, 10 subtle points, adaptations for QA, pre-flight checklist, unit tests skeleton"
project: FedCondGraphRAG
---

# DANCE Reference Implementation Guide (CRITICAL: no official code)

> **How to use this file.** **PHẢI load mỗi khi code trong `fedcond_grag/graph_condensation/`** vì DANCE không có code công khai và mọi chi tiết phải đến từ paper. File này là PRIMARY reference. File `03_CLIENT_CONDENSATION.md` chỉ là spec ngắn; pseudo-code Python-style, công thức chính xác từng Eq., và 10 subtle points dễ sai đều nằm ở đây.

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

---

## See also
- **High-level spec & integration point**: `03_CLIENT_CONDENSATION.md`
- **Khi inject DANCE thành "local_train()" trong gfl trainer**: `11_INT_GFL.md` §48
- **Hyperparam values + ambiguity giữa α/β và λ_1/λ_3**: `08_APPENDIX_HYPERPARAMS.md`
- **Surrogate task formula (node-type CE + link-pred + KL alignment)** đã được chi tiết ở §39.3
- **Pre-flight checklist** (§40) và **unit tests skeleton** (§41) trước khi chạy stage B
