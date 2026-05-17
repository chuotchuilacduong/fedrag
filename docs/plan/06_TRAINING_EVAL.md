---
file: 06_TRAINING_EVAL.md
title: Training Schedule, Stages & Evaluation
load_priority: task-load
prerequisites: [01_OVERVIEW.md]
related: [03_CLIENT_CONDENSATION.md, 04_SERVER_CONDENSATION.md, 05_INFERENCE_PROMPTING.md, 11_INT_GFL.md]
covers_sections: "Part IV §21-22 (Stages A-E + milestone timeline); Part V §23-26 (baselines, ablations, metrics, reporting)"
project: FedCondGraphRAG
---

# Training Schedule, Stages & Evaluation

> **How to use this file.** Load khi planning lịch trình project, hoặc khi setup ablation/baseline runs. Lưu ý: **Milestone timeline ở file này đã được cập nhật trong `11_INT_GFL.md` §54** với M4a (gfl sanity baseline) và M6b (free baselines) — đọc cả hai.

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

---

## See also
- **Cập nhật timeline với gfl milestone M4a + M6b**: `11_INT_GFL.md` §54
- **Free baselines từ gfl (FedAvg, FedSage+, FedGTA, FedGM, FedC4, FedGVD)**: `11_INT_GFL.md` §49
- **DANCE baseline (phải tự implement)**: `10_DANCE_REFERENCE.md` toàn bộ
- **Stage A/B/D entry scripts location**: `09_INT_HOST_REPO.md` §28 (folder layout)
- **Hyperparam search ranges (mix ratio, α, β)**: `08_APPENDIX_HYPERPARAMS.md`
