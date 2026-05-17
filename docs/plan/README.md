---
file: README.md
title: "FedCondGraphRAG — Implementation Plan (Index)"
load_priority: always-load
project: FedCondGraphRAG
---

# FedCondGraphRAG — Implementation Plan

> **Federated Graph Condensation for Retrieval-Augmented Generation over Textual Graphs**
>
> Một framework kết hợp:
> - **LinearRAG** (ICLR'26) — relation-free Tri-Graph + two-stage retrieval
> - **DANCE** (ICML 2025-style) — federated graph condensation cho TAG
> - **G-Retriever** (NeurIPS 2024) — graph soft-prompt cho LLM
> - **OpenFGL / gfl** (NeurIPS 2024) — federated learning infrastructure

---

## 0. Cách dùng repo plan này

Plan ban đầu là 1 file Markdown 2500+ dòng — quá dài để agent giữ context. Đã được chia thành **11 file nội dung + 1 README + 1 prompt**. Mỗi file:

- Có **YAML frontmatter** ở đầu: `load_priority`, `prerequisites`, `related`, `covers_sections`.
- Có **section "See also"** ở cuối với cross-references đến file khác.
- Có thể được load **độc lập** mà vẫn hiểu được nhờ prerequisites + cross-refs.

→ **Agent chỉ load các file cần thiết cho task hiện tại**, thay vì toàn bộ plan.

---

## 1. File Map

| # | File | Load Priority | Lines | Content |
|---|---|---|---|---|
| 0 | `README.md` | always-load | ~250 | (file này) — index + file selector |
| 1 | `01_OVERVIEW.md` | always-load | 213 | Research narrative (motivation, RQs, contributions); scope MVV; code structure |
| 2 | `02_DATA_AND_TRIGRAPH.md` | task-load | 127 | HotpotQA loading, federated partition, Tri-Graph spec |
| 3 | `03_CLIENT_CONDENSATION.md` | task-load | 258 | Stage B: S-E-P motif + DANCE text cond + topology recon + output |
| 4 | `04_SERVER_CONDENSATION.md` | task-load | 99 | Stage C: anchor gradient + PGE + gradient matching |
| 5 | `05_INFERENCE_PROMPTING.md` | task-load | 96 | Stage D: query-time retrieval + dual graph prompting |
| 6 | `06_TRAINING_EVAL.md` | task-load | 124 | Stages A-E, milestones, baselines, ablations, metrics |
| 7 | `07_DEBUG_RISKS_ORDER.md` | task-load | 102 | Per-module debug checklist, base risks (R1-R7), implementation order |
| 8 | `08_APPENDIX_HYPERPARAMS.md` | reference | 86 | Default hyperparameters (with source) + codebase relationship |
| 9 | `09_INT_HOST_REPO.md` | task-load | 499 | Integration tier-1+2: G-Retriever fork + LinearRAG vendor ("copy and fix") |
| 10 | `10_DANCE_REFERENCE.md` | **task-load (critical)** | 589 | DANCE pseudo-code + 10 subtle points + adaptations (no official code!) |
| 11 | `11_INT_GFL.md` | task-load | 575 | Integration tier-3: OpenFGL/gfl infrastructure + custom `fedcond_qa` algorithm |
| 12 | `AGENT_PROMPT.md` | always-load | ~120 | Prompt template để invoke agent |

**Tổng**: ~3160 dòng, chia trung bình ~240 dòng/file. File lớn nhất (10, 11) đều dưới 600 dòng — vừa với context của agent code.

---

## 2. Load Priority Levels

- **`always-load`** (3 files): `README.md`, `AGENT_PROMPT.md`, `01_OVERVIEW.md`. Mọi session đều load.
- **`task-load`** (8 files): Load **chỉ khi task hiện tại đụng đến module đó**.
- **`reference`** (1 file): `08_APPENDIX_HYPERPARAMS.md`. Lookup-only, không cần đọc tuần tự.

---

## 3. File Selector Logic (cho agent)

Khi nhận task mới, agent chọn file để load theo bảng dưới. **Cột "Required"** = bắt buộc; **"Suggested"** = tốt nếu fit trong context.

| Task pattern (keyword trong yêu cầu user) | Required | Suggested |
|---|---|---|
| "setup repo", "clone", "vendor", "import path", "git clone", "initial repo structure" | `01`, `09` | `11`, `07` |
| "load HotpotQA", "process dataset", "federated partition", "split clients" | `01`, `02` | `09`, `11` |
| "build Tri-Graph", "entity extraction", "spaCy", "trigraph_builder" | `01`, `02` | `09` |
| "motif selection", "S-E-P", "entity anchor", "PageRank scoring" | `01`, `03` | — |
| "client condensation", "DANCE", "neighbor gating", "text condensation", "self-expression", "topology reconstruction" | `01`, `03`, **`10`** | `11` |
| "entmax", "STE", "straight-through", "difficulty score", "alignment loss" | `01`, **`10`** | `03` |
| "PGE", "server condensation", "anchor gradient", "gradient matching", "FedGM extend" | `01`, `04`, `11` | `10` |
| "DualGraphLLM", "dual encoder", "z_e", "z_c", "graph prompt", "soft prompt" | `01`, `05`, `09` | — |
| "inference pipeline", "LinearRAG retrieve", "global graph retrieve", "query-time" | `01`, `05` | `09`, `04` |
| "train", "baseline run", "ablation", "evaluation", "metrics", "EM/F1" | `01`, `06` | `11` (baselines) |
| "FedAvg", "FedSage+", "FedGTA", "FedC4", "FedGVD", "fl_algorithm" | `01`, `11`, `06` | `04` |
| "debug", "stuck", "error", "bug" | `01`, `07` | (relevant module) |
| "hyperparameter", "default value", "B_0", "B_1", "B_tok", "α", "β" | `01`, `08` | `10` (if DANCE-related) |
| "research narrative", "motivation", "contributions", "RQ" | `01` | — |

**Rule of thumb**: nếu task đụng đến module X, load `01 + 0X + (related files in 0X frontmatter)`.

---

## 4. Dependency Graph (file references)

```text
                    01_OVERVIEW (always)
                    /     |     \     \
              02_DATA  03_CLIENT  04_SERVER  05_INFERENCE
                |     /     |       |          |
                |   10_DANCE |     11_GFL      |
                |       |    \     /           |
                +-------+----09_INT_HOST_REPO--+
                              |
                       08_APPENDIX (lookup-only)
                       07_DEBUG_RISKS (debug-only)
                       06_TRAINING_EVAL (planning/eval)
```

Edges = "file X references file Y in its body". Đọc cross-reference footer của mỗi file để biết explicit links.

---

## 5. 3-Tier Stack Summary

```text
┌──────────────────────────────────────────────────────────┐
│   FedCondGraphRAG (custom logic ~1K-1.5K LoC)            │
│   - Tri-Graph wrap, S-E-P motif, DANCE adapt,             │
│   - Dual graph prompting, fedcond_qa custom algorithm     │
└──────────────────────────────────────────────────────────┘
       │                       │                   │
       ▼                       ▼                   ▼
┌─────────────┐         ┌──────────────┐    ┌──────────────┐
│ G-Retriever │         │ LinearRAG    │    │ OpenFGL/gfl  │
│ NeurIPS'24  │         │ ICLR'26      │    │ NeurIPS'24   │
│ ~5K reused  │         │ ~3K reused   │    │ ~10K reused  │
│ (host repo) │         │ (vendored)   │    │ (vendored)   │
└─────────────┘         └──────────────┘    └──────────────┘

Tổng reuse: ~18K LoC ; viết mới: ~1-1.5K LoC ; ratio reuse = 92%
```

Chi tiết integration của mỗi tier: `09_INT_HOST_REPO.md` (tier 1+2), `11_INT_GFL.md` (tier 3).

---

## 6. Đọc Plan Lần Đầu — Suggested Order

Người mới đến project nên đọc theo thứ tự:

1. `README.md` (file này, 10 phút)
2. `01_OVERVIEW.md` (research narrative + scope, 30 phút)
3. `09_INT_HOST_REPO.md` §28 only (folder layout overview, 10 phút)
4. `11_INT_GFL.md` §55 only (3-tier diagram, 5 phút)
5. **Stop**. Sau bước 4, đã đủ context tổng quan. Các file còn lại load on-demand theo task.

Tổng thời gian onboarding: **~1 giờ**.

---

## 7. Khi Plan Thay Đổi

Mọi cập nhật phải duy trì invariant:
- Mỗi file có YAML frontmatter hợp lệ.
- Cross-references trong "See also" còn valid.
- Bảng file map ở §1 của README phản ánh đúng line count.

Khi thêm Part mới (ví dụ Part XII), tạo file mới `12_*.md` và:
1. Add row vào §1 (file map).
2. Update §3 (selector logic) nếu cần.
3. Update §4 (dependency graph).
4. Add references vào "See also" của file gốc liên quan.

---

## 8. Reference

Plan này tổng hợp ý tưởng từ:

- **G-Retriever** — He et al., NeurIPS 2024. arXiv:2402.07630. [Code](https://github.com/XiaoxinHe/G-Retriever).
- **LinearRAG** — Zhuang et al., ICLR'26. arXiv:2510.10114. [Code](https://github.com/DEEP-PolyU/LinearRAG).
- **DANCE** — Chen et al., 2026. arXiv:2601.16519. *(no official code)*
- **OpenFGL / gfl** — Li et al., NeurIPS 2024. arXiv:2408.16288. [Code](https://github.com/xkLi-Allen/OpenFGL) / [fork](https://github.com/chuotchuilacduong/gfl).
