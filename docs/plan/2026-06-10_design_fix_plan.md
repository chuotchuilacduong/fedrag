# Plan sửa thiết kế — 2026-06-10

Dựa trên review `docs/flow.md` + kiểm chứng code trực tiếp. Phạm vi đã chốt với
tác giả; mục nào chưa duyệt được đánh dấu *(đề xuất kèm)*.

## Trạng thái (cập nhật 2026-06-10 chiều)

- ✅ **WP1** — xong: GraphTextFusion identity-additive, không còn nn.Linear;
  cache `condensed_graph.pt` cũ đã rename `.pre_wp1.bak`, rebuild bằng
  `scripts/build_client_pipeline.py --dataset musique`.
- ✅ **WP2** — xong: `score_and_select` cosine top-k + softmax;
  `select_chunks` dùng `score_chunks` + `topk_softmax` có sẵn.
- ⏸️ **WP3** — HOÃN theo quyết định tác giả (xử lý sau).
- ✅ **WP4** — xong: `resolve_prompt_template()` trong graph_llm.py:
  Qwen Instruct (eos `<|im_end|>`) → ChatML; Qwen base (eos `<|endoftext|>`) →
  plain `"" / "\nAnswer:" / <|endoftext|>` — verify trực tiếp trên Qwen2.5-1.5B
  fp32: plain format trả đúng "Paris", ChatML trên base chỉ echo input;
  Llama giữ nguyên. Bỏ hack `split("</s>")`; thêm `exact_match`/`token_f1`
  vào evaluate.py; `_eval_split_acc` trả {hit, em, f1}, log wandb.
- ✅ **WP5** (phần WP1/2/4) — flow.md đã cập nhật.
- 🧪 Tests: 68 passed, 2 skipped. Sửa kèm 2 lỗi có sẵn ngoài scope:
  `dual_graph_llm._encode_one_graph` StopIteration với encoder không tham số;
  `client._load_ppr_node_map` crash khi args.dataset là list.
- ✅ **Full run 33k data (2026-06-11)**: 3 client × 5 round × full pool
  (~10.6k sample/client), Qwen2.5-1.5B, 1 epoch/round, seed 42.
  wandb: shared-fulldata-33k-postfix (956x4k3l). Kết quả theo round
  (val/test = hit, metric cũ):
  R1 22.0/17.5 · R2 24.5/15.0 · R3 24.5/17.5 · R4 27.0/19.5 (%).
  EM/F1 cuối: val 25.5/35.2, test 17.5/28.5. Loss 1.61→1.31, còn giảm.
  **Baseline cũ (pre-fix, cùng metric): test_acc 2.0% → giờ 19.5% (~10×).**
  Lỗi còn lại chủ yếu là sai multi-hop thật (nhầm thực thể gần đúng:
  "jose mourinho" vs "roberto di matteo", "mary pickford" vs "jack pickford")
  — không còn lỗi format. Trend chưa bão hòa → đáng chạy thêm round.
- ✅ Smoke run fl-train (2026-06-11): 3 client × 2 round × 50 sample/client,
  Qwen2.5-1.5B, seed 42. Kết quả: train loss ~3.4–3.6 (ổn định), eval
  val: hit 25% / EM 20% / F1 28.4; test: hit 5% / EM 0% / F1 7.1 (20 sample).
  Predictions sạch, đúng dạng câu trả lời, không còn `</s>`/lặp rác.
  So sánh: run cũ 5 round × 1000 sample chỉ đạt test_acc 2.0%.
  Sửa kèm trong lúc smoke: bos_text rỗng (base-model template) làm tokenizer
  trả empty FLOAT tensor → crash nn.Embedding; đã guard trong graph_llm.py.

## Kết quả kiểm chứng các khẳng định

| # | Khẳng định | Kết quả kiểm chứng |
|---|---|---|
| 1 | "Projection đã bỏ hoàn toàn linear layer, thay bằng fixed parameter" | **Chưa đúng với code hiện tại.** `graph_text_fusion.py:17-18` vẫn còn `nn.Linear(graph_dim, out_dim)` × 2 (random, không bao giờ train). Chỉ có **gate** là fixed buffer 0.5 (`:22`). → WP1 |
| 2 | "desc lấy sẵn để chạy nhanh; chạy thực tế client sẽ retrieve riêng từng query" | Ghi nhận là design intent (oracle mode để dev nhanh). Code hiện tại desc = gold evidence; chế độ per-query retrieval chưa có code. → ghi vào docs, ngoài scope plan này |
| 3 | "Synthetic global graph là đủ — sẽ ablation" | Câu hỏi mở, chấp nhận. Các mode ablation (`no_synthetic`, `evidence_only`, …) đã có sẵn trong `dual_graph_mode` |
| 4 | "Score dùng cơ chế tính điểm riêng từng node, rank theo công thức" | **Đúng cho anchor selection** (`anchor_node_selector.py`: `log1p(deg_S)+log1p(deg_P)+λ_idf·IDF+λ_pr·PageRank` + MMR; neighbor score `mention + 0.1·centrality + cosine`). **Nhưng tầng text-condensation vẫn first-k tùy tiện**: `score_and_select` (`neighbor_gating.py:91`) lấy k phần tử đầu theo thứ tự node-ID, `g_v` không dùng; `select_chunks` (`chunk_selection.py:89`) lấy 8 chunk đầu theo thứ tự flatten. → WP2 |
| 5 | Phê bình "repr_align ép khớp 1 vector pooled" (review trước) | **Phê bình SAI** — `representation_alignment_loss` (`repr_align.py:85`) đã là node-level attention-reconstruction: `A_m = softmax(H_m·H_synᵀ/√H)`, `L = Σ‖H_m − A_m·H_syn‖²_F / Σ N_j`. flow.md đã được đính chính. Điểm yếu còn lại: (a) một chiều — anchor phải tái tạo được từ syn, nhưng syn node không bị ràng về manifold anchor; (b) bỏ qua node type (entity anchor có thể tái tạo từ passage syn); (c) round 1 chạy với encoder random; (d) FedAvg trễ 1 round. → WP3 |
| 6 | EOS/template | **Xác nhận lỗi.** `graph_llm.py:14-16` hardcode Llama-2: `BOS='<s>[INST]'`, `EOS_USER='[/INST]'`, `EOS='</s>'`. Với Qwen tokenizer chúng thành text thường → model học sinh chuỗi "</s>", phải vá bằng `pred.split("</s>")[0]` (`dual_graph_llm.py:263`). → WP4 |

---

## WP1 — Stage B: bỏ random Linear projection trong GraphTextFusion

**Mục tiêu:** feature anchor graph nằm trong không gian MiniLM gốc, để cosine
retrieval ở Stage D (`mean(G_ev.x)` vs `synthetic_x`) có nghĩa.

**Thay đổi:**
- `fedcond_grag/client/stage_b_condense/graph_text_fusion.py`
  - Xóa `self.graph_proj`, `self.text_proj`.
  - `forward`: `fused = self.norm(graph_embeddings + gate * text_embeddings)`.
  - Yêu cầu `graph_dim == text_dim` (đều 384 MiniLM) — raise nếu lệch; xóa `out_dim`.
  - Giữ `LayerNorm` (affine γ=1, β=0 không bao giờ train → chỉ là chuẩn hóa, deterministic).
  - Cập nhật helper `fuse()` tương ứng.
- `client_condensor.py`: bỏ tham số `out_dim` truyền vào `GraphTextFusion` (`:83`),
  bỏ `out_dim` khỏi `ClientCondensationConfig` nếu không còn nơi dùng.

**Hệ quả phải xử lý:**
- `processed/{dataset}/client_*/condensed_graph.pt` build bằng fusion cũ → **xóa cache
  và build lại** (`fedcond_grag preprocess --force` hoặc xóa file để client tự build round 0).
- Synthetic graph khởi tạo từ anchor feature → tự đúng lại sau khi cache mới.

**Test/nghiệm thu:**
- `pytest tests/test_stage_b_graph_condensation.py` (sửa test nếu đang assert có proj).
- Sanity: `cosine(condensed.x, trigraph.x[core_ids])` phải cao rõ rệt (cùng không gian);
  trước fix gần như trực giao.

## WP2 — Stage B: chấm điểm cosine thay first-k trong text condensation

**Mục tiêu:** chọn láng giềng/chunk theo độ liên quan với node lõi thay vì theo
thứ tự node-ID / thứ tự flatten.

**Thay đổi:**
- `neighbor_gating.py: score_and_select` — dùng `g_v` (đang bị bỏ qua):
  `scores = neighbor_text_embs @ g_v / √d` → top-k theo score, weight đều 1/k
  (hoặc `topk_softmax(scores, k)` — chọn một, ghi chú lại). Giữ chữ ký hàm.
- `chunk_selection.py: select_chunks` — thay khối uniform-first-k (`:88-91`) bằng
  `weights = topk_softmax(score_chunks(g_v, chunks_tensor), budget)`.
  Hai hàm `score_chunks`/`topk_softmax` **đã viết sẵn** trong file, hiện không được gọi.

**Hệ quả:** cùng nhóm cache với WP1 — build lại `condensed_graph.pt` một lần cho cả hai.

**Test/nghiệm thu:**
- Unit test mới: node lõi có 2 láng giềng (1 liên quan, 1 nhiễu, ID nhiễu nhỏ hơn) —
  trước fix chọn nhiễu, sau fix chọn liên quan.
- `pytest tests/test_stage_b_graph_condensation.py`.

## WP3 — Stage C: bổ sung per-type MMD alignment (đã duyệt)

**Mục tiêu:** bổ sung mục tiêu đối xứng, có nhận biết node-type, bên cạnh
attention-reconstruction hiện tại (giữ làm default vì khớp paper §3.2).

**Thay đổi:**
- `repr_align.py` thêm:
  - `rbf_mmd(h_a, h_b, bandwidth=None)` — RBF kernel, bandwidth theo median heuristic,
    unbiased estimator; trả scalar.
  - `per_type_mmd(h_syn, syn_types, anchor_h_list, anchor_type_list)` —
    MMD riêng từng type t ∈ {E,S,P}, trọng số theo tỉ lệ node type trong anchors;
    type vắng mặt → bỏ qua.
- `server.py`:
  - `precompute_anchor_reprs` trả thêm `node_type` từng graph (hoặc hàm mới
    `precompute_anchor_reprs_typed`).
  - `server_repr_align_step` / `server_combined_step`: tính loss theo flag mới.
  - Flag: `--repr-align-objective {recon, mmd, recon+mmd}` (main.py + stage_c config),
    default `recon` (giữ nguyên hành vi hiện tại); `--mmd-weight` (default 1.0).
- *(đề xuất kèm — chưa duyệt, sửa nhỏ, nên làm cùng lúc):*
  - Dời `_fedavg_model_weights()` lên **trước** vòng optimize trong `execute()`
    (`server.py:137` → trước `:113`) — Stage C dùng weights round hiện tại thay vì trễ 1 round.
  - Round chưa có FedAvg weights (round 1): skip repr_align/mmd, chỉ chạy
    gradient_match hoặc bỏ qua optimize — tránh align theo encoder random.

**Test/nghiệm thu:**
- Unit: MMD(X, X) ≈ 0; MMD(X, X+shift) > MMD(X, X); per-type bỏ qua type rỗng.
- Ablation run (sau khi WP1/2 xong, cache mới): 3 client × 5 round × 1000 sample,
  mode `shared`, so `recon` vs `mmd` vs `recon+mmd` trên val_acc + loss_match.

## WP4 — Stage D: prompt/EOS native theo tokenizer (đã duyệt)

**Mục tiêu:** bỏ template Llama-2 hardcode; model không học sinh "</s>" như text.

**Thay đổi:**
- `fedcond_grag/model/graph_llm.py`:
  - Thay hằng module-level `BOS/EOS_USER/EOS` bằng template suy ra từ tokenizer
    trong `__init__` (self.bos_text / self.eos_user_text / self.eos_text):
    - Qwen (`<|im_start|>` trong vocab / chat_template có "im_start"):
      `bos = "<|im_start|>user\n"`, `eos_user = "<|im_end|>\n<|im_start|>assistant\n"`,
      `eos = tokenizer.eos_token` (`<|im_end|>` hoặc `<|endoftext|>` tùy bản — lấy từ tokenizer, không hardcode).
    - Llama: giữ `<s>[INST]` / `[/INST]` / `</s>` như cũ.
    - Fallback: bos_token/eos_token của tokenizer, eos_user = `"\nAnswer: "`.
  - Mọi chỗ tokenize hằng cũ (`:152, :193-194, :275`) đổi sang attr instance.
- `fedcond_grag/model/dual_graph_llm.py`:
  - `forward`/`inference` (`:124-125, :204`) dùng attr template từ parent.
  - Xóa hack `pred = [p.split("</s>")[0] ...]` (`:263`) — generate dừng đúng
    `eos_token_id`, `skip_special_tokens=True` lo phần còn lại.
- Eval — `fedcond_grag/utils/evaluate.py` thêm:
  - `exact_match(pred, label)` = `normalize(pred) == normalize(label)`.
  - `token_f1(pred, label)` — F1 trên token sau normalize (chuẩn SQuAD/MuSiQue).
- `trainer.py:_eval_split_acc`: tính cả 3 metric (hit substring cũ, EM, F1);
  return dict, log `round/val_{hit,em,f1}`, `round/test_{hit,em,f1}` lên wandb;
  cập nhật bảng summary.

**Lưu ý:** label format đổi → run/checkpoint cũ không so sánh trực tiếp được với
run mới. Chấp nhận (không có checkpoint cần giữ).

**Test/nghiệm thu:**
- `pytest tests/test_stage_d_dual_prompting.py` (cập nhật assert template nếu có).
- Smoke: 2 round, `--max-train-per-client 50`, Qwen-1.5B — kiểm tra sample
  predictions không còn "</s>"/garbage đuôi; loss giảm bình thường.

## WP5 — Docs

- `docs/flow.md`: cập nhật theo WP1–4 sau khi merge (fusion formula, selection,
  flag mới, metric mới). Phần repr_align đã đính chính hôm nay (per-node recon).
- Ghi rõ trong flow.md: chế độ desc hiện tại là **oracle** (gold evidence, để dev
  nhanh — theo xác nhận tác giả); chế độ per-query client retrieval là runtime
  mode dự kiến, chưa có code.

## Thứ tự thực hiện & phụ thuộc

```
WP1 + WP2  (cùng đụng Stage B, build lại cache MỘT lần)   ~ nửa ngày
   ↓ cache mới
WP3        (độc lập với WP4)                              ~ 1 ngày kể cả ablation
WP4        (độc lập, có thể làm song song WP3)            ~ nửa ngày + smoke run
   ↓
Smoke cuối: full pipeline 3 client × 5 round × 1000 sample, so metric với
run baseline 2026-06-10 (test_acc 2.0%) — kỳ vọng EM/F1 tăng rõ sau WP4.
   ↓
WP5 docs
```

## Rủi ro

- WP1 đổi phân bố feature anchor → synthetic graph + Stage C loss scale đổi theo;
  λ_div/λ_deg có thể cần tune lại (theo dõi loss_match round đầu).
- WP4: nếu Qwen chat template thiếu trong tokenizer bản local thì dùng fallback —
  kiểm tra `tokenizer.chat_template is not None` trước.
- Build lại cache Stage B trên musique mất thời gian (PPR map không cần build lại —
  không phụ thuộc fusion).
