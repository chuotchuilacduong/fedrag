# SETUP — chạy repo trên máy mới (fresh machine runbook)

Quy trình đầy đủ từ máy trắng đến kết quả trên WandB. Mỗi bước idempotent —
chạy lại không phá gì, `--force` để rebuild.

---

## 0. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| OS | Linux (đã test WSL2) |
| GPU | NVIDIA, ≥8GB VRAM (Qwen-1.5B bf16 hoặc Qwen-7B 4-bit); 24GB (RTX 4090) cho Qwen-7B bf16 như paper |
| Đĩa | ~50GB (model + dataset + artifacts) |
| Python | 3.11 (conda env) |

## 1. Môi trường

```bash
git clone <repo-url> fedrag && cd fedrag

conda create -n fedrag python=3.11 -y
conda activate fedrag

# PyTorch + PyG trước (chọn bản CUDA khớp driver), rồi phần còn lại
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric torch_scatter
pip install -r requirements.txt
```

Model LLM tự tải từ HuggingFace ở lần chạy đầu (Qwen2.5-1.5B-Instruct ~3GB,
Qwen2.5-7B-Instruct ~15GB). Nếu mạng chậm/hay đứt, tải trước cho chắc:

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct     # hoặc Qwen2.5-7B-Instruct
```

## 2. WandB (tuỳ chọn nhưng nên có)

Tạo file `.env` ở repo root (đã nằm trong `.gitignore`):

```
WANDB_API_KEY=<key từ wandb.ai/authorize>
WANDB_PROJECT=fedcond-graphrag
```

Không có key → training vẫn chạy, metrics ghi ra `/tmp/fl_metrics.jsonl` + log.

## 3. Chuẩn bị dữ liệu — MỘT lệnh

```bash
python main.py preprocess --dataset hotpotqa --num-clients 3
```

Lệnh này chạy tuần tự 6 bước (tự skip bước đã có output):

| Bước | Script | Output |
|---|---|---|
| 1 | `setup_datasets` — tải benchmark (bản HippoRAG chuẩn, 1000 câu dev + corpus) | `dataset/linearrag/<ds>/{chunks,questions}.json` |
| 2 | `preprocess_data` — chia corpus thành N shard client (idx % N) | `processed/<ds>/client_<m>/chunks.json` |
| 3 | `build_client_pipeline` — Stage A (tri-graph) → Stage B (anchor condensation) | `trigraph.pt`, `condensed_graph.pt` |
| 4 | `build_fedcond_qa_dataset` — records + splits 80/10/10 + question embeddings | `dataset/fedcond_qa/{records.jsonl, split/, q_embs.pt}` |
| 5 | `preprocess_fedcond_qa` — PPR node maps per client (evidence retrieval lúc train) | `processed/<ds>/client_<m>/ppr_node_map.pt` |
| 6 | `build_passage_anchors` — chỉ khi `--with-passage-anchors` (cho `--top-r-passages`) | `passage_embs.pt`, `passage_node_map.pt` |

Ghi chú:
- **Nhiều dataset song song**: bước 4 mặc định ghi vào `dataset/fedcond_qa`
  (bị ghi đè nếu đổi dataset). Dùng `--qa-out-root dataset/fedcond_qa_musique`
  và truyền cùng path cho fl-train qua `--qa-data-root`.
- **Dataset `medical`** là corpus private: tự đặt `chunks.json`/`questions.json`
  vào `dataset/linearrag/medical/` rồi chạy lệnh trên (bước 1 tự skip).
- Bước 3 là bước nặng nhất (NER + encode toàn corpus) — lần đầu có thể mất
  hàng chục phút đến vài giờ tuỳ máy; các lần sau skip nhờ cache.
- Stage B.3.5 refinement (KL retrieval-preserving) chạy ở fl-train round 0
  và tự cache thành `condensed_graph_refined.pt` — không cần bước riêng.

## 4. Train FedRAG (paper Algorithm 1)

```bash
# GPU 8GB — Qwen 1.5B bf16
python main.py fl-train --dataset hotpotqa --num-clients 3 --num-rounds 5 \
  --local-epochs 3 --llm-model-name qwen2.5-1.5b-instruct \
  --local-batch-size 1 --eval-batch-size 4 --dual-graph-mode both --use-cuda \
  --wandb-run-name fedrag-hotpotqa

# GPU 8GB — Qwen 7B (4-bit bắt buộc)
python main.py fl-train --dataset hotpotqa --num-clients 3 --num-rounds 5 \
  --local-epochs 3 --llm-model-name qwen2.5-7b-instruct --llm-load-in-4bit \
  --llm-gradient-checkpointing --local-batch-size 1 --eval-batch-size 2 \
  --dual-graph-mode both --use-cuda --wandb-run-name fedrag-hotpotqa-7b

# GPU 24GB — setup paper (Qwen 7B bf16)
python main.py fl-train --dataset hotpotqa --num-clients 3 --num-rounds 5 \
  --local-epochs 3 --llm-model-name qwen2.5-7b-instruct \
  --local-batch-size 4 --dual-graph-mode both --use-cuda
```

Mode mặc định là `fedrag` (Algorithm 1: Phase 0 khởi tạo Θ_syn từ anchors;
mỗi round sau: prompt tuning soft-retrieval → K_mem bước memory adaptation →
server aggregate Δ theo Eq. 18–19). Thuật toán legacy: `--server-stage-c-mode
gradient_match|repr_align|both`. Toàn bộ knob: `docs/WORKFLOW.md` §7a.

Baseline so sánh (ví dụ FlexLoRA — hàng "Federated LLM" trong Table 1):

```bash
python main.py fl-train --dataset hotpotqa --num-clients 3 --num-rounds 5 \
  --local-epochs 3 --llm-frozen False --lora-agg-method flexlora \
  --llm-model-name qwen2.5-1.5b-instruct --dual-graph-mode none \
  --max-txt-len 0 --local-batch-size 1 --use-cuda
```

## 5. Kiểm tra

```bash
python -m pytest tests/ -q          # unit tests (Stage B/C/D/E)
tail -f fltrain_run.log             # log run đang chạy (nếu redirect vào đây)
cat /tmp/fl_metrics.jsonl           # metrics thô per-round
```

Kết quả per-round (loss, hit/EM/F1 val+test, L_cond, syn-mem QA loss) lên
WandB project `fedcond-graphrag`, trục x `comm_round` (round 0 = baseline
trước training).

## 6. Cấu trúc script dữ liệu (tham khảo)

Một script = một bước, orchestrated bởi `main.py preprocess`:

```
scripts/
├── setup_datasets.py           # bước 1 — nguồn benchmark DUY NHẤT (HippoRAG copies)
├── preprocess_data.py          # bước 2 — chia shard client
├── build_client_pipeline.py    # bước 3 — Stage A→B (KHÔNG còn Stage C offline)
├── build_fedcond_qa_dataset.py # bước 4 — QA records/splits/q_embs (--out-root)
├── preprocess_fedcond_qa.py    # bước 5 — PPR node maps (tên lịch sử; chỉ build PPR maps)
└── build_passage_anchors.py    # bước 6 — optional, cho --top-r-passages
```

Đã xoá: `download_hotpotqa.py`, `download_musique.py` (trùng chức năng với
`setup_datasets.py` nhưng tải nguồn khác → kết quả không so sánh được);
Stage C offline trong `build_client_pipeline.py` (Θ_syn được server khởi tạo
ở fl-train, không có artifact per-client).
