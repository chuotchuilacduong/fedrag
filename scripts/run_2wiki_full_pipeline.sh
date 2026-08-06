#!/usr/bin/env bash
# Full pipeline for 2WikiMultiHop: download → preprocess → fl-train (ablations)
# Mirrors the 3 most recent musique runs on qwen2.5-7b.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(conda run -n fedrag which python)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

log() { echo ""; echo "=== $(date '+%Y-%m-%d %H:%M:%S') $* ==="; echo ""; }

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 0 — Download full 2WikiMultiHop
# ---------------------------------------------------------------------------
log "Step 0 — Download full 2WikiMultiHop (train + dev)"
conda run -n fedrag python scripts/download_2wikimultihop.py --splits train dev \
    2>&1 | tee "$LOG_DIR/download_2wiki.log"

# ---------------------------------------------------------------------------
# Step 1 — Preprocess: Stage A (Tri-Graph) + Stage B (client condense)
# ---------------------------------------------------------------------------
log "Step 1 — Preprocess 2wikimultihop for 3 clients"
conda run -n fedrag python main.py preprocess \
    --dataset 2wikimultihop \
    --num-clients 3 \
    --entity-ratio 0.05 \
    2>&1 | tee "$LOG_DIR/preprocess_2wiki.log"

# ---------------------------------------------------------------------------
# Step 2 — Build fedcond_qa dataset for 2wikimultihop
# ---------------------------------------------------------------------------
log "Step 2 — Build fedcond_qa QA dataset"
conda run -n fedrag python scripts/build_fedcond_qa_dataset.py \
    --dataset 2wikimultihop \
    2>&1 | tee "$LOG_DIR/build_qa_2wiki.log"

# ---------------------------------------------------------------------------
# Common fl-train args (same as 3 most recent musique runs)
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --dataset 2wikimultihop
    --num-clients 3
    --num-rounds 5
    --seed 42
    --use-cuda
    --llm-model-name qwen2.5-7b
    --llm-load-in-4bit
    --llm-gradient-checkpointing
    --local-epochs 1
    --eval-every 1
    --max-eval-samples 200
    --wandb-group 2wiki-full
)

# ---------------------------------------------------------------------------
# Step 3a — Ablation: text_only (no graph)
# ---------------------------------------------------------------------------
log "Step 3a — fl-train: text_only ablation"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode text_only \
    --wandb-run-name "text-only-7b-2wiki-fulldata" \
    --wandb-tags ablation text_only 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_text_only_7b.log"

# ---------------------------------------------------------------------------
# Step 3b — Ablation: shared graph, no FedAvg
# ---------------------------------------------------------------------------
log "Step 3b — fl-train: shared + no-fedavg ablation"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode shared \
    --no-fedavg \
    --wandb-run-name "no-fedavg-7b-2wiki-fulldata" \
    --wandb-tags ablation no_fedavg 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_no_fedavg_7b.log"

log "All 2WikiMultiHop runs complete"
