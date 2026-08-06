#!/usr/bin/env bash
# Resume 2WikiMultiHop pipeline from PPR step onward.
# (download + preprocess + build_qa already done)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

log() { echo ""; echo "=== $(date '+%Y-%m-%d %H:%M:%S') $* ==="; echo ""; }

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 2b — PPR node map (per-client, required before fl-train)
# ---------------------------------------------------------------------------
log "Step 2b — Build PPR node maps for 2wikimultihop (3 clients)"
conda run -n fedrag python scripts/preprocess_fedcond_qa.py \
    --dataset 2wikimultihop \
    2>&1 | tee "$LOG_DIR/preprocess_fedcond_qa_2wiki.log"

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
# Step 3a — Ablation: text_only
# ---------------------------------------------------------------------------
log "Step 3a — fl-train: text_only ablation"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode text_only \
    --wandb-run-name "text-only-7b-2wiki-fulldata" \
    --wandb-tags ablation text_only 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_text_only_7b.log"

# ---------------------------------------------------------------------------
# Step 3b — Ablation: shared + no-fedavg
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
