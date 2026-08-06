#!/usr/bin/env bash
# Wait for all 3 ppr_node_map.pt files to appear, then run fl-train ablations.
# Launch this after the 3 parallel PPR jobs (ppr_2wiki_client{0,1,2}.log) are started.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

log() { echo ""; echo "=== $(date '+%Y-%m-%d %H:%M:%S') $* ==="; echo ""; }

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Wait for all 3 PPR node maps
# ---------------------------------------------------------------------------
log "Waiting for PPR node maps (3 clients)..."
for c in 0 1 2; do
    TARGET="$PROJECT_ROOT/processed/2wikimultihop/client_${c}/ppr_node_map.pt"
    PROGRESS="$PROJECT_ROOT/processed/2wikimultihop/client_${c}/ppr_progress.json"
    echo "  Waiting for client_${c} to complete..."
    # Wait for the file to exist AND ppr_progress.json to be absent (= run complete)
    until [[ -f "$TARGET" && ! -f "$PROGRESS" ]]; do
        sleep 60
    done
    SIZE=$(python3 -c "import torch; m=torch.load('$TARGET', weights_only=True); print(f'shape={tuple(m.shape)}')" 2>/dev/null || echo "check failed")
    echo "  client_${c} done: $SIZE"
done
log "All 3 PPR node maps ready"

# Verify shapes are correct (should be [180030, 5])
python3 -c "
import torch, sys
for c in range(3):
    m = torch.load(f'processed/2wikimultihop/client_{c}/ppr_node_map.pt', weights_only=True)
    if m.shape[0] < 100000:
        print(f'ERROR: client_{c} ppr_node_map shape={tuple(m.shape)} — too small, expected ~180030 rows')
        sys.exit(1)
    print(f'  client_{c}: {tuple(m.shape)} OK')
"

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
# Step 3a — Main: FedCondGraphRAG (shared encoder + FedAvg)
# ---------------------------------------------------------------------------
log "fl-train: fedcond main run (shared + FedAvg)"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode shared \
    --wandb-run-name "fedcond-7b-2wiki-fulldata" \
    --wandb-tags fedcond shared 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_fedcond_7b.log"

# ---------------------------------------------------------------------------
# Step 3b — Ablation: text_only
# ---------------------------------------------------------------------------
log "fl-train: text_only ablation"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode text_only \
    --wandb-run-name "text-only-7b-2wiki-fulldata" \
    --wandb-tags ablation text_only 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_text_only_7b.log"

# ---------------------------------------------------------------------------
# Step 3c — Ablation: shared + no-fedavg
# ---------------------------------------------------------------------------
log "fl-train: shared + no-fedavg ablation"
conda run -n fedrag python main.py fl-train \
    "${COMMON_ARGS[@]}" \
    --dual-graph-mode shared \
    --no-fedavg \
    --wandb-run-name "no-fedavg-7b-2wiki-fulldata" \
    --wandb-tags ablation no_fedavg 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_no_fedavg_7b.log"

log "All 2WikiMultiHop runs complete"
