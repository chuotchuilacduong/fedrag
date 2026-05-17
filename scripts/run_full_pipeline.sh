#!/usr/bin/env bash
# Full FedCondGraphRAG pipeline: preprocess → federated loop (Stage C + D)
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#   bash scripts/run_full_pipeline.sh --rounds 5 --clients 4
#
# All variables below can be overridden via environment or CLI flags.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override with env vars or --flag value)
# ---------------------------------------------------------------------------
DATASET="${DATASET:-hotpotqa}"
NUM_CLIENTS="${NUM_CLIENTS:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"          # round 0 = Stage C bootstrap; 1+ = Stage D + FedAvg
CLIENT_FRAC="${CLIENT_FRAC:-1.0}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-processed}"
QA_DATA_ROOT="${QA_DATA_ROOT:-dataset/fedcond_qa}"
CONDA_ENV="${CONDA_ENV:-fedcond}"

# LLM
LLM_MODEL_NAME="${LLM_MODEL_NAME:-7b}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-meta-llama/Llama-2-7b-hf}"

# GNN
GNN_MODEL_NAME="${GNN_MODEL_NAME:-gt}"
GNN_MODEL_NAME_C="${GNN_MODEL_NAME_C:-gat}"
GNN_IN_DIM="${GNN_IN_DIM:-384}"
GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM:-384}"
GNN_NUM_LAYERS="${GNN_NUM_LAYERS:-4}"
GNN_NUM_HEADS="${GNN_NUM_HEADS:-4}"

# Stage C (server condensation)
NUM_GLOBAL_SYN_NODES="${NUM_GLOBAL_SYN_NODES:-512}"
SERVER_CONDENSE_ITERS="${SERVER_CONDENSE_ITERS:-50}"
HID_DIM="${HID_DIM:-384}"
NUM_LAYERS="${NUM_LAYERS:-4}"

# Stage D (local training)
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LOCAL_LR="${LOCAL_LR:-1e-5}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-4}"
RETRIEVAL_TOP_R="${RETRIEVAL_TOP_R:-16}"

# Hardware
USE_CUDA="${USE_CUDA:-1}"
GPUID="${GPUID:-0}"

# ---------------------------------------------------------------------------
# CLI flag parsing (--key value pairs override defaults above)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2";             shift 2 ;;
        --clients)        NUM_CLIENTS="$2";          shift 2 ;;
        --rounds)         NUM_ROUNDS="$2";           shift 2 ;;
        --seed)           SEED="$2";                 shift 2 ;;
        --llm-path)       LLM_MODEL_PATH="$2";       shift 2 ;;
        --gnn-in-dim)     GNN_IN_DIM="$2";           shift 2 ;;
        --gnn-hidden-dim) GNN_HIDDEN_DIM="$2";       shift 2 ;;
        --local-epochs)   LOCAL_EPOCHS="$2";         shift 2 ;;
        --local-lr)       LOCAL_LR="$2";             shift 2 ;;
        --syn-nodes)      NUM_GLOBAL_SYN_NODES="$2"; shift 2 ;;
        --no-cuda)        USE_CUDA=0;                shift 1 ;;
        *) echo "[warn] unknown flag: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    # Fall back to conda env
    PYTHON="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    PYTHON="python"
fi

CUDA_FLAG=""
[[ "$USE_CUDA" == "1" ]] && CUDA_FLAG="--use-cuda"

log() { echo ""; echo "=== $* ==="; echo ""; }

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 0 — Verify environment
# ---------------------------------------------------------------------------
log "Checking environment"
"$PYTHON" -c "import torch, torch_geometric; print('torch', torch.__version__, '| pyg', torch_geometric.__version__)"

# ---------------------------------------------------------------------------
# Step 1 — Preprocess: Stage A (Tri-Graph) + Stage B (client condense)
#           for all clients
# ---------------------------------------------------------------------------
log "Step 1 — Preprocess (Stage A → B) for $NUM_CLIENTS client(s)"
"$PYTHON" main.py preprocess \
    --dataset "$DATASET" \
    --num-clients "$NUM_CLIENTS"

# ---------------------------------------------------------------------------
# Step 2 — Federated loop: Stage C bootstrap (round 0) + Stage C+D (round 1+)
# ---------------------------------------------------------------------------
log "Step 2 — Federated loop ($NUM_ROUNDS rounds, $NUM_CLIENTS clients)"
echo "  round 0       : Stage C bootstrap (gradient matching → synthetic graph)"
echo "  round 1 .. $((NUM_ROUNDS-1))  : Stage D local train + FedAvg + Stage C update"
echo ""

"$PYTHON" main.py fl-train \
    --dataset           "$DATASET" \
    --num-clients       "$NUM_CLIENTS" \
    --num-rounds        "$NUM_ROUNDS" \
    --client-frac       "$CLIENT_FRAC" \
    --seed              "$SEED" \
    --data-root         "$DATA_ROOT" \
    --qa-data-root      "$QA_DATA_ROOT" \
    $CUDA_FLAG \
    --gpuid             "$GPUID" \
    \
    --llm-model-name    "$LLM_MODEL_NAME" \
    --llm-model-path    "$LLM_MODEL_PATH" \
    \
    --gnn-model-name    "$GNN_MODEL_NAME" \
    --gnn-model-name-c  "$GNN_MODEL_NAME_C" \
    --gnn-in-dim        "$GNN_IN_DIM" \
    --gnn-hidden-dim    "$GNN_HIDDEN_DIM" \
    --gnn-num-layers    "$GNN_NUM_LAYERS" \
    --gnn-num-heads     "$GNN_NUM_HEADS" \
    \
    --num-global-syn-nodes   "$NUM_GLOBAL_SYN_NODES" \
    --server-condense-iters  "$SERVER_CONDENSE_ITERS" \
    --hid-dim                "$HID_DIM" \
    --num-layers             "$NUM_LAYERS" \
    \
    --local-epochs      "$LOCAL_EPOCHS" \
    --local-lr          "$LOCAL_LR" \
    --local-batch-size  "$LOCAL_BATCH_SIZE" \
    --retrieval-top-r   "$RETRIEVAL_TOP_R"

log "Pipeline complete"
