#!/usr/bin/env bash
# Ablation: text_only + LoRA (QLoRA) with PPR-retrieved desc — 2wikimultihop.
# Identical config to run_2wiki_text_only_lora.sh (wandb om2241bx) except
# --desc-source ppr: the LLM text input is the PPR passage node text instead
# of gold evidence, i.e. the realistic retrieval-RAG setting (R@5 ~22.9%).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

conda run --no-capture-output -n fedrag python main.py fl-train \
    --dataset 2wikimultihop \
    --num-clients 3 \
    --num-rounds 5 \
    --seed 42 \
    --use-cuda \
    --llm-model-name qwen2.5-7b \
    --llm-load-in-4bit \
    --llm-gradient-checkpointing \
    --local-epochs 1 \
    --eval-every 1 \
    --max-eval-samples 200 \
    --wandb-group 2wiki-full \
    --dual-graph-mode text_only \
    --llm-frozen False \
    --desc-source ppr \
    --wandb-run-name "text-only-lora-7b-2wiki-ppr-desc" \
    --wandb-tags ablation text_only lora 7b ppr_desc \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_text_only_lora_ppr_7b.log"
