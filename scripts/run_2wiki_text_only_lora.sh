#!/usr/bin/env bash
# Ablation: text_only + LoRA finetune (QLoRA) — 2wikimultihop, 3 clients, 5 rounds.
# Compares against the frozen-LLM text_only run to measure LoRA effectiveness.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

conda run -n fedrag python main.py fl-train \
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
    --wandb-run-name "text-only-lora-7b-2wiki-fulldata" \
    --wandb-tags ablation text_only lora 7b \
    2>&1 | tee "$LOG_DIR/fl_train_2wiki_text_only_lora_7b.log"
