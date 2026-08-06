#!/usr/bin/env bash
# Eval-only: full-data shared+frozen (GNN-only) checkpoint on the 2wikifed
# 987-question bench, for direct comparison with LinearRAG/HippoRAG.
# Runs concurrently with the musique full-data training, so eval batch is
# kept small and expandable segments enabled.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

python -u main.py fl-train --dataset 2wikifed --qa-data-root dataset/fedcond_qa/2wikifed \
  --num-clients 3 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit \
  --max-eval-samples 1000 --eval-batch-size 2 \
  --dual-graph-mode shared --llm-frozen True --desc-source ppr \
  --eval-only --load-checkpoint checkpoints/shared-frozen-7b-2wiki-ppr-desc-best.pt \
  --wandb-group 2wikifed-bench --wandb-run-name eval-shared-frozen-fulldata-on-2wikifed \
  --wandb-tags 2wikifed eval_only shared frozen gnn_only fulldata_ckpt \
  2>&1 | tee logs/eval_shared_frozen_on_2wikifed.log
echo "===== eval exit: ${PIPESTATUS[0]} ====="
echo "===== SHARED-FROZEN BENCH EVAL DONE ====="
