#!/usr/bin/env bash
# Eval-only on the 2wikifed bench: MERGED checkpoint = shared-frozen GNN
# (graph_encoder+projector from shared-frozen-7b-2wiki-ppr-desc-best) + LoRA
# LLM weights (from text-only-lora-7b-2wiki-ppr-desc-r2). Tests whether the
# trained graph channel and the finetuned LLM compose. Waits for the
# shared-frozen bench eval to release its VRAM first.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1; }

for attempt in 1 2 3 4 5; do
  until [ "$(free_mib)" -ge 7500 ]; do
    echo "attempt $attempt: only $(free_mib) MiB free, waiting..."; sleep 180
  done
  echo "===== merged eval attempt $attempt ($(free_mib) MiB free) ====="
  python -u main.py fl-train --dataset 2wikifed --qa-data-root dataset/fedcond_qa/2wikifed \
    --num-clients 3 --seed 42 --use-cuda \
    --llm-model-name qwen2.5-7b --llm-load-in-4bit \
    --max-eval-samples 1000 --eval-batch-size 2 \
    --dual-graph-mode shared --llm-frozen False --desc-source ppr \
    --eval-only --load-checkpoint checkpoints/merged-sharedgnn-fulllora-2wiki.pt \
    --wandb-group 2wikifed-bench --wandb-run-name eval-merged-sharedgnn-fulllora-2wikifed \
    --wandb-tags 2wikifed eval_only shared merged gnn_plus_lora fulldata_ckpt \
    2>&1 | tee logs/eval_merged_on_2wikifed.log
  echo "===== merged eval exit: ${PIPESTATUS[0]} ====="
  if grep -q "test  :" logs/eval_merged_on_2wikifed.log; then
    break
  fi
  echo "attempt $attempt produced no test metrics (likely OOM), retrying in 5m..."
  sleep 300
done
echo "===== MERGED BENCH EVAL DONE ====="
