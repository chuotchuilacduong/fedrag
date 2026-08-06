#!/usr/bin/env bash
# Orchestration only (no pipeline/chain code changed): musiquefed's PPR node
# maps were stale after the dedup rebuild of its train/val questions, so
# rebuild them (--force) before any training, then hand off to the existing
# desc-source ablations chain (nodesc / hippo-desc / linearrag-desc for
# musiquefed + hotpotqafed, Qwen2.5-7B text_only + LoRA).
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "===== musiquefed PPR rebuild (post-dedup, --force) ====="
python -u scripts/preprocess_fedcond_qa.py --dataset musiquefed --force \
  2>&1 | tee logs/musiquefed_ppr_rebuild.log
echo "===== musiquefed ppr rebuild exit: ${PIPESTATUS[0]} ====="

echo "===== handing off to desc-source ablations chain ====="
exec bash scripts/run_fed_bench_desc_ablations_chain.sh
