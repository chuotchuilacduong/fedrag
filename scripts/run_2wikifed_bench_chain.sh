#!/usr/bin/env bash
# 3-system fair comparison on the shared 2wiki bench (same corpus + test questions):
#   1) FedCondGraphRAG fl-train on 2wikifed (text_only + LoRA, Qwen2.5-7B 4-bit,
#      desc from PPR retrieval; test = the 987 upfed bench questions)
#   2) HippoRAG QA on 2wikimultihop_upfed with local Qwen2.5-7B reader — the
#      index/OpenIE from the Jul 5 retrieval-only run is cached and reused.
# LinearRAG results already exist: output/baselines/linearrag_local{,_qwen7b}/.
# Steps run sequentially so they never compete for VRAM.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "===== step 1: fl-train 2wikifed (text_only + LoRA, ppr desc) ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root dataset/fedcond_qa/2wikifed \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --wandb-group 2wikifed-bench --wandb-run-name text-only-lora-7b-2wikifed \
  --wandb-tags bench 2wikifed text_only lora 7b ppr_desc \
  2>&1 | tee logs/fl_train_2wikifed.log
echo "===== step 1 exit: ${PIPESTATUS[0]} ====="

echo "===== step 2: HippoRAG QA on 2wikimultihop_upfed (Qwen2.5-7B local) ====="
python -u main.py hipporag-baseline \
  --dataset 2wikimultihop_upfed --num-clients 3 --split all \
  --local-llm --local-llm-batch-size 8 --local-llm-max-gpu-mem 7GiB \
  --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
  --retrieval-top-k 20 \
  2>&1 | tee logs/hipporag_qa_7b.log
echo "===== step 2 exit: ${PIPESTATUS[0]} ====="
echo "===== 2WIKIFED BENCH CHAIN DONE ====="
