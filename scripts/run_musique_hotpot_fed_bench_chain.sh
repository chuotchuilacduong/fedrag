#!/usr/bin/env bash
# FedRAG on the musique / hotpotqa upfed benches, mirroring the 2wikifed
# recipe end-to-end: preprocess -> Stage A/B/C graphs -> fedcond_qa + PPR
# maps -> text_only+LoRA fine-tune (small train split, 310/client, 5 rounds)
# -> eval on the 999-question bench test split.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

for DS in musiquefed hotpotqafed; do
  echo "===== [$DS] step 1: partition chunks ====="
  python -u scripts/preprocess_data.py --dataset "$DS" --num_clients 3 \
    2>&1 | tee "logs/${DS}_preprocess.log"
  echo "===== [$DS] step 1 exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] step 2: Stage A/B/C graphs ====="
  python -u scripts/build_client_pipeline.py --dataset "$DS" \
    2>&1 | tee "logs/${DS}_client_pipeline.log"
  echo "===== [$DS] step 2 exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] step 3: fedcond_qa dataset + splits ====="
  python -u scripts/build_fedcond_qa_dataset.py --dataset "$DS" \
    --out-root "dataset/fedcond_qa/$DS" \
    2>&1 | tee "logs/${DS}_qa_dataset.log"
  echo "===== [$DS] step 3 exit: ${PIPESTATUS[0]} ====="
  mkdir -p "dataset/fedcond_qa/$DS/split"
  cp "processed/$DS/split/"*.txt "dataset/fedcond_qa/$DS/split/"

  echo "===== [$DS] step 4: PPR node maps ====="
  python -u scripts/preprocess_fedcond_qa.py --dataset "$DS" \
    2>&1 | tee "logs/${DS}_ppr_maps.log"
  echo "===== [$DS] step 4 exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] step 5: fl-train text_only + LoRA (5 rounds) ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
    --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
    --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
    --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
    --wandb-group "$DS-bench" --wandb-run-name "text-only-lora-7b-$DS" \
    --wandb-tags "$DS" bench text_only lora 7b ppr_desc \
    2>&1 | tee "logs/fl_train_$DS.log"
  echo "===== [$DS] step 5 exit: ${PIPESTATUS[0]} ====="
done
echo "===== MUSIQUE/HOTPOT FED BENCH CHAIN DONE ====="
