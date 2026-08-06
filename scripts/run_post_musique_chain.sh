#!/usr/bin/env bash
# Replacement for the tail of run_2wikifed_xl_train.sh (whose wrapper was
# killed; the in-flight musique training python survives on its own).
# New GPU order per user priority: after musique finishes,
#   1. nodesc control on 2wiki full data (question-only LoRA, 1 training
#      round at the full 48k/client budget — direct comparison with
#      text-only-lora-7b-2wiki-ppr-desc-r2 round 1, EM 50.5)
#   2. hotpotqa full-data ppr-desc (unchanged from the old chain)
#   3. nodesc control on musique full data (mirrors the 5-round ppr run)
#   4. top10-desc retry (unchanged)
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

MUSIQUE_PID="${MUSIQUE_PID:?set MUSIQUE_PID to the running musique python PID}"
echo "waiting for musique training (PID $MUSIQUE_PID) to finish..."
while kill -0 "$MUSIQUE_PID" 2>/dev/null; do sleep 300; done
echo "musique training finished."
sleep 60

echo "===== full-data 2wiki: nodesc control (question-only LoRA, 1 training round) ====="
python -u main.py fl-train --dataset 2wikimultihop \
  --num-clients 3 --num-rounds 2 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --dual-graph-mode text_only --llm-frozen False --desc-source none \
  --wandb-group 2wiki-full --wandb-run-name text-only-lora-7b-2wiki-nodesc \
  --wandb-tags fulldata text_only lora 7b nodesc question_only \
  2>&1 | tee logs/fl_train_2wiki_nodesc_full.log
echo "===== 2wiki nodesc exit: ${PIPESTATUS[0]} ====="

echo "===== full-data hotpotqa: text_only + LoRA + ppr desc (78.3k train) ====="
python -u main.py fl-train --dataset hotpotqa --qa-data-root dataset/fedcond_qa/hotpotqa \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --wandb-group hotpotqa-full --wandb-run-name text-only-lora-7b-hotpotqa-ppr-desc \
  --wandb-tags fulldata text_only lora 7b ppr_desc hotpotqa \
  2>&1 | tee logs/fl_train_hotpotqa_ppr_desc.log
echo "===== hotpotqa exit: ${PIPESTATUS[0]} ====="

echo "===== full-data musique: nodesc control (question-only LoRA, 5 rounds) ====="
python -u main.py fl-train --dataset musique --qa-data-root dataset/fedcond_qa/musique \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --dual-graph-mode text_only --llm-frozen False --desc-source none \
  --wandb-group musique-full --wandb-run-name text-only-lora-7b-musique-nodesc \
  --wandb-tags fulldata text_only lora 7b nodesc question_only musique \
  2>&1 | tee logs/fl_train_musique_nodesc.log
echo "===== musique nodesc exit: ${PIPESTATUS[0]} ====="

echo "===== top10 retry: batch 2, expandable segments ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root dataset/fedcond_qa/2wikifed \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
  --local-batch-size 2 --eval-batch-size 4 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --ppr-map-name ppr_node_map_top10.pt --max-txt-len 1200 \
  --wandb-group 2wikifed-bench --wandb-run-name text-only-lora-7b-2wikifed-top10desc \
  --wandb-tags 2wikifed text_only lora 7b top10_desc \
  2>&1 | tee logs/fl_train_2wikifed_top10desc.log
echo "===== top10 retry exit: ${PIPESTATUS[0]} ====="
echo "===== POST-MUSIQUE CHAIN DONE ====="
