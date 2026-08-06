#!/usr/bin/env bash
# Push FedRAG performance on the 2wikifed bench. Two levers:
#   A) richer retrieval context: top-10 PPR passages in desc (map suffix _top10,
#      max-txt-len raised so 10 passages are not truncated at 512 tokens)
#   B) more training data: restart the killed full-data 2wiki ppr-desc LoRA run
#      (144k questions vs 915 on the bench), save its checkpoint, then eval that
#      checkpoint on the 2wikifed bench (checkpoint = GNN/projector/LoRA only,
#      dataset-agnostic).
# Waits for the graph-mode ablation chain to release the GPU first.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
mkdir -p logs

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "waiting for PID $WAIT_PID (graph-mode chain) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
  echo "PID $WAIT_PID finished."
fi

QA_ROOT=dataset/fedcond_qa/2wikifed

echo "===== stage A1: build top-10 PPR maps (2wikifed, 3 clients) ====="
python -u scripts/preprocess_fedcond_qa.py --dataset 2wikifed \
  --top_k_passages 10 --out-suffix _top10 \
  2>&1 | tee logs/ppr_top10_2wikifed.log
echo "===== stage A1 exit: ${PIPESTATUS[0]} ====="

echo "===== stage A2: fl-train with top-10 desc ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root "$QA_ROOT" \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --ppr-map-name ppr_node_map_top10.pt --max-txt-len 1200 --eval-batch-size 8 \
  --wandb-group 2wikifed-bench --wandb-run-name text-only-lora-7b-2wikifed-top10desc \
  --wandb-tags 2wikifed text_only lora 7b top10_desc \
  2>&1 | tee logs/fl_train_2wikifed_top10desc.log
echo "===== stage A2 exit: ${PIPESTATUS[0]} ====="

echo "===== stage B1: full-data 2wiki ppr-desc LoRA restart (144k questions) ====="
python -u main.py fl-train --dataset 2wikimultihop \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --wandb-group 2wiki-full --wandb-run-name text-only-lora-7b-2wiki-ppr-desc-r2 \
  --wandb-tags fulldata text_only lora 7b ppr_desc restart \
  2>&1 | tee logs/fl_train_2wiki_ppr_desc_r2.log
echo "===== stage B1 exit: ${PIPESTATUS[0]} ====="

echo "===== stage B2: eval full-data checkpoint on the 2wikifed bench ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root "$QA_ROOT" \
  --num-clients 3 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit \
  --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --eval-only --load-checkpoint checkpoints/text-only-lora-7b-2wiki-ppr-desc-r2.pt \
  --wandb-group 2wikifed-bench --wandb-run-name eval-fulldata-ckpt-on-2wikifed \
  --wandb-tags 2wikifed eval_only fulldata_ckpt \
  2>&1 | tee logs/eval_fulldata_ckpt_2wikifed.log
echo "===== stage B2 exit: ${PIPESTATUS[0]} ====="
echo "===== PUSH-PERF CHAIN DONE ====="
