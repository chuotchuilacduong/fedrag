#!/usr/bin/env bash
# Train on the 10x-expanded 2wikifed_xl bench train split (~3,300/client), then
# retry the OOM-killed top-10-desc run with a smaller batch. Waits for the
# push-perf chain (full-data restart) to release the GPU first, and for the
# CPU-side xl build to finish.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "waiting for PID $WAIT_PID (push-perf chain) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
  echo "PID $WAIT_PID finished."
fi
until grep -q "XL BUILD DONE" logs/build_2wikifed_xl.log 2>/dev/null; do
  echo "waiting for xl build..."; sleep 300
done

CKPT_2WIKI=checkpoints/text-only-lora-7b-2wiki-ppr-desc-r2.pt
for DS in musique hotpotqa; do
  echo "===== transfer eval: 2wiki full-data ckpt on $DS (zero-shot) ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    --num-clients 3 --seed 42 --use-cuda \
    --llm-model-name qwen2.5-7b --llm-load-in-4bit \
    --max-eval-samples 200 \
    --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
    --eval-only --load-checkpoint "$CKPT_2WIKI" \
    --wandb-group "$DS-full" --wandb-run-name "eval-2wiki-ckpt-on-$DS" \
    --wandb-tags transfer eval_only fulldata_ckpt "$DS" \
    2>&1 | tee "logs/eval_2wiki_ckpt_on_$DS.log"
  echo "===== transfer eval $DS exit: ${PIPESTATUS[0]} ====="
done

echo "===== fused-desc eval: cross-client retrieval upper bound (~1h) ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root dataset/fedcond_qa/2wikifed \
  --num-clients 3 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit \
  --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False \
  --desc-source file --desc-file dataset/fedcond_qa/2wikifed/desc_fused.jsonl \
  --eval-only --load-checkpoint checkpoints/text-only-lora-7b-2wikifed.pt \
  --wandb-group 2wikifed-bench --wandb-run-name eval-fused-desc-2wikifed \
  --wandb-tags 2wikifed eval_only fused_desc upper_bound \
  2>&1 | tee logs/eval_fused_desc_2wikifed.log
echo "===== fused-desc eval exit: ${PIPESTATUS[0]} ====="

echo "===== full-data 2wiki: shared + FROZEN LLM (GNN-only channel, 3 training rounds) ====="
python -u main.py fl-train --dataset 2wikimultihop \
  --num-clients 3 --num-rounds 4 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --max-train-per-client 24000 \
  --dual-graph-mode shared --llm-frozen True --desc-source ppr \
  --wandb-group 2wiki-full --wandb-run-name shared-frozen-7b-2wiki-ppr-desc \
  --wandb-tags fulldata shared frozen gnn_only 7b ppr_desc \
  2>&1 | tee logs/fl_train_2wiki_shared_frozen.log
echo "===== shared-frozen exit: ${PIPESTATUS[0]} ====="

echo "===== xl train: text_only + LoRA on 2wikifed_xl ====="
python -u main.py fl-train --dataset 2wikifed_xl --qa-data-root dataset/fedcond_qa/2wikifed_xl \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --wandb-group 2wikifed-bench --wandb-run-name text-only-lora-7b-2wikifed-xl \
  --wandb-tags 2wikifed_xl text_only lora 7b expanded_train \
  2>&1 | tee logs/fl_train_2wikifed_xl.log
echo "===== xl train exit: ${PIPESTATUS[0]} ====="

echo "===== full-data musique: text_only + LoRA + ppr desc (31.9k train) ====="
python -u main.py fl-train --dataset musique --qa-data-root dataset/fedcond_qa/musique \
  --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
  --local-epochs 1 --eval-every 1 --max-eval-samples 200 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --wandb-group musique-full --wandb-run-name text-only-lora-7b-musique-ppr-desc \
  --wandb-tags fulldata text_only lora 7b ppr_desc musique \
  2>&1 | tee logs/fl_train_musique_ppr_desc.log
echo "===== musique exit: ${PIPESTATUS[0]} ====="

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
echo "===== XL TRAIN CHAIN DONE ====="
