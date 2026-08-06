#!/usr/bin/env bash
# REORDERED (user priority): baselines/comparison results FIRST, FedRAG re-train SECOND.
# Part B (now first): eval-only each existing text_only -best.pt checkpoint to
#   regenerate its val+test metrics at full (2-decimal) precision — the training
#   logs only kept 1 decimal and /tmp/fl_metrics.jsonl is reset per run.
# Part A (now second): re-run FedRAG FULL model (graph GNN + LoRA both trained,
#   --dual-graph-mode shared --llm-frozen False), desc ppr, 3 rounds, on
#   musiquefed + hotpotqafed. Distinct checkpoint/run names so the earlier
#   text_only runs are not clobbered.
# Every run's /tmp/fl_metrics.jsonl is copied to metrics/*.jsonl right after it
# finishes (trainer resets that path on the next run).
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs metrics

BASE=(--num-clients 3 --seed 42 --use-cuda
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000)

save_metrics() { cp /tmp/fl_metrics.jsonl "metrics/$1.jsonl" 2>/dev/null; }

eval_only() {  # $1=DS $2=tag $3=ckpt ; $4.. = desc args
  local DS="$1" TAG="$2" CKPT="$3"; shift 3
  echo "===== [regen] $TAG eval-only ($CKPT) ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${BASE[@]}" --num-rounds 1 \
    --dual-graph-mode text_only --llm-frozen False \
    --eval-only --load-checkpoint "checkpoints/$CKPT" \
    "$@" \
    2>&1 | tee "logs/eval_regen_${TAG}.log"
  echo "===== [regen] $TAG exit: ${PIPESTATUS[0]} ====="
  save_metrics "regen_${TAG}"
}

# =============== Part B FIRST: eval-only regen (2-decimal) ================
export WANDB_MODE=offline   # eval-only regen: no need to pollute wandb
for DS in musiquefed hotpotqafed; do
  eval_only "$DS" "${DS}_nodesc"          "text-only-lora-7b-${DS}-nodesc-best.pt"          --desc-source none
  eval_only "$DS" "${DS}_hippo_desc"      "text-only-lora-7b-${DS}-hippo-desc-best.pt"      --desc-source file --desc-file "dataset/fedcond_qa/$DS/desc_hipporag.jsonl"
  eval_only "$DS" "${DS}_linearrag_desc"  "text-only-lora-7b-${DS}-linearrag-desc-best.pt"  --desc-source file --desc-file "dataset/fedcond_qa/$DS/desc_linearrag.jsonl"
  eval_only "$DS" "${DS}_ppr_desc"        "text-only-lora-7b-${DS}-best.pt"                 --desc-source ppr
done
echo "===== PART B (regen) DONE ====="
unset WANDB_MODE            # re-enable wandb logging for the training runs

# =============== Part A SECOND: graph + LoRA (shared mode) ================
for DS in musiquefed hotpotqafed; do
  echo "===== [$DS] FULL FedRAG: graph(GNN)+LoRA, shared mode, ppr desc, 3 rounds ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${BASE[@]}" --num-rounds 3 \
    --dual-graph-mode shared --llm-frozen False --desc-source ppr \
    --wandb-group "$DS-bench" --wandb-run-name "graph-lora-7b-$DS" \
    --wandb-tags "$DS" bench graph_lora shared 7b ppr_desc \
    2>&1 | tee "logs/fl_train_${DS}_graph_lora.log"
  echo "===== [$DS] graph_lora exit: ${PIPESTATUS[0]} ====="
  save_metrics "${DS}_graph_lora"
done

echo "===== GRAPH-LORA + REGEN CHAIN DONE ====="
