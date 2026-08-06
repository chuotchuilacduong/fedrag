#!/usr/bin/env bash
# Graph-contribution ablations on the 2wikifed bench (ppr desc, 987 test q).
# Settles what the graph soft-prompt channel adds in the current setting:
#   1) shared + LoRA        — do graph tokens improve over text_only+LoRA (hit 30.3)?
#   2) no_synthetic + LoRA  — z_c from the LOCAL evidence graph instead of the
#      server synthetic graph; (shared - no_synthetic) isolates the server
#      global graph's contribution specifically.
#   3) shared + frozen LLM  — graph-only adaptation channel (full-data gold-desc
#      era reached ~48 EM this way; this is the retrieval-grounded version).
# Waits for the broadcast-eval chain to release the GPU first.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
mkdir -p logs

WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "waiting for PID $WAIT_PID (broadcast chain) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
  echo "PID $WAIT_PID finished."
fi

QA_ROOT=dataset/fedcond_qa/2wikifed

fl_train() {  # $1 = dual-graph-mode, $2 = llm-frozen, $3 = run suffix, $4 = log name
  python -u main.py fl-train --dataset 2wikifed --qa-data-root "$QA_ROOT" \
    --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
    --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
    --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
    --dual-graph-mode "$1" --llm-frozen "$2" --desc-source ppr \
    --wandb-group 2wikifed-bench --wandb-run-name "$3" \
    --wandb-tags ablation 2wikifed graph_mode 7b "$1" \
    2>&1 | tee "logs/$4.log"
  echo "===== $3 exit: ${PIPESTATUS[0]} ====="
}

echo "===== run 1: shared + LoRA ====="
fl_train shared False shared-lora-7b-2wikifed fl_train_2wikifed_shared_lora

echo "===== run 2: no_synthetic + LoRA ====="
fl_train no_synthetic False nosyn-lora-7b-2wikifed fl_train_2wikifed_nosyn_lora

echo "===== run 3: shared + frozen ====="
fl_train shared True shared-frozen-7b-2wikifed fl_train_2wikifed_shared_frozen

echo "===== GRAPH MODE ABLATION CHAIN DONE ====="
