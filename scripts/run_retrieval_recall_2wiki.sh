#!/usr/bin/env bash
# Measure retrieval R@1/2/5/10 on full 2wikimultihop by re-running LinearRAG
# retrieval with top_k=10 per client (resumable; see eval_retrieval_recall.py).
#
# CPU-only (CUDA_VISIBLE_DEVICES="") so the training GPU is untouched, and at
# most 2 clients run in parallel (~13GB RAM each) to stay inside free memory.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

run() {
  # --no-capture-output: stream stdout straight to the log instead of conda
  # buffering it until process exit (which makes progress invisible).
  nice -n 10 conda run --no-capture-output -n fedrag python scripts/eval_retrieval_recall.py \
    --dataset 2wikimultihop --client-id "$1" --top-k 10 \
    >> "logs/retrieval_recall_client$1.log" 2>&1
}

run 0 &
run 1 &
wait -n          # as soon as one of the two finishes...
run 2 &          # ...start the third
wait

conda run -n fedrag python scripts/eval_retrieval_recall.py \
  --dataset 2wikimultihop --top-k 10 --aggregate \
  2>&1 | tee logs/retrieval_recall_summary.log
