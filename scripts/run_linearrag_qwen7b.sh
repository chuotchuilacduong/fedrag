#!/usr/bin/env bash
# LinearRAG local-only baseline with a local Qwen2.5-7B-Instruct (4-bit) reader
# on the 3 *_upfed benches. Fair-comparison counterpart to the gpt-4o-mini run
# (output/baselines/linearrag_local/) — same retrieval, same prompt, 7B reader.
#
# The GPU is shared with other users' jobs, so this waits for enough free VRAM
# before each dataset and retries on OOM; the sqlite LLM cache makes every
# retry resume where the last attempt died.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEED_MIB="${NEED_MIB:-6500}"
BATCH="${BATCH:-2}"
OUT=output/baselines/linearrag_local_qwen7b

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

wait_for_vram() {
  while true; do
    f=$(free_mib)
    if [ "$f" -ge "$NEED_MIB" ]; then
      echo "[vram] ${f}MiB free >= ${NEED_MIB}MiB — starting"
      return
    fi
    echo "[vram] only ${f}MiB free, waiting for ${NEED_MIB}MiB..."
    sleep 60
  done
}

mkdir -p logs "$OUT"
for ds in 2wikimultihop_upfed hotpotqa_upfed musique_upfed; do
  for attempt in 1 2 3 4 5; do
    wait_for_vram
    echo "===== START $ds (attempt $attempt) ====="
    if python -u main.py linearrag-baseline --dataset "$ds" --num-clients 3 \
        --local-llm --local-llm-batch-size "$BATCH" \
        --output-dir "$OUT"; then
      echo "===== DONE $ds ====="
      break
    fi
    echo "===== RETRY $ds (attempt $attempt failed) ====="
    sleep 120
    [ "$attempt" -eq 5 ] && echo "===== FAILED $ds after 5 attempts ====="
  done
done
echo "===== ALL DONE ====="
