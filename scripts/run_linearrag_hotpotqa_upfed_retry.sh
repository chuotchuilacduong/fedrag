#!/usr/bin/env bash
# Retry LinearRAG QA 7B on hotpotqa_upfed (batch 8 OOM'd next to the hotpotqa
# full-data training). Runs after the hipporag hotpotqa retry, batch 2.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

until grep -q "HIPPORAG HOTPOTQA RETRY DONE" logs/hipporag_hotpotqa_retry_wrap.log 2>/dev/null; do
  echo "waiting for hipporag hotpotqa retry..."; sleep 300
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1; }
for attempt in 1 2 3 4 5; do
  until [ "$(free_mib)" -ge 6800 ]; do echo "only $(free_mib) MiB free, waiting..."; sleep 300; done
  echo "===== linearrag hotpotqa_upfed retry attempt $attempt ====="
  python -u main.py linearrag-baseline \
    --dataset hotpotqa_upfed --num-clients 3 \
    --local-llm --local-llm-batch-size 2 --local-llm-gpu-mem 5GiB \
    --output-dir output/baselines/linearrag_local_qwen7b \
    2>&1 | tee logs/linearrag_qa_7b_hotpotqa_upfed.log
  rc=${PIPESTATUS[0]}
  echo "===== retry attempt $attempt exit: $rc ====="
  [ "$rc" -eq 0 ] && break
  sleep 600
done
echo "===== LINEARRAG HOTPOTQA RETRY DONE ====="
