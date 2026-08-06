#!/usr/bin/env bash
# Retry HippoRAG QA 7B on hotpotqa_upfed after the baselines chain finishes.
# First attempt OOM'd during the ~6-doc local OpenIE top-up while the hotpotqa
# full-data training was ramping up; retry with batch 2 and a tighter cap.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

until grep -q "UPFED BASELINES QA7B DONE" logs/upfed_baselines_qa7b_wrap.log 2>/dev/null; do
  echo "waiting for baselines chain..."; sleep 300
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1; }
for attempt in 1 2 3 4 5; do
  until [ "$(free_mib)" -ge 6800 ]; do echo "only $(free_mib) MiB free, waiting..."; sleep 300; done
  echo "===== hipporag hotpotqa_upfed retry attempt $attempt ====="
  python -u main.py hipporag-baseline \
    --dataset hotpotqa_upfed --num-clients 3 --split all \
    --precomputed-openie-path dataset/ner/hotpotqa/openie_results_ner_propfacts_llama70b.json \
    --local-llm --local-llm-batch-size 2 --local-llm-max-gpu-mem 5GiB \
    --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
    --retrieval-top-k 20 \
    2>&1 | tee logs/hipporag_qa_7b_hotpotqa_upfed.log
  rc=${PIPESTATUS[0]}
  echo "===== retry attempt $attempt exit: $rc ====="
  [ "$rc" -eq 0 ] && break
  sleep 600
done
echo "===== HIPPORAG HOTPOTQA RETRY DONE ====="
