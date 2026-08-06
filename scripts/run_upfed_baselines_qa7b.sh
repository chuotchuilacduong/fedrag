#!/usr/bin/env bash
# HippoRAG + LinearRAG QA with the Qwen2.5-7B local reader on the
# musique_upfed / hotpotqa_upfed benches (999 q each, indexes cached Jul 5).
# Runs in the spare VRAM window next to the musique full-data training:
# waits for the merged-ckpt eval to finish, then guards on free VRAM.
# LinearRAG 7B goes to a separate output root so the gpt-4o-mini summaries
# are not overwritten.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

until grep -q "MERGED BENCH EVAL DONE" logs/eval_merged_bench_wrap.log 2>/dev/null; do
  echo "waiting for merged bench eval..."; sleep 180
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1; }
wait_vram() { until [ "$(free_mib)" -ge 6800 ]; do echo "only $(free_mib) MiB free, waiting..."; sleep 180; done; }

for DS in musique_upfed hotpotqa_upfed; do
  NER_DIR="${DS%_upfed}"
  wait_vram
  echo "===== HippoRAG QA 7B on $DS (propfacts OpenIE) ====="
  python -u main.py hipporag-baseline \
    --dataset "$DS" --num-clients 3 --split all \
    --precomputed-openie-path "dataset/ner/$NER_DIR/openie_results_ner_propfacts_llama70b.json" \
    --local-llm --local-llm-batch-size 8 --local-llm-max-gpu-mem 6GiB \
    --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
    --retrieval-top-k 20 \
    2>&1 | tee "logs/hipporag_qa_7b_$DS.log"
  echo "===== hipporag $DS exit: ${PIPESTATUS[0]} ====="
done

for DS in musique_upfed hotpotqa_upfed; do
  wait_vram
  echo "===== LinearRAG QA 7B on $DS ====="
  python -u main.py linearrag-baseline \
    --dataset "$DS" --num-clients 3 \
    --local-llm --local-llm-batch-size 8 --local-llm-gpu-mem 6GiB \
    --output-dir output/baselines/linearrag_local_qwen7b \
    2>&1 | tee "logs/linearrag_qa_7b_$DS.log"
  echo "===== linearrag $DS exit: ${PIPESTATUS[0]} ====="
done
echo "===== UPFED BASELINES QA7B DONE ====="
