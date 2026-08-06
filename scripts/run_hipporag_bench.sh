#!/usr/bin/env bash
# HippoRAG local-only baseline on the 2wiki sub-benchmark (3 clients).
# Local Qwen2.5-7B-Instruct 4-bit for OpenIE + fact rerank (no API key),
# all-MiniLM-L6-v2 embeddings (same encoder as LinearRAG).
# Resumable: OpenIE + LLM calls are cached (sqlite + per-client openie json).
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag

mkdir -p logs
python main.py hipporag-baseline \
  --dataset 2wikimultihop_bench \
  --num-clients 3 \
  --split all \
  --retrieval-only \
  --local-llm \
  --local-llm-batch-size "${BATCH:-8}" \
  --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
  --retrieval-top-k 20 \
  2>&1 | tee logs/hipporag_bench.log

python scripts/eval_bench_recall.py --bench 2wikimultihop_bench \
  --hipporag-split all --linearrag-top-k 10 2>&1 | tee logs/bench_recall_summary.log
