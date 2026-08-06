#!/usr/bin/env bash
# Broadcast-eval protocol on the shared 2wiki bench: every client answers ALL
# 987 test questions on its own local shard (corpus partition unchanged);
# metrics = average over the 3 clients. Ownership-protocol results stay in
# their original output dirs — everything here writes to new locations.
#   0) PPR anchors for every question on every client (ppr_node_map_all.pt)
#   1) FedRAG eval-only from the trained ppr-desc checkpoint (--eval-broadcast)
#   2) LinearRAG --all-questions (retrieval + Qwen2.5-7B QA)  -> linearrag_local_bcast/
#   3) HippoRAG  --all-questions (retrieval + Qwen2.5-7B QA)  -> hipporag_local/2wikifed/test/
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
mkdir -p logs

QA_ROOT=dataset/fedcond_qa/2wikifed

echo "===== stage 0: PPR anchors for all questions (3 clients) ====="
python -u scripts/preprocess_fedcond_qa.py --dataset 2wikifed --all-questions \
  2>&1 | tee logs/ppr_all_2wikifed.log
echo "===== stage 0 exit: ${PIPESTATUS[0]} ====="

echo "===== stage 1: FedRAG eval-only broadcast (ppr-desc checkpoint) ====="
python -u main.py fl-train --dataset 2wikifed --qa-data-root "$QA_ROOT" \
  --num-clients 3 --seed 42 --use-cuda \
  --llm-model-name qwen2.5-7b --llm-load-in-4bit \
  --max-eval-samples 1000 \
  --dual-graph-mode text_only --llm-frozen False --desc-source ppr \
  --eval-only --eval-broadcast \
  --load-checkpoint checkpoints/text-only-lora-7b-2wikifed.pt \
  --wandb-group 2wikifed-bench --wandb-run-name eval-bcast-7b-2wikifed-ppr \
  --wandb-tags broadcast eval_only 2wikifed \
  2>&1 | tee logs/eval_bcast_fedrag_2wikifed.log
echo "===== stage 1 exit: ${PIPESTATUS[0]} ====="

echo "===== stage 2: LinearRAG broadcast (retrieval + Qwen-7B QA) ====="
python -u main.py linearrag-baseline --dataset 2wikifed --num-clients 3 \
  --all-questions --split-file "$QA_ROOT/split/test_indices.txt" \
  --local-llm --local-llm-batch-size 4 \
  --output-dir output/baselines/linearrag_local_bcast \
  2>&1 | tee logs/linearrag_bcast_2wikifed.log
echo "===== stage 2 exit: ${PIPESTATUS[0]} ====="

echo "===== stage 3: HippoRAG broadcast (retrieval + Qwen-7B QA) ====="
python -u main.py hipporag-baseline \
  --dataset 2wikifed --qa-data-root "$QA_ROOT" --num-clients 3 \
  --split test --all-questions \
  --local-llm --local-llm-batch-size 8 --local-llm-max-gpu-mem 7GiB \
  --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
  --retrieval-top-k 20 \
  --precomputed-openie-path dataset/ner/2wikimultihopqa/openie_results_ner_propfacts_llama70b.json \
  2>&1 | tee logs/hipporag_bcast_2wikifed.log
echo "===== stage 3 exit: ${PIPESTATUS[0]} ====="
echo "===== BROADCAST EVAL CHAIN DONE ====="
