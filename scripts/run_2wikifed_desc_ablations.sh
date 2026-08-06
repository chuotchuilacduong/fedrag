#!/usr/bin/env bash
# Retriever-swap ablations on the shared 2wiki bench: fine-tune the same
# text_only+LoRA reader with desc = LinearRAG retrieval and desc = HippoRAG
# retrieval (train + val + test questions all retrieved by the same system,
# so train/eval inputs are consistent). Compare against:
#   ppr desc    (text-only-lora-7b-2wikifed):        hit 30.3 / EM 29.8 / F1 34.0
#   no desc     (text-only-lora-7b-2wikifed-nodesc): hit 22.8 / EM 22.5 / F1 26.6
# Stages run sequentially to avoid VRAM contention.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
mkdir -p logs

QA_ROOT=dataset/fedcond_qa/2wikifed

fl_train() {  # $1 = desc file, $2 = run suffix
  python -u main.py fl-train --dataset 2wikifed --qa-data-root "$QA_ROOT" \
    --num-clients 3 --num-rounds 5 --seed 42 --use-cuda \
    --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing \
    --local-epochs 1 --eval-every 1 --max-eval-samples 1000 \
    --dual-graph-mode text_only --llm-frozen False \
    --desc-source file --desc-file "$1" \
    --wandb-group 2wikifed-bench --wandb-run-name "text-only-lora-7b-2wikifed-$2" \
    --wandb-tags ablation 2wikifed text_only lora 7b "$2"
}

check_coverage() {  # $1 = desc file
  python - "$1" <<'EOF'
import json, sys
ids = {json.loads(l)["id"] for l in open("dataset/fedcond_qa/2wikifed/records.jsonl") if l.strip()}
descs = {json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}
missing = ids - descs
print(f"records={len(ids)} descs={len(descs)} missing={len(missing)}")
sys.exit(1 if missing else 0)
EOF
}

echo "===== stage 1: LinearRAG retrieval over all 2001 2wikifed questions ====="
python -u main.py linearrag-baseline --dataset 2wikifed --num-clients 3 \
  --retrieval-only \
  2>&1 | tee logs/linearrag_retrieval_2wikifed.log
echo "===== stage 1 exit: ${PIPESTATUS[0]} ====="

echo "===== stage 2: build LinearRAG desc file + fl-train ====="
python -u scripts/build_desc_from_baseline_retrieval.py --source linearrag \
  --dataset 2wikifed --out "$QA_ROOT/desc_linearrag.jsonl" \
  && check_coverage "$QA_ROOT/desc_linearrag.jsonl" \
  && fl_train "$QA_ROOT/desc_linearrag.jsonl" linearrag-desc \
    2>&1 | tee logs/fl_train_2wikifed_linearrag_desc.log
echo "===== stage 2 exit: $? ====="

echo "===== stage 3: HippoRAG retrieval over all 2001 2wikifed questions ====="
python -u main.py hipporag-baseline \
  --dataset 2wikifed --qa-data-root "$QA_ROOT" --num-clients 3 --split all \
  --retrieval-only --local-llm --local-llm-batch-size 8 --local-llm-max-gpu-mem 7GiB \
  --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
  --retrieval-top-k 20 \
  --precomputed-openie-path dataset/ner/2wikimultihopqa/openie_results_ner_propfacts_llama70b.json \
  2>&1 | tee logs/hipporag_retrieval_2wikifed.log
echo "===== stage 3 exit: ${PIPESTATUS[0]} ====="

echo "===== stage 4: build HippoRAG desc file + fl-train ====="
python -u scripts/build_desc_from_baseline_retrieval.py --source hipporag \
  --dataset 2wikifed --split all --out "$QA_ROOT/desc_hipporag.jsonl" \
  && check_coverage "$QA_ROOT/desc_hipporag.jsonl" \
  && fl_train "$QA_ROOT/desc_hipporag.jsonl" hippo-desc \
    2>&1 | tee logs/fl_train_2wikifed_hippo_desc.log
echo "===== stage 4 exit: $? ====="
echo "===== DESC ABLATION CHAIN DONE ====="
