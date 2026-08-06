#!/usr/bin/env bash
# User-priority chain: on the musiquefed / hotpotqafed benches, run the
# LoRA-finetuned-Qwen reader with (1) no desc, (2) HippoRAG-retrieved desc,
# (3) LinearRAG-retrieved desc — before any ppr-desc runs. Mirrors the
# 2wikifed desc-source ablation. hotpotqafed still needs its preprocessing
# (steps 1-4 of run_musique_hotpot_fed_bench_chain.sh), done here inline.
# ppr-desc runs for both benches are appended LAST (deprioritized).
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

FL_COMMON=(--num-clients 3 --num-rounds 3 --seed 42 --use-cuda
  --llm-model-name qwen2.5-7b --llm-load-in-4bit --llm-gradient-checkpointing
  --local-epochs 1 --eval-every 1 --max-eval-samples 1000
  --dual-graph-mode text_only --llm-frozen False)

retrieval_and_desc() {  # $1 = fed dataset name, $2 = ner subdir
  local DS="$1" NER="$2"
  echo "===== [$DS] LinearRAG retrieval (all 2028 questions, ownership) ====="
  python -u main.py linearrag-baseline --dataset "$DS" --num-clients 3 \
    --retrieval-only 2>&1 | tee "logs/linearrag_retrieval_$DS.log"
  echo "===== [$DS] linearrag retrieval exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] HippoRAG retrieval (propfacts OpenIE) ====="
  python -u main.py hipporag-baseline --dataset "$DS" --num-clients 3 --split all \
    --precomputed-openie-path "dataset/ner/$NER/openie_results_ner_propfacts_llama70b.json" \
    --retrieval-only \
    --local-llm --local-llm-batch-size 2 --local-llm-max-gpu-mem 5GiB \
    --embedding-name "Transformers/sentence-transformers/all-MiniLM-L6-v2" \
    --retrieval-top-k 20 2>&1 | tee "logs/hipporag_retrieval_$DS.log"
  echo "===== [$DS] hipporag retrieval exit: ${PIPESTATUS[0]} ====="

  python scripts/build_desc_from_baseline_retrieval.py --source linearrag \
    --dataset "$DS" --out "dataset/fedcond_qa/$DS/desc_linearrag.jsonl"
  python scripts/build_desc_from_baseline_retrieval.py --source hipporag \
    --dataset "$DS" --out "dataset/fedcond_qa/$DS/desc_hipporag.jsonl"
  echo "===== [$DS] desc files built ====="
}

train_trio() {  # $1 = fed dataset name
  local DS="$1"
  echo "===== [$DS] fl-train nodesc (question-only LoRA) ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${FL_COMMON[@]}" --desc-source none \
    --wandb-group "$DS-bench" --wandb-run-name "text-only-lora-7b-$DS-nodesc" \
    --wandb-tags "$DS" bench text_only lora 7b nodesc \
    2>&1 | tee "logs/fl_train_${DS}_nodesc.log"
  echo "===== [$DS] nodesc exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] fl-train hippo-desc ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${FL_COMMON[@]}" --desc-source file --desc-file "dataset/fedcond_qa/$DS/desc_hipporag.jsonl" \
    --wandb-group "$DS-bench" --wandb-run-name "text-only-lora-7b-$DS-hippo-desc" \
    --wandb-tags "$DS" bench text_only lora 7b hippo_desc \
    2>&1 | tee "logs/fl_train_${DS}_hippo_desc.log"
  echo "===== [$DS] hippo-desc exit: ${PIPESTATUS[0]} ====="

  echo "===== [$DS] fl-train linearrag-desc ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${FL_COMMON[@]}" --desc-source file --desc-file "dataset/fedcond_qa/$DS/desc_linearrag.jsonl" \
    --wandb-group "$DS-bench" --wandb-run-name "text-only-lora-7b-$DS-linearrag-desc" \
    --wandb-tags "$DS" bench text_only lora 7b linearrag_desc \
    2>&1 | tee "logs/fl_train_${DS}_linearrag_desc.log"
  echo "===== [$DS] linearrag-desc exit: ${PIPESTATUS[0]} ====="
}

# ---------------- musiquefed (already preprocessed) ----------------
retrieval_and_desc musiquefed musique
train_trio musiquefed

# ---------------- hotpotqafed: preprocessing first -----------------
echo "===== [hotpotqafed] step 1: partition chunks ====="
python -u scripts/preprocess_data.py --dataset hotpotqafed --num_clients 3 \
  2>&1 | tee logs/hotpotqafed_preprocess.log
echo "===== exit: ${PIPESTATUS[0]} ====="
echo "===== [hotpotqafed] step 2: Stage A/B/C graphs ====="
python -u scripts/build_client_pipeline.py --dataset hotpotqafed \
  2>&1 | tee logs/hotpotqafed_client_pipeline.log
echo "===== exit: ${PIPESTATUS[0]} ====="
echo "===== [hotpotqafed] step 3: fedcond_qa dataset + splits ====="
python -u scripts/build_fedcond_qa_dataset.py --dataset hotpotqafed \
  --out-root dataset/fedcond_qa/hotpotqafed \
  2>&1 | tee logs/hotpotqafed_qa_dataset.log
echo "===== exit: ${PIPESTATUS[0]} ====="
mkdir -p dataset/fedcond_qa/hotpotqafed/split
cp processed/hotpotqafed/split/*.txt dataset/fedcond_qa/hotpotqafed/split/
echo "===== [hotpotqafed] step 4: PPR node maps ====="
python -u scripts/preprocess_fedcond_qa.py --dataset hotpotqafed \
  2>&1 | tee logs/hotpotqafed_ppr_maps.log
echo "===== exit: ${PIPESTATUS[0]} ====="

retrieval_and_desc hotpotqafed hotpotqa
train_trio hotpotqafed

# ---------------- deprioritized: ppr-desc runs ----------------------
for DS in musiquefed hotpotqafed; do
  echo "===== [$DS] fl-train ppr-desc (deprioritized rerun) ====="
  python -u main.py fl-train --dataset "$DS" --qa-data-root "dataset/fedcond_qa/$DS" \
    "${FL_COMMON[@]}" --desc-source ppr \
    --wandb-group "$DS-bench" --wandb-run-name "text-only-lora-7b-$DS" \
    --wandb-tags "$DS" bench text_only lora 7b ppr_desc \
    2>&1 | tee "logs/fl_train_$DS.log"
  echo "===== [$DS] ppr exit: ${PIPESTATUS[0]} ====="
done
echo "===== FED BENCH DESC ABLATIONS CHAIN DONE ====="
