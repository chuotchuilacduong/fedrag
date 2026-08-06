#!/usr/bin/env bash
# Full preprocessing (Stage 0 -> A/B/C -> QA dataset -> PPR maps) for the two
# datasets that have never been fully processed: musique, then hotpotqa.
#
# - Every step skips work whose output already exists; PPR checkpoints per
#   batch, so a killed run resumes for free.
# - QA datasets go to per-dataset dirs (dataset/fedcond_qa/<ds>) so the legacy
#   shared dataset/fedcond_qa (currently 2wikimultihop) is left untouched.
#   fl-train must be given --qa-data-root dataset/fedcond_qa/<ds>.
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1

run_dataset() {
  ds="$1"
  echo "########## PREPROCESS $ds ##########"

  echo "===== [$ds] step 0: split chunks per client ====="
  python -u scripts/preprocess_data.py --dataset "$ds" --num_clients 3 \
    || { echo "===== [$ds] FAILED step 0 ====="; return 1; }

  echo "===== [$ds] step 1-3: Stage A->B->C per client ====="
  python -u scripts/build_client_pipeline.py --dataset "$ds" \
    || { echo "===== [$ds] FAILED stage A-C ====="; return 1; }

  echo "===== [$ds] step 4: build fedcond_qa dataset ====="
  python -u scripts/build_fedcond_qa_dataset.py --dataset "$ds" \
    --out-root "dataset/fedcond_qa/$ds" \
    || { echo "===== [$ds] FAILED fedcond_qa build ====="; return 1; }

  echo "===== [$ds] step 5: PPR node maps (3 clients, sequential) ====="
  for c in 0 1 2; do
    python -u scripts/preprocess_fedcond_qa.py --dataset "$ds" --client-id "$c" \
      || { echo "===== [$ds] FAILED PPR client_$c ====="; return 1; }
  done

  echo "########## DONE $ds ##########"
}

run_dataset musique
run_dataset hotpotqa
echo "########## ALL PREPROCESSING DONE ##########"
