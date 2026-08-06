#!/usr/bin/env bash
# CPU-only build of 2wikifed_xl (runs alongside GPU training safely).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedrag
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

echo "===== xl step 1: build dataset ====="
python -u scripts/build_2wikifed_xl.py --num-extra 9000 || exit 1
echo "===== xl step 2: fedcond_qa records ====="
python -u scripts/build_fedcond_qa_dataset.py --dataset 2wikifed_xl \
  --out-root dataset/fedcond_qa/2wikifed_xl || exit 1
cp processed/2wikifed_xl/split/*.txt dataset/fedcond_qa/2wikifed_xl/split/
echo "===== xl step 3: PPR maps (3 clients, CPU) ====="
python -u scripts/preprocess_fedcond_qa.py --dataset 2wikifed_xl
echo "===== XL BUILD DONE ====="
