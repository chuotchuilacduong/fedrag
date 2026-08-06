#!/usr/bin/env bash
# Run PPR preprocessing for 2wikimultihop — resumes from checkpoint if exists.
# Fixed version: SpaCy cached, DPR limited to top-100, GPU entity similarity.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

echo "=== Starting PPR (fast) for 2wikimultihop ==="

# Run all 3 clients sequentially with checkpoint resume
for cid in 0 1 2; do
    echo ""
    echo "--- client_$cid ---"
    conda run -n fedrag python scripts/preprocess_fedcond_qa.py \
        --dataset 2wikimultihop \
        --client-id $cid \
        2>&1 | tee -a "$LOG_DIR/ppr_2wiki_client${cid}.log"
done

echo ""
echo "=== PPR complete for all clients ==="
