#!/usr/bin/env bash
# Sweep: {flexlora, rolora} x {hotpotqa, musique, 2wikimultihop}, 5 rounds each.
# Text-only-question LoRA ablation: --dual-graph-mode none (no graph
# soft-prompt tokens) + QA cache rebuilt with --top-k-desc 0 (no retrieved
# passages either) -- the LLM sees only the question. Only the LoRA adapter
# is trained/saved; the graph encoder/projector are untouched.
#
# flora is intentionally excluded -- fedcond_grag/server/lora_aggregate/flora.py
# merges the aggregated LoRA delta directly into the base weight, which
# requires a dense (non-quantized) tensor. This project runs Qwen2.5-7B-Instruct
# in 4-bit (--llm-load-in-4bit) to fit a 16GB card; full fp16 would need
# ~15.2GB for weights alone and is not safe here. Run flora separately with a
# smaller model (e.g. qwen2.5-1.5b-instruct, no --llm-load-in-4bit) if needed.
#
# Usage: bash scripts/run_lora_sweep.sh
# Resumable: preprocess and fl-train are both safe to re-run (preprocess skips
# built artifacts; fl-train just starts a fresh run per combo). The desc-strip
# step re-runs every time since it must run after preprocess's own QA-cache
# build (which populates desc) and there's no cheap way to tell the two apart
# on disk.

set -uo pipefail  # NOT -e: one failed combo must not abort the rest of the sweep

DATASETS=(hotpotqa musique 2wikimultihop)
METHODS=(flexlora rolora)
NUM_CLIENTS=3
NUM_ROUNDS=5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/lora_sweep"
CKPT_DIR="$ROOT/checkpoints"
mkdir -p "$LOG_DIR"

FAILED=()
SUCCEEDED=()

for ds in "${DATASETS[@]}"; do
    echo "==================================================================="
    echo "[preprocess] $ds"
    echo "==================================================================="
    if ! python main.py preprocess --dataset "$ds" --num-clients "$NUM_CLIENTS" \
            2>&1 | tee "$LOG_DIR/preprocess_${ds}.log"; then
        echo "!! preprocess failed for $ds -- skipping its fl-train runs"
        for method in "${METHODS[@]}"; do
            FAILED+=("$ds/$method (preprocess)")
        done
        continue
    fi

    echo "[strip desc] $ds -- rebuilding QA cache with --top-k-desc 0"
    if ! python scripts/build_fedcond_qa_dataset.py --dataset "$ds" --top-k-desc 0 \
            2>&1 | tee "$LOG_DIR/strip_desc_${ds}.log"; then
        echo "!! desc-strip failed for $ds -- skipping its fl-train runs"
        for method in "${METHODS[@]}"; do
            FAILED+=("$ds/$method (strip-desc)")
        done
        continue
    fi

    for method in "${METHODS[@]}"; do
        run_name="fedrag-${ds}-${method}-5r"
        ckpt_path="$CKPT_DIR/${ds}/${method}/best.pt"
        log_path="$LOG_DIR/${ds}_${method}.log"

        echo "==================================================================="
        echo "[fl-train] dataset=$ds method=$method -> $ckpt_path"
        echo "==================================================================="

        if python main.py fl-train --dataset "$ds" --num-clients "$NUM_CLIENTS" \
                --num-rounds "$NUM_ROUNDS" --local-epochs 3 \
                --llm-model-name qwen2.5-7b-instruct --llm-load-in-4bit \
                --local-batch-size 2 --eval-batch-size 8 \
                --dual-graph-mode none --use-cuda \
                --llm-frozen False --lora-agg-method "$method" \
                --save-best --save-best-path "$ckpt_path" \
                --wandb-run-name "$run_name" \
                --wandb-tags fedrag "$ds" "$method" lora \
                2>&1 | tee "$log_path"; then
            SUCCEEDED+=("$ds/$method")
        else
            echo "!! fl-train failed for $ds/$method (see $log_path)"
            FAILED+=("$ds/$method")
        fi
    done
done

echo
echo "==================================================================="
echo "SWEEP SUMMARY"
echo "==================================================================="
echo "Succeeded (${#SUCCEEDED[@]}):"
printf '  %s\n' "${SUCCEEDED[@]}"
echo "Failed (${#FAILED[@]}):"
printf '  %s\n' "${FAILED[@]}"
echo "Checkpoints under: $CKPT_DIR/<dataset>/<method>/best.pt"
echo "Logs under:        $LOG_DIR/"

[ "${#FAILED[@]}" -eq 0 ]
