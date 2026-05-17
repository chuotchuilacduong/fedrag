#!/usr/bin/env bash
# FedCondGraphRAG runner. Stage A→B→C preprocessing must already be done:
#     python main.py preprocess --dataset hotpotqa
# (or invoke scripts/build_client_pipeline.py directly.)

set -euo pipefail

for seed in 0 1 2 3; do
    # Stage D — dual-prompting fit, frozen LLM
    python main.py train \
        --dataset fedcond_qa --model_name dual_graph_llm --llm_frozen True \
        --gnn_in_dim 384 --gnn_hidden_dim 384 --gnn_in_dim_c 384 --gnn_hidden_dim_c 384 \
        --gnn_model_name gt --gnn_model_name_c gat --seed "$seed"

    # Stage D — dual-prompting fit, LoRA-tuned LLM
    python main.py train \
        --dataset fedcond_qa --model_name dual_graph_llm --llm_frozen False \
        --gnn_in_dim 384 --gnn_hidden_dim 384 --gnn_in_dim_c 384 --gnn_hidden_dim_c 384 \
        --gnn_model_name gt --gnn_model_name_c gat --seed "$seed"
done

# Federated Stage C round loop (single round example):
# python main.py fl-train --dataset hotpotqa --num-clients 8 --num-rounds 1
