python main.py fl-train --dataset hotpotqa --num-clients 3 --num-rounds 3 \
  --local-epochs 3 --llm-model-name qwen2.5-7b-instruct --llm-load-in-4bit \
  --llm-gradient-checkpointing --local-batch-size 1 --eval-batch-size 2 \
  --dual-graph-mode both --use-cuda \
  --wandb-run-name fedrag-hotpotqa-3c3r-qwen7b-4bit --wandb-tags fedrag hotpotqa qwen2.5-7b