# HippoRAG Local-Only Baseline

This baseline vendors upstream HippoRAG unchanged under
`third_party/HippoRAG` and runs one HippoRAG instance per FedCondGraphRAG
client shard.

No client uploads anchors, model weights, documents, or synthetic graphs. Each
client indexes only:

```text
processed/<dataset>/client_<id>/chunks.json
```

Queries are assigned by the same local ownership rule used by the preprocessing
maps:

```text
question_index % num_clients == client_id
```

Run:

```bash
python main.py hipporag-baseline \
  --dataset 2wikimultihop \
  --num-clients 3 \
  --split test \
  --llm-base-url https://api.openai.com/v1 \
  --llm-name gpt-4o-mini \
  --embedding-name nvidia/NV-Embed-v2
```

For a smoke test:

```bash
python main.py hipporag-baseline \
  --dataset 2wikimultihop \
  --num-clients 3 \
  --split test \
  --client-id 0 \
  --max-docs-per-client 100 \
  --max-questions-per-client 10
```

To reuse pre-extracted OpenIE/NER results, pass the upstream JSON file. The
wrapper copies it into each client save directory using the filename HippoRAG
expects for the selected reader model.

```bash
python main.py hipporag-baseline \
  --dataset musique \
  --num-clients 1 \
  --client-id 0 \
  --precomputed-openie-path /path/to/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json
```

Outputs:

```text
output/baselines/hipporag_local/<dataset>/<split>/summary.json
output/baselines/hipporag_local/<dataset>/<split>/client_<id>/metrics.json
output/baselines/hipporag_local/<dataset>/<split>/client_<id>/predictions.jsonl
```

Metrics are computed by upstream HippoRAG:

- Retrieval: `Recall@k`
- QA: `ExactMatch`, `F1`

The wrapper lives under `fedcond_grag/baselines/graph_rag/` and only converts
local LinearRAG chunks into HippoRAG's expected `title\ntext` document format
before passing gold docs/answers into HippoRAG.
