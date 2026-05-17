# FedCondGraphRAG Evaluation Results

## gfl Tri-Graph Baselines

These rows track the free gfl baselines from plan `11_INT_GFL.md` section 49.
The current metric is the `condensation_qa` proxy task accuracy: node-type
classification over the Hotpot Tri-Graph adapter. This is not final QA EM/F1.

| Date | Baseline | Dataset | Task | Clients | Rounds | Model | Partition | Best Local Test Accuracy | Global Model Test Accuracy | Upload / Round | Download / Round | Notes |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 2026-05-14 | FedAvg | hotpot_trigraph | condensation_qa | 5 | 3 | gcn | subgraph_fl_louvain | 0.7136 | 0.7480 | 68.01 KB | 68.01 KB | Built from `linear-rag/hotpotqa/questions.json`; S-E/P-E edges only. |
