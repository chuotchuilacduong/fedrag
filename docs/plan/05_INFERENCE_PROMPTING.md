---
file: 05_INFERENCE_PROMPTING.md
title: Inference Retrieval & Dual Graph Prompting (Stage D)
load_priority: task-load
prerequisites: [01_OVERVIEW.md]
related: [09_INT_HOST_REPO.md, 04_SERVER_CONDENSATION.md, 02_DATA_AND_TRIGRAPH.md]
covers_sections: "Part III §19 (Query-time retrieval: LinearRAG + global graph); §20 (Dual GNN encoders + late fusion + LLM)"
project: FedCondGraphRAG
---

# Inference Retrieval & Dual Graph Prompting (Stage D)

> **How to use this file.** Spec của stage D — pipeline inference end-to-end với LLM. **PHẢI load `09_INT_HOST_REPO.md` cùng** khi implement vì stage D dựa heavy trên G-Retriever pattern (GraphLLM → DualGraphLLM subclass). File 09 chứa code mẫu của `DualGraphLLM` và dataset class.

---

### 19. Query-time Retrieval

#### 19.1 LinearRAG evidence branch

Input: `q`. Output:

- `P_q`: top-K passages qua **(stage 1)** entity activation + **(stage 2)** PPR trên entity-passage subgraph.
- `E_q`: evidence graph chứa entity activated, sentence chứa entity activated, passage trong `P_q`, cùng các cạnh S-E, P-E.

`E_q` là **query-specific và text-grounded**.

#### 19.2 Global graph branch

Input: `z_q = Enc(q)`. Output `G_global(q)`:

```text
seed = TopR_{i}( cos(z_q, X_global[i]) )
expand 1-hop theo A_global
G_global(q) = induced subgraph
```

`G_global(q)` là **embedding-only, no text**.

#### Checkpoints (query-time)

- `|V(E_q)| ≤ budget_E`, `|V(G_global(q))| ≤ budget_C`.
- Retrieval deterministic với seed cố định.
- Full global graph **không bao giờ** được pass thẳng vào LLM.

### 20. Dual Graph Prompting (Option B)

#### 20.1 Hai GNN encoders

```text
z_e = MLP_e( POOL( GNN_e(E_q) ) ) ∈ R^{d_l}
z_c = MLP_c( POOL( GNN_c(G_global(q)) ) ) ∈ R^{d_l}
```

`GNN_e` là Graph Transformer trên Tri-Graph với edge type embedding. `GNN_c` đơn giản hơn (GCN/GAT).

#### 20.2 Late fusion (first version)

```text
soft_prompt = [z_e ; z_c]    # 2 tokens
```

Sau:
- gated: `α · z_e + (1-α) · z_c`,
- query-aware: `α = σ(W [z_q; z_e; z_c])`,
- attention fusion: cho query làm Q, [z_e, z_c] làm K/V.

#### 20.3 LLM input (G-Retriever style)

```text
text_input  = textualize(E_q) ⊕ "\n\nRetrieved passages:\n" ⊕ P_q ⊕ "\n\nQuestion: " ⊕ q
h_t         = TextEmbedder(text_input)
input_seq   = [z_e ; z_c ; h_t]
Y           = LLM(input_seq)
```

Loss: standard causal LM trên token answer.

#### Checkpoints (prompting)

- `z_e.shape == z_c.shape == [d_l]` (= 4096 cho Llama-2-7B).
- Pipeline chạy được khi mask 1 trong 2 token (cho ablation).
- Text input không vượt context length (truncate `P_q` theo ngân sách).

---

---

## See also
- **DualGraphLLM subclass code + Dataset class** (BẮT BUỘC khi implement): `09_INT_HOST_REPO.md` §29 Step 6-7
- **LinearRAG retrieval wrapper** (`LinearRAGRetriever.retrieve()`): `09_INT_HOST_REPO.md` §29 Step 4
- **Global graph retrieval (top-r + 1-hop trên A_global)**: `04_SERVER_CONDENSATION.md` (output) + `09_INT_HOST_REPO.md` §31
- **Hyperparams (top-K passages, top-R seed, LLM model name)**: `08_APPENDIX_HYPERPARAMS.md`
- **QA metrics + ablation table**: `06_TRAINING_EVAL.md`
- **Debug checklist cho LLM (z_e/z_c shape, context length, random token baseline)**: `07_DEBUG_RISKS_ORDER.md` §27.7
