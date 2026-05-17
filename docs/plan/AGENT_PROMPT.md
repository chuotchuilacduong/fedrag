---
file: AGENT_PROMPT.md
title: Agent Invocation Prompt
load_priority: always-load
project: FedCondGraphRAG
---

# Agent Invocation Prompt

> Copy block dưới vào turn đầu khi mở session làm việc với agent (Claude Code, Cursor, Aider, hoặc tương tự). Block này thiết lập context tối thiểu + rules để agent điều hướng plan.

---

## A. Prompt template (copy nguyên block dưới)

```text
You are implementing the FedCondGraphRAG project — a federated graph
condensation framework for retrieval-augmented generation, combining
LinearRAG (Tri-Graph), DANCE (federated condensation), G-Retriever
(graph soft prompt), and OpenFGL/gfl (federated infrastructure).

The implementation plan is split into 12 markdown files in `docs/plan/`.
Before doing anything, read `docs/plan/README.md` — it explains the
file map and how to choose which files to load for a given task.

Rules:

1. ALWAYS-LOAD: Read `docs/plan/README.md` and `docs/plan/01_OVERVIEW.md`
   in every fresh session. Re-skim them whenever switching to a new
   task.

2. TASK-LOAD: For the user's current task, consult section §3 of
   README ("File Selector Logic") to pick which `0X_*.md` files to
   load. Load each file fully — do NOT skim. Pay attention to:
     - YAML frontmatter (prerequisites, related)
     - "See also" footer (cross-references to other files)

3. CRITICAL FILE: When ANY code in `fedcond_grag/graph_condensation/`
   or anything touching DANCE methodology is being written, ALSO load
   `10_DANCE_REFERENCE.md`. DANCE has no official code — this file is
   the only authoritative reference and contains 10 subtle points that
   are easy to get wrong from paper alone.

4. DO NOT WRITE FROM SCRATCH. The plan prioritizes "copy and fix" over
   re-implementation. Before writing ANY module:
     - Check `09_INT_HOST_REPO.md` for G-Retriever + LinearRAG file
       mappings (which file to copy from upstream).
     - Check `11_INT_GFL.md` for FL infrastructure (FGLTrainer,
       FedGMServer, partitioning) that may already exist in gfl.
     - Only write new code for: DANCE re-implementation, S-E-P motif
       selection, dual graph LLM subclass, glue code.

5. RESPECT THE DESIGN INVARIANT: The topology `Sentence – Entity –
   Passage` must be preserved throughout client condensation. NEVER
   add direct Sentence–Passage edges to the main graph. NEVER select
   sentence and passage nodes independently of entity bridges.

6. HYPERPARAMETERS: Always check `08_APPENDIX_HYPERPARAMS.md` for
   defaults. The "Source" column tells you whether a value comes
   from DANCE Table 5, G-Retriever default, or our choice. Note the
   ambiguity between α/β (Eq. 13) and λ_1/λ_3 (Algo 4) in DANCE —
   read `10_DANCE_REFERENCE.md` §38 Point 9 before tuning these.

7. WHEN UNCERTAIN about an API of an upstream library
   (LinearRAG.entity_list, FedGMServer methods, etc.), do NOT guess.
   Run the verification scripts in `09_INT_HOST_REPO.md` §34 first.

8. AFTER EACH STEP: Run the verification checkpoint listed in
   `09_INT_HOST_REPO.md` §35 (for tier-1/2 setup) or `11_INT_GFL.md`
   §50.6 (for tier-3 setup). Do NOT proceed until the checkpoint
   passes.

9. WRITE COMMIT MESSAGES that reference plan sections, e.g.
   `feat: implement S-E-P motif selector (plan §11)` or
   `fix: respect DANCE B_2=2 budget (plan 10_DANCE §37.3 + 08_APPENDIX)`.

10. IF THE PLAN IS WRONG: If you find that the plan disagrees with
    the actual upstream code (e.g., FedGMServer method signature is
    different), prefer the upstream code and flag the discrepancy in
    your reply for the user to decide. Do not silently amend the plan.

11. ENVIRONMENT: i have created conda environment name fedcond in wsl. Everything you need to do with env, do in that environment
Begin by stating which files you are about to load, and why. Then load
them, summarize what you understood, and propose a concrete next step.
```

---

## B. Conditional addendum

Tuỳ task, paste thêm 1 trong các block bổ sung dưới đây vào prompt:

### B.1 Khi setup repo lần đầu

```text
TASK: Initial repo setup.
- Clone G-Retriever as host repo, branch `fedcond-grag`.
- Vendor LinearRAG under `fedcond_grag/external/linearrag/`.
- Vendor gfl under `fedcond_grag/external/gfl/`.
- Rewrite import paths in both vendored libraries.
- Verify imports via the smoke tests in 09 §35 and 11 §50.6.

Required reading: README, 01, 09, 11.
```

### B.2 Khi implement Stage B (client condensation)

```text
TASK: Implement Stage B (client-side graph condensation).
- Modules to write: motif_core_selector, text_bank, neighbor_gating,
  chunk_selection, graph_text_fusion, topology_reconstruction,
  client_condensor.
- This is the highest-risk part because DANCE has no official code.
- Use the pseudo-code in 10 §37 as ground truth.
- Implement in this order: KNN topology baseline first (10 §38 Point 1),
  then self-expression. entmax+STE is phase-2 — start with topk+softmax.

Required reading: README, 01, 03, 10. Suggested: 08, 11.
```

### B.3 Khi implement Stage C (server condensation)

```text
TASK: Implement Stage C (server-side condensation via fedcond_qa).
- Subclass FedGMServer (from vendored gfl/flcore/fedgm/).
- Override init_synthetic_graph, compute_anchor_gradients,
  server_condense_step.
- Add PGE adjacency learning on top of FedGM logic.
- Register fedcond_qa in gfl's algorithm registry.

Required reading: README, 01, 04, 11. Suggested: 03, 10 (for surrogate
task loss formula).
```

### B.4 Khi implement Stage D (dual prompting + LLM)

```text
TASK: Implement Stage D (dual graph prompting with LLM).
- Subclass GraphLLM as DualGraphLLM in src/model/dual_graph_llm.py.
- Add condensed_encoder (GAT), projector_c.
- Override encode_graphs() to produce both z_e and z_c.
- Concatenate as soft prompt: [z_e ; z_c ; text_emb].
- Create dataset class hotpot_fedcond following webqsp.py template.

Required reading: README, 01, 05, 09. Suggested: 06 (for ablation
choices: z_e-only, z_c-only, random z_c).
```

### B.5 Khi chạy baselines

```text
TASK: Run free baselines from gfl on Tri-Graph.
- Verify Tri-Graph adapter (hotpot_trigraph) registered in gfl.
- Run: fedavg.
- Each is 1-line CLI as listed in 11 §49.
- Collect results into the main eval table (see 06 §23).

Required reading: README, 01, 11, 06. Suggested: 02 (Tri-Graph format).
```

### B.6 Khi debug

```text
TASK: Debug an issue in {module_name}.
- Check the per-module debug checklist in 07 §27 first.
- For DANCE-related modules, also check 10 §40 (pre-flight) and §41
  (unit tests skeleton).
- For gfl integration issues, check 11 §53 (R8-R13).
- For copy-and-fix migration issues, check 09 §35 (per-step
  checkpoints).

Required reading: README, 01, 07, + the file covering the module.
```

---

## C. Anti-patterns to avoid

Khi agent vi phạm các pattern dưới đây, dừng lại và đọc lại file plan tương ứng:

| Anti-pattern | Counter-action |
|---|---|
| Write `entity_extractor.py` from scratch using spaCy | Stop → wrap `LinearRAG.index()` (09 §29 Step 5) |
| Write FedAvg/FedSage+/FedGM from scratch | Stop → use gfl flags (11 §49) |
| Use class labels in DANCE loss | Stop → adapt to node-type CE (10 §39.3) |
| Add Sentence–Passage edges to Tri-Graph | Stop → S-E-P invariant (02 §10.1) |
| Skip difficulty score; score all 2-hop neighbors | Stop → pre-filter (10 §38 Point 3) |
| Use `c_v` (Eq. 7) directly in fusion `x_v` | Stop → use `t̃_v` only (10 §38 Point 2 + 7) |
| Update model delta `Δω` and upload | Stop → upload condensed graph `C_m` (10 §38 Point 10) |
| Implement node_condensation every round | Stop → every 10 rounds (10 §38 Point 5) |
| Tune α and β as if they are λ_1 and λ_3 | Stop → read 10 §38 Point 9 + 08 note on ambiguity |
| Mix entmax forward with vanilla softmax backward | Stop → need straight-through estimator (10 §38 Point 1) |

---

## D. Example first turn

User says: *"Help me start implementing Stage B."*

Good agent response opening:

```
I'll start by reading the relevant plan files for Stage B.

Files I'm about to load:
- docs/plan/README.md (always-load)
- docs/plan/01_OVERVIEW.md (always-load — project context)
- docs/plan/03_CLIENT_CONDENSATION.md (Stage B spec)
- docs/plan/10_DANCE_REFERENCE.md (CRITICAL — DANCE has no official code)
- docs/plan/08_APPENDIX_HYPERPARAMS.md (for B_0/B_1/B_2/B_tok defaults)

Reading now...

[after reading]

Key constraints I noted:
1. Topology invariant: never select sentence/passage independently of
   entity anchors (03 §11.1).
2. Budgets per DANCE Table 5: B_0=1, B_1=3, B_2=2 (08 + 10 §37.3).
3. 2-hop must be pre-filtered by difficulty score (10 §38 Point 3).
4. Text encoder MUST be frozen (10 §38 Point 4).
5. Implement KNN topology baseline first, then self-expression
   (10 §37.4 + plan suggested order).

Concrete next step proposal: I'll start with
`fedcond_grag/graph_condensation/text_bank.py` because it's the lowest
dependency (no GNN needed, just frozen encoder + chunker + cache).
Then `neighbor_gating.py`, then `chunk_selection.py`. Topology and
fusion last.

Shall I proceed?
```

→ Notice: agent **states which files**, **summarizes what it learned**, **proposes a concrete step**, **asks for confirmation**. Don't dive into code immediately.
