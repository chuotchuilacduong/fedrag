<claude-mem-context>
# Memory Context

# [fedrag] recent context, 2026-07-01 3:06pm GMT+9

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (19,458t read) | 288,663t work | 93% savings

### Jun 27, 2026
1034 4:35p 🔵 PPR Process Still Running — 62GB RAM, 106% CPU, 240+ Minutes for client_0
1035 4:36p 🔵 client_1 PPR Not Yet Started — Sequential Processing Means Days of Work Remaining
1036 4:37p 🔵 LinearRAGConfig Defaults — BFS Mode Active, iteration_threshold=0.5 Limits Propagation
1037 " 🔵 2wikimultihop Has 180,030 Questions — Root Cause of Multi-Day PPR Runtime Confirmed
1039 4:39p 🔵 RTX 4090 Has 13GB Free VRAM But GPU Vectorized Retrieval Is Disabled
1040 " 🔴 SpacyNER.__init__ Loads Model for Validation But Discards It — Fix: Cache as self._nlp
1041 " 🔴 ner.py Fixed: SpaCy Model Now Cached as self._nlp in __init__
1038 " 🔴 Critical Bug: SpaCy Model Loaded Freshly on Every question_ner() Call — 180k Times Per Client
1042 4:42p 🔴 ner.py Fully Fixed: question_ner Uses Cached Model + New batch_question_ner Added
1043 4:43p 🔴 calculate_passage_scores() Fixed: DPR Candidates Capped at top_k×20, Entity Text Cache Added
1044 " 🔴 get_seed_entities() Gets GPU Acceleration via entity_embeddings_gpu Optional Path
1045 4:44p 🔴 retrieve() Now Pins Entity Embeddings to GPU at Start — Enables Fast Per-Query Similarity Search
1046 " 🔴 EvidenceLinearRAG._prepare() Also Gets GPU Entity Embedding Pin — PPR Preprocessing Path Now Fully Accelerated
1047 " 🔴 retrieve_with_evidence() Fully Batched: Single SpaCy pipe() + Single embedding.encode() Per 500-Question Batch
1048 " 🟣 New Script: run_ppr_fast.sh — Runs PPR Preprocessing Per-Client with Checkpoint Resume
1049 4:45p 🔴 SpaCy NER Fix Verified Working — Cached Model and batch_question_ner() Confirmed
1050 " ✅ Optimized PPR Process Restarted — client_0 Resuming from Batch 5000/180030
1051 5:13p 🔵 ppr_node_map.pt must span all 180k global question indices despite per-client FL-split
1052 5:17p 🔵 PPR Python process stdout goes to unread pipe — new-run log lines silently discarded
1053 " 🔵 PPR steady-state speed confirmed at 600ms/q with 3 parallel clients on 124GB machine
1054 6:13p 🔵 ppr_node_map.pt must be global shape [180030, top_k] despite per-client FL-split
1055 6:39p 🔵 client_0 crashed and was restarted from checkpoint at local question 13000/60010
1056 8:38p 🔵 Monitor b1wtyi6dc produces spurious inflated ETA at round-number milestones (40%, 50%)
1057 10:03p 🔵 ppr_node_map.pt uses global shape [180030, top_k] despite per-client processing
1058 11:08p 🟣 PPR preprocessing completed for all 3 FedCondGraphRAG clients on 2wikimultihop
1059 11:38p 🔵 FL training script for 2wikimultihop runs 2 sequential ablations: text_only and shared+no-fedavg
1060 11:40p 🔵 Previous fl-train run crashed: "No PPR anchor nodes for sample idx=28263" — old stale ppr_node_map.pt from before preprocessing
1061 " 🔵 Previously failing sample idx=28263 now has valid PPR anchors — fl-train will succeed
1062 " 🔵 FedCondGraphRAG dual-graph-mode has 9 choices; trainer auto-generates wandb run names from mode
1063 11:41p 🔵 Previous musique fl-train run confirmed: wandb "self_expr_topo_7b_full", ~0.94 s/step on 26094 train samples per client
1064 " 🔵 Complete wandb run history reveals FedCondGraphRAG experiment progression across musique and 2wikimultihop datasets
1065 11:42p 🔵 Full musique experiment history: "shared" (full FedCondGraphRAG) run exists — Jun 11 without 4bit, Jun 17-18 with 4bit
1066 " ✅ scripts/run_2wiki_fl_train.sh updated: added full FedCondGraphRAG run (shared+FedAvg) as Step 3a before ablations
1067 " 🟣 FL training pipeline launched — PID 2962895, PPR verified, fedcond main run starting at 23:42 KST
### Jun 28, 2026
1068 2:38a 🔵 fl-train fedcond run has been running ~2.75h (277min CPU) but log not updating — training in progress, not stuck
1069 " 🔵 fl_train_2wiki_fedcond_7b.log is 0 bytes — PYTHONUNBUFFERED may not be flushing to tee, or output going only to wandb
1070 2:39a 🔵 FL training confirmed healthy: GPU at 100% utilization, 23.6GB/24GB VRAM, PID 2963152 using 157% CPU and 10.1% RAM
1071 " 🔵 FL train PID 2963152 running 2h57m elapsed, 4h39m CPU time; wandb run ID qlufywq6 confirmed active
1072 2:40a 🔵 wandb captures stdout via "wrap_raw" redirect — explains why tee receives no output; full run config confirmed from debug.log
1074 " 🔵 FL training Round 1 status check — client_1 still in progress, no new milestones since last check
1075 5:00a 🔵 FedCondGraphRAG eval metrics definition: "hit" = substring match, EM = exact_match, F1 = token-level F1
1076 5:01a 🔵 Full eval metric pipeline confirmed: hit=normalized substring, EM=SQuAD exact match, F1=token-level; eval distributes samples across clients with on-the-fly PPR retrieval
1077 " 🔵 FedCondGraphRAG trainer data split loading: train/val/test from split/*.txt files, FL partitioned by i%n_clients, eval capped at max_eval_samples=200
1078 " 🔵 2wikimultihop dataset split sizes confirmed: 144023 train / 18002 val / 18002 test from dataset/fedcond_qa/split/
1079 " 🔵 2wikimultihop QA dataset: 180030 total records, test samples are multi-hop film/director questions with short factual answers
1080 " 🔵 Dataset split generation: simple contiguous 80/10/10 index ranges, not stratified — train=0..143999, val=144000..162029, test=162030..180029
1081 5:07a 🔵 wandb run qlufywq6 has no wandb-summary.json and no history files — metrics not being logged to wandb cloud despite active run
1082 9:24a 🔵 FedCondGraphRAG ablation question repeated — answer deferred to Steps 3b/3c completion
1083 12:24p 🔵 FedCondGraphRAG Step 3a Round 3 per-client loss profiles confirmed — c0 plateaus, c1 monotonically declines
1084 2:54p 🔵 FedCondGraphRAG Step 3a Round 3 complete — 3 rounds done, Round 4 started, ~19:35 KST finish
S694 Repeated status check — no new tool output beyond Round 4 client_0 step 3600 already captured (Jun 28, 3:10 PM)
S695 FedCondGraphRAG 2wikimultihop FL training monitoring — Step 3a Round 4 (final), client_0 at step 6000/12002, loss=0.0365 (Jun 28, 3:24 PM)
S696 Repeated status check — no new tool output beyond Round 4 client_0 step 6000 already captured (Jun 28, 3:25 PM)
S697 FedCondGraphRAG 2wikimultihop FL training monitoring — Step 3a Round 4 (final), client_0 at step 7200/12002, loss=0.0359 (Jun 28, 3:39 PM)
S698 Repeated status check — no new tool output beyond Round 4 client_0 step 7200 already captured (Jun 28, 3:40 PM)
S699 FedCondGraphRAG 2wikimultihop FL training monitoring — Step 3a Round 4 (final), client_0 at step 9600/12002, loss=0.0365 (Jun 28, 3:54 PM)
S700 Repeated status check — no new tool output beyond Round 4 client_0 step 9600 already captured (Jun 28, 3:55 PM)
S701 FedCondGraphRAG 2wikimultihop FL training monitoring — Step 3a Round 4 (final), client_0 at step 10800/12002, ~10min to finish (Jun 28, 4:09 PM)
S702 Repeated status check — no new tool output beyond Round 4 client_0 step 10800 already captured (Jun 28, 4:10 PM)
S703 FedCondGraphRAG 2wikimultihop FL training monitoring — Step 3a Round 4, client_0 done, client_1 started at step 1200/12002, loss=0.0322 (Jun 28, 4:25 PM)
**Investigated**: wandb/run-20260627_234305-qlufywq6/files/output.log monitored; Round 4 client_0 completion and client_1 start confirmed

**Learned**: - client_0 Round 4 complete; client_1 started at step 1200/12002, loss=0.0322
    - client_1 Round 4 opening loss (0.0322) dramatically lower than Round 3 opening (0.0462) — FedAvg Round 3→4 drove strong step-change
    - client_1 opening at 0.0322 is already lower than client_0's entire Round 4 range (~0.035-0.037) — suggests c1 benefits more from FedAvg
    - ~92min estimated for client_1 to complete Round 4

**Completed**: - Rounds 1-3 fully complete; Round 4 (final): client_0 done; client_1 at step 1200/12002 (just started), loss=0.0322
    - Cross-round loss trajectory confirmed: R1~0.07 → R2~0.047-0.059 → R3~0.039-0.041 → R4 c0~0.036, c1 opening 0.0322

**Next Steps**: - client_1 finishes in ~92min → client_2 (~99min) → final eval → Step 3a complete ~19:35 KST Jun 28
    - Steps 3b (text_only, ~24.5h) and 3c (shared+no-fedavg, ~24.5h) auto-launch sequentially
    - Ablation comparison (3a vs 3b vs 3c) needed to answer retrieval text/graph contribution question


Access 289k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>