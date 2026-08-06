# Experimental Setting — Data Preparation (draft)

## Datasets and Corpus Construction

We evaluate on three multi-hop QA benchmarks: MuSiQue, 2WikiMultiHopQA, and
HotpotQA. To enable a direct comparison with centralized graph-based RAG
systems, we adopt the evaluation protocol of HippoRAG: for each dataset,
1,000 questions are sampled from the validation split, and the retrieval
corpus is formed as the union of all candidate passages (supporting passages
and distractors) associated with the sampled questions. This yields corpora
of 11,656 passages (MuSiQue), 6,119 passages (2WikiMultiHopQA), and 9,811
passages (HotpotQA). We reuse the exact question sets and corpora released
with HippoRAG, so our centralized baseline numbers are directly comparable
to previously published results.

## Federated Partitioning

We simulate a cross-silo federated setting with $C{=}3$ clients. Each corpus
is partitioned into $C$ disjoint shards: passages are shuffled with a fixed
seed (42) and assigned round-robin, so each client indexes roughly one third
of the corpus and no passage is shared across clients. Questions are likewise
assigned to clients by the ownership rule $\mathrm{idx} \bmod C$, mirroring
the assignment used throughout our federated pipeline. For 2WikiMultiHopQA,
the shards are inherited from our full-data federated preprocessing; 987 of
the 1,000 upstream questions and 6,066 of the 6,119 passages are retained
after matching against the local shards (client lists are truncated to equal
length to preserve the round-robin invariant).

Because the supporting passages of a multi-hop question are scattered across
shards, a single client can rarely observe all gold evidence locally. The
resulting per-client oracle recall — the fraction of gold passages present in
the local shard — is 31.8–34.7% on 2WikiMultiHopQA, 30.9–35.4% on HotpotQA,
and 46.5–50.5% on MuSiQue. This ceiling is a structural property of the
federated partition and upper-bounds any purely local retriever.

| | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|
| Questions (total / per client) | 999 / 333 | 987 / 329 | 999 / 333 |
| Passages per client | 3,886 / 3,885 / 3,885 | 1,983 / 2,053 / 2,030 | 3,271 / 3,270 / 3,270 |
| Per-client oracle recall (%) | 50.5 / 46.5 / 48.2 | 31.8 / 31.9 / 34.7 | 30.9 / 35.4 / 32.6 |

## Training and Evaluation Splits

The benchmark questions above are reserved exclusively for testing. To train
FedCondGraphRAG without leakage, we augment the benchmark with additional
questions drawn from the original training region of the full dataset
(disjoint from the region the test questions were sampled from), keeping only
questions whose gold passages are all contained in the benchmark corpus. For
2WikiMultiHopQA this produces 915 training and 99 validation questions on top
of the 987 test questions; train/val/test blocks are interleaved across
clients so that the $\mathrm{idx} \bmod C$ ownership rule holds within every
split.

## Identical Inputs Across Systems

All compared systems consume the same per-client chunk files. Chunks are
serialized as `"<id>:<title>: <text>"`; each system's adapter reconstructs
its native input format from this shared representation, guaranteeing that
FedCondGraphRAG, LinearRAG, and HippoRAG index byte-identical passage sets
per client.

System-specific preprocessing is then run per client shard:

- **HippoRAG** requires OpenIE (NER + triple extraction) over the corpus. For
  2WikiMultiHopQA we reuse the PropRAG-released Llama-3.3-70B extractions,
  which cover the benchmark corpus almost completely; since the released
  files provide entity-annotated propositions but no (subject, predicate,
  object) triples, we synthesize triples from consecutive entity pairs within
  each proposition, using the proposition sentence as the predicate. The
  conversion is question-blind (per-passage only) and introduces no
  evaluation leakage. For HotpotQA and MuSiQue, OpenIE is extracted with
  GPT-4o-mini using HippoRAG's own extraction prompts.
- **LinearRAG** builds its passage, sentence, and entity embedding stores
  over each client shard with its default spaCy NER and encoder.
- **FedCondGraphRAG** runs its full preprocessing pipeline per client
  (tri-graph construction, graph condensation, and per-question PPR
  retrieval maps) on the same shards.

No system observes passages outside its client shard at any point;
cross-client information may flow only through the mechanisms under study
(e.g., server-side aggregation in FedCondGraphRAG).
