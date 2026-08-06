"""Build the `2wikifed` dataset: FedCondGraphRAG on the HippoRAG benchmark.

Goal: run the full fedrag FL pipeline on EXACTLY the corpus + test questions
of the upstream HippoRAG 2wiki benchmark (processed/2wikimultihop_upfed), so
fedrag / LinearRAG / HippoRAG are all measured on the same federated bench.

Construction (paper recipe: 1,000 dev queries; corpus = union of their
candidate passages — verified corpus == union(question contexts)):

- chunks.json  = concat of the three upfed shard files. Chunk texts keep their
  original global "g:" prefixes, and every upfed shard chunk satisfies
  g % 3 == client_id, so partition_linearrag_chunks reproduces the upfed
  shards exactly.
- questions.json = [train | val | test], each block interleaved c0,c1,c2 so
  question_idx % 3 == owning client throughout (block sizes are multiples
  of 3):
    * test  (987)  = the upfed bench questions, same per-client order;
    * train (930) + val (99) = train-split questions of the full 2wiki dataset
      (indices < 144,024 — disjoint from the dev-region test questions) whose
      gold titles ALL exist in the bench corpus, 343 per client, split
      310 train / 33 val.
- split/ indices are contiguous blocks (train 0..929, val 930..1028,
  test 1029..2015) — written here; the 80/10/10 default of
  build_fedcond_qa_dataset.py must be OVERWRITTEN with these files.

After this script:
  python scripts/preprocess_data.py --dataset 2wikifed --num_clients 3
  python scripts/build_client_pipeline.py --dataset 2wikifed
  python scripts/build_fedcond_qa_dataset.py --dataset 2wikifed --out-root dataset/fedcond_qa/2wikifed
  cp processed/2wikifed/split/*.txt dataset/fedcond_qa/2wikifed/split/
  python scripts/preprocess_fedcond_qa.py --dataset 2wikifed --client-id {0,1,2}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402

NUM_CLIENTS = 3
TRAIN_PER_CLIENT = 310
VAL_PER_CLIENT = 33
UPFED = _ROOT / "processed" / "2wikimultihop_upfed"
OUT_LINEARRAG = _ROOT / "dataset" / "linearrag" / "2wikifed"
OUT_SPLIT = _ROOT / "processed" / "2wikifed" / "split"


def main() -> None:
    # --- corpus: concat upfed shards, sanity-check the g%3 invariant --------
    chunks: list[str] = []
    bench_titles: set[str] = set()
    for cid in range(NUM_CLIENTS):
        shard = json.loads((UPFED / f"client_{cid}" / "chunks.json").read_text())
        for c in shard:
            g = int(str(c).split(":", 1)[0])
            assert g % NUM_CLIENTS == cid, f"chunk {g} not owned by client {cid}"
            bench_titles.add(norm_passage_title(c))
        chunks.extend(shard)
    print(f"corpus: {len(chunks)} chunks, {len(bench_titles)} titles")

    # --- test block: the upfed bench questions, same order ------------------
    bench_questions = json.loads((UPFED / "questions.json").read_text())
    test_per_client = [
        [q for j, q in enumerate(bench_questions) if j % NUM_CLIENTS == cid]
        for cid in range(NUM_CLIENTS)
    ]
    test_ids = {q["id"] for q in bench_questions}

    # --- train/val: eligible train-split questions of the full dataset ------
    full = json.loads((_ROOT / "processed" / "2wikimultihop" / "questions.json").read_text())
    train_end = 144024
    per_client_pool: list[list[dict]] = [[] for _ in range(NUM_CLIENTS)]
    need = TRAIN_PER_CLIENT + VAL_PER_CLIENT
    for gi in range(train_end):
        q = full[gi]
        if q["id"] in test_ids:
            continue
        gt = {norm_passage_title(t) for t, _ in q.get("evidence", [])}
        if not gt or not gt <= bench_titles:
            continue
        cid = gi % NUM_CLIENTS
        if len(per_client_pool[cid]) < need:
            per_client_pool[cid].append(q)
        if all(len(p) >= need for p in per_client_pool):
            break
    avail = min(len(p) for p in per_client_pool)
    assert avail > VAL_PER_CLIENT + 50, f"too few eligible train questions per client: {avail}"
    train_per_client = min(TRAIN_PER_CLIENT, avail - VAL_PER_CLIENT)
    need = train_per_client + VAL_PER_CLIENT
    print(f"eligible per client: {[len(p) for p in per_client_pool]} -> "
          f"train {train_per_client} + val {VAL_PER_CLIENT} each")

    def interleave(block_per_client: list[list[dict]]) -> list[dict]:
        n = len(block_per_client[0])
        return [block_per_client[cid][j] for j in range(n) for cid in range(NUM_CLIENTS)]

    train_block = interleave([p[:train_per_client] for p in per_client_pool])
    val_block = interleave([p[train_per_client:need] for p in per_client_pool])
    test_block = interleave(test_per_client)
    questions = train_block + val_block + test_block
    # strip helper keys; keep the canonical fields
    questions = [
        {k: q[k] for k in ("id", "source", "question", "answer", "question_type", "evidence") if k in q}
        for q in questions
    ]
    n_train, n_val, n_test = len(train_block), len(val_block), len(test_block)
    print(f"questions: train {n_train} + val {n_val} + test {n_test} = {len(questions)}")

    OUT_LINEARRAG.mkdir(parents=True, exist_ok=True)
    (OUT_LINEARRAG / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (OUT_LINEARRAG / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))

    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    (OUT_SPLIT / "train_indices.txt").write_text("\n".join(map(str, range(0, n_train))) + "\n")
    (OUT_SPLIT / "val_indices.txt").write_text(
        "\n".join(map(str, range(n_train, n_train + n_val))) + "\n")
    (OUT_SPLIT / "test_indices.txt").write_text(
        "\n".join(map(str, range(n_train + n_val, len(questions)))) + "\n")
    print(f"wrote {OUT_LINEARRAG} and split files in {OUT_SPLIT}")


if __name__ == "__main__":
    main()
