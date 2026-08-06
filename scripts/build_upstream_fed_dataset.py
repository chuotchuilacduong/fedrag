"""Build `musiquefed` / `hotpotqafed`: FedCondGraphRAG on an upstream bench.

Generalisation of build_2wikifed_dataset.py to the musique / hotpotqa upfed
benches. One difference matters: those upfed shards do NOT satisfy the
"global chunk index % 3 == client_id" invariant (the corpus was shuffled and
split contiguously by build_upstream_fed_bench.py), while fedrag's
partition_linearrag_chunks assigns chunks by index % num_clients. So chunks
are RE-PREFIXED here: the j-th chunk of upfed client c becomes "3*j+c:..." —
partitioning the concatenated file by idx % 3 then reproduces the upfed
shards exactly (passage text and per-client membership unchanged; only the
numeric prefix differs, which nothing downstream keys on except ownership).

Questions = [train | val | test], each block interleaved c0,c1,c2 so
question_idx % 3 == owning client:
  * test = the 999 upfed bench questions, upfed order;
  * train/val = train-region questions of the full dataset (disjoint ids)
    whose gold titles are best covered by the bench corpus. Fully-covered
    questions are preferred; if a client's pool runs short the best
    partially-covered ones fill in (matches the federated reality — per-shard
    oracle recall is ~1/3 anyway).

After this script (same recipe as 2wikifed):
  python scripts/preprocess_data.py --dataset <name> --num_clients 3
  python scripts/build_client_pipeline.py --dataset <name>
  python scripts/build_fedcond_qa_dataset.py --dataset <name> --out-root dataset/fedcond_qa/<name>
  cp processed/<name>/split/*.txt dataset/fedcond_qa/<name>/split/
  python scripts/preprocess_fedcond_qa.py --dataset <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402

NUM_CLIENTS = 3
TRAIN_PER_CLIENT = 310
VAL_PER_CLIENT = 33

SOURCES = {
    "musique": ("musique_upfed", "musiquefed"),
    "hotpotqa": ("hotpotqa_upfed", "hotpotqafed"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    args = parser.parse_args()
    upfed_name, out_name = SOURCES[args.dataset]
    upfed = _ROOT / "processed" / upfed_name

    # --- corpus: re-prefix each shard so idx % 3 == client_id ---------------
    chunks_per_client: list[list[str]] = []
    bench_titles: set[str] = set()
    for cid in range(NUM_CLIENTS):
        shard = json.loads((upfed / f"client_{cid}" / "chunks.json").read_text())
        renum = []
        for j, c in enumerate(shard):
            _, _, body = str(c).partition(":")
            renum.append(f"{NUM_CLIENTS * j + cid}:{body}")
            bench_titles.add(norm_passage_title(body))
        chunks_per_client.append(renum)
    max_len = max(len(s) for s in chunks_per_client)
    chunks = [chunks_per_client[cid][j]
              for j in range(max_len) for cid in range(NUM_CLIENTS)
              if j < len(chunks_per_client[cid])]
    print(f"corpus: {len(chunks)} chunks, {len(bench_titles)} titles, "
          f"per client {[len(s) for s in chunks_per_client]}")

    # --- test block: the upfed bench questions, same per-client order -------
    bench_questions = json.loads((upfed / "questions.json").read_text())
    assert len(bench_questions) % NUM_CLIENTS == 0
    test_per_client = [
        [q for j, q in enumerate(bench_questions) if j % NUM_CLIENTS == cid]
        for cid in range(NUM_CLIENTS)
    ]
    test_ids = {q["id"] for q in bench_questions}

    # --- train/val: best-covered train-region questions of the full dataset -
    full = json.loads((_ROOT / "processed" / args.dataset / "questions.json").read_text())
    train_idx = [int(l) for l in
                 (_ROOT / "dataset" / "fedcond_qa" / args.dataset / "split" / "train_indices.txt")
                 .read_text().splitlines() if l.strip()]
    train_end = max(train_idx) + 1
    print(f"full dataset: {len(full)} questions, train region 0..{train_end - 1}")

    need = TRAIN_PER_CLIENT + VAL_PER_CLIENT
    pools: list[list[tuple[float, int, dict]]] = [[] for _ in range(NUM_CLIENTS)]
    seen_ids = set(test_ids)  # the full datasets contain duplicate question ids
    for gi in range(train_end):
        q = full[gi]
        if q["id"] in seen_ids:
            continue
        seen_ids.add(q["id"])
        gt = {norm_passage_title(str(t) + ": x") for t, _ in q.get("evidence", []) if t}
        if not gt:
            continue
        k = len(gt & bench_titles)
        if k == 0:
            continue
        pools[gi % NUM_CLIENTS].append((k / len(gt), gi, q))
    for p in pools:
        p.sort(key=lambda x: (-x[0], x[1]))
    n_full_cov = [sum(1 for r, _, _ in p if r == 1.0) for p in pools]
    print(f"eligible per client: {[len(p) for p in pools]} "
          f"(fully covered: {n_full_cov}) -> need {need} each")
    avail = min(len(p) for p in pools)
    assert avail >= need, f"only {avail} eligible train questions per client (< {need})"
    picked = [[q for _, _, q in p[:need]] for p in pools]
    cov = [sum(r for r, _, _ in p[:need]) / need for p in pools]
    print(f"mean gold-title coverage of picked train/val: "
          f"{[f'{c:.2f}' for c in cov]}")

    def interleave(block_per_client: list[list[dict]]) -> list[dict]:
        n = len(block_per_client[0])
        return [block_per_client[cid][j] for j in range(n) for cid in range(NUM_CLIENTS)]

    train_block = interleave([p[:TRAIN_PER_CLIENT] for p in picked])
    val_block = interleave([p[TRAIN_PER_CLIENT:need] for p in picked])
    test_block = interleave(test_per_client)
    questions = [
        {k: q[k] for k in ("id", "source", "question", "answer", "question_type", "evidence") if k in q}
        for q in train_block + val_block + test_block
    ]
    n_train, n_val, n_test = len(train_block), len(val_block), len(test_block)
    print(f"questions: train {n_train} + val {n_val} + test {n_test} = {len(questions)}")

    out_lin = _ROOT / "dataset" / "linearrag" / out_name
    out_lin.mkdir(parents=True, exist_ok=True)
    (out_lin / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (out_lin / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))

    out_split = _ROOT / "processed" / out_name / "split"
    out_split.mkdir(parents=True, exist_ok=True)
    (out_split / "train_indices.txt").write_text("\n".join(map(str, range(0, n_train))) + "\n")
    (out_split / "val_indices.txt").write_text(
        "\n".join(map(str, range(n_train, n_train + n_val))) + "\n")
    (out_split / "test_indices.txt").write_text(
        "\n".join(map(str, range(n_train + n_val, len(questions)))) + "\n")
    print(f"wrote {out_lin} and split files in {out_split}")


if __name__ == "__main__":
    main()
