"""Build `2wikifed_xl`: 2wikifed with a 10x larger train split.

Appends up to --num-extra train-region questions from the full 2wikimultihop
dataset (disjoint from 2wikifed's existing 2001 questions) whose gold titles
are AT LEAST PARTIALLY covered by the bench corpus, sorted by coverage ratio
so the best-grounded questions come first. Only 48 questions have 100%
coverage left, so partial coverage is required for any real expansion — which
matches the federated reality anyway (per-shard oracle recall is ~33% even
for fully-covered questions).

Layout: processed/2wikifed_xl/client_*/ symlinks every heavy artifact from
processed/2wikifed (chunks, trigraph, condensed/synthetic graphs, text bank,
linearrag_cache) and gets its own questions.json + split; val/test indices
are byte-identical to 2wikifed so results stay comparable.

Appended questions are interleaved c0,c1,c2 starting at index 2001 (2001%3==0)
so the idx % 3 ownership rule holds.

After this script:
  python scripts/build_fedcond_qa_dataset.py --dataset 2wikifed_xl --out-root dataset/fedcond_qa/2wikifed_xl
  cp processed/2wikifed_xl/split/*.txt dataset/fedcond_qa/2wikifed_xl/split/
  python scripts/preprocess_fedcond_qa.py --dataset 2wikifed_xl
  python main.py fl-train --dataset 2wikifed_xl --qa-data-root dataset/fedcond_qa/2wikifed_xl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.utils.evaluate import norm_passage_title  # noqa: E402

SRC = _ROOT / "processed" / "2wikifed"
DST = _ROOT / "processed" / "2wikifed_xl"
FULL_TRAIN_REGION = 144024
NUM_CLIENTS = 3
LINK_ARTIFACTS = [
    "chunks.json", "trigraph.pt", "condensed_graph.pt", "synthetic_graph.pt",
    "text_bank.pt", "linearrag_cache",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-extra", type=int, default=9000,
                        help="Appended train questions (multiple of 3).")
    args = parser.parse_args()
    num_extra = args.num_extra - (args.num_extra % NUM_CLIENTS)

    base_questions = json.loads((SRC / "questions.json").read_text())
    n_base = len(base_questions)
    assert n_base % NUM_CLIENTS == 0, f"base question count {n_base} not divisible by 3"
    used_ids = {str(q.get("id")) for q in base_questions}

    titles: set[str] = set()
    for cid in range(NUM_CLIENTS):
        for c in json.loads((SRC / f"client_{cid}" / "chunks.json").read_text()):
            t = str(c).split(":", 2)[1]
            titles.add(norm_passage_title(t + ": x"))
    print(f"bench corpus titles: {len(titles)}")

    full_q = json.loads((_ROOT / "processed" / "2wikimultihop" / "questions.json").read_text())
    candidates: list[tuple[float, int, dict]] = []
    for i, q in enumerate(full_q):
        if i >= FULL_TRAIN_REGION:
            break
        if str(q.get("id")) in used_ids:
            continue
        golds = {norm_passage_title(str(t) + ": x")
                 for t, _ in q.get("evidence", []) if t}
        if not golds:
            continue
        k = len(golds & titles)
        if k == 0:
            continue
        candidates.append((k / len(golds), k, q))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    extra = [q for _, _, q in candidates[:num_extra]]
    print(f"candidates: {len(candidates)}, taking {len(extra)} (best coverage first)")

    questions = base_questions + extra

    # --- write processed/2wikifed_xl -------------------------------------
    DST.mkdir(exist_ok=True)
    (DST / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))
    for cid in range(NUM_CLIENTS):
        cdir = DST / f"client_{cid}"
        cdir.mkdir(exist_ok=True)
        for name in LINK_ARTIFACTS:
            link = cdir / name
            target = SRC / f"client_{cid}" / name
            if not target.exists():
                print(f"WARN: missing {target}")
                continue
            if not link.exists():
                link.symlink_to(target.resolve())

    # --- splits: same val/test as 2wikifed, train = old + appended -------
    src_split = SRC / "split"
    dst_split = DST / "split"
    dst_split.mkdir(exist_ok=True)
    old = {name: [int(l) for l in (src_split / f"{name}_indices.txt").read_text().splitlines() if l.strip()]
           for name in ("train", "val", "test")}
    new_train = old["train"] + list(range(n_base, n_base + len(extra)))
    (dst_split / "train_indices.txt").write_text("\n".join(map(str, new_train)))
    (dst_split / "val_indices.txt").write_text("\n".join(map(str, old["val"])))
    (dst_split / "test_indices.txt").write_text("\n".join(map(str, old["test"])))

    print(f"2wikifed_xl: {len(questions)} questions "
          f"(train {len(new_train)}, val {len(old['val'])}, test {len(old['test'])})")
    per_client = [sum(1 for i in new_train if i % NUM_CLIENTS == c) for c in range(NUM_CLIENTS)]
    print(f"train per client: {per_client}")


if __name__ == "__main__":
    main()
