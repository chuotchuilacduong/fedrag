"""Build a cross-client FUSED desc file for the 2wikifed test set.

Simulates a global mechanism that lets a client see evidence retrieved on ALL
clients' shards: for each test question, the top passages of the three clients'
broadcast LinearRAG retrievals (output/baselines/linearrag_local_bcast) are
rank-interleaved (c0#1, c1#1, c2#1, c0#2, ...), deduplicated, and the top
--top-k become the desc. Evaluating the existing bench checkpoint with this
desc (fl-train --eval-only --desc-source file) upper-bounds what a
text-carrying global graph could add over single-shard retrieval.

Val/train questions (not covered by the broadcast run) fall back to the
single-owner desc from desc_linearrag.jsonl so --desc-source file never
crashes; only the TEST numbers of this eval are meaningful.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PREFIX_RE = re.compile(r"^\d+:\s*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--out", default="dataset/fedcond_qa/2wikifed/desc_fused.jsonl")
    args = parser.parse_args()

    bcast = _ROOT / "output" / "baselines" / "linearrag_local_bcast" / "2wikifed"
    per_client: list[dict[str, list[str]]] = []
    for cid in range(args.num_clients):
        rows: dict[str, list[str]] = {}
        with (bcast / f"client_{cid}" / "predictions.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows[str(r["id"])] = [
                    _PREFIX_RE.sub("", str(p), count=1).strip()
                    for p in r.get("retrieved_passages", [])
                ]
        per_client.append(rows)
        print(f"client_{cid}: {len(rows)} broadcast retrievals")

    test_ids = set(per_client[0])
    for rows in per_client[1:]:
        if set(rows) != test_ids:
            print("ERROR: client question-id sets differ")
            sys.exit(1)

    fused: dict[str, str] = {}
    for qid in test_ids:
        seen: set[str] = set()
        merged: list[str] = []
        max_rank = max(len(per_client[c][qid]) for c in range(args.num_clients))
        for rank in range(max_rank):
            for c in range(args.num_clients):
                plist = per_client[c][qid]
                if rank < len(plist) and plist[rank] not in seen:
                    seen.add(plist[rank])
                    merged.append(plist[rank])
        fused[qid] = "\n\n".join(merged[: args.top_k])

    # fallback for train/val ids so --desc-source file covers every record
    n_fallback = 0
    fallback_path = _ROOT / "dataset" / "fedcond_qa" / "2wikifed" / "desc_linearrag.jsonl"
    with fallback_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r["id"])
            if rid not in fused:
                fused[rid] = str(r.get("desc", ""))
                n_fallback += 1

    out_path = _ROOT / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for rid, desc in fused.items():
            f.write(json.dumps({"id": rid, "desc": desc}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(fused)} descs ({len(test_ids)} fused test, "
          f"{n_fallback} owner-desc fallback) -> {out_path}")


if __name__ == "__main__":
    main()
