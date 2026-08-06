"""Build a per-question desc jsonl from a baseline's retrieval predictions.

Converts the retrieval output of the HippoRAG / LinearRAG baseline runners
into the {"id", "desc"} jsonl consumed by `fl-train --desc-source file
--desc-file <out>` — the "swap the retriever, keep the reader" ablation.

  hipporag : output/baselines/hipporag_local/<dataset>/<split>/client_*/predictions.jsonl
             desc = top-K entries of the "docs" field ("Title\nBody" texts)
  linearrag: output/baselines/linearrag_local/<dataset>/client_*/predictions.jsonl
             desc = top-K entries of "retrieved_passages" ("<g>:<title>: <body>"
             chunk strings; the numeric global-id prefix is stripped, matching
             the PPR desc format)

Usage:
  python scripts/build_desc_from_baseline_retrieval.py --source linearrag \
      --dataset 2wikifed --out dataset/fedcond_qa/2wikifed/desc_linearrag.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PREFIX_RE = re.compile(r"^\d+:\s*")

TOP_K_DESC = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["hipporag", "linearrag"])
    parser.add_argument("--dataset", default="2wikifed")
    parser.add_argument("--split", default="all",
                        help="hipporag output split subdir (default: all)")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=TOP_K_DESC)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.source == "hipporag":
        base = _ROOT / "output" / "baselines" / "hipporag_local" / args.dataset / args.split
        field = "docs"
    else:
        base = _ROOT / "output" / "baselines" / "linearrag_local" / args.dataset
        field = "retrieved_passages"

    rows: dict[str, str] = {}
    for cid in range(args.num_clients):
        pred_path = base / f"client_{cid}" / "predictions.jsonl"
        if not pred_path.exists():
            print(f"ERROR: {pred_path} not found")
            sys.exit(1)
        n = 0
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                passages = [str(p) for p in r.get(field, [])][: args.top_k]
                if args.source == "linearrag":
                    passages = [_PREFIX_RE.sub("", p, count=1).strip() for p in passages]
                rid = str(r["id"])
                if rid in rows:
                    print(f"ERROR: duplicate question id {rid} (client_{cid})")
                    sys.exit(1)
                rows[rid] = "\n\n".join(passages)
                n += 1
        print(f"client_{cid}: {n} questions from {pred_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid, desc in rows.items():
            f.write(json.dumps({"id": rid, "desc": desc}, ensure_ascii=False) + "\n")
    empty = sum(1 for d in rows.values() if not d)
    print(f"Wrote {len(rows)} descs ({empty} empty) -> {out_path}")


if __name__ == "__main__":
    main()
