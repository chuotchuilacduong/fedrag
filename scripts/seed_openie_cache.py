"""Seed baselines/hipporag and baselines/comorag's per-client OpenIE cache
from an externally precomputed `openie_results_ner_*.json` file (e.g. from
https://github.com/ReLink-Inc/PropRAG's `outputs/<dataset>/` -- verified to
cover the exact same corpus documents as this project's dataset/raw/*_corpus.json).

Both HippoRAG's and ComoRAG's `index()` build `ner_results_dict`/
`triple_results_dict` from the ENTIRE cache file's contents (not filtered to
the client's own chunk_keys) and then assert
`len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)`.
Seeding every client with the full global cache (all documents, every
client) makes that assertion fail for any client whose shard is a strict
subset of the corpus (i.e. every client, once num_clients > 1) -- and,
short of the assertion, would also leak entities/triples from documents
outside a client's own shard into its graph, defeating the point of the
"client only sees its own passages" baseline. Each client must be seeded
with ONLY the cache entries for its own passage shard (same `idx %
num_clients` partition as `_load_client_docs` in hipporag/comorag/flare's
own client_runner.py), matched by exact passage-text equality (both
HippoRAG's and ComoRAG's own cache-key recompute is a hash of `passage`).

Usage:
    python scripts/seed_openie_cache.py --dataset musique --cache-file /tmp/proprag_cache/musique_openie.json --num-clients 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fedcond_grag.baselines.hipporag.client_runner import DEFAULT_SAVE_ROOT as HIPPORAG_SAVE_ROOT
from fedcond_grag.baselines.hipporag.client_runner import HIPPO_DATASET_NAMES
from fedcond_grag.baselines.comorag.client_runner import DEFAULT_SAVE_ROOT as COMORAG_SAVE_ROOT

RAW_DIR = _ROOT / "dataset" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(HIPPO_DATASET_NAMES))
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--llm-name", default="qwen2.5:1.5b-instruct")
    args = parser.parse_args()

    cache_file = Path(args.cache_file)
    if not cache_file.is_file():
        raise FileNotFoundError(cache_file)

    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    docs_by_passage = {doc["passage"]: doc for doc in cache["docs"]}

    raw_name = HIPPO_DATASET_NAMES[args.dataset]
    corpus = json.loads((RAW_DIR / f"{raw_name}_corpus.json").read_text(encoding="utf-8"))

    # HippoRAG (pip package) keeps ':' as-is; ComoRAG (vendored) replaces both
    # '/' and ':' with '_' -- see each package's own openie_results_path build.
    hippo_name = f"openie_results_ner_{args.llm_name.replace('/', '_')}.json"
    comorag_name = f"openie_results_ner_{args.llm_name.replace('/', '_').replace(':', '_')}.json"

    for client_id in range(args.num_clients):
        client_passages = [
            f"{item['title']}\n{item['text']}"
            for i, item in enumerate(corpus)
            if i % args.num_clients == client_id
        ]
        missing = [p for p in client_passages if p not in docs_by_passage]
        if missing:
            raise RuntimeError(
                f"client_{client_id}: {len(missing)}/{len(client_passages)} passages not found in "
                f"{cache_file} -- cache does not cover this corpus, aborting."
            )
        client_docs = [docs_by_passage[p] for p in client_passages]
        payload = json.dumps({"docs": client_docs})

        for save_root, filename in (
            (HIPPORAG_SAVE_ROOT, hippo_name),
            (COMORAG_SAVE_ROOT, comorag_name),
        ):
            dest_dir = Path(save_root) / args.dataset / f"client_{client_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            dest.write_text(payload, encoding="utf-8")
            print(f"Seeded {dest} ({len(client_docs)} docs)")


if __name__ == "__main__":
    main()
