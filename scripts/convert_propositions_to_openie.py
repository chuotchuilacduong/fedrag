"""Convert PropRAG proposition extractions into HippoRAG OpenIE format.

PropRAG's released files (dataset/ner/<ds>/openie_results_ner_*.json, extracted
with Llama-3.3-70B) carry per-passage `propositions` [{text, entities}] but an
empty `extracted_triples` list, so HippoRAG builds a graph with zero facts and
degrades to plain DPR. This script synthesizes (subject, predicate, object)
facts from the propositions:

- k >= 2 entities: one triple per consecutive entity pair, with the full
  proposition sentence as the predicate — the graph keeps its entity->passage
  structure and the fact text is even richer than a terse relation.
- k == 1 entity: a self-triple (e, text, e) so the entity still becomes a
  phrase node reachable by fact-based retrieval.

The conversion is question-blind (per-passage only), so unlike the dataset's
gold `evidences` triples it introduces no eval leakage. Label results as
"HippoRAG 2 (proposition facts, 70B)" — the fact construction differs from
upstream's LLM triple extraction.

Usage:
  python scripts/convert_propositions_to_openie.py \
    --src dataset/ner/2wikimultihopqa/openie_results_ner_meta-llama_llama-3.3-70b-instruct.json \
    --dst dataset/ner/2wikimultihopqa/openie_results_ner_propfacts_llama70b.json
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    payload = json.load(open(args.src))
    docs = payload["docs"] if isinstance(payload, dict) else payload

    n_triples = 0
    n_props = 0
    for doc in docs:
        triples: list[list[str]] = []
        for prop in doc.get("propositions", []) or []:
            text = str(prop.get("text", "")).strip()
            ents = [str(e).strip() for e in (prop.get("entities") or []) if str(e).strip()]
            if not text or not ents:
                continue
            n_props += 1
            if len(ents) == 1:
                triples.append([ents[0], text, ents[0]])
            else:
                for a, b in zip(ents, ents[1:]):
                    triples.append([a, text, b])
        doc["extracted_triples"] = triples
        n_triples += len(triples)

    out = payload if isinstance(payload, dict) else {"docs": docs}
    out["docs"] = docs
    json.dump(out, open(args.dst, "w"), ensure_ascii=False)
    print(f"{len(docs)} docs, {n_props} propositions -> {n_triples} synthesized triples "
          f"(avg {n_triples / max(len(docs), 1):.1f}/doc). Wrote {args.dst}")


if __name__ == "__main__":
    main()
