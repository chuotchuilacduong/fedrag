"""Build passage embeddings and passage→graph-node mapping for FedCondQA.

Outputs (under dataset/fedcond_qa/):
    passage_embs.pt       [N, 10, 384]  float16  — MiniLM embeddings for each
                                                   of the 10 retrieved passages
                                                   per record.
    passage_node_map.pt   [N, 10, 2]    int32    — (client_id, node_id) for each
                                                   passage, or (-1, -1) when no
                                                   matching trigraph node was
                                                   found.

Both artefacts use the title prefix to match a record passage against the
trigraph passage nodes (node_type=2) — the trigraph stores passages as
"<seq>:Title: body" while the record stores "Title: body".

This script is single-shot: rerun only if the trigraphs or records change.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


_TITLE_PREFIX_RE = re.compile(r"^\d+:")


def normalise_title(text: str) -> str:
    text = str(text)
    text = _TITLE_PREFIX_RE.sub("", text, count=1)
    head, _, _ = text.partition(":")
    return head.strip().lower()


def build_global_title_map(processed_root: Path, num_clients: int) -> dict[str, tuple[int, int]]:
    """title → (client_id, node_id) for every passage node in any client trigraph."""
    title_map: dict[str, tuple[int, int]] = {}
    for cid in range(num_clients):
        tg_path = processed_root / f"client_{cid}" / "trigraph.pt"
        print(f"  loading {tg_path}", flush=True)
        g = torch.load(tg_path, map_location="cpu", weights_only=False)
        psg_mask = g["node_type"] == 2
        idxs = psg_mask.nonzero().flatten().tolist()
        node_texts = g["node_text"]
        for i in idxs:
            title = normalise_title(node_texts[i])
            if title and title not in title_map:
                title_map[title] = (cid, i)
        print(f"    client_{cid}: cumulative unique titles = {len(title_map)}", flush=True)
    return title_map


def load_records(records_path: Path) -> list[dict]:
    out: list[dict] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def encode_passages(
    records: list[dict],
    encoder: SentenceTransformer,
    batch_size: int = 256,
    device: str = "cuda",
    max_slots: int = 30,
) -> torch.Tensor:
    """Returns [N, max_slots, 384] float16 passage embeddings.

    Records missing the kth passage are zero-padded (kept as zeros).
    """
    n = len(records)
    out = torch.zeros(n, max_slots, encoder.get_sentence_embedding_dimension(), dtype=torch.float16)

    # Flatten all passages so the encoder can batch them efficiently.
    texts: list[str] = []
    positions: list[tuple[int, int]] = []      # (record_idx, passage_idx) for each text
    for i, rec in enumerate(records):
        passages = rec.get("retrieved_passages", []) or []
        for k, p in enumerate(passages[:max_slots]):
            texts.append(str(p))
            positions.append((i, k))

    print(f"  encoding {len(texts)} passages...", flush=True)
    embs = encoder.encode(
        texts,
        batch_size=batch_size,
        device=device,
        show_progress_bar=True,
        convert_to_numpy=False,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    embs = embs.to(torch.float16).cpu()

    for (i, k), e in zip(positions, embs):
        out[i, k] = e
    return out


def build_node_map(
    records: list[dict],
    title_map: dict[str, tuple[int, int]],
    max_slots: int = 30,
) -> torch.Tensor:
    """Returns [N, max_slots, 2] int32 — (client_id, node_id) or (-1, -1)."""
    n = len(records)
    out = torch.full((n, max_slots, 2), -1, dtype=torch.int32)
    n_hits, n_total = 0, 0
    for i, rec in enumerate(tqdm(records, desc="  mapping passages → nodes")):
        passages = rec.get("retrieved_passages", []) or []
        for k, p in enumerate(passages[:max_slots]):
            title = normalise_title(p)
            hit = title_map.get(title)
            n_total += 1
            if hit is not None:
                out[i, k, 0] = hit[0]
                out[i, k, 1] = hit[1]
                n_hits += 1
    print(f"  passage → node hits: {n_hits}/{n_total} = {100*n_hits/max(n_total,1):.2f}%", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", default="dataset/fedcond_qa")
    parser.add_argument("--processed-root", default="processed/hotpotqa")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-embs", action="store_true", help="Skip passage embedding build")
    parser.add_argument("--skip-map", action="store_true", help="Skip passage→node map build")
    parser.add_argument("--max-slots", type=int, default=30,
                        help="Max passages per record to map (default 30 = 10 per client × 3 clients)")
    args = parser.parse_args()

    qa_root = Path(args.qa_root)
    processed_root = Path(args.processed_root)
    records_path = qa_root / "records.jsonl"

    print(f"[1/3] loading records from {records_path}", flush=True)
    records = load_records(records_path)
    print(f"    loaded {len(records)} records", flush=True)

    if not args.skip_map:
        print("[2/3] building global title → node map", flush=True)
        title_map = build_global_title_map(processed_root, args.num_clients)
        node_map = build_node_map(records, title_map, max_slots=args.max_slots)
        out_map = qa_root / "passage_node_map.pt"
        torch.save(node_map, out_map)
        print(f"    saved {out_map}  ({tuple(node_map.shape)})", flush=True)
    else:
        print("[2/3] skipped (--skip-map)", flush=True)

    if not args.skip_embs:
        print(f"[3/3] encoding passages with {args.encoder}", flush=True)
        encoder = SentenceTransformer(args.encoder, device=args.device)
        embs = encode_passages(records, encoder, batch_size=args.batch_size,
                               device=args.device, max_slots=args.max_slots)
        out_embs = qa_root / "passage_embs.pt"
        torch.save(embs, out_embs)
        print(f"    saved {out_embs}  ({tuple(embs.shape)}, {embs.dtype})", flush=True)
    else:
        print("[3/3] skipped (--skip-embs)", flush=True)

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
