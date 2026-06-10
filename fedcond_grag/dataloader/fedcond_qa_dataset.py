from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_PATH = "dataset/fedcond_qa"


class FedCondQADataset(Dataset):
    """Cached QA dataset for FedCondGraphRAG dual graph prompting.

    Expected files under ``root``:
    - ``records.jsonl`` / ``manifest.jsonl`` / ``dataset.jsonl``
    - ``q_embs.pt``  — [Q, 384] question embeddings for on-the-fly retrieval
    - optional split files in ``split/{train,val,test}_indices.txt``

    Evidence graphs are retrieved on-the-fly from each client's local
    trigraph during training (no pre-built cached_graphs/ needed).
    """

    def __init__(self, root: str | os.PathLike | None = None,
                 top_r_passages: int = 0, top_r_anchor: int | None = None):
        super().__init__()
        self.root = Path(root or os.environ.get("FEDCOND_QA_PATH", DEFAULT_PATH))
        self.prompt = "Please answer the given question using the retrieved passages and graphs."
        self.graph = None
        self.graph_type = "Tri-Graph + Condensed Graph"
        self.records = self._load_records()
        # 0 disables re-ranking and keeps legacy behaviour; >0 keeps the top-r
        # passages by cos-sim(q_emb, passage_emb) and exposes anchor node ids.
        self.top_r_passages = int(top_r_passages)
        # Cap on number of passages used as graph subgraph anchors. Defaults to
        # top_r_passages, but you usually want fewer anchors than passages —
        # all 10 passages as text + only the top 3 as graph anchors avoids
        # building huge subgraphs while keeping text coverage high.
        self.top_r_anchor = int(top_r_anchor) if top_r_anchor is not None else self.top_r_passages

        q_embs_path = self.root / "q_embs.pt"
        if q_embs_path.exists():
            self.q_embs: torch.Tensor | None = torch.load(
                q_embs_path, map_location="cpu", weights_only=True
            )
        else:
            self.q_embs = None

        # Per-client PPR node maps are loaded by FedCondQAClient directly from
        # processed/{dataset}/client_{c}/ppr_node_map.pt — not by this dataset.

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        record_id = str(record.get("id", record.get("_id", index)))
        question_text = record.get("question", "")
        answer = record.get("answer", record.get("label", ""))
        if isinstance(answer, list):
            answer = "|".join(str(item) for item in answer)

        passages = record.get("retrieved_passages", []) or []

        desc = self._load_description(record_id, record)

        item: dict = {
            "idx": index,          # dataset integer index — used by client to look up ppr_node_map
            "id": record_id,
            "question": f"Question: {question_text}\nAnswer: ",
            "label": str(answer).lower(),
            "desc": desc,
            "retrieved_passages": passages,
        }
        if self.q_embs is not None:
            item["q_emb"] = self.q_embs[index]   # [384] — used for on-the-fly retrieval
        return item

    def get_idx_split(self) -> dict[str, list[int]]:
        split_dir = self.root / "split"
        paths = {
            "train": split_dir / "train_indices.txt",
            "val":   split_dir / "val_indices.txt",
            "test":  split_dir / "test_indices.txt",
        }
        if all(path.exists() for path in paths.values()):
            return {name: self._read_indices(path) for name, path in paths.items()}

        n = len(self.records)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)
        return {
            "train": list(range(0, train_end)),
            "val":   list(range(train_end, val_end)),
            "test":  list(range(val_end, n)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        for filename in ("records.jsonl", "manifest.jsonl", "dataset.jsonl"):
            path = self.root / filename
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]
        raise FileNotFoundError(
            f"No FedCondQA manifest found under {self.root}. "
            "Expected records.jsonl, manifest.jsonl, or dataset.jsonl."
        )

    def _load_description(self, record_id: str, record: dict) -> str:
        desc_path = self.root / "cached_desc" / f"{record_id}.txt"
        if desc_path.exists():
            return desc_path.read_text(encoding="utf-8")
        # Prefer the pre-built desc (clean MuSiQue evidence paragraphs) over
        # retrieved_passages, which is a noisy 20-passage BM25 dump full of distractors.
        desc = str(record.get("desc", "")).strip()
        if desc:
            return desc
        passages = record.get("retrieved_passages", [])
        if passages:
            return "\n\nRetrieved passages:\n" + "\n".join(str(p) for p in passages)
        return ""

    def _read_indices(self, path: Path) -> list[int]:
        with path.open("r", encoding="utf-8") as handle:
            return [int(line.strip()) for line in handle if line.strip()]
