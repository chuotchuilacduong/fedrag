from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_PATH = "dataset/fedcond_qa"


class FedCondQADataset(Dataset):
    """Cached QA dataset for FedCondGraphRAG dual graph prompting.

    Works with any QA dataset preprocessed into the FedCond cache format.

    Expected files under ``root``:
    - ``records.jsonl`` / ``manifest.jsonl`` / ``dataset.jsonl``
    - evidence graphs in ``cached_graphs/{id}.pt`` or ``evidence_graphs/{id}.pt``
    - condensed graphs in ``cached_condensed_graphs/{id}.pt`` or ``condensed_graphs/{id}.pt``
    - optional descriptions in ``cached_desc/{id}.txt``
    - optional split files in ``split/{train,val,test}_indices.txt``
    """

    def __init__(self, root: str | os.PathLike | None = None):
        super().__init__()
        self.root = Path(root or os.environ.get("FEDCOND_QA_PATH", DEFAULT_PATH))
        self.prompt = "Please answer the given question using the retrieved passages and graphs."
        self.graph = None
        self.graph_type = "Tri-Graph + Condensed Graph"
        self.records = self._load_records()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        record_id = str(record.get("id", record.get("_id", index)))
        graph = self._load_graph(record_id, "cached_graphs", "evidence_graphs")
        condensed_graph = self._load_graph(record_id, "cached_condensed_graphs", "condensed_graphs")
        desc = self._load_description(record_id, record)
        question_text = record.get("question", "")
        answer = record.get("answer", record.get("label", ""))
        if isinstance(answer, list):
            answer = "|".join(str(item) for item in answer)

        return {
            "id": record_id,
            "question": f"Question: {question_text}\nAnswer: ",
            "label": str(answer).lower(),
            "graph": graph,
            "evidence_graph": graph,
            "condensed_graph": condensed_graph,
            "desc": desc,
            "retrieved_passages": record.get("retrieved_passages", []),
        }

    def get_idx_split(self):
        split_dir = self.root / "split"
        paths = {
            "train": split_dir / "train_indices.txt",
            "val": split_dir / "val_indices.txt",
            "test": split_dir / "test_indices.txt",
        }
        if all(path.exists() for path in paths.values()):
            return {name: self._read_indices(path) for name, path in paths.items()}

        n = len(self.records)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        return {
            "train": list(range(0, train_end)),
            "val": list(range(train_end, val_end)),
            "test": list(range(val_end, n)),
        }

    def _load_records(self):
        for filename in ("records.jsonl", "manifest.jsonl", "dataset.jsonl"):
            path = self.root / filename
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]
        raise FileNotFoundError(
            f"No FedCondQA manifest found under {self.root}. "
            "Expected records.jsonl, manifest.jsonl, or dataset.jsonl."
        )

    def _load_graph(self, record_id: str, *directories: str):
        candidates = []
        for directory in directories:
            candidates.append(self.root / directory / f"{record_id}.pt")
        if record_id.isdigit():
            for directory in directories:
                candidates.append(self.root / directory / f"{int(record_id)}.pt")
        for path in candidates:
            if path.exists():
                return torch.load(path, map_location="cpu")
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing cached graph for id={record_id}. Tried: {tried}")

    def _load_description(self, record_id: str, record: dict) -> str:
        desc_path = self.root / "cached_desc" / f"{record_id}.txt"
        if desc_path.exists():
            return desc_path.read_text(encoding="utf-8")
        if record_id.isdigit():
            numeric_desc_path = self.root / "cached_desc" / f"{int(record_id)}.txt"
            if numeric_desc_path.exists():
                return numeric_desc_path.read_text(encoding="utf-8")

        passages = record.get("retrieved_passages", [])
        if passages:
            return "\n\nRetrieved passages:\n" + "\n".join(str(passage) for passage in passages)
        return str(record.get("desc", ""))

    def _read_indices(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return [int(line.strip()) for line in handle if line.strip()]
