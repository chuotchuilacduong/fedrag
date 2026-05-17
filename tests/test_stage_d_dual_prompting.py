import json
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Data

from fedcond_grag.dataloader.fedcond_qa_dataset import FedCondQADataset
from fedcond_grag.model import load_model
from fedcond_grag.model.dual_graph_llm import DualGraphLLM
from fedcond_grag.utils.collate import collate_fn


class IdentityGraphEncoder(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        return x, edge_attr


def _graph(x_offset: float = 0.0) -> Data:
    x = torch.tensor([[1.0 + x_offset, 0.0], [0.0, 1.0 + x_offset], [1.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1), 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _minimal_dual_model(mode: str = "both") -> DualGraphLLM:
    model = DualGraphLLM.__new__(DualGraphLLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(device=torch.device("cpu"))
    model.graph_encoder = IdentityGraphEncoder()
    model.condensed_encoder = IdentityGraphEncoder()
    model.projector = nn.Identity()
    model.projector_c = nn.Identity()
    model.dual_graph_mode = mode
    return model


def test_dual_graph_llm_registered():
    assert load_model["dual_graph_llm"] is DualGraphLLM


def test_dual_graph_encoding_returns_two_prompt_tokens():
    batch = collate_fn(
        [
            {"graph": _graph(0.0), "condensed_graph": _graph(1.0), "id": "a"},
            {"graph": _graph(2.0), "condensed_graph": _graph(3.0), "id": "b"},
        ]
    )
    model = _minimal_dual_model()

    z_e, z_c = model.encode_graphs(batch)

    assert z_e.shape == (2, 2)
    assert z_c.shape == (2, 2)
    assert not torch.allclose(z_e, z_c)


def test_dual_graph_ablation_modes_mask_expected_token():
    batch = collate_fn([{"graph": _graph(), "condensed_graph": _graph(1.0), "id": "a"}])

    z_e, z_c = _minimal_dual_model("evidence_only").encode_graphs(batch)
    assert torch.count_nonzero(z_e) > 0
    assert torch.count_nonzero(z_c) == 0

    z_e, z_c = _minimal_dual_model("condensed_only").encode_graphs(batch)
    assert torch.count_nonzero(z_e) == 0
    assert torch.count_nonzero(z_c) > 0


def test_collate_batches_evidence_and_condensed_graphs():
    batch = collate_fn(
        [
            {"graph": _graph(), "evidence_graph": _graph(), "condensed_graph": _graph(1.0), "id": "a"},
            {"graph": _graph(), "evidence_graph": _graph(), "condensed_graph": _graph(2.0), "id": "b"},
        ]
    )

    assert batch["graph"].num_graphs == 2
    assert batch["evidence_graph"].num_graphs == 2
    assert batch["condensed_graph"].num_graphs == 2


def test_fedcond_qa_dataset_loads_cached_dual_graphs(tmp_path):
    root = tmp_path / "fedcond_qa"
    (root / "cached_graphs").mkdir(parents=True)
    (root / "cached_condensed_graphs").mkdir()
    (root / "cached_desc").mkdir()
    (root / "split").mkdir()

    with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "q1", "question": "Who wrote the book?", "answer": "Ada", "retrieved_passages": ["p1"]}) + "\n")
    torch.save(_graph(), root / "cached_graphs" / "q1.pt")
    torch.save(_graph(1.0), root / "cached_condensed_graphs" / "q1.pt")
    (root / "cached_desc" / "q1.txt").write_text("Evidence graph text", encoding="utf-8")
    (root / "split" / "train_indices.txt").write_text("0\n", encoding="utf-8")
    (root / "split" / "val_indices.txt").write_text("0\n", encoding="utf-8")
    (root / "split" / "test_indices.txt").write_text("0\n", encoding="utf-8")

    dataset = FedCondQADataset(root=root)
    item = dataset[0]

    assert item["id"] == "q1"
    assert item["question"].startswith("Question: Who wrote")
    assert item["label"] == "ada"
    assert item["graph"].num_nodes == 3
    assert item["evidence_graph"].num_nodes == 3
    assert item["condensed_graph"].num_nodes == 3
    assert dataset.get_idx_split() == {"train": [0], "val": [0], "test": [0]}
