"""Smoke test for the FedCondGraphRAG package imports."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fedcond_grag import load_client, load_server, load_task
from fedcond_grag.cli import main as cli_main
from fedcond_grag.client.client import FedCondQAClient
from fedcond_grag.client.stage_a_trigraph import build_trigraph_for_client
from fedcond_grag.client.stage_b_condense import ClientCondensor
from fedcond_grag.client.stage_d_retrieve import (
    EvidenceLinearRAG,
    GlobalGraphRetriever,
    LinearRAGRetriever,
    build_evidence_graph,
)
from fedcond_grag.dataloader import FedCondQADataset, load_dataset, load_hotpot, load_linearrag
from fedcond_grag.linearrag import LinearRAG, LinearRAGConfig
from fedcond_grag.model import DualGraphLLM, GraphLLM, llama_model_path, load_model
from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE
from fedcond_grag.server.stage_c_aggregate.surrogate import SurrogateGNN
from fedcond_grag.trainer import FedTrainer


def main() -> None:
    for obj in (
        load_client, load_server, load_task,
        cli_main,
        FedCondQAClient, FedCondQAServer,
        build_trigraph_for_client, ClientCondensor,
        EvidenceLinearRAG, GlobalGraphRetriever, LinearRAGRetriever, build_evidence_graph,
        FedCondQADataset, load_hotpot, load_linearrag,
        LinearRAG, LinearRAGConfig,
        DualGraphLLM, GraphLLM, llama_model_path, load_model,
        TypeAwarePGE, SurrogateGNN, FedTrainer,
        load_dataset,
    ):
        assert obj is not None, obj
    print("All FedCondGraphRAG imports OK.")


if __name__ == "__main__":
    main()
