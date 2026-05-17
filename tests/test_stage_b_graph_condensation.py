import torch
from torch_geometric.data import Data

from fedcond_grag.client.stage_b_condense import (
    ClientCondensationConfig,
    ClientCondensor,
    MotifSelectorConfig,
    TextBank,
    select_motif_core,
)
from fedcond_grag.client.stage_b_condense.chunk_selection import topk_softmax
from fedcond_grag.client.stage_b_condense.graph_text_fusion import GraphTextFusion
from fedcond_grag.client.stage_b_condense.neighbor_gating import hierarchical_text_condensation, score_and_select
from fedcond_grag.client.stage_b_condense.text_bank import HashTextEncoder, build_text_bank, load_text_bank, save_text_bank
from fedcond_grag.client.stage_b_condense.topology_reconstruction import (
    ENTITY,
    PASSAGE,
    SENTENCE,
    knn_topology,
    self_expressive_topology,
)


def _toy_trigraph(dim=16):
    # Nodes: E0, E1, S0, S1, S2, P0, P1, P2
    node_type = torch.tensor([ENTITY, ENTITY, SENTENCE, SENTENCE, SENTENCE, PASSAGE, PASSAGE, PASSAGE])
    undirected_edges = [
        (2, 0), (3, 0), (5, 0), (6, 0),
        (3, 1), (4, 1), (6, 1), (7, 1),
    ]
    edges = []
    for src, dst in undirected_edges:
        edges.extend([(src, dst), (dst, src)])
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    torch.manual_seed(7)
    x = torch.randn(node_type.numel(), dim)
    return Data(x=x, edge_index=edge_index, node_type=node_type)


def _toy_text_bank(num_nodes=8, dim=16):
    torch.manual_seed(11)
    chunks = [torch.randn(3, dim) for _ in range(num_nodes)]
    node = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
    return TextBank(node_embeddings=node, chunk_embeddings=chunks, encoder_name="toy", dim=dim)


def test_topk_softmax_respects_budget():
    scores = torch.arange(10, dtype=torch.float32)
    weights = topk_softmax(scores, k=3)
    assert weights.shape == scores.shape
    assert int((weights > 0).sum()) == 3
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_neighbor_gating_and_chunk_selection_respect_budgets():
    graph = _toy_trigraph()
    bank = _toy_text_bank()
    t_tilde, contexts, traces = hierarchical_text_condensation(
        core_node_ids=[0, 2],
        edge_index=graph.edge_index,
        graph_embeddings=graph.x,
        node_text_embeddings=bank.node_embeddings,
        chunk_embeddings=bank.chunk_embeddings,
        budgets=(1, 3, 2),
        chunk_budget=4,
    )
    assert t_tilde.shape == (2, graph.x.size(1))
    assert set(contexts) == {0, 2}
    for trace in traces.values():
        assert len(trace.hops[0].node_ids) <= 1
        assert len(trace.hops[1].node_ids) <= 3
        assert len(trace.hops[2].node_ids) <= 2
        assert trace.chunks.weights.numel() <= 4


def test_text_bank_encoder_frozen_and_cache_roundtrip(tmp_path):
    encoder = HashTextEncoder(dim=16)
    bank = build_text_bank(["alpha beta", "gamma delta epsilon"], encoder=encoder, dim=16)
    assert sum(p.requires_grad for p in encoder.parameters()) == 0
    assert bank.node_embeddings.shape == (2, 16)
    assert len(bank.chunk_embeddings) == 2

    path = tmp_path / "text_bank.pt"
    save_text_bank(bank, path)
    loaded = load_text_bank(path)
    assert torch.allclose(loaded.node_embeddings, bank.node_embeddings)
    assert loaded.dim == 16


def test_motif_selection_preserves_entity_bridges():
    graph = _toy_trigraph()
    selection = select_motif_core(
        graph,
        config=MotifSelectorConfig(entity_ratio=1.0, sentence_budget=2, passage_budget=2),
    )
    selected_types = graph.node_type[selection.core_node_ids]
    assert set(selected_types.tolist()) == {ENTITY, SENTENCE, PASSAGE}

    selected_entities = set(selection.core_node_ids[selected_types == ENTITY].tolist())
    for node_id, node_t in zip(selection.core_node_ids.tolist(), selected_types.tolist()):
        if node_t in (SENTENCE, PASSAGE):
            neighbors = graph.edge_index[1][graph.edge_index[0] == node_id].tolist()
            assert selected_entities.intersection(neighbors)


def test_topologies_are_symmetric_sparse_and_have_no_sentence_passage_edges():
    graph = _toy_trigraph()
    x = graph.x
    text = _toy_text_bank().node_embeddings
    node_type = graph.node_type

    for result in [
        knn_topology(x, node_type=node_type, text_embeddings=text, k=2),
        self_expressive_topology(x, text, node_type=node_type, candidate_size=4, iterations=3, final_k=2),
    ]:
        assert torch.allclose(result.adjacency, result.adjacency.T)
        assert torch.diag(result.adjacency).abs().sum() == 0
        assert int((result.adjacency > 0).sum(dim=1).max()) <= 2
        for src, dst in result.edge_index.T.tolist():
            pair = {int(node_type[src]), int(node_type[dst])}
            assert pair != {SENTENCE, PASSAGE}


def test_client_condensor_outputs_numeric_upload_only():
    graph = _toy_trigraph()
    bank = _toy_text_bank()
    condensor = ClientCondensor(
        graph_dim=graph.x.size(1),
        text_dim=bank.node_embeddings.size(1),
        config=ClientCondensationConfig(
            motif=MotifSelectorConfig(entity_ratio=1.0, sentence_budget=2, passage_budget=2),
            topology_method="knn",
            knn_k=2,
            chunk_budget=4,
        ),
    )
    condensed, artifacts = condensor(graph, text_bank=bank, return_artifacts=True)
    assert condensed.x.dim() == 2
    assert condensed.edge_index.shape[0] == 2
    assert condensed.edge_weight.numel() == condensed.edge_index.size(1)
    assert condensed.node_type.dtype == torch.long
    assert not any(isinstance(value, str) for value in condensed.__dict__.values())
    assert artifacts.evidence_traces


def test_fusion_gate_shape_and_range():
    torch.manual_seed(13)
    fusion = GraphTextFusion(16, 16)
    x, gate = fusion(torch.randn(6, 16), torch.randn(6, 16))
    assert x.shape == (6, 16)
    assert gate.shape == (6,)
    assert torch.all((gate >= 0) & (gate <= 1))
