from argparse import Namespace

import torch
from torch_geometric.data import Data

from fedcond_grag import load_client, load_server, load_task
from fedcond_grag.client.client import FedCondQAClient
from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE
from fedcond_grag.server.stage_c_aggregate.surrogate import (
    SurrogateGNN,
    edge_index_to_dense,
    parameter_gradients,
    surrogate_loss,
)


def _toy_anchor(num_features: int = 8) -> Data:
    node_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    x = torch.randn(node_type.numel(), num_features)
    edge_index = torch.tensor(
        [
            [0, 2, 0, 3, 1, 4, 1, 5, 2, 0, 3, 0, 4, 1, 5, 1],
            [2, 0, 3, 0, 4, 1, 5, 1, 0, 2, 0, 3, 1, 4, 1, 5],
        ],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index, node_type=node_type, y=node_type)
    data.num_global_classes = 3
    return data


def _args(tmp_path, *, num_syn_nodes: int = 9) -> Namespace:
    return Namespace(
        fl_algorithm="fedcond_qa",
        task="condensation_qa",
        dataset=["toy"],
        model=["gcn"],
        hid_dim=16,
        num_layers=2,
        dropout=0.0,
        lr=0.01,
        weight_decay=0.0,
        optim="adam",
        metrics=["accuracy"],
        num_clients=1,
        num_global_syn_nodes=num_syn_nodes,
        server_condense_iters=2,
        condense_iters=2,
        local_epochs=0,
        lr_feat=0.01,
        lr_adj=0.01,
        pge_hidden=16,
        pge_topk=2,
        type_emb_dim=4,
        surrogate_type_weight=1.0,
        surrogate_link_weight=0.5,
        surrogate_align_weight=0.1,
        match_norm_weight=0.0,
        condense_refresh_every=10,
        preserve_sep_topology=True,
        use_cuda=False,
        gpuid=0,
        dp_mech="no_dp",
        dp_epsilon=0.0,
        dp_delta=1e-5,
        dp_clip=1.0,
        train_val_test="default_split",
        processing="raw",
        processing_percentage=0.1,
        feature_mask_prob=0.1,
        homo_injection_ratio=0.0,
        hete_injection_ratio=0.0,
        debug=False,
        wandb_name="test",
        log_root=str(tmp_path),
        data_root=str(tmp_path),
    )


def test_surrogate_loss_and_gradients_are_finite():
    torch.manual_seed(7)
    graph = _toy_anchor()
    model = SurrogateGNN(input_dim=graph.x.size(1), hidden_dim=16, output_dim=3, num_layers=2)
    adj = edge_index_to_dense(graph.edge_index, graph.num_nodes)

    output = surrogate_loss(model, graph.x, adj, graph.node_type, edge_index=graph.edge_index)
    grads = parameter_gradients(output.loss, list(model.parameters()))

    assert torch.isfinite(output.loss)
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_type_aware_pge_sparse_preserves_sep_topology():
    torch.manual_seed(11)
    graph = _toy_anchor()
    pge = TypeAwarePGE(feature_dim=graph.x.size(1), hidden_dim=16, type_emb_dim=4, topk=2, preserve_sep=True)

    adj = pge(graph.x, graph.node_type)

    assert torch.allclose(adj, adj.T)
    assert torch.count_nonzero(torch.diag(adj)) == 0
    assert int((adj > 0).sum(dim=1).max().item()) <= 2
    for src, dst in (adj > 0).nonzero(as_tuple=False).tolist():
        pair = {int(graph.node_type[src]), int(graph.node_type[dst])}
        assert pair in ({0, 1}, {0, 2})


def test_fedcond_server_condense_step_exports_synthetic_graph(tmp_path):
    torch.manual_seed(13)
    device = torch.device("cpu")
    args = _args(tmp_path, num_syn_nodes=9)
    graph = _toy_anchor()
    message_pool = {"sampled_clients": [0], "client_0": {"anchor_graph": graph}, "round": 0}

    server = FedCondQAServer(args, graph, str(tmp_path), message_pool, device)
    server.execute()
    synthetic = server.export_synthetic_graph()

    assert server.synthetic_x is not None
    assert server.pge is not None
    assert synthetic.x.size(0) == args.num_global_syn_nodes
    assert synthetic.node_type.unique().numel() == 3
    assert torch.isfinite(torch.tensor(server.train_loss_match))


def test_gfl_registry_loads_fedcond_qa_components(tmp_path):
    torch.manual_seed(17)
    device = torch.device("cpu")
    args = _args(tmp_path, num_syn_nodes=6)
    graph = _toy_anchor()
    message_pool = {"sampled_clients": [0], "round": 0}

    task = load_task(args, 0, graph, str(tmp_path), device)
    client = load_client(args, 0, graph, str(tmp_path), message_pool, device)
    assert task.__class__.__name__ == "CondensationQATask"
    assert isinstance(client, FedCondQAClient)

    client.execute()
    client.send_message()
    server = load_server(args, graph, str(tmp_path), message_pool, device)
    assert isinstance(server, FedCondQAServer)
    server.execute()
    assert "synthetic_x" in message_pool["server"]
