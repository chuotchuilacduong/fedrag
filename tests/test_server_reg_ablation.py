"""Regularization-diversity ablation semantics (--disable-server-reg).

Verifies the isolation the ablation depends on:
  * the "full" arm runs nonzero server-side L_reg in Phase I (init) and
    performs the Phase II server L_reg refinement step;
  * the "no-reg" arm's Phase-I aux loss is exactly zero and Phase II skips
    the server L_reg step entirely (not merely zero-weighted);
  * client-side loss coefficients (lambda_div_mem/lambda_deg_mem, and by
    extension lambda_reg_mem/lambda_gm/lambda_align_mem, which this ablation
    never touches) are identical between the two arms.
"""

from __future__ import annotations

import inspect
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Data

from fedcond_grag.client import client as client_module
from fedcond_grag.client.stage_e_memory.synthetic_memory import LocalSyntheticMemory
from fedcond_grag.server.server import FedCondQAServer
from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE


def _toy_anchor(num_features: int = 8) -> Data:
    node_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    x = torch.randn(node_type.numel(), num_features)
    edge_index = torch.tensor(
        [[0, 2, 0, 3, 1, 4, 1, 5, 2, 0, 3, 0, 4, 1, 5, 1],
         [2, 0, 3, 0, 4, 1, 5, 1, 0, 2, 0, 3, 1, 4, 1, 5]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index, node_type=node_type, y=node_type)
    data.num_global_classes = 3
    return data


def _args(*, disable_server_reg: bool, num_syn_nodes: int = 6, condense_iters: int = 3) -> Namespace:
    return Namespace(
        num_clients=1,
        num_global_syn_nodes=num_syn_nodes,
        server_condense_iters=condense_iters,
        condense_iters=condense_iters,
        local_epochs=0,
        lr_feat=0.05,
        lr_adj=0.05,
        pge_hidden=16,
        pge_topk=2,
        type_emb_dim=4,
        surrogate_type_weight=1.0,
        surrogate_link_weight=0.5,
        match_norm_weight=0.0,
        preserve_sep_topology=True,
        use_cuda=False,
        gpuid=0,
        hid_dim=16,
        num_layers=2,
        dropout=0.0,
        server_stage_c_mode="fedrag",
        gnn_model_name_c="gcn",
        gnn_in_dim_c=8,
        gnn_hidden_dim_c=8,
        gnn_num_layers_c=2,
        gnn_num_heads_c=1,
        gnn_dropout=0.0,
        repr_proj_out_dim=8,
        eta_agg=1.0,
        eta_reg=1e-2,
        server_reg_steps=1,
        lambda_div=0.1,
        lambda_deg=0.05,
        # Client-side terms this ablation must never touch, held identical
        # across both arms.
        lambda_div_mem=0.1,
        lambda_deg_mem=0.05,
        lambda_reg_mem=0.01,
        lambda_gm=0.1,
        lambda_align_mem=0.1,
        disable_server_reg=disable_server_reg,
    )


def _make_server(args: Namespace, graph: Data) -> FedCondQAServer:
    device = torch.device("cpu")
    message_pool = {"sampled_clients": [0], "client_0": {"anchor_graph": graph}, "round": 0}
    return FedCondQAServer(args, graph, "/tmp", message_pool, device)


def test_full_arm_phase1_uses_nonzero_server_reg():
    torch.manual_seed(0)
    graph = _toy_anchor()
    server = _make_server(_args(disable_server_reg=False), graph)
    server.execute()

    c = server.last_phase1_components
    assert c is not None
    assert c["div"] != 0.0
    assert c["deg"] != 0.0
    assert c["total"] == pytest.approx(c["align"] + c["div"] + c["deg"], rel=1e-5)


def test_noreg_arm_phase1_aux_loss_is_exactly_zero():
    torch.manual_seed(0)
    graph = _toy_anchor()
    server = _make_server(_args(disable_server_reg=True), graph)
    server.execute()

    c = server.last_phase1_components
    assert c is not None
    assert c["div"] == 0.0
    assert c["deg"] == 0.0
    assert c["total"] == pytest.approx(c["align"], rel=1e-5)


def _run_phase2(disable_server_reg: bool) -> tuple[FedCondQAServer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    graph = _toy_anchor()
    server = _make_server(_args(disable_server_reg=disable_server_reg), graph)
    server.execute()  # round 0: Phase-I bootstrap

    x_before = server.synthetic_x.detach().clone()
    zero_delta = {
        "x": torch.zeros_like(x_before),
        "pge": {k: torch.zeros_like(v) for k, v in server.pge.state_dict().items()},
    }
    server.message_pool["round"] = 1
    server.message_pool["sampled_clients"] = [0]
    server.message_pool["client_0"] = {"syn_delta": zero_delta, "num_samples": 5}
    # No anchor_graph this round -- fedrag mode must still proceed via deltas.
    server.execute()
    return server, x_before, server.synthetic_x.detach().clone()


def test_full_arm_phase2_runs_server_reg_step():
    server, x_before, x_after = _run_phase2(disable_server_reg=False)
    c = server.last_phase2_components
    assert c is not None
    assert c["enabled"] is True
    assert c["reg_steps"] >= 1
    # Delta was exactly zero, so any change in X_syn is due solely to the
    # server L_reg gradient step.
    assert not torch.allclose(x_before, x_after)


def test_noreg_arm_phase2_skips_server_reg_step():
    server, x_before, x_after = _run_phase2(disable_server_reg=True)
    c = server.last_phase2_components
    assert c == {"enabled": False, "reg_steps": 0, "total": 0.0, "div": 0.0, "deg": 0.0}
    # Delta was exactly zero and no reg step ran -- X_syn must be unchanged.
    assert torch.allclose(x_before, x_after)


def test_client_side_lambda_coefficients_identical_across_arms():
    full = _args(disable_server_reg=False)
    noreg = _args(disable_server_reg=True)
    for key in ("lambda_div_mem", "lambda_deg_mem", "lambda_reg_mem", "lambda_gm", "lambda_align_mem"):
        assert getattr(full, key) == getattr(noreg, key), key


def test_adapt_synthetic_memory_reads_mem_suffixed_lambdas_not_server_ones():
    """Guards the exact bug this ablation must not reintroduce: client.py's
    Stage-E adaptation must read lambda_div_mem/lambda_deg_mem, never the bare
    lambda_div/lambda_deg the server-side ablation flips."""
    src = inspect.getsource(client_module.FedCondQAClient.adapt_synthetic_memory)
    assert '"lambda_div_mem"' in src
    assert '"lambda_deg_mem"' in src
    assert 'getattr(self.args, "lambda_div",' not in src
    assert 'getattr(self.args, "lambda_deg",' not in src


def test_local_synthetic_memory_regularization_uses_given_coefficients():
    torch.manual_seed(0)
    node_type = torch.tensor([0, 0, 1, 1, 2, 2])
    x = torch.randn(6, 4)
    pge = TypeAwarePGE(feature_dim=4, hidden_dim=8, type_emb_dim=2, topk=2)
    mem = LocalSyntheticMemory(x, node_type, pge, target_degree=4.0)

    z_syn = torch.randn(6, 4, requires_grad=True)
    adj = torch.rand(6, 6)
    zero = mem.regularization(z_syn, adj, lambda_syn_div=0.0, lambda_deg=0.0)
    nonzero = mem.regularization(z_syn, adj, lambda_syn_div=0.1, lambda_deg=0.05)
    assert float(zero) == 0.0
    assert float(nonzero) != 0.0
