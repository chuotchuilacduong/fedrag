"""Stage E — client-side query-conditioned synthetic memory adaptation tests.

Covers the FedRAG Phase 1 building blocks without loading an LLM: local
Θ_syn reconstruction from the server broadcast, differentiable soft synthetic
retrieval, gradient flow to X_syn and θ_PGE, client-side alignment /
regularization losses, delta export, and server-side delta aggregation.
"""

import torch
from torch import nn

from fedcond_grag.client.stage_e_memory import (
    LocalSyntheticMemory,
    aggregate_syn_deltas,
    gradient_matching_loss,
)
from fedcond_grag.model.gnn import GCN
from fedcond_grag.server.stage_c_aggregate.pge import TypeAwarePGE

KG = 8
DIM = 16
PROMPT_DIM = 12
PGE_CONFIG = {
    "feature_dim": DIM,
    "hidden_dim": 32,
    "type_emb_dim": 4,
    "topk": 3,
    "preserve_sep": False,
}


def make_broadcast_state(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    pge = TypeAwarePGE(**PGE_CONFIG)
    return {
        "x": torch.randn(KG, DIM),
        "node_type": torch.randint(0, 3, (KG,)),
        "pge_state": {k: v.clone() for k, v in pge.state_dict().items()},
        "pge_config": dict(PGE_CONFIG),
        "target_degree": 4.0,
    }


def make_prompt_module() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(1)
    encoder = GCN(
        in_channels=DIM, hidden_channels=DIM, out_channels=DIM,
        num_layers=2, dropout=0.0,
    )
    projector = nn.Sequential(nn.Linear(DIM, PROMPT_DIM))
    return encoder, projector


def test_from_broadcast_reconstructs_theta_syn():
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    assert torch.allclose(mem.x.detach(), state["x"])
    assert torch.equal(mem.node_type, state["node_type"].long())
    for key, value in mem.pge.state_dict().items():
        assert torch.allclose(value, state["pge_state"][key])
    assert mem.x.requires_grad
    assert all(p.requires_grad for p in mem.pge.parameters())
    assert mem.target_degree == 4.0


def test_soft_context_shapes_and_simplex():
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    encoder, projector = make_prompt_module()
    z_syn, adj_soft = mem.encode(encoder, projector)
    assert z_syn.shape == (KG, PROMPT_DIM)
    assert adj_soft.shape == (KG, KG)

    query = torch.randn(3, PROMPT_DIM)
    z_c = mem.soft_context(query, z_syn, tau=0.1)
    assert z_c.shape == (3, PROMPT_DIM)
    # z_c rows are convex combinations of z_syn rows → bounded by extremes
    assert z_c.max() <= z_syn.max() + 1e-5
    assert z_c.min() >= z_syn.min() - 1e-5


def test_gradients_flow_to_x_and_pge():
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    encoder, projector = make_prompt_module()

    z_syn, adj_soft = mem.encode(encoder, projector)
    z_c = mem.soft_context(torch.randn(2, PROMPT_DIM), z_syn, tau=0.1)
    loss = z_c.pow(2).sum() + mem.regularization(z_syn, adj_soft)
    loss.backward()

    assert mem.x.grad is not None and mem.x.grad.abs().sum() > 0
    pge_grads = [p.grad for p in mem.pge.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in pge_grads)


def test_alignment_loss_zero_when_client_matches_synthetic():
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    z_syn = torch.randn(KG, PROMPT_DIM)
    # A client rep equal to one synthetic node is reconstructed almost exactly
    # once the softmax sharpens; use a scaled copy to sharpen the row softmax.
    z_client = z_syn[:2] * 1.0
    loss_same = mem.alignment_loss(z_client, z_syn * 100.0)
    loss_far = mem.alignment_loss(z_client + 10.0, z_syn * 100.0)
    assert loss_same.item() < loss_far.item()
    # Empty inputs are safe
    assert mem.alignment_loss(torch.zeros(0, PROMPT_DIM), z_syn).item() == 0.0


def test_delta_matches_parameter_drift():
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    with torch.no_grad():
        mem.x.add_(1.0)
        first_pge_param = next(iter(mem.pge.parameters()))
        first_pge_param.add_(0.5)
    delta = mem.delta(state)
    assert torch.allclose(delta["x"], torch.ones(KG, DIM))
    pge_keys = list(mem.pge.state_dict().keys())
    moved = delta["pge"][pge_keys[0]]
    assert torch.allclose(moved, torch.full_like(moved, 0.5))
    untouched = delta["pge"][pge_keys[1]]
    assert torch.allclose(untouched, torch.zeros_like(untouched))


def test_gradient_matching_loss_bounds():
    g = [torch.randn(4, 4), torch.randn(7)]
    zero_loss = gradient_matching_loss(g, [t.clone() for t in g])
    assert abs(zero_loss.item()) < 1e-5
    flipped = gradient_matching_loss(g, [-t for t in g])
    assert abs(flipped.item() - 2.0) < 1e-5
    # None entries (allow_unused) are skipped in lockstep
    partial = gradient_matching_loss([g[0], None], [g[0].clone(), None])
    assert abs(partial.item()) < 1e-5


def test_gradient_matching_loss_differentiable():
    g_loc = [torch.randn(5)]
    g_syn = [torch.randn(5, requires_grad=True)]
    loss = gradient_matching_loss(g_loc, g_syn)
    loss.backward()
    assert g_syn[0].grad is not None


def test_aggregate_syn_deltas_weighted_average():
    d1 = {"x": torch.ones(2, 3), "pge": {"w": torch.ones(4)}}
    d2 = {"x": torch.full((2, 3), 3.0), "pge": {"w": torch.full((4,), 3.0)}}
    agg = aggregate_syn_deltas([(d1, 1), (d2, 3)])
    # (1*1 + 3*3) / 4 = 2.5
    assert torch.allclose(agg["x"], torch.full((2, 3), 2.5))
    assert torch.allclose(agg["pge"]["w"], torch.full((4,), 2.5))
    assert aggregate_syn_deltas([]) is None
    assert aggregate_syn_deltas([(None, 5), (d1, 0)]) is None


def test_adaptation_step_reduces_surrogate_objective():
    """A few Adam steps on the local memory should reduce a fixed objective."""
    state = make_broadcast_state()
    mem = LocalSyntheticMemory.from_broadcast(state, torch.device("cpu"))
    encoder, projector = make_prompt_module()
    optimizer = torch.optim.Adam(mem.parameters(), lr=1e-2)
    target = torch.randn(2, PROMPT_DIM)
    query = torch.randn(2, PROMPT_DIM)

    def objective():
        z_syn, adj_soft = mem.encode(encoder, projector)
        z_c = mem.soft_context(query, z_syn, tau=0.1)
        return (z_c - target).pow(2).sum() + 0.01 * mem.regularization(z_syn, adj_soft)

    start = objective().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    end = objective().item()
    assert end < start

    delta = mem.delta(state)
    assert delta["x"].abs().sum() > 0
