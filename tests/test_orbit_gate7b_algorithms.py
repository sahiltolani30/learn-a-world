"""Gate 7B Test Suite: Algorithmic Verification & Loss Correctness.

Verifies:
1. Exact mathematical loss fixtures for Expectile loss, CQL penalty, and Quantile Huber loss.
2. Architectures: DiscreteQCritic, ExpectileValueCritic, DistributionalValueCritic, ResidualPolicyHead.
3. Parameter isolation: 0 shared parameters between critics, actor, and frozen world model.
4. Execution correctness: IQL, Discrete CQL, and CRR gradient steps on synthetic batches.
5. Invariants: finite gradients, 17-action discrete bounds, zero NaN/Inf.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate7b_algorithms import (
    cql_conservative_penalty,
    crr_training_step,
    discrete_cql_training_step,
    expectile_loss,
    iql_training_step,
    quantile_huber_loss,
    soft_update,
)
from eval.gate7b_models import (
    DiscreteQCritic,
    DistributionalValueCritic,
    ExpectileValueCritic,
    ResidualPolicyHead,
)
from true_jepa.config import JEPAConfig
from true_jepa.model import TrueJEPA


# ---------------------------------------------------------------------------
# Test 1: Hand-Calculated Mathematical Loss Fixtures
# ---------------------------------------------------------------------------

def test_expectile_loss_hand_calculated_fixture():
    """Verify asymmetric L2 expectile loss against hand-calculated values."""
    tau = 0.7
    # Case 1: Positive error (underestimation) u = +2.0 -> weight = 0.7, loss = 0.7 * 4.0 = 2.8
    u_pos = torch.tensor([2.0])
    loss_pos = expectile_loss(u_pos, tau=tau)
    assert math.isclose(loss_pos.item(), 2.8, rel_tol=1e-5)

    # Case 2: Negative error (overestimation) u = -2.0 -> weight = 1 - 0.7 = 0.3, loss = 0.3 * 4.0 = 1.2
    u_neg = torch.tensor([-2.0])
    loss_neg = expectile_loss(u_neg, tau=tau)
    assert math.isclose(loss_neg.item(), 1.2, rel_tol=1e-5)

    # Case 3: Zero error -> loss = 0.0
    u_zero = torch.tensor([0.0])
    loss_zero = expectile_loss(u_zero, tau=tau)
    assert math.isclose(loss_zero.item(), 0.0, abs_tol=1e-7)


def test_cql_conservative_penalty_hand_calculated_fixture():
    """Verify discrete CQL conservative penalty against hand-calculated values."""
    # Construct Q = [0.0, ln(2.0)] with taken action = 0
    # logsumexp([0, ln(2)]) = ln(exp(0) + exp(ln(2))) = ln(1 + 2) = ln(3)
    # Q(a_real = 0) = 0.0
    # Penalty = ln(3) - 0.0 = ln(3) approx 1.098612
    q = torch.tensor([[0.0, math.log(2.0)]])
    action = torch.tensor([0])
    penalty = cql_conservative_penalty(q, action)
    expected = math.log(3.0)
    assert math.isclose(penalty.item(), expected, rel_tol=1e-5)


def test_quantile_huber_loss_hand_calculated_fixture():
    """Verify quantile Huber loss against hand-calculated values."""
    # Pred = 0.0, Target = 1.0, tau = 0.5, kappa = 1.0
    # diff = 1.0 <= kappa -> Huber = 0.5 * (1.0)^2 = 0.5
    # weight = |0.5 - I(diff < 0)| = |0.5 - 0.0| = 0.5
    # Quantile loss = 0.5 * 0.5 = 0.25
    pred = torch.tensor([[0.0]])
    target = torch.tensor([[1.0]])
    taus = torch.tensor([0.5])
    loss = quantile_huber_loss(pred, target, taus, kappa=1.0)
    assert math.isclose(loss.item(), 0.25, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Architecture Dimensions & Forward Passes
# ---------------------------------------------------------------------------

def test_gate7b_model_architectures_and_forward():
    """Verify all Gate 7B model heads accept state_dim 448 and output expected shapes."""
    B = 4
    state_dim = 448
    num_actions = 17
    x = torch.randn(B, state_dim)

    # 1. DiscreteQCritic: twin Q output
    q_critic = DiscreteQCritic(state_dim=state_dim, num_actions=num_actions)
    q1, q2 = q_critic(x)
    assert q1.shape == (B, num_actions)
    assert q2.shape == (B, num_actions)
    actions = torch.randint(0, num_actions, (B,))
    q1_a, q2_a = q_critic.evaluate_actions(x, actions)
    assert q1_a.shape == (B,)
    assert q2_a.shape == (B,)

    # 2. ExpectileValueCritic
    v_critic = ExpectileValueCritic(state_dim=state_dim)
    v = v_critic(x)
    assert v.shape == (B,)

    # 3. DistributionalValueCritic
    dist_critic = DistributionalValueCritic(state_dim=state_dim, num_quantiles=8)
    quantiles = dist_critic(x)
    assert quantiles.shape == (B, 8)
    exp_val = dist_critic.expected_value(x)
    assert exp_val.shape == (B,)

    # 4. ResidualPolicyHead
    res_policy = ResidualPolicyHead(state_dim=state_dim, num_actions=num_actions)
    logits = res_policy(x)
    assert logits.shape == (B, num_actions)


# ---------------------------------------------------------------------------
# Test 3: Optimizer Isolation & Parameter Ownership
# ---------------------------------------------------------------------------

def test_gate7b_parameter_isolation():
    """Verify actor and critic optimizers share zero parameters with each other or the world model."""
    cfg = JEPAConfig(token_dim=64, belief_dim=64, predictor_mlp_dim=64)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    q_critic = DiscreteQCritic(state_dim=448, num_actions=17)
    v_critic = ExpectileValueCritic(state_dim=448)
    actor_head = nn.Linear(448, 17)

    q_params = set(q_critic.parameters())
    v_params = set(v_critic.parameters())
    actor_params = set(actor_head.parameters())
    wm_params = set(model.parameters())

    assert len(q_params & v_params) == 0
    assert len(q_params & actor_params) == 0
    assert len(v_params & actor_params) == 0
    assert len(q_params & wm_params) == 0
    assert len(actor_params & wm_params) == 0


# ---------------------------------------------------------------------------
# Test 4: Algorithmic Step Execution (IQL, CQL, CRR)
# ---------------------------------------------------------------------------

def test_gate7b_algorithm_step_executions():
    """Verify IQL, CQL, and CRR gradient steps execute without error and produce finite metrics."""
    cfg = JEPAConfig(token_dim=64, belief_dim=64, predictor_mlp_dim=64)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")
    model.eval()

    device = torch.device("cpu")
    B, T = 2, 4
    batch = {
        "obs": torch.randint(0, 256, (B, T + 1, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, 17, (B, T), dtype=torch.int64),
        "rewards": torch.randn(B, T),
        "terminated": torch.zeros(B, T, dtype=torch.bool),
        "valid": torch.ones(B, T, dtype=torch.bool),
    }

    # Test IQL Step
    state_dim = model.heads.policy[0].in_features
    q_critic = DiscreteQCritic(state_dim=state_dim, num_actions=17)
    v_critic = ExpectileValueCritic(state_dim=state_dim)
    v_target = ExpectileValueCritic(state_dim=state_dim)
    v_target.load_state_dict(v_critic.state_dict())
    actor_head = nn.Linear(state_dim, 17)

    q_opt = torch.optim.Adam(q_critic.parameters(), lr=1e-3)
    v_opt = torch.optim.Adam(v_critic.parameters(), lr=1e-3)
    act_opt = torch.optim.Adam(actor_head.parameters(), lr=1e-3)

    iql_res = iql_training_step(
        model=model,
        q_critic=q_critic,
        v_critic=v_critic,
        v_target=v_target,
        actor_head=actor_head,
        batch=batch,
        q_optimizer=q_opt,
        v_optimizer=v_opt,
        actor_optimizer=act_opt,
        device=device,
    )
    assert np.isfinite(iql_res["q_loss"])
    assert np.isfinite(iql_res["v_loss"])
    assert np.isfinite(iql_res["actor_loss"])

    # Test Discrete CQL Step
    q_cql = DiscreteQCritic(state_dim=state_dim, num_actions=17)
    q_cql_target = DiscreteQCritic(state_dim=state_dim, num_actions=17)
    q_cql_target.load_state_dict(q_cql.state_dict())
    cql_q_opt = torch.optim.Adam(q_cql.parameters(), lr=1e-3)
    cql_act_opt = torch.optim.Adam(actor_head.parameters(), lr=1e-3)

    cql_res = discrete_cql_training_step(
        model=model,
        q_critic=q_cql,
        q_target=q_cql_target,
        actor_head=actor_head,
        batch=batch,
        q_optimizer=cql_q_opt,
        actor_optimizer=cql_act_opt,
        device=device,
    )
    assert np.isfinite(cql_res["total_q_loss"])
    assert np.isfinite(cql_res["actor_loss"])
    assert np.isfinite(cql_res["cql_penalty"])

    # Test CRR Step
    crr_res = crr_training_step(
        model=model,
        q_critic=q_cql,
        q_target=q_cql_target,
        actor_head=actor_head,
        batch=batch,
        q_optimizer=cql_q_opt,
        actor_optimizer=cql_act_opt,
        device=device,
    )
    assert np.isfinite(crr_res["q_loss"])
    assert np.isfinite(crr_res["actor_loss"])
    assert 0.0 <= crr_res["filter_pass_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Soft Update Invariant
# ---------------------------------------------------------------------------

def test_polyak_soft_update():
    """Verify target network softly interpolates towards source network."""
    src = nn.Linear(4, 4, bias=False)
    tgt = nn.Linear(4, 4, bias=False)
    src.weight.data.fill_(1.0)
    tgt.weight.data.fill_(0.0)

    soft_update(tgt, src, tau=0.1)
    # Expected: (1 - 0.1)*0 + 0.1*1 = 0.1
    assert torch.allclose(tgt.weight.data, torch.tensor(0.1))


# ---------------------------------------------------------------------------
# Test 6: Gate 7B Artifacts and Integrity
# ---------------------------------------------------------------------------

def test_gate7b_pipeline_artifacts():
    """Verify that Gate 7B produces required dataset manifests, plots, and metrics."""
    out_dir = Path("results/gate7b")
    if not out_dir.exists() or not (out_dir / "gate7b_experiment_metrics.json").exists():
        pytest.skip("Gate 7B execution has not generated results/gate7b yet")

    manifest = out_dir / "gate7b_dataset_manifest.json"
    plots = out_dir / "gate7b_comparison_plots.png"
    metrics = out_dir / "gate7b_experiment_metrics.json"

    assert manifest.exists(), "Dataset manifest must exist"
    assert plots.exists(), "Comparison plots must exist"
    assert metrics.exists(), "Metrics manifest must exist"

    import json
    with open(metrics) as f:
        m = json.load(f)
    assert "benchmark_results" in m
    assert "gate7_champion" in m["benchmark_results"]
    assert "candidate_b_iql" in m["benchmark_results"]
    assert "candidate_c_cql" in m["benchmark_results"]
    assert "candidate_d_crr" in m["benchmark_results"]

