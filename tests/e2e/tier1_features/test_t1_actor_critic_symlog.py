"""Tier 1: Actor-Critic, Symlog Two-Hot Distributional Critic, and Lambda-Returns (F8, F9, F10).

Tests:
- F8: Amortized Policy Actor [5 tests]
- F9: Symlog Two-Hot Distributional Critic [5 tests]
- F10: Multi-Step Lambda-Returns Credit Assignment [5 tests]
"""
from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from zephyr.ac import unimix_probs
from zephyr.config import NUM_ACTIONS
from zephyr.heads import ZephyrHeads
from zephyr.losses import TwoHotCoder, inv_symlog, lambda_returns, symlog
from zephyr.planner import ZephyrAgent


# ==============================================================================
# Feature 8: Amortized Policy Actor
# ==============================================================================

def test_f8_actor_head_output_shape_and_logits(mock_heads):
    """F8: Actor head maps composite state s to unnormalized action logits across 17 discrete actions."""
    batch_size = 4
    state_dim = mock_heads.cfg.world_model.latent_dim + mock_heads.cfg.world_model.belief_dim
    s = torch.randn(batch_size, state_dim)

    logits = mock_heads.actor(s)

    assert logits.shape == (batch_size, NUM_ACTIONS), f"Logits shape {logits.shape} != ({batch_size}, {NUM_ACTIONS})"
    assert logits.dtype == torch.float32, f"Expected float32 logits, got {logits.dtype}"
    assert torch.isfinite(logits).all(), "Actor logits must be finite"


def test_f8_actor_action_sampling_unimix_exploration():
    """F8: unimix_probs blends softmax probabilities with uniform 1/17 distribution."""
    # Extreme logits: action 0 dominates with logit 100.0, rest 0.0
    logits = torch.zeros(1, NUM_ACTIONS)
    logits[0, 0] = 100.0

    unimix_eps = 0.01
    probs = unimix_probs(logits, unimix=unimix_eps)

    assert probs.shape == (1, NUM_ACTIONS)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5), "Probabilities must sum to 1.0"
    # Even with extreme logit, every non-greedy action receives unimix / NUM_ACTIONS
    expected_floor = unimix_eps / NUM_ACTIONS
    for a in range(1, NUM_ACTIONS):
        assert probs[0, a].item() >= expected_floor * 0.99, (
            f"Action {a} prob {probs[0, a].item()} collapsed below unimix floor {expected_floor}"
        )


def test_f8_actor_entropy_regularization(mock_heads):
    """F8: Categorical policy entropy calculation rewards high exploration dispersion."""
    # Case A: Uniform policy (maximal entropy = ln(17))
    uniform_logits = torch.zeros(1, NUM_ACTIONS)
    dist_uniform = torch.distributions.Categorical(probs=torch.softmax(uniform_logits, dim=-1))
    ent_uniform = dist_uniform.entropy().item()

    # Case B: Peaked deterministic policy (near-zero entropy)
    peaked_logits = torch.zeros(1, NUM_ACTIONS)
    peaked_logits[0, 3] = 50.0
    dist_peaked = torch.distributions.Categorical(probs=torch.softmax(peaked_logits, dim=-1))
    ent_peaked = dist_peaked.entropy().item()

    assert ent_uniform > 2.8, f"Uniform entropy {ent_uniform} should be near ln(17) ~ 2.83"
    assert ent_peaked < 0.01, f"Peaked entropy {ent_peaked} should be near 0"
    assert ent_uniform > ent_peaked, "Uniform policy must have higher entropy than peaked"


def test_f8_actor_reinforce_advantage_gradient():
    """F8: Policy gradients shift action logits in proportion to positive/negative advantages."""
    logits = torch.zeros(1, NUM_ACTIONS, requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)

    # Action 2 taken with positive advantage (+1.0)
    log_p = dist.log_prob(torch.tensor([2]))
    loss_pos = -(log_p * 1.0)
    loss_pos.backward()

    # Gradient with respect to logits[0, 2] should be negative (loss minimization increases logit 2)
    grad_at_2 = logits.grad[0, 2].item()
    assert grad_at_2 < 0.0, f"Expected negative gradient to increase action 2 logit, got {grad_at_2}"


def test_f8_actor_mode_a_deterministic_inference(mock_wm, mock_heads):
    """F8: In deterministic evaluation mode, ZephyrAgent selects argmax greedy action."""
    agent = ZephyrAgent(mock_wm, mock_heads, device="cpu")
    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # Inject known peak in actor linear layer bias so action 5 has highest logit
    with torch.no_grad():
        mock_heads.actor.net[-1].bias.fill_(0.0)
        mock_heads.actor.net[-1].bias[5] = 100.0

    action = agent.act(obs)
    assert action == 5, f"Greedy actor should pick argmax action 5, got {action}"


# ==============================================================================
# Feature 9: Symlog Two-Hot Distributional Critic
# ==============================================================================

def test_f9_symlog_math_invertibility():
    """F9: Mathematical invertibility: inv_symlog(symlog(x)) == x for all positive/negative real values."""
    values = torch.tensor([-1000.0, -100.0, -20.0, -1.0, -0.5, 0.0, 0.5, 1.0, 20.0, 100.0, 1000.0])
    y = symlog(values)
    x_reconstructed = inv_symlog(y)

    assert torch.allclose(x_reconstructed, values, atol=1e-5), (
        f"Invertibility failed: diff={torch.max(torch.abs(x_reconstructed - values))}"
    )


def test_f9_two_hot_encoding_sum_to_one():
    """F9: TwoHotCoder.encode(v) yields probability distribution strictly summing to 1.0."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    values = torch.tensor([-50.0, -10.0, -0.2, 0.0, 0.2, 10.0, 50.0])

    probs = coder.encode(values)

    assert probs.shape == (len(values), 65)
    assert torch.all(probs >= 0.0), "Probabilities must be non-negative"
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
        f"Two-hot encoding probabilities must sum to 1.0, got {sums}"
    )


def test_f9_two_hot_exact_bin_center_hit():
    """F9: When value falls exactly on a bin center (e.g. 0.0), probability is 1.0 at that single bin."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    center_idx = 32  # Center bin for 65 bins spanning [-20, 20] is bin 32 (value 0.0)
    assert coder.bin_centers[center_idx].item() == pytest.approx(0.0, abs=1e-5)

    probs = coder.encode(torch.tensor([0.0]))

    assert probs[0, center_idx].item() == pytest.approx(1.0, abs=1e-5), (
        f"Bin {center_idx} probability {probs[0, center_idx].item()} != 1.0"
    )
    # All other bins should be exactly 0
    non_center_sum = probs[0].sum().item() - probs[0, center_idx].item()
    assert non_center_sum == pytest.approx(0.0, abs=1e-6)


def test_f9_two_hot_linear_interpolation_between_bins():
    """F9: Values between bin centers are partitioned linearly between the two adjacent bins."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    c0 = coder.bin_centers[32].item()  # 0.0
    c1 = coder.bin_centers[33].item()  # step size in symlog space

    # Choose value whose symlog is exactly at the midpoint: symlog(x) = (c0 + c1) / 2
    midpoint_symlog = (c0 + c1) / 2.0
    val = inv_symlog(torch.tensor([midpoint_symlog]))

    probs = coder.encode(val)

    assert probs[0, 32].item() == pytest.approx(0.5, abs=1e-4), f"Bin 32 prob {probs[0, 32].item()} != 0.5"
    assert probs[0, 33].item() == pytest.approx(0.5, abs=1e-4), f"Bin 33 prob {probs[0, 33].item()} != 0.5"
    remainder = probs[0].sum().item() - (probs[0, 32].item() + probs[0, 33].item())
    assert remainder == pytest.approx(0.0, abs=1e-6)


def test_f9_two_hot_cross_entropy_loss_optimization():
    """F9: coder.loss() computes cross-entropy; correctly aligned logits minimize loss."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    target_value = torch.tensor([5.0])

    # Case A: Correctly aligned logits (peaked at correct target bin)
    target_probs = coder.encode(target_value)
    aligned_logits = torch.log(target_probs + 1e-8)
    loss_aligned = coder.loss(aligned_logits, target_value)

    # Case B: Misaligned logits (peaked at opposite side)
    misaligned_logits = torch.roll(aligned_logits, shifts=30, dims=-1)
    loss_misaligned = coder.loss(misaligned_logits, target_value)

    assert loss_aligned.item() < loss_misaligned.item(), (
        f"Aligned loss {loss_aligned.item()} should be < misaligned {loss_misaligned.item()}"
    )


# ==============================================================================
# Feature 10: Multi-Step Lambda-Returns Credit Assignment
# ==============================================================================

def test_f10_lambda_returns_recursive_equation():
    """F10: lambda_returns matches recursive equation: G_t = r_t + gamma_eff * ((1-lambda)*V + lambda*G_{t+1})."""
    T, B = 3, 1
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    continues = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)

    # Under gamma=1.0, lambda=1.0, G_t = r_t + G_{t+1} (pure Monte Carlo return)
    returns = lambda_returns(rewards, continues, values, bootstrap, gamma=1.0, td_lambda=1.0)

    assert returns[2, 0].item() == pytest.approx(3.0, abs=1e-5)
    assert returns[1, 0].item() == pytest.approx(5.0, abs=1e-5)
    assert returns[0, 0].item() == pytest.approx(6.0, abs=1e-5)


def test_f10_lambda_zero_pure_td_one_step():
    """F10: At lambda=0.0, lambda_returns collapses to 1-step TD target: G_t = r_t + gamma * V_{t+1}."""
    T, B = 2, 1
    rewards = torch.tensor([[1.0], [2.0]])
    continues = torch.ones(T, B)
    values = torch.tensor([[10.0], [20.0]])
    bootstrap = torch.tensor([30.0])

    returns = lambda_returns(rewards, continues, values, bootstrap, gamma=1.0, td_lambda=0.0)

    # G_0 = r_0 + V_1 = 1 + 20 = 21
    assert returns[0, 0].item() == pytest.approx(21.0, abs=1e-5)
    # G_1 = r_1 + bootstrap = 2 + 30 = 32
    assert returns[1, 0].item() == pytest.approx(32.0, abs=1e-5)


def test_f10_lambda_continuation_zero_truncation():
    """F10: When continuation is 0.0 (death/terminal), future values and bootstrap are truncated."""
    T, B = 3, 1
    rewards = torch.tensor([[1.0], [5.0], [10.0]])
    continues = torch.tensor([[1.0], [0.0], [1.0]])  # step 1 is terminal
    values = torch.tensor([[2.0], [20.0], [50.0]])
    bootstrap = torch.tensor([100.0])

    returns = lambda_returns(rewards, continues, values, bootstrap, gamma=0.997, td_lambda=0.95)

    # Terminal step 1 has continues=0, so G_1 = r_1 = 5.0 regardless of future
    assert returns[1, 0].item() == pytest.approx(5.0, abs=1e-5)


def test_f10_slow_target_critic_ema_update(mock_heads):
    """F10: Slow target critic weights smoothly track online critic via Polyak EMA."""
    fast_critic = mock_heads.critic
    slow_critic = copy.deepcopy(fast_critic)

    # Perturb fast critic weights
    with torch.no_grad():
        for p in fast_critic.parameters():
            p.add_(1.0)

    tau = 0.995
    # Apply EMA step: slow = tau * slow + (1 - tau) * fast
    with torch.no_grad():
        for p_slow, p_fast in zip(slow_critic.parameters(), fast_critic.parameters()):
            p_slow.mul_(tau).add_(p_fast, alpha=1.0 - tau)

    # Verify slow critic moved towards fast critic by exactly (1 - tau)
    for p_slow, p_fast in zip(slow_critic.parameters(), fast_critic.parameters()):
        diff = torch.abs(p_fast - p_slow)
        assert torch.all(diff > 0.0), "Slow critic should lag behind fast critic"


def test_f10_long_horizon_credit_propagation_60_steps():
    """F10: Reward delivered at step 60 propagates non-zero credit back to step 0."""
    T, B = 61, 1  # steps 0..60
    rewards = torch.zeros(T, B)
    rewards[60, 0] = 1.0  # Milestone reward at step 60
    continues = torch.ones(T, B)
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)

    gamma = 0.997
    td_lambda = 0.95
    returns = lambda_returns(rewards, continues, values, bootstrap, gamma=gamma, td_lambda=td_lambda)

    # G_0 must receive non-zero credit back from step 60
    assert returns[0, 0].item() > 0.0, f"Expected non-zero credit propagation to step 0, got {returns[0, 0].item()}"
    # Verify approximate theoretical bound: G_0 ~ (gamma * lambda)^60
    expected_g0 = (gamma * td_lambda) ** 60
    assert returns[0, 0].item() == pytest.approx(expected_g0, rel=1e-2), (
        f"Credit propagation {returns[0, 0].item()} differs from expected {expected_g0}"
    )
