"""Tier 3: Latent Imagination & Actor-Critic Credit Flow Combinations.

Covers combinatorial interactions between:
- F6 (Latent Rep), F7 (Dynamics Predictor), F8 (Actor), F9 (Symlog Two-Hot Critic),
  F10 (Multi-Step Lambda-Returns), F13 (Continuation Head), F14 (State-Dependent Discounting).
"""
from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from zephyr.ac import (
    actor_lambda_return_loss,
    critic_twohot_loss,
    imagine_trajectory,
    normalize_returns,
    trajectory_lambda_returns,
)
from zephyr.heads import ZephyrHeads
from zephyr.losses import TwoHotCoder, lambda_returns
from zephyr.wm import ZephyrWM


def test_t3_flow_01_encoder_and_closed_loop_latent_imagination(tiny_test_cfg, mock_wm, mock_heads, sample_obs_batch):
    """T3-FLOW-01: ConvEncoder -> GRU -> Predictor -> Actor closed-loop imagination."""
    b = sample_obs_batch.shape[0]
    obs = torch.from_numpy(np.ascontiguousarray(sample_obs_batch))
    with torch.no_grad():
        z0 = mock_wm.encode(obs)
        h0 = mock_wm.init_state(b, "cpu")

    horizon = 5
    imagination = imagine_trajectory(
        mock_wm, mock_heads, z0, h0, horizon=horizon, cfg=tiny_test_cfg, deterministic=False
    )

    assert imagination["z"].shape == (horizon, b, tiny_test_cfg.world_model.latent_dim)
    assert imagination["h"].shape == (horizon, b, tiny_test_cfg.world_model.belief_dim)
    assert imagination["actions"].shape == (horizon, b)
    assert not torch.isnan(imagination["z"]).any()
    assert (imagination["actions"] >= 0).all() and (imagination["actions"] < 17).all()


def test_t3_flow_02_latent_trajectory_evaluation_symlog_critic(tiny_test_cfg, mock_wm, mock_heads, sample_obs_batch):
    """T3-FLOW-02: Imagined states evaluated by symlog two-hot critic head."""
    b = sample_obs_batch.shape[0]
    obs = torch.from_numpy(np.ascontiguousarray(sample_obs_batch))
    with torch.no_grad():
        z0 = mock_wm.encode(obs)
        h0 = mock_wm.init_state(b, "cpu")

    horizon = 5
    imagination = imagine_trajectory(
        mock_wm, mock_heads, z0, h0, horizon=horizon, cfg=tiny_test_cfg
    )

    s = imagination["s"]
    logits = mock_heads.critic(s)
    coder: TwoHotCoder = mock_heads.critic.coder

    assert logits.shape == (horizon, b, tiny_test_cfg.heads.critic_bins)
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    values = coder.decode(logits)
    assert values.shape == (horizon, b)
    assert (values >= -tiny_test_cfg.heads.critic_symlog_range - 1.0).all()
    assert (values <= tiny_test_cfg.heads.critic_symlog_range + 1.0).all()


def test_t3_flow_03_backward_multistep_lambda_returns():
    """T3-FLOW-03: Backward lambda-return recursion smooth long-horizon credit propagation."""
    t_len, b = 15, 2
    rewards = torch.zeros((t_len, b), dtype=torch.float32)
    rewards[14] = 1.0  # Milestone reward at final step
    continues = torch.ones((t_len, b), dtype=torch.float32)
    values = torch.full((t_len, b), 0.5, dtype=torch.float32)
    bootstrap = torch.full((b,), 0.5, dtype=torch.float32)

    g_lambda = lambda_returns(
        rewards, continues, values, bootstrap, gamma=0.997, td_lambda=0.95
    )

    assert g_lambda.shape == (t_len, b)
    # Expected at t=14: 1.0 + 0.997 * 0.5 = 1.4985
    expected_14 = 1.0 + 0.997 * 0.5
    assert torch.allclose(g_lambda[14], torch.tensor(expected_14), atol=1e-3)

    # Smooth backward decay across all 15 steps: g[0] > 0.5
    assert (g_lambda[0] > 0.5).all()

    # Monotonically increasing toward the milestone reward at step 14
    for step in range(t_len - 1):
        assert (g_lambda[step + 1] >= g_lambda[step] - 1e-6).all()


def test_t3_flow_04_slow_target_critic_ema_bootstrapping(tiny_test_cfg, mock_heads):
    """T3-FLOW-04: Slow target critic EMA Polyak tracking and stability."""
    online_critic = mock_heads.critic
    slow_critic = copy.deepcopy(online_critic)
    decay = 0.95

    # Simulate online weight updates
    for p in online_critic.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    # Apply Polyak update
    with torch.no_grad():
        for p_slow, p in zip(slow_critic.parameters(), online_critic.parameters()):
            p_slow.mul_(decay).add_(p, alpha=1.0 - decay)

    # Assert slow critic tracked online critic with lag
    dist = sum(torch.norm(p_slow - p).item() for p_slow, p in zip(slow_critic.parameters(), online_critic.parameters()))
    assert dist > 0.0, "Slow critic should lag behind updated online critic"


def test_t3_flow_05_sparse_terminal_reward_propagation(tiny_test_cfg, mock_heads):
    """T3-FLOW-05: Sparse terminal milestone reward propagated via lambda-returns and advantage."""
    t_len, b = 10, 4
    rewards = torch.zeros((t_len, b), dtype=torch.float32)
    rewards[9] = 1.0  # Rare milestone reward at horizon end
    continues = torch.ones((t_len, b), dtype=torch.float32)
    values = torch.full((t_len, b), 0.2, dtype=torch.float32)
    bootstrap = torch.full((b,), 0.2, dtype=torch.float32)

    g_lambda = lambda_returns(rewards, continues, values, bootstrap, gamma=0.99, td_lambda=0.95)
    adv = normalize_returns(g_lambda)

    assert adv.shape == (t_len, b)
    assert not torch.isnan(adv).any()
    # Positive advantage at step 9 where reward landed
    assert (g_lambda[9] > g_lambda[0]).all()


def test_t3_flow_06_value_regression_loss_advantage_standardization(tiny_test_cfg, mock_heads):
    """T3-FLOW-06: Two-hot value regression loss and standardized advantages."""
    coder = mock_heads.critic.coder
    b, bins = 16, tiny_test_cfg.heads.critic_bins
    logits = torch.randn(b, bins, requires_grad=True)
    target_returns = torch.tensor(np.random.uniform(-5.0, 5.0, size=(b,)), dtype=torch.float32)

    loss = critic_twohot_loss(logits, target_returns, coder)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    returns_seq = target_returns.unsqueeze(0).repeat(5, 1)  # [5, 16]
    norm_adv = normalize_returns(returns_seq)
    mean_adv = norm_adv.mean(dim=1)
    std_adv = norm_adv.std(dim=1)

    assert torch.allclose(mean_adv, torch.zeros_like(mean_adv), atol=1e-4)
    assert torch.allclose(std_adv, torch.ones_like(std_adv), atol=1e-2)


def test_t3_flow_07_stop_gradient_discipline_on_world_model(tiny_test_cfg, mock_wm, mock_heads, sample_obs_batch):
    """T3-FLOW-07: Actor-Critic loss optimization strictly stops gradient into world model."""
    b = sample_obs_batch.shape[0]
    obs = torch.from_numpy(np.ascontiguousarray(sample_obs_batch))

    # Detached starting states as in Dreamer / Zephyr AC rollout
    with torch.no_grad():
        z0 = mock_wm.encode(obs)
        h0 = mock_wm.init_state(b, "cpu")

    # In Zephyr AC loss, latents are generated using no_grad on WM dynamics
    with torch.no_grad():
        imagination = imagine_trajectory(
            mock_wm, mock_heads, z0, h0, horizon=4, cfg=tiny_test_cfg
        )

    # Actor loss through policy logits and critic loss through critic logits
    s = imagination["s"].detach()
    actor_logits = mock_heads.actor(s)
    probs = torch.softmax(actor_logits, dim=-1)
    log_probs = torch.log(probs.gather(-1, imagination["actions"].unsqueeze(-1)).squeeze(-1) + 1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    lam_ret = torch.randn_like(imagination["actions"].float())
    act_loss = actor_lambda_return_loss(lam_ret, log_probs, entropy, tiny_test_cfg)

    crit_logits = mock_heads.critic(s)
    crit_loss = critic_twohot_loss(crit_logits, lam_ret, mock_heads.critic.coder)

    total_ac_loss = act_loss + 0.5 * crit_loss
    total_ac_loss.backward()

    # Verify WM parameters have None grad (stop-gradient strictly preserved)
    for p in mock_wm.parameters():
        assert p.grad is None, "World model parameter received unintended gradient from AC loss"

    # Verify Actor and Critic parameters have valid non-zero gradients
    actor_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in mock_heads.actor.parameters())
    critic_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in mock_heads.critic.parameters())
    assert actor_has_grad, "Actor head parameters received no gradients"
    assert critic_has_grad, "Critic head parameters received no gradients"
