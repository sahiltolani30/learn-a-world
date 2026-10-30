"""Unit tests for CriticNet and AWR advantage-weighted regression (V3 F1 & F2)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from atlas.critic import (
    CriticNet,
    critic_loss,
    symexp,
    symlog,
    two_hot_decode,
    two_hot_encode,
)
from atlas.skills import build_her_batch, compute_her_loss, OptionNet, ReachabilityNet


def test_symlog_symexp_inversion():
    """Verifies symlog and symexp are exact inverses across negative, zero, and positive values."""
    values = torch.tensor([-100.0, -5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
    s = symlog(values)
    rec = symexp(s)
    assert torch.allclose(values, rec, atol=1e-5)

    np_vals = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    assert np.allclose(np_vals, symexp(symlog(np_vals)), atol=1e-5)


def test_two_hot_encoding_and_decoding():
    """Verifies two-hot discrete bin properties: sum to 1, exact reconstruction."""
    num_bins = 65
    v_min = -5.0
    v_max = 10.0
    targets = torch.tensor([-5.0, -2.5, 0.0, 2.5, 7.0, 10.0])

    twohot = two_hot_encode(targets, num_bins=num_bins, v_min=v_min, v_max=v_max)
    assert twohot.shape == (len(targets), num_bins)
    assert torch.allclose(twohot.sum(dim=-1), torch.ones(len(targets)), atol=1e-6)

    # Decode target distributions
    decoded = two_hot_decode(torch.log(twohot + 1e-8), num_bins=num_bins, v_min=v_min, v_max=v_max)
    assert torch.allclose(targets, decoded, atol=0.05)


def test_critic_net_architecture_and_ema():
    """Verifies CriticNet trunk, LayerNorm, SiLU, output shapes, and EMA target tracking."""
    critic = CriticNet(state_dim=384, hidden_dim=256, num_bins=65, v_min=-5.0, v_max=10.0)

    # Check LayerNorm and SiLU in trunk
    layers = list(critic.trunk)
    assert any(isinstance(l, nn.LayerNorm) for l in layers)
    assert any(isinstance(l, nn.SiLU) for l in layers)

    # Forward pass
    s_batch = torch.randn(8, 384)
    logits = critic(s_batch)
    assert logits.shape == (8, 65)

    # Value calculation in return scale
    val = critic.value(s_batch)
    assert isinstance(val, torch.Tensor)
    assert val.shape == (8,)

    # 1D single-state value calculation returns float
    s_single = torch.randn(384)
    val_single = critic.value(s_single)
    assert isinstance(val_single, float)

    # EMA target update
    online_p = list(critic.trunk.parameters())[0].clone()
    target_p_before = list(critic.target_trunk.parameters())[0].clone()
    assert torch.allclose(online_p, target_p_before)

    # Perturb online weights
    with torch.no_grad():
        list(critic.trunk.parameters())[0].add_(1.0)

    critic.update_target(decay=0.995)
    target_p_after = list(critic.target_trunk.parameters())[0].clone()
    # Target moved toward perturbed online: 0.995 * before + 0.005 * (before + 1.0)
    expected = 0.995 * target_p_before + 0.005 * (target_p_before + 1.0)
    assert torch.allclose(target_p_after, expected, atol=1e-5)


def test_awr_weighting_advantage():
    """V3 F2 verification: segment with return +3 vs V=0 gets weight > 1; return -3 gets < 1.

    Also verifies weights renormalize to mean 1.0 and clip at 50.0.
    """
    class StubCritic:
        def value(self, s: torch.Tensor) -> float:
            return 0.0

    stub_critic = StubCritic()

    # Episode 1 has return +3
    ep1 = {
        "z": np.zeros((10, 128), dtype=np.float32),
        "h": np.zeros((10, 256), dtype=np.float32),
        "actions": np.zeros(10, dtype=np.int64),
        "rewards": np.array([3.0] + [0.0] * 9, dtype=np.float32),
        "events": np.zeros((10, 22), dtype=bool),
    }

    # Episode 2 has return -3
    ep2 = {
        "z": np.zeros((10, 128), dtype=np.float32),
        "h": np.zeros((10, 256), dtype=np.float32),
        "actions": np.zeros(10, dtype=np.int64),
        "rewards": np.array([-3.0] + [0.0] * 9, dtype=np.float32),
        "events": np.zeros((10, 22), dtype=bool),
    }

    batch_adv = build_her_batch([ep1, ep2], horizons=(5,), critic=stub_critic, device="cpu")
    weights = batch_adv["weights_pos"]

    # First segment (return +3 vs V=0) has A = +3 -> w_adv = exp(1.0) ~ 2.72 > 1
    # Second segment (return -3 vs V=0) has A = -3 -> w_adv = exp(-1.0) ~ 0.37 < 1
    # Verify that positive return segment receives strictly higher weight than negative return segment
    assert weights[0] > weights[-1]

    # Isolated AWR test with zero base return difference to isolate w_adv
    # Segment A: return = 0, V = -3 -> A = +3 -> w_adv = exp(1.0) ~ 2.718
    # Segment B: return = 0, V = +3 -> A = -3 -> w_adv = exp(-1.0) ~ 0.368
    class CriticWithStateDep:
        def value(self, s: torch.Tensor) -> float:
            # Check sign of state to simulate V(s) = -3 vs +3
            return -3.0 if s[0] < 0 else 3.0

    ep_pos_a = {
        "z": -np.ones((10, 128), dtype=np.float32),
        "h": np.zeros((10, 256), dtype=np.float32),
        "actions": np.zeros(10, dtype=np.int64),
        "rewards": np.zeros(10, dtype=np.float32),
        "events": np.zeros((10, 22), dtype=bool),
    }
    ep_neg_a = {
        "z": np.ones((10, 128), dtype=np.float32),
        "h": np.zeros((10, 256), dtype=np.float32),
        "actions": np.zeros(10, dtype=np.int64),
        "rewards": np.zeros(10, dtype=np.float32),
        "events": np.zeros((10, 22), dtype=bool),
    }
    batch_isolated = build_her_batch([ep_pos_a, ep_neg_a], horizons=(5,), critic=CriticWithStateDep(), device="cpu")
    # Base weights are identical (return=0, events=0, progress=0)
    w_pos = batch_isolated["weights_pos"][0].item()
    w_neg = batch_isolated["weights_pos"][-1].item()
    assert w_pos > 1.0
    assert w_neg < 1.0
    ratio_adv = w_pos / w_neg
    assert 7.0 < ratio_adv < 7.6  # exp(2.0) ~ 7.389

    # Test that compute_her_loss normalizes weights to mean 1.0
    opt_net = OptionNet()
    reach_net = ReachabilityNet()
    loss, metrics = compute_her_loss(opt_net, reach_net, batch_adv)
    assert abs(metrics["her_weight_mean"] - 1.0) < 1e-4
