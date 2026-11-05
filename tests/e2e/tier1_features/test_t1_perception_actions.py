"""Tier 1: Perception & Actions (F1, F2).

Tests:
- F1: Pixels-Only Input (64x64x3 RGB) [5 tests]
- F2: 17 Discrete Actions [5 tests]
"""
from __future__ import annotations

import inspect
import numpy as np
import pytest
import torch

from zephyr.config import NUM_ACTIONS
from zephyr.planner import ZephyrAgent
from zephyr.vec_env import ZephyrVecEnv
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 1: Pixels-Only Input (64x64x3 RGB)
# ==============================================================================

def test_f1_observation_shape_and_dtype(vec_env_factory):
    """F1: Environment observation returns strictly uint8 [N, 64, 64, 3] arrays."""
    venv = vec_env_factory(num_envs=2, base_seed=7000, max_steps=100)
    obs, infos = venv.reset()

    assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)}"
    assert obs.shape == (2, 64, 64, 3), f"Observation shape {obs.shape} != (2, 64, 64, 3)"
    assert obs.dtype == np.uint8, f"Observation dtype {obs.dtype} != uint8"
    assert int(obs.min()) >= 0 and int(obs.max()) <= 255, (
        f"Pixel values must be bounded in [0, 255], got min={obs.min()}, max={obs.max()}"
    )


def test_f1_visual_encoder_tensor_contract(mock_wm):
    """F1: Visual encoder accepts uint8 [B, 64, 64, 3] and outputs float32 [B, latent_dim] LayerNorm latents."""
    batch_size = 4
    raw_obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)

    z = mock_wm.encode(raw_obs)

    assert isinstance(z, torch.Tensor), f"Expected torch.Tensor, got {type(z)}"
    assert z.shape == (batch_size, mock_wm.cfg.world_model.latent_dim), (
        f"Encoded shape {z.shape} != ({batch_size}, {mock_wm.cfg.world_model.latent_dim})"
    )
    assert z.dtype == torch.float32, f"Encoded dtype {z.dtype} != torch.float32"
    assert torch.isfinite(z).all(), "Encoded latents must not contain NaNs or Infinities"

    # LayerNorm ensures standard deviation across features is normalized
    per_vector_std = z.std(dim=-1)
    assert torch.all(per_vector_std > 0.1), f"Latent std {per_vector_std} indicates collapsed representation"


def test_f1_encoder_invariance_to_batch_size(mock_wm):
    """F1: Encoding a single observation produces identical latents whether processed alone or batched."""
    mock_wm.eval()
    obs_batch = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8)
    obs_single = obs_batch[0:1]

    with torch.no_grad():
        z_batch = mock_wm.encode(obs_batch)
        z_single = mock_wm.encode(obs_single)

    assert torch.allclose(z_batch[0], z_single[0], atol=1e-4), (
        f"Encoder batch invariance violated: diff={torch.max(torch.abs(z_batch[0] - z_single[0]))}"
    )


def test_f1_encoder_gradient_flow(tiny_test_cfg):
    """F1: Gradients flow from latents back through all encoder layers without vanishing/exploding."""
    wm = ZephyrWM(tiny_test_cfg).train()
    obs = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)

    z = wm.encode(obs)
    # Note: LayerNorm sum is zero, so use quadratic loss (z**2).sum() to test gradient flow
    loss = (z ** 2).sum()
    loss.backward()

    for name, param in wm.encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient non-finite for {name}"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0.0, f"Vanishing gradient for {name}: norm={grad_norm}"


def test_f1_agent_act_accepts_pure_rgb_only(mock_wm, mock_heads):
    """F1: ZephyrAgent.act() interface accepts strictly raw uint8 RGB observations."""
    agent = ZephyrAgent(mock_wm, mock_heads, device="cpu")
    sig = inspect.signature(agent.act)
    params = list(sig.parameters.keys())

    assert params == ["obs_uint8"], f"Expected signature ['obs_uint8'], got {params}"

    sample_obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    action = agent.act(sample_obs)

    assert isinstance(action, (int, np.integer)), f"Expected integer action, got {type(action)}"
    assert 0 <= action < NUM_ACTIONS, f"Action {action} outside discrete space [0, {NUM_ACTIONS - 1}]"


# ==============================================================================
# Feature 2: 17 Discrete Actions
# ==============================================================================

def test_f2_discrete_action_space_validity(vec_env_factory):
    """F2: Environment accepts all 17 discrete actions without exceptions."""
    venv = vec_env_factory(num_envs=1, base_seed=7000, max_steps=100)
    venv.reset()

    for act in range(NUM_ACTIONS):
        obs, rewards, dones, infos = venv.step(np.array([act]))
        assert obs.shape == (1, 64, 64, 3), f"Observation shape {obs.shape} invalid on action {act}"
        assert rewards.shape == (1,), f"Rewards shape {rewards.shape} invalid on action {act}"
        assert dones.shape == (1,), f"Dones shape {dones.shape} invalid on action {act}"
        assert len(infos) == 1, f"Expected 1 info dict, got {len(infos)}"


def test_f2_action_noop_invariance(crafter_env_factory):
    """F2: Action 0 (noop) maintains position while advancing time."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    obs, info = env.reset(seed=7000)
    init_pos = list(env._env._player.pos)

    obs2, reward, terminated, truncated, info2 = env.step(0)
    new_pos = list(env._env._player.pos)

    assert init_pos == new_pos, f"Action 0 (noop) moved player from {init_pos} to {new_pos}"


def test_f2_directional_movement_effects(crafter_env_factory):
    """F2: Directional actions (1, 2, 3, 4) update player facing vector."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    # Action 1: move_left (facing [-1, 0])
    env.step(1)
    assert list(env._env._player.facing) == [-1, 0], (
        f"Action 1 should face [-1, 0], got {env._env._player.facing}"
    )

    # Action 2: move_right (facing [1, 0])
    env.step(2)
    assert list(env._env._player.facing) == [1, 0], (
        f"Action 2 should face [1, 0], got {env._env._player.facing}"
    )

    # Action 3: move_up (facing [0, -1])
    env.step(3)
    assert list(env._env._player.facing) == [0, -1], (
        f"Action 3 should face [0, -1], got {env._env._player.facing}"
    )

    # Action 4: move_down (facing [0, 1])
    env.step(4)
    assert list(env._env._player.facing) == [0, 1], (
        f"Action 4 should face [0, 1], got {env._env._player.facing}"
    )


def test_f2_action_embedding_layer_contract(mock_wm):
    """F2: World model embedding processes all valid action indices [0..16]."""
    batch_size = 3
    z = torch.randn(batch_size, mock_wm.cfg.world_model.latent_dim)
    h = mock_wm.init_state(batch_size, z.device)

    for action_idx in [0, 5, 10, 16]:
        actions = torch.full((batch_size,), action_idx, dtype=torch.int64)
        h_next = mock_wm.belief_step(z, actions, h)
        assert h_next.shape == (batch_size, mock_wm.cfg.world_model.belief_dim), (
            f"Belief state shape {h_next.shape} invalid for action {action_idx}"
        )
        assert torch.isfinite(h_next).all(), f"Belief state non-finite for action {action_idx}"


def test_f2_invalid_action_rejection(vec_env_factory):
    """F2: Out-of-bounds action indices (< 0 or >= 17) raise ValueError."""
    venv = vec_env_factory(num_envs=1, base_seed=7000, max_steps=100)
    venv.reset()

    with pytest.raises(ValueError, match="actions must be in 0..16"):
        venv.step(np.array([17]))

    with pytest.raises(ValueError, match="actions must be in 0..16"):
        venv.step(np.array([-1]))
