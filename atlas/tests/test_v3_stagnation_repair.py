"""Unit tests for F6 stagnation repair and telemetry."""

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig
from atlas.goals import GoalEntry


def _vector_with_cos(target: np.ndarray, cos_val: float) -> np.ndarray:
    """Constructs a unit vector having exact cosine with target."""
    ortho = np.random.randn(len(target)).astype(np.float32)
    ortho -= float(np.dot(ortho, target)) * target
    ortho /= np.linalg.norm(ortho)
    sin_val = np.sqrt(max(0.0, 1.0 - cos_val ** 2))
    v = cos_val * target + sin_val * ortho
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_stagnation_repair_rising_cos_no_abort():
    """V3 F6: 12 flat-psi steps with rising cos -> no abort."""
    cfg = AtlasConfig()
    cfg.skills.stagnation_patience = 24
    cfg.skills.wander_steps = 6
    cfg.skills.arrival_cosine = 0.85

    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    goal_key = np.zeros(128, dtype=np.float32)
    goal_key[0] = 1.0  # unit vector

    goal = GoalEntry(
        key=goal_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.active_goal = goal
    agent.mode = "NAVIGATE"
    agent.progress_best = 0.50

    # Simulate 12 steps where reach_net psi is flat at 0.50, but visual cos is rising:
    # 0.52, 0.54, 0.56, ... (rising cos improves progress_t = max(psi, cos))
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(12):
        sim_cos = 0.52 + 0.02 * i
        mock_z = _vector_with_cos(goal_key, sim_cos)
        agent.backbone.encode = lambda o, z_mock=mock_z: torch.from_numpy(z_mock).float().unsqueeze(0)

        # Mock reach_net to return flat 0.50
        agent.reach_net = lambda s, g: torch.tensor([0.50])

        act, telem = agent.act(obs, reward_prev=0.0, step=i + 1)
        assert telem["mode"] == "NAVIGATE", f"Step {i+1} aborted prematurely"

    assert agent.stagnation_counter == 0, "Stagnation counter should reset on rising progress"


def test_stagnation_repair_flat_progress_aborts_at_24_steps():
    """V3 F6: 24 steps flat on both psi and cos -> abort; wander block exactly 6 steps."""
    cfg = AtlasConfig()
    cfg.skills.stagnation_patience = 24
    cfg.skills.wander_steps = 6
    cfg.skills.arrival_cosine = 0.85

    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    goal_key = np.zeros(128, dtype=np.float32)
    goal_key[0] = 1.0

    goal = GoalEntry(
        key=goal_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.active_goal = goal
    agent.mode = "NAVIGATE"
    agent.progress_best = 0.70  # Higher than current 0.50

    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # Flat mock: cos is 0.50 (flat < 0.85), psi is 0.50 (flat)
    mock_z = _vector_with_cos(goal_key, 0.50)
    agent.backbone.encode = lambda o, z_mock=mock_z: torch.from_numpy(z_mock).float().unsqueeze(0)
    agent.reach_net = lambda s, g: torch.tensor([0.50])

    for step in range(1, 24):
        act, telem = agent.act(obs, reward_prev=0.0, step=step)
        assert telem["mode"] == "NAVIGATE", f"Step {step} should not abort before 24 flat steps"
        assert agent.stagnation_counter == step

    # Step 24: stagnation reaches patience 24 -> aborts into WANDER
    act, telem = agent.act(obs, reward_prev=0.0, step=24)
    assert telem["mode"] == "WANDER"
    assert telem["abort_reason"] == "stagnation"
    # wander_steps_remaining initialized to wander_steps - 1 = 5
    assert agent.wander_steps_remaining == 5

    # Exactly 5 more steps of WANDER remaining (total 6 steps in WANDER block)
    for w_step in range(5):
        act, telem = agent.act(obs, reward_prev=0.0, step=25 + w_step)
        assert telem["mode"] == "WANDER"

    # Next step: wander block expired, returns to FALLBACK (or selects new goal)
    act, telem = agent.act(obs, reward_prev=0.0, step=30)
    assert telem["mode"] != "WANDER"
