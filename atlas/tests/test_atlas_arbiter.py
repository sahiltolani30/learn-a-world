"""Unit tests for AtlasAgent and arbiter mode machine."""

import inspect
import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig
from atlas.goals import GoalEntry


def test_agent_act_strict_signature():
    sig = inspect.signature(AtlasAgent.act)
    params = list(sig.parameters.keys())
    assert params == ["self", "obs_uint8", "reward_prev", "step"], (
        f"AtlasAgent.act signature violates strictness invariant: {params}"
    )


def test_safety_override_fires():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    # Stub safety_net so all actions have risk 0.95 except action 1 which has safe risk 0.05
    def stubbed_risk_all(s):
        r = torch.full((s.shape[0], 17), 0.95)
        r[:, 1] = 0.05
        return r

    agent.safety_net.risk_all_actions = stubbed_risk_all
    # Force intended action to action 0
    fb.prior_logits = lambda s, m: torch.tensor([[10.0] + [0.0] * 16])
    agent.epsilon = 0.0

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    action, telemetry = agent.act(obs, reward_prev=0.0, step=0)

    assert telemetry["override_fired"] is True
    assert action == 1
    assert telemetry["risk_intended"] >= 0.5


def test_mode_transitions():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # 1. Bank empty -> FALLBACK
    action, telemetry = agent.act(obs, reward_prev=0.0, step=0)
    assert telemetry["mode"] == "FALLBACK"
    assert "r_int" in telemetry

    # 2. Add distant goal -> NAVIGATE
    distant_key = np.ones(128, dtype=np.float32) * -5.0
    distant_goal = GoalEntry(
        key=distant_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.goals.add(distant_goal)
    agent.active_goal = distant_goal

    action, telemetry = agent.act(obs, reward_prev=0.0, step=1)
    assert telemetry["mode"] == "NAVIGATE"

    # 3. Add arrival goal (same view as encoded obs) -> transitions to PROBE
    with torch.no_grad():
        z_cur = fb.encode(obs).squeeze(0).cpu().numpy()

    arrived_goal = GoalEntry(
        key=z_cur,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.active_goal = arrived_goal

    action, telemetry = agent.act(obs, reward_prev=0.0, step=2)
    assert telemetry["mode"] == "PROBE"
