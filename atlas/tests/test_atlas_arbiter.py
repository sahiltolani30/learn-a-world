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
    # Force intended action to action 0 via epistemic disagreement
    agent.epistemic_net.disagreement_all_actions = lambda s: torch.tensor([[10.0] + [0.0] * 16])
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


def test_epsilon_tail_in_navigate():
    """1000 act() calls with epsilon=0.3 in NAVIGATE mode produce >= 3 distinct actions."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.epsilon = 0.3

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    distant_goal = GoalEntry(
        key=np.ones(128, dtype=np.float32) * -5.0,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.goals.add(distant_goal)

    actions = []
    for step in range(1000):
        # Keep goal active so agent stays in NAVIGATE mode
        agent.active_goal = distant_goal
        agent.mode = "NAVIGATE"
        agent.stagnation_counter = 0
        agent.nav_steps = 0
        action, _ = agent.act(obs, reward_prev=0.0, step=step)
        actions.append(action)

    assert len(set(actions)) >= 3, f"Expected >= 3 distinct actions with eps=0.3, got {len(set(actions))}"


def test_timeout_wander_ten_steps():
    """Option timeout produces exactly 10 WANDER steps before goal reselection."""
    cfg = AtlasConfig()
    cfg.skills.max_option_steps = 3
    cfg.skills.wander_steps = 10
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.epsilon = 0.0

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    distant_goal = GoalEntry(
        key=np.ones(128, dtype=np.float32) * -5.0,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.goals.add(distant_goal)
    agent.active_goal = distant_goal

    wander_streaks = []
    current_streak = 0
    modes = []
    for step in range(16):
        _, telemetry = agent.act(obs, reward_prev=0.0, step=step)
        mode = telemetry["mode"]
        modes.append(mode)
        if mode == "WANDER":
            current_streak += 1
        else:
            if current_streak > 0:
                wander_streaks.append(current_streak)
                current_streak = 0
    if current_streak > 0:
        wander_streaks.append(current_streak)

    assert len(wander_streaks) > 0 and wander_streaks[0] == 10, (
        f"Expected 10 WANDER steps in first streak, got {wander_streaks}. Mode sequence: {modes}"
    )


def test_probe_honest_success_credit():
    """Probe without belief unlock calls mark_used with success=False."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    goal = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.goals.add(goal)
    agent.active_goal = goal
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 1

    recorded_success = []
    orig_mark_used = agent.goals.mark_used

    def spy_mark_used(key, success, current_iter=0):
        recorded_success.append(success)
        return orig_mark_used(key, success, current_iter)

    agent.goals.mark_used = spy_mark_used

    # Run step: probe ends
    agent.act(obs, reward_prev=0.0, step=1)
    assert len(recorded_success) == 1
    assert recorded_success[0] is False, "Expected probe without belief unlock to report success=False"


def test_frozen_memory_protects_bank():
    """frozen_memory=True keeps goal n_try unchanged after 50 probes."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.frozen_memory = True

    goal = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[0],
        kind="event",
    )
    agent.goals.add(goal)
    initial_n_try = agent.goals.entries[0].n_try

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    for step in range(50):
        agent.active_goal = agent.goals.entries[0]
        agent.mode = "PROBE"
        agent.probe_steps_remaining = 1
        agent.act(obs, reward_prev=0.0, step=step)

    assert agent.goals.entries[0].n_try == initial_n_try == 0


def test_prior_weight_zero_ignores_prior():
    """prior_weight=0 makes FALLBACK independent of prior net."""
    cfg = AtlasConfig()
    cfg.backbone.prior_weight = 0.0
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.epsilon = 0.0
    agent.mode = "FALLBACK"
    agent.active_goal = None
    agent.probe_steps_remaining = 0
    agent.wander_steps_remaining = 0

    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # Stub prior with preference for action 0
    fb.prior_logits = lambda s, m: torch.tensor([[100.0] + [0.0] * 16])
    action1, _ = agent.act(obs, reward_prev=0.0, step=0)

    # Reset and stub prior with preference for action 10
    agent.reset_episode()
    fb.prior_logits = lambda s, m: torch.tensor([[0.0] * 10 + [100.0] + [0.0] * 6])
    action2, _ = agent.act(obs, reward_prev=0.0, step=0)

    assert action1 == action2, f"Actions differed ({action1} vs {action2}) despite prior_weight=0.0"
