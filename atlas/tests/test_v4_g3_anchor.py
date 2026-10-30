"""Verification tests for G3 Visual Station Anchor (atlas/arbiter.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.config import AtlasConfig, NUM_ACTIONS, NUM_ACHIEVEMENTS
from atlas.goals import GoalEntry, PredecessorLift


class StubDeltaBackbone:
    def __init__(self):
        self.device = "cpu"
        self.has_learned_delta_head = True

    def encode(self, obs_uint8):
        return torch.randn(1, 128)

    def recurrent_step(self, z, a_prev, h_prev):
        return torch.randn(1, 256)

    def composite(self, z, h):
        return torch.cat([z, h], dim=-1)

    def event_logits_all_actions(self, s, mask):
        probs = torch.full((s.shape[0], NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.05)
        probs[:, 8, 20] = 0.90  # table (20)
        probs[:, 11, 15] = 0.90  # pickaxe (15)
        return probs

    def inventory_delta(self, s, action):
        n = action.shape[0] if torch.is_tensor(action) and action.ndim > 0 else 1
        deltas = torch.zeros(n, 16)
        if torch.is_tensor(action):
            for i, a in enumerate(action.view(-1)):
                if int(a) == 11:
                    deltas[i, 1] = -1.0
        return deltas

    def prior_logits_weighted(self, s, mask):
        return torch.zeros(s.shape[0], NUM_ACTIONS)

    def event_probs(self, s, a, mask):
        return torch.full((s.shape[0], NUM_ACHIEVEMENTS), 0.05)

    def q_mean(self, s, a):
        return torch.tensor([0.0])


class StubReadoutNet(torch.nn.Module):
    def __init__(self, wood_have: float = 0.0):
        super().__init__()
        self.wood_have = wood_have

    def forward(self, s):
        out = torch.zeros(s.shape[0], 16)
        out[:, 1] = self.wood_have
        out[:, 12:16] = 9.0
        return out, None


def test_g3_station_anchor_lifecycle():
    """Verify station anchor:
    (1) Pending pickaxe goal, table (20) just unlocked -> active goal becomes artifact-kind with key = tagged z.
    (2) Walking with cos >= threshold -> PROBE entered (not WANDER).
    (3) Deficit satisfied -> artifact goal released and pending target restored.
    (4) No pending target at unlock -> no anchor.
    """
    cfg = AtlasConfig()
    cfg.skills.arrival_cosine = 0.70
    backbone = StubDeltaBackbone()
    agent = AtlasAgent(cfg=cfg, backbone=backbone)
    agent.readout_net = StubReadoutNet(wood_have=0.0)  # Deficit: needs 1 wood

    # Setup lift: Table (20) is predecessor of Pickaxe (15)
    agent.lift = PredecessorLift()
    agent.lift.ci_low[20, 15] = 1.50

    # Pending target goal: Pickaxe (15)
    g_pickaxe = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[15],
        kind="event",
    )
    agent.target_goal_pending = g_pickaxe

    # Step 0: Initial step
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    agent.act(obs, reward_prev=0.0, step=0)

    # Simulate unlocking table (20) at step 1
    # Mock event prediction that unlocks 20
    agent.last_event_probs = torch.zeros(1, 22)
    agent.last_event_probs[0, 20] = 0.99
    # realized reward +1.0 triggers unlock in belief.update
    action, telemetry = agent.act(obs, reward_prev=1.0, step=1)

    # (1) Active goal should now be artifact-kind anchored at table
    assert agent.active_goal is not None, "Station anchor should be active"
    assert agent.active_goal.kind == "artifact", f"Expected kind='artifact', got {agent.active_goal.kind}"
    assert agent.active_goal.event_dims == [15]
    assert 20 in agent.episode_artifact_views

    # (2) When cos >= arrival_threshold: NAVIGATE enters PROBE
    anchor_key = agent.active_goal.key
    # Mock encode to return the anchor key (cos = 1.0)
    agent.backbone.encode = lambda obs_u: torch.from_numpy(anchor_key).unsqueeze(0)
    agent.mode = "NAVIGATE"
    agent.act(obs, reward_prev=0.0, step=2)
    assert agent.mode == "PROBE", f"Expected arrival into PROBE, got {agent.mode}"

    # (3) When deficit is satisfied (have wood >= 1.0)
    agent.readout_net.wood_have = 2.0  # deficit satisfied
    agent.probe_steps_remaining = 1  # 1 step left in PROBE mode
    agent.act(obs, reward_prev=0.0, step=3)
    # Artifact goal released and pending pickaxe restored
    assert agent.active_goal is g_pickaxe, "Pending pickaxe goal should be restored after deficit satisfied"


def test_g3_no_anchor_without_pending_target():
    """(4) No pending target at unlock -> no anchor (behavior unchanged)."""
    cfg = AtlasConfig()
    backbone = StubDeltaBackbone()
    agent = AtlasAgent(cfg=cfg, backbone=backbone)
    agent.target_goal_pending = None

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    agent.act(obs, reward_prev=0.0, step=0)

    agent.last_event_probs = torch.zeros(1, 22)
    agent.last_event_probs[0, 20] = 0.99
    agent.act(obs, reward_prev=1.0, step=1)

    assert agent.active_goal is None or agent.active_goal.kind != "artifact"
