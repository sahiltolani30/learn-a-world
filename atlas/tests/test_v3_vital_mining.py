"""Unit tests for F7 windowed realized-vital mining and play-time homeostatic gating."""

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig
from atlas.goals import GoalBank, GoalEntry
from atlas.selflearn import mine_goals_and_lift


def test_windowed_realized_vital_mining_drink_event():
    """V3 F7: synthetic episode with a +1 drink event at t=40 mines a vital goal at water view;

    no goal mined for +0.2 drift.
    """
    cfg = AtlasConfig()
    cfg.goals.vital_window = 30
    cfg.goals.vital_min_delta = 1.0  # 1.0 unit raw Crafter vital increase

    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    T = 60
    z = np.random.randn(T, 128).astype(np.float32)
    h = np.random.randn(T, 256).astype(np.float32)
    actions = np.zeros(T, dtype=np.int64)
    events = np.zeros((T, 22), dtype=bool)
    mask_before = np.zeros((T, 22), dtype=bool)

    # In Crafter vitals target vector (dims 12..15 in inventory), values are divided by 16.0
    # Create inventory array: baseline vital is 5/16.0 = 0.3125
    inventory = np.zeros((T, 16), dtype=np.float32)
    inventory[:, 12:16] = 5.0 / 16.0

    # At t=40, player drinks water: drink vital (dim 14) increases by +1 unit (to 6.0/16.0)
    # The restoration takes effect at t=41
    inventory[41:, 14] = 6.0 / 16.0

    # Also add small +0.2 drift on food (dim 13) at t=10..t=20 (e.g. 5.2/16.0, < 1.0 unit)
    inventory[15:25, 13] = 5.2 / 16.0

    # Unique visual key at t=40 (the water view)
    water_view = np.ones(128, dtype=np.float32)
    water_view /= np.linalg.norm(water_view)
    z[40] = water_view

    ep = {
        "z": z,
        "h": h,
        "actions": actions,
        "events": events,
        "mask_before": mask_before,
        "inventory": inventory,
    }

    mine_goals_and_lift(agent, [ep], iter_idx=1)

    vital_goals = [g for g in agent.goals.entries if g.kind == "vital"]
    assert len(vital_goals) >= 1, "Expected at least 1 vital goal mined"

    # Verify that the vital goal was mined around the drink restoration (t=40)
    # and NOT for the +0.2 drift at t=10
    has_water_goal = any(np.dot(g.key, water_view) > 0.95 for g in vital_goals)
    assert has_water_goal, "Water view at t=40 should be stored as vital goal"

    # Test episode with only +0.2 drift (no +1 restoration)
    agent.goals = GoalBank(cfg.goals)
    inventory_drift_only = np.zeros((T, 16), dtype=np.float32)
    inventory_drift_only[:, 12:16] = 5.0 / 16.0
    inventory_drift_only[15:25, 13] = 5.2 / 16.0  # +0.2 units

    ep_drift = {
        "z": z,
        "h": h,
        "actions": actions,
        "events": events,
        "mask_before": mask_before,
        "inventory": inventory_drift_only,
    }

    mine_goals_and_lift(agent, [ep_drift], iter_idx=1)
    vital_goals_drift = [g for g in agent.goals.entries if g.kind == "vital"]
    assert len(vital_goals_drift) == 0, "No vital goal should be mined for +0.2 unit drift"


def test_vital_goals_homeostatic_selection():
    """V3 F7: vital goals are selectable when readout vitals fall below 0.35,

    and low vitals enable vital goals in GoalBank.select.
    """
    cfg = AtlasConfig()
    cfg.goals.vital_select_below = 0.35

    bank = GoalBank(cfg.goals)
    # Add 1 vital goal and 1 event goal
    vital_goal = GoalEntry(
        key=np.ones(128, dtype=np.float32) / np.sqrt(128),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[],
        kind="vital",
    )
    event_goal = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.ones((17, 22), dtype=np.float32) * 0.5,
        event_dims=[0],
        kind="event",
    )
    event_goal.key[0] = 1.0

    bank.add(vital_goal)
    bank.add(event_goal)

    z = np.zeros(128, dtype=np.float32)
    h = np.zeros(256, dtype=np.float32)
    m_soft = np.zeros(22, dtype=np.float32)

    class StubReadout(torch.nn.Module):
        def __init__(self, vital_val: float):
            super().__init__()
            self.vital_val = vital_val
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, s):
            out = torch.ones(16) * 0.5
            out[12:16] = self.vital_val
            return out.unsqueeze(0), torch.ones(16).unsqueeze(0)

    # 1. Healthy agent: vitals = 0.50 (above 0.35) -> vital goals NOT selectable
    readout_healthy = StubReadout(vital_val=0.50)
    selected_healthy, _ = bank.select(
        z, h, m_soft,
        psi_fn=lambda s, g: 0.8,
        readout_net=readout_healthy,
    )
    assert selected_healthy is not None
    assert selected_healthy.kind == "event", "Healthy agent should not select vital goal"

    # 2. Thirsty/hungry agent: vitals = 0.20 (below 0.35) -> vital goal IS selectable
    readout_low = StubReadout(vital_val=0.20)
    # If event goal is removed or locked, vital goal is chosen
    bank.entries = [vital_goal]
    selected_low, score = bank.select(
        z, h, m_soft,
        psi_fn=lambda s, g: 0.8,
        readout_net=readout_low,
    )
    assert selected_low is not None
    assert selected_low.kind == "vital", "Low-vitals agent must be able to select vital goal"
