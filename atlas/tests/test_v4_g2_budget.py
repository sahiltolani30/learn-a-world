"""Verification tests for G2 Chain-Aware Material Budget (atlas/arbiter.py, atlas/goals.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.config import AtlasConfig, NUM_ACTIONS, NUM_ACHIEVEMENTS
from atlas.goals import GoalEntry, GoalBank, PredecessorLift


class StubDeltaBackbone:
    """Stub backbone returning controlled learned delta head costs and event affordances."""
    def __init__(self, action_table_cost: float = 2.0, action_pickaxe_cost: float = 1.0):
        self.device = "cpu"
        self.has_learned_delta_head = True
        self.action_table_cost = action_table_cost
        self.action_pickaxe_cost = action_pickaxe_cost

    def encode(self, obs_uint8):
        return torch.randn(1, 128)

    def recurrent_step(self, z, a_prev, h_prev):
        return torch.randn(1, 256)

    def composite(self, z, h):
        return torch.cat([z, h], dim=-1)

    def event_logits_all_actions(self, s, mask):
        # Action 8 is best affordance for table (event 20)
        # Action 11 is best affordance for wood pickaxe (event 15)
        # Action 5 is best affordance for wood (event 6)
        probs = torch.full((s.shape[0], NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.01)
        probs[:, 8, 20] = 0.90  # table
        probs[:, 11, 15] = 0.90  # wood pickaxe
        probs[:, 5, 6] = 0.90   # wood
        return probs

    def inventory_delta(self, s, action):
        # Item 1 is wood. Negative delta represents consumption.
        deltas = torch.zeros(1, 16)
        a = int(action.item()) if torch.is_tensor(action) else int(action)
        if a == 8:
            # Table costs 2 wood
            deltas[0, 1] = -self.action_table_cost
        elif a == 11:
            # Pickaxe costs 1 wood
            deltas[0, 1] = -self.action_pickaxe_cost
        return deltas

    def q_mean(self, s, a):
        return torch.tensor([0.0])


class StubReadoutNet(torch.nn.Module):
    def __init__(self, wood_have: float = 1.0):
        super().__init__()
        self.wood_have = wood_have

    def forward(self, s):
        out = torch.zeros(s.shape[0], 16)
        out[:, 1] = self.wood_have
        out[:, 12:16] = 9.0  # healthy vitals
        return out, None


def test_g2_chain_budget_computation():
    """Verify chain budget sums learned costs along lift graph:
    table-action: 2 wood, pickaxe-action: 1 wood -> chain budget 3.
    have=1 -> deficit 2 -> cycles 2.
    """
    cfg = AtlasConfig()
    cfg.goals.material_gating_learned = True
    backbone = StubDeltaBackbone(action_table_cost=2.0, action_pickaxe_cost=1.0)
    readout = StubReadoutNet(wood_have=1.0)

    agent = AtlasAgent(cfg=cfg, backbone=backbone)
    agent.readout_net = readout

    # Setup lift: Table (20) is predecessor of Pickaxe (15) with ci_low > 1.2
    agent.lift = PredecessorLift()
    agent.lift.ci_low[20, 15] = 1.50
    agent.lift.lift[20, 15] = 3.0

    # ItemEventMap: item 1 (wood) -> event 6 (collect_wood)
    agent.item_event_map = {1: 6}

    # Target goal: Pickaxe (event 15)
    g_pickaxe = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[15],
        kind="event",
    )
    # Goal in bank for wood (event 6)
    g_wood = GoalEntry(
        key=np.ones(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[6],
        kind="event",
    )
    agent.goals.entries = [g_pickaxe, g_wood]

    # Compute chain budget for Pickaxe (15)
    budget, deficit_diffs, d_star = agent.compute_chain_budget(
        target_dims=[15],
        s=torch.zeros(1, 384),
        all_event_probs=backbone.event_logits_all_actions(torch.zeros(1, 384), None).squeeze(0),
    )

    # Budget for wood (item 1) should be 2.0 (table) + 1.0 (pickaxe) = 3.0
    assert abs(budget[1] - 3.0) < 1e-4, f"Expected chain budget 3.0, got {budget[1]}"
    # With have=1.0, deficit is 3.0 - 1.0 = 2.0
    assert abs(deficit_diffs[1] - 2.0) < 1e-4, f"Expected deficit 2.0, got {deficit_diffs[1]}"
    assert d_star == 1, f"Expected d_star=1 (wood), got {d_star}"


def test_g2_single_unlock_dropout_fix():
    """Verify that when active_material_deficit_dim is active, GoalBank.select
    treats the deficit item's event dimension as locked even if m_soft is 1.0.
    """
    cfg = AtlasConfig()
    bank = GoalBank(cfg=cfg.goals)

    # Wood goal (event 6)
    g_wood = GoalEntry(
        key=np.ones(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[6],
        kind="event",
    )
    # Set high action probability for wood
    g_wood.action_profile[5, 6] = 0.95
    bank.entries = [g_wood]

    # Agent has already unlocked wood: m_soft[6] = 1.0
    m_soft = np.zeros(22, dtype=np.float32)
    m_soft[6] = 1.0

    z = np.zeros(128, dtype=np.float32)
    h = np.zeros(256, dtype=np.float32)

    def dummy_psi(s, k):
        return 0.8

    # Case A: without active material deficit dim, unlock term is 0 (unlock drop-out)
    chosen_normal, score_normal = bank.select(
        z=z, h=h, m_soft=m_soft, psi_fn=dummy_psi,
        active_material_deficit_dim=None, item_event_map={1: 6},
    )

    # Case B: with active_material_deficit_dim=1 (wood), m_soft[6] is forced to 0
    chosen_active, score_active = bank.select(
        z=z, h=h, m_soft=m_soft, psi_fn=dummy_psi,
        active_material_deficit_dim=1, item_event_map={1: 6},
    )
    assert chosen_active is not None
    # Score with active deficit should be significantly higher due to unlock potential (> +0.8)
    assert score_active > score_normal + 0.8, f"Expected boost in unlock score: {score_active:.2f} vs {score_normal:.2f}"
