"""Unit tests for F3 quantity-aware subgoal emission and F8 craft-execution step."""

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig, NUM_ACHIEVEMENTS, NUM_ACTIONS
from atlas.goals import GoalBank, GoalEntry, PredecessorLift


class StubReadout(torch.nn.Module):
    def __init__(self, have_wood: float):
        super().__init__()
        self.have_wood = have_wood
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, s):
        vec = torch.zeros(16)
        vec[0] = self.have_wood  # dim 0 = wood (log1p scale)
        return vec.unsqueeze(0), torch.ones(16).unsqueeze(0)


def test_quantity_aware_subgoal_emission_and_cycles():
    """V3 F3 verification:

    goal with learned cost 2 wood, have 0 -> selects wood-cloud goal with cycles=2;
    after 1 simulated collection (have=1) -> cycles=1; have >= cost -> original goal.
    Test that no substitution occurs when no ci>1.2 edge exists.
    """
    cfg = AtlasConfig()
    cfg.goals.material_gating_learned = True
    cfg.goals.lift_ci_threshold = 1.2
    cfg.goals.max_cycles = 4

    fb = FrozenBackbone(config=cfg, device="cpu")
    if not hasattr(fb.model, "loaded_component_keys"):
        fb.model.loaded_component_keys = set()
    fb.model.loaded_component_keys.add("state_delta_head")

    # Mock inventory_delta to return cost 2 wood: log1p(2) = 1.0986 for action 8 (place_table)
    def mock_inv_delta(s, a):
        # returns [B, 16]
        deltas = torch.zeros((len(a), 16))
        for idx, act in enumerate(a):
            if int(act.item()) == 8:
                deltas[idx, 0] = -1.10  # consumes ~2 wood
        return deltas

    fb.inventory_delta = mock_inv_delta

    # Dim 0: collect_wood; Dim 1: place_table
    WOOD_DIM = 0
    TABLE_DIM = 1

    wood_key = np.zeros(128, dtype=np.float32)
    wood_key[0] = 1.0
    wood_goal = GoalEntry(
        key=wood_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[WOOD_DIM],
        kind="event",
    )

    table_key = np.zeros(128, dtype=np.float32)
    table_key[1] = 1.0
    table_action_profile = np.zeros((17, 22), dtype=np.float32)
    table_action_profile[8, TABLE_DIM] = 0.80  # place_table is top affordance action
    table_goal = GoalEntry(
        key=table_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=table_action_profile,
        event_dims=[TABLE_DIM],
        kind="event",
    )

    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.goals.add(wood_goal)
    agent.goals.add(table_goal)

    # 1. Lift graph has prerequisite edge: wood precedes table with ci_low = 2.5 > 1.2
    agent.lift.ci_low[WOOD_DIM, TABLE_DIM] = 2.5
    agent.lift.lift[WOOD_DIM, TABLE_DIM] = 3.0

    # Case A: have = 0 wood (log1p = 0.0) -> cost 1.10 - have 0.0 = 1.10 -> ceil = 2 cycles
    agent.readout_net = StubReadout(have_wood=0.0)
    agent.reset_episode()

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    # Agent should emit material subgoal targeting wood-cloud with expected_cycles = 2
    assert agent.current_goal_mode == "material_subgoal"
    assert agent.material_subgoal_cycles == 2
    assert agent.target_goal_pending is not None
    assert TABLE_DIM in agent.target_goal_pending.event_dims
    assert WOOD_DIM in agent.active_goal.event_dims

    # Case B: simulated 1 wood collected (have = log1p(1) = 0.693)
    # Deficit = 1.10 - 0.693 = 0.407 -> ceil = 1 cycle
    agent.readout_net = StubReadout(have_wood=np.log1p(1.0))
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    assert agent.current_goal_mode == "material_subgoal"
    assert agent.material_subgoal_cycles == 1

    # Case C: have >= cost (have = 2 wood -> log1p(2) = 1.0986 >= 1.10 - margin)
    # No deficit! Target goal is directly pursued without material substitution
    agent.readout_net = StubReadout(have_wood=np.log1p(2.0))
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    # Should NOT emit material subgoal
    assert agent.current_goal_mode != "material_subgoal"
    assert agent.target_goal_pending is None

    # Case D: have = 0 wood, but NO lift edge exists (ci_low <= 1.2)
    agent.readout_net = StubReadout(have_wood=0.0)
    agent.lift.ci_low[WOOD_DIM, TABLE_DIM] = 0.0
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    # Honest failure: no substitution occurs when no learned predecessor exists
    assert agent.current_goal_mode != "material_subgoal"
    assert agent.target_goal_pending is None

    # Case E: lift edge exists, but ItemEventMap has NO confident link (dim 0 -> None)
    agent.lift.ci_low[WOOD_DIM, TABLE_DIM] = 2.5
    agent.item_event_map = {0: None}
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    # Honest failure: dim with no confident link emits NO material subgoal
    assert agent.current_goal_mode != "material_subgoal"
    assert agent.target_goal_pending is None

    # Case F: ItemEventMap has confident link (dim 0 -> WOOD_DIM)
    agent.item_event_map = {0: WOOD_DIM}
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)
    assert agent.current_goal_mode == "material_subgoal"
    assert agent.material_subgoal_cycles == 2


def test_craft_execution_step():
    """V3 F8 verification:

    table goal, have >= cost, affordance 0.4 -> PROBE chosen without navigation (EXECUTE-CRAFT);
    affordance 0.01 -> material / navigation path;
    post-place tagged-view routing works.
    """
    cfg = AtlasConfig()
    cfg.goals.material_gating_learned = True
    cfg.goals.probe_affordance_min = 0.05
    cfg.goals.probe_steps = 6

    fb = FrozenBackbone(config=cfg, device="cpu")
    if not hasattr(fb.model, "loaded_component_keys"):
        fb.model.loaded_component_keys = set()
    fb.model.loaded_component_keys.add("state_delta_head")

    TABLE_DIM = 1
    WOOD_DIM = 0

    table_key = np.zeros(128, dtype=np.float32)
    table_key[1] = 1.0
    table_goal = GoalEntry(
        key=table_key,
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[TABLE_DIM],
        kind="event",
    )
    table_goal.action_profile[8, TABLE_DIM] = 0.80

    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.goals.add(table_goal)
    agent.lift.ci_low[WOOD_DIM, TABLE_DIM] = 2.5
    agent.lift.lift[WOOD_DIM, TABLE_DIM] = 3.0

    # Sufficient wood (2 wood)
    agent.readout_net = StubReadout(have_wood=np.log1p(2.0))

    # Mock delta: action 8 costs 2 wood
    def mock_inv_delta(s, a):
        deltas = torch.zeros((len(a), 16))
        for idx, act in enumerate(a):
            if int(act.item()) == 8:
                deltas[idx, 0] = -1.10
        return deltas

    fb.inventory_delta = mock_inv_delta

    # Subtest 1: affordance is 0.40 (>= 0.05) -> EXECUTE-CRAFT chosen immediately (PROBE mode, no navigation)
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    def mock_event_logits_high(s, m):
        logits = torch.zeros((1, 17, 22))
        logits[0, 8, TABLE_DIM] = 0.40  # affordance 0.40
        return logits

    fb.event_logits_all_actions = mock_event_logits_high
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    assert telem["mode"] == "PROBE", "Should transition immediately to PROBE for EXECUTE-CRAFT"
    assert telem["goal_mode"] == "execute_craft"
    assert agent.nav_steps == 0

    # Subtest 2: affordance is 0.01 (< 0.05) -> EXECUTE-CRAFT NOT emitted; normal navigation pursued
    def mock_event_logits_low(s, m):
        logits = torch.zeros((1, 17, 22))
        logits[0, 8, TABLE_DIM] = 0.01  # affordance below 0.05
        return logits

    fb.event_logits_all_actions = mock_event_logits_low
    agent.reset_episode()
    act, telem = agent.act(obs, reward_prev=0.0, step=0)

    assert telem["goal_mode"] != "execute_craft"
    assert telem["mode"] == "NAVIGATE"
