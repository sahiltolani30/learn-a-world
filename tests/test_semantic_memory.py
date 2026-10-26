"""
tests/test_semantic_memory.py
==============================

Unit tests for Dual-Memory Cognitive World Model Engine:
- ShortTermWorkingMemory: rolling FIFO, landmark persistence, object permanence.
- LongTermSemanticMemory: precondition gating, cosine matching, serialization.
- AutonomousMemoryWriter: online breakthrough detection.
- RecurrentMPCPlanner integration: memory-guided planning.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from planning.semantic_memory import (
    AutonomousMemoryWriter,
    DopaminePersonalBestTracker,
    LongTermSemanticMemory,
    NUM_ACHIEVEMENTS,
    ShortTermWorkingMemory,
)


def test_short_term_working_memory_basic():
    wm = ShortTermWorkingMemory(capacity=16, latent_dim=128, recurrent_dim=256)
    assert len(wm.latent_window) == 0
    assert wm.current_step == 0

    # Step 1: Add a step
    z_0 = torch.randn(128)
    h_0 = torch.zeros(256)
    wm.update(z_t=z_0, action=2, reward=0.0, next_h=h_0)
    assert len(wm.latent_window) == 1
    assert wm.current_step == 1
    assert wm.action_window[0] == 2

    # Step 2: Test landmark anchor placement (e.g. table placed)
    z_table = torch.randn(128)
    wm.update(z_t=z_table, action=11, reward=1.0, next_h=h_0, unlocked_event="place_table")
    assert wm.has_anchor("place_table")
    retrieved_anchor = wm.get_anchor("place_table")
    assert retrieved_anchor is not None
    assert torch.allclose(retrieved_anchor, z_table.flatten(), atol=1e-5)

    # Reset
    wm.reset()
    assert len(wm.latent_window) == 0
    assert not wm.has_anchor("place_table")


def test_long_term_semantic_memory_precondition_gating():
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")
    assert len(ltm) == 0

    # Query on empty memory returns None
    z_query = torch.randn(128)
    inv = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    act, sim, name = ltm.query(z_query, inv)
    assert act is None
    assert sim == 0.0

    # Add Entry 1: Pickaxe craft requires Wood and Table (indices 6 and 17)
    wood_idx = CRAFTER_ACHIEVEMENTS.index("collect_wood")
    table_idx = CRAFTER_ACHIEVEMENTS.index("place_table")
    pick_idx = CRAFTER_ACHIEVEMENTS.index("make_wood_pickaxe")

    p_pick = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    p_pick[wood_idx] = 1
    p_pick[table_idx] = 1

    outcome_pick = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    outcome_pick[pick_idx] = 1

    # Key is a normalized vector
    key_pick = torch.randn(128)
    key_pick = key_pick / key_pick.norm()

    ltm.add_entry(
        key=key_pick,
        preconditions=p_pick,
        action=15,  # make_wood_pickaxe
        outcome=outcome_pick,
        realized_return=5.1,
        outcome_name="make_wood_pickaxe",
    )
    assert len(ltm) == 1

    # TEST CASE A: Agent does NOT have wood or table
    inv_empty = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    act_a, sim_a, _ = ltm.query(key_pick, inv_empty, sim_threshold=0.50)
    assert act_a is None, "Should block retrieval because preconditions are not met!"

    # TEST CASE B: Agent has table and wood, and visual matches key
    inv_ready = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    inv_ready[wood_idx] = 1
    inv_ready[table_idx] = 1

    act_b, sim_b, name_b = ltm.query(key_pick, inv_ready, sim_threshold=0.50)
    assert act_b == 15, "Should retrieve action 15 (make_wood_pickaxe)!"
    assert sim_b > 0.99, f"Cosine similarity to identical key should be ~1.0, got {sim_b}"
    assert name_b == "make_wood_pickaxe"

    # TEST CASE C: Agent ALREADY has pickaxe -> Should not retrieve pickaxe again
    inv_has_pick = inv_ready.copy()
    inv_has_pick[pick_idx] = 1
    act_c, _, _ = ltm.query(key_pick, inv_has_pick, sim_threshold=0.50)
    assert act_c is None, "Should block retrieval because milestone is already unlocked!"


def test_long_term_semantic_memory_save_load(tmp_path):
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")
    key = torch.randn(128)
    p = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    o = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    o[2] = 1
    ltm.add_entry(key=key, preconditions=p, action=3, outcome=o, realized_return=2.0, outcome_name="test_event")

    save_p = tmp_path / "test_memory.pt"
    ltm.save(save_p)
    assert save_p.exists()

    ltm2 = LongTermSemanticMemory(latent_dim=128, device="cpu")
    ltm2.load(save_p)
    assert len(ltm2) == 1
    assert ltm2.actions[0].item() == 3
    assert ltm2.outcome_names[0] == "test_event"


def test_autonomous_memory_writer():
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")
    writer = AutonomousMemoryWriter(ltm)

    # Step without achievement
    z_0 = torch.randn(128)
    info_empty = {"achievements": {"collect_wood": 0}}
    res, surge = writer.step(z_t=z_0, action=1, reward=0.0, info=info_empty)
    assert res is None
    assert surge == 0.0
    assert len(ltm) == 0

    # Step with achievement unlock
    info_wood = {"achievements": {"collect_wood": 1}}
    res_wood, surge_wood = writer.step(z_t=z_0, action=2, reward=1.0, info=info_wood)
    assert res_wood == "collect_wood"
    assert len(ltm) == 1
    assert ltm.actions[0].item() == 2
    assert ltm.outcome_names[0] == "collect_wood"


def test_recurrent_mpc_planner_with_semantic_memory():
    model = JEPAv3(latent_dim=128, recurrent_dim=256, num_actions=17)
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")

    # Add a learned pickaxe memory
    wood_idx = CRAFTER_ACHIEVEMENTS.index("collect_wood")
    table_idx = CRAFTER_ACHIEVEMENTS.index("place_table")
    pick_idx = CRAFTER_ACHIEVEMENTS.index("make_wood_pickaxe")

    p = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    p[wood_idx] = 1
    p[table_idx] = 1
    o = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
    o[pick_idx] = 1

    # Fake table visual key
    fake_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    obs_t = torch.from_numpy(fake_obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        key_table = model.encode(obs_t, use_target=True).squeeze(0)

    ltm.add_entry(key=key_table, preconditions=p, action=15, outcome=o, realized_return=5.1, outcome_name="make_wood_pickaxe")

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=3,
        num_candidates=8,
        cem_iterations=1,
        semantic_memory=ltm,
        memory_sim_threshold=0.50,
        memory_inject_ratio=0.50,
        device="cpu",
    )
    belief = AgentBeliefV3(recurrent_dim=256)
    # Set belief to have wood and table
    inv = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
    inv[wood_idx] = 1.0
    inv[table_idx] = 1.0
    belief.set_mask(inv)

    # When facing the exact matching image, planner should query memory and return a plan
    action, debug = planner.plan(fake_obs, belief)
    assert 0 <= action < 17
    assert "memory_retrieved_action" in debug
    assert debug["memory_retrieved_action"] == 15.0
    assert debug["memory_match_sim"] > 0.90


def test_dopamine_personal_best_tracker():
    tracker = DopaminePersonalBestTracker(dopamine_scale=0.50)
    tracker.reset_episode()

    # Step 1: No milestones
    surge_0 = tracker.update_step(reward=0.0, newly_unlocked=[])
    assert surge_0 == 0.0

    # Step 2: First-ever discovery of wood (unprecedented achievement)
    surge_1 = tracker.update_step(reward=1.0, newly_unlocked=["collect_wood"])
    assert surge_1 > 0.0, "First-ever discovery should trigger dopamine surge!"
    assert "collect_wood" in tracker.lifetime_unlocked_set

    # End episode 1, start episode 2
    tracker.reset_episode()
    # Step 1 in ep 2: wood already seen, but achieves 2 milestones (new milestone count PB)
    tracker.update_step(reward=1.0, newly_unlocked=["collect_wood"])
    surge_2 = tracker.update_step(reward=1.0, newly_unlocked=["place_table"])
    assert surge_2 > 0.0, "Breaking milestone count PB should trigger dopamine surge!"
    assert tracker.lifetime_best_milestones == 2

