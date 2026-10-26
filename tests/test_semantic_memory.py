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
    CRAFTER_INVENTORY_KEYS,
    DopaminePersonalBestTracker,
    LongTermSemanticMemory,
    NUM_ACHIEVEMENTS,
    NUM_INVENTORY_ITEMS,
    ShortTermWorkingMemory,
    inventory_dict_to_vector,
    vector_to_inventory_dict,
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


def test_dynamic_backpack_16d_vector_conversion():
    # Test None / empty dict starts with standard Crafter initial vital stats
    default_vec = inventory_dict_to_vector(None)
    assert len(default_vec) == NUM_INVENTORY_ITEMS
    assert default_vec[CRAFTER_INVENTORY_KEYS.index("health")] == 9.0
    assert default_vec[CRAFTER_INVENTORY_KEYS.index("food")] == 9.0
    assert default_vec[CRAFTER_INVENTORY_KEYS.index("drink")] == 9.0
    assert default_vec[CRAFTER_INVENTORY_KEYS.index("energy")] == 9.0
    assert default_vec[CRAFTER_INVENTORY_KEYS.index("wood")] == 0.0

    # Test conversion of active game inventory dict
    inv_dict = {"wood": 3, "stone": 1, "wood_pickaxe": 1, "health": 8}
    vec = inventory_dict_to_vector(inv_dict)
    assert vec[CRAFTER_INVENTORY_KEYS.index("wood")] == 3.0
    assert vec[CRAFTER_INVENTORY_KEYS.index("stone")] == 1.0
    assert vec[CRAFTER_INVENTORY_KEYS.index("wood_pickaxe")] == 1.0
    assert vec[CRAFTER_INVENTORY_KEYS.index("health")] == 8.0
    assert vec[CRAFTER_INVENTORY_KEYS.index("sapling")] == 0.0

    # Test roundtrip reconstruction
    recon_dict = vector_to_inventory_dict(vec)
    assert recon_dict["wood"] == 3
    assert recon_dict["stone"] == 1
    assert recon_dict["wood_pickaxe"] == 1
    assert recon_dict["health"] == 8


def test_gaussian_rbf_kernel_regression_backpack_conditioning():
    """
    Pillar 4 Test: Verifies that Gaussian RBF Nadaraya-Watson kernel regression
    correctly evaluates candidate actions purely based on visual and 16-D backpack similarity,
    with zero hardcoded rules.
    """
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu", rbf_sigma=0.80)

    # Visual key for a crafting table tile
    torch.manual_seed(42)
    key_table = torch.randn(128)
    key_table = key_table / key_table.norm()

    # Experience 1: Agent stood at table with wood=1, pressed craft pickaxe (action 15) -> G = 1.0
    inv_with_wood = inventory_dict_to_vector({"wood": 1, "health": 9, "food": 9, "drink": 9, "energy": 9})
    ltm.add_entry(
        key=key_table,
        inventory=inv_with_wood,
        action=15,  # craft pickaxe
        inventory_delta=inventory_dict_to_vector({"wood": -1, "wood_pickaxe": 1}),
        realized_return=1.0,
        outcome_name="make_wood_pickaxe",
    )

    # Experience 2: Agent stood at table with wood=0, pressed craft pickaxe (action 15) -> G = 0.0 (Failed!)
    inv_without_wood = inventory_dict_to_vector({"wood": 0, "health": 9, "food": 9, "drink": 9, "energy": 9})
    ltm.add_entry(
        key=key_table,
        inventory=inv_without_wood,
        action=15,  # craft pickaxe
        inventory_delta=np.zeros(16, dtype=np.float32),
        realized_return=0.0,
        outcome_name="fruitless_craft",
    )

    # SCENARIO A: Agent is currently standing at the table with wood=0
    current_inv_0 = inventory_dict_to_vector({"wood": 0, "health": 9, "food": 9, "drink": 9, "energy": 9})
    q_all_0, sim_all_0 = ltm.query_all_actions(key_table, current_inv_0)
    # The failed memory (G=0.0) is matched strongly (distance 0), while success memory (wood=1) is distant
    # Expected return for crafting with wood=0 is strongly suppressed!
    assert q_all_0[15].item() < 0.40, f"Expected low Q for craft with 0 wood, got {q_all_0[15].item()}"

    # SCENARIO B: Agent is currently standing at the table with wood=1
    current_inv_1 = inventory_dict_to_vector({"wood": 1, "health": 9, "food": 9, "drink": 9, "energy": 9})
    q_all_1, sim_all_1 = ltm.query_all_actions(key_table, current_inv_1)
    # The success memory (G=1.0) is matched strongly (distance 0)
    assert q_all_1[15].item() > 0.60, f"Expected high Q for craft with 1 wood, got {q_all_1[15].item()}"
    assert sim_all_1[15].item() > 0.95

    # Query directly returns action 15 when holding wood
    best_act, best_sim, best_name = ltm.query(key_table, current_inv_1, sim_threshold=0.50)
    assert best_act == 15
    assert best_sim > 0.95


def test_mistake_learning_and_counterfactual_failure_suppression():
    """
    Pillar 2 Test: Learning from mistakes.
    When an action produces delta_inv = 0 and r = 0, it is recorded with G = 0.0.
    Gaussian RBF naturally suppresses it without any handcrafted 'if wood == 0' rules.
    """
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu", rbf_sigma=1.50)
    key_stone = torch.randn(128)
    key_stone = key_stone / key_stone.norm()

    # Fruitless attempt to mine stone with bare hands (wood_pickaxe=0): G = 0.0
    inv_bare_hands = inventory_dict_to_vector({"wood_pickaxe": 0, "health": 9, "food": 9, "drink": 9, "energy": 9})
    ltm.add_entry(
        key=key_stone,
        inventory=inv_bare_hands,
        action=2,  # do/mine
        inventory_delta=np.zeros(16, dtype=np.float32),
        realized_return=0.0,
        outcome_name="fruitless_act_2",
    )

    q_vals, max_sims = ltm.query_all_actions(key_stone, inv_bare_hands)
    assert q_vals[2].item() == 0.0, "Action that failed should have Q_memory = 0.0"
    best_act, _, _ = ltm.query(key_stone, inv_bare_hands, sim_threshold=0.50)
    assert best_act is None, "Fruitless action with Q=0 must never be recommended as a breakthrough!"


def test_temporal_monte_carlo_return_backpropagation():
    """
    Pillar 3 Test: Retrospective Monte Carlo Credit Assignment.
    Future dopamine surge and tool reward flow backward along the trajectory:
    G_t = sum_k gamma^k (r_{t+k} + dopamine_{t+k})
    Elevates the estimated return of tree-chopping naturally!
    """
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")
    dopamine_tracker = DopaminePersonalBestTracker(dopamine_scale=0.50)
    writer = AutonomousMemoryWriter(memory_bank=ltm, dopamine_tracker=dopamine_tracker, gamma=0.90)
    writer.reset()

    # Step 0: Agent chops tree -> collects wood (reward = 1.0, inv wood 0 -> 1)
    z_tree = torch.randn(128)
    info_step0 = {
        "inventory": {"wood": 1, "health": 9, "food": 9, "drink": 9, "energy": 9},
        "achievements": {"collect_wood": 1},
    }
    writer.step(z_t=z_tree, action=2, reward=1.0, info=info_step0)

    # Step 1: Agent moves towards clearing (reward = 0.0)
    z_move = torch.randn(128)
    info_step1 = {
        "inventory": {"wood": 1, "health": 9, "food": 9, "drink": 9, "energy": 9},
        "achievements": {"collect_wood": 1},
    }
    writer.step(z_t=z_move, action=4, reward=0.0, info=info_step1)

    # Step 2: Agent crafts pickaxe at table (reward = 1.0 + dopamine surge PB!)
    z_table = torch.randn(128)
    info_step2 = {
        "inventory": {"wood": 0, "wood_pickaxe": 1, "health": 9, "food": 9, "drink": 9, "energy": 9},
        "achievements": {"collect_wood": 1, "make_wood_pickaxe": 1},
    }
    _, surge = writer.step(z_t=z_table, action=15, reward=1.0, info=info_step2)
    assert surge > 0.0, "Crafting pickaxe for the first time should trigger dopamine PB surge!"

    # Game over: call end_episode consolidation
    consolidated_count = writer.end_episode(final_reward=2.0 + surge, gamma=0.90)
    assert consolidated_count >= 1

    # Check that tree-chopping step received the discounted future return from pickaxe
    # Step 2 return: 1.0 + surge
    # Step 1 return: 0.9 * (1.0 + surge)
    # Step 0 return: 1.0 + 0.9^2 * (1.0 + surge) > 1.80!
    expected_g0 = 1.0 + (0.90 ** 2) * (1.0 + surge)

    # Find the tree chopping memory in the memory bank
    tree_mem_idx = [i for i, name in enumerate(ltm.outcome_names) if "wood" in name]
    assert len(tree_mem_idx) > 0
    tree_return = ltm.returns[tree_mem_idx[0]].item()
    assert tree_return >= 1.50, f"Tree chopping return should be elevated by future pickaxe, got {tree_return}"


def test_experience_diary_de_duplication():
    """
    Verifies that identical transitions do not duplicate memory entries,
    and update return with max return achieved.
    """
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")
    key = torch.randn(128)
    key = key / key.norm()
    inv = inventory_dict_to_vector({"wood": 1})

    # Add first entry with return 1.0
    ltm.add_entry(key=key, inventory=inv, action=2, realized_return=1.0, outcome_name="tree_1")
    assert len(ltm) == 1
    assert ltm.returns[0].item() == 1.0

    # Add almost identical entry (cosine sim ~ 1.0, same inv, same action) with higher return 2.5
    ltm.add_entry(key=key, inventory=inv, action=2, realized_return=2.5, outcome_name="tree_2")
    assert len(ltm) == 1, "Memory bank must not duplicate identical entries!"
    assert ltm.returns[0].item() == 2.5, "Return should be updated to higher return!"


def test_autonomous_backward_goal_chaining():
    """
    Verifies that the agent autonomously deduces sub-goals backwards from its Experience Diary
    WITHOUT human hardcoded recipes:
    1. Empty backpack + Table memory -> deduces it needs wood and selects producer Action 5!
    2. 2 Wood in backpack -> recognizes deficit is 0 and switches to Action 8 (Place table)!
    3. Table placed -> automatically shifts frontier goal to Action 11 (Make wood pickaxe)!
    """
    ltm = LongTermSemanticMemory(latent_dim=128, device="cpu")

    # Key vectors
    z_tree = torch.randn(128)
    z_table = torch.randn(128)
    z_craft = torch.randn(128)

    # 1. Experience 1: Punching tree gives wood (Action 5)
    delta_wood = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)
    delta_wood[0] = 1.0  # wood +1
    ltm.add_entry(
        key=z_tree,
        inventory=inventory_dict_to_vector({}),
        action=5,
        inventory_delta=delta_wood,
        realized_return=5.0,
        outcome_name="collect_wood",
    )

    # 2. Experience 2: Placing table consumes 2 wood (Action 8)
    delta_table = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)
    delta_table[0] = -2.0  # wood -2
    ltm.add_entry(
        key=z_table,
        inventory=inventory_dict_to_vector({"wood": 2}),
        action=8,
        inventory_delta=delta_table,
        realized_return=2.5,
        outcome_name="place_table",
    )

    # 3. Experience 3: Crafting pickaxe at table consumes 1 wood (Action 11)
    delta_pick = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)
    delta_pick[0] = -1.0  # wood -1
    delta_pick[6] = 1.0   # wood_pickaxe +1
    ltm.add_entry(
        key=z_craft,
        inventory=inventory_dict_to_vector({"wood": 1}),
        action=11,
        inventory_delta=delta_pick,
        realized_return=1.8,
        outcome_name="make_wood_pickaxe",
    )

    assert len(ltm) == 3

    # Case 1: Agent spawns at step 0 with empty backpack
    empty_inv = inventory_dict_to_vector({})
    subgoal = ltm.deduce_active_subgoal(empty_inv)
    assert subgoal is not None
    assert subgoal["status"] == "HARVESTING_INPUT", "Empty hands should deduce harvesting input!"
    assert subgoal["missing_resource_name"] == "wood", "Missing resource should be wood!"
    assert subgoal["deficit"] == 2.0, "Deficit for table should be 2 wood!"
    assert subgoal["producer_action"] == 5, "Producer action should be Action 5 (Punch Tree)!"
    assert subgoal["target_milestone"] == "place_table"

    # Case 2: Agent has gathered 2 wood
    inv_2_wood = inventory_dict_to_vector({"wood": 2})
    subgoal2 = ltm.deduce_active_subgoal(inv_2_wood)
    assert subgoal2 is not None
    assert subgoal2["status"] == "READY_TO_EXECUTE", "Having 2 wood should switch to ready to execute!"
    assert subgoal2["target_action"] == 8, "Target action should be Action 8 (Place table)!"
    assert subgoal2["deficit"] == 0.0

    # Case 3: Table is already placed in this episode, agent has 1 wood
    inv_1_wood = inventory_dict_to_vector({"wood": 1})
    subgoal3 = ltm.deduce_active_subgoal(inv_1_wood, achieved_names={"place_table", "collect_wood"})
    assert subgoal3 is not None
    assert subgoal3["status"] == "READY_TO_EXECUTE", "With table placed and 1 wood, ready to craft pickaxe!"
    assert subgoal3["target_milestone"] == "make_wood_pickaxe"
    assert subgoal3["target_action"] == 11, "Target action should be Action 11 (Make wood pickaxe)!"



