import numpy as np
import pytest
import torch
import torch.nn as nn

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


def test_gate10_action_ranking_shifts_with_active_consequence_scoring():
    """Gate 10 Criterion: Transition-aware mode changes relevant action rankings (Ablation C vs B)."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    torch.manual_seed(42)
    np.random.seed(42)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Mock predict_transition: Give action 11 (make_wood_pickaxe) a strong positive resource delta and reward
    orig_predict_transition = model.predict_transition

    def mock_predict_transition(s_cand, a_cand):
        trans = orig_predict_transition(s_cand, a_cand)
        # Identify branches where candidate action is 11 (pickaxe craft)
        is_craft = (a_cand == 11).float()
        # Add high positive delta on wood_pickaxe (index 6) and reward
        boosted_delta = trans["state_delta"].clone()
        boosted_delta[:, 6] = boosted_delta[:, 6] + 2.0 * is_craft
        boosted_reward = trans["reward"].clone() + 2.0 * is_craft
        return {
            "state_delta": boosted_delta,
            "reward": boosted_reward,
            "done_prob": trans["done_prob"],
            "uncertainty": trans["uncertainty"],
        }

    model.predict_transition = mock_predict_transition

    # Run Ablation B: Shadow transition mode (transition_scoring=False)
    torch.manual_seed(2050)
    np.random.seed(2050)
    planner_b = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=32,
        cem_iterations=2,
        use_transition_model=True,
        transition_scoring=False,
        strict_inference=True,
        device=dev,
    )
    belief_b = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    act_b, dbg_b = planner_b.plan(frame, belief_b)

    # Run Ablation C: Active consequence scoring mode (transition_scoring=True)
    torch.manual_seed(2050)
    np.random.seed(2050)
    planner_c = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=32,
        cem_iterations=2,
        use_transition_model=True,
        transition_scoring=True,
        transition_weight=1.0,
        transition_resource_weight=1.0,
        strict_inference=True,
        device=dev,
    )
    belief_c = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    act_c, dbg_c = planner_c.plan(frame, belief_c)

    # Assertions for Gate 10:
    # 1. Scoring is active in C, inactive in B
    assert dbg_b["transition_scoring_active"] is False
    assert dbg_c["transition_scoring_active"] is True

    # 2. Candidate scores differ significantly due to consequence weighting
    assert dbg_c["best_score"] != dbg_b["best_score"]
    assert dbg_c["reward_sum"] > dbg_b["reward_sum"]

    # 3. Action selection shifted towards the productive consequence action
    print(f"\nAblation B Action: {act_b} (Score: {dbg_b['best_score']:.4f})")
    print(f"Ablation C Action: {act_c} (Score: {dbg_c['best_score']:.4f})")
    assert act_c == 11, f"Expected active consequence scoring to select action 11 (craft pickaxe), got {act_c}"
    assert act_b != act_c, f"Expected action ranking to change between Ablation B ({act_b}) and C ({act_c})"


def test_gate10_death_penalty_avoids_fatal_action():
    """Gate 10 Criterion: Predicted termination probability penalizes fatal branches in Ablation C."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    torch.manual_seed(99)
    np.random.seed(99)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    orig_predict_transition = model.predict_transition

    # Force action 1 (e.g. walking left into lava/zombie) to predict 100% death probability
    def mock_predict_transition(s_cand, a_cand):
        trans = orig_predict_transition(s_cand, a_cand)
        is_fatal = (a_cand == 1).float()
        done_prob = torch.clamp(trans["done_prob"] + 0.95 * is_fatal, 0.0, 1.0)
        return {
            "state_delta": trans["state_delta"],
            "reward": trans["reward"],
            "done_prob": done_prob,
            "uncertainty": trans["uncertainty"],
        }

    model.predict_transition = mock_predict_transition

    # Ablation C with high death penalty
    planner_c = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=32,
        cem_iterations=2,
        use_transition_model=True,
        transition_scoring=True,
        transition_done_penalty=5.0,
        strict_inference=True,
        device=dev,
    )
    belief_c = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    torch.manual_seed(1234)
    np.random.seed(1234)
    act_c, dbg_c = planner_c.plan(frame, belief_c)

    # Action 1 should be completely avoided due to severe death avoidance penalty
    assert act_c != 1, f"Expected death avoidance to avoid fatal action 1, got {act_c}"


def test_gate10_strict_landmark_guidance_conditioned_on_learned_readout():
    """Gate 10 Criterion: Strict mode activates table landmark guidance using learned StateReadoutHead."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    torch.manual_seed(101)
    np.random.seed(101)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Create a table landmark in working memory at current position
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    obs_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    z_anc = model.encode(obs_t, use_target=False).detach().cpu()
    belief.working_memory.landmark_anchors["structure"] = z_anc
    belief.working_memory.landmark_step_created["structure"] = 0
    belief.working_memory.anchor_offsets["structure"] = np.array([0.0, 0.0], dtype=np.float32)

    # Case 1: Learned state predicts 0 wood (should NOT pull to table)
    def mock_predict_state_no_wood(s):
        x = torch.zeros((s.shape[0], 16), dtype=torch.float32)
        logvar = torch.zeros_like(x)
        return x, logvar

    model.predict_state = mock_predict_state_no_wood

    planner_no_wood = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=16,
        cem_iterations=1,
        use_transition_model=True,
        transition_scoring=True,
        strict_inference=True,
        device=dev,
    )
    torch.manual_seed(777)
    np.random.seed(777)
    _, dbg_no_wood = planner_no_wood.plan(frame, belief)

    # Case 2: Learned state predicts 2 wood (SHOULD pull to table to craft)
    def mock_predict_state_has_wood(s):
        x = torch.zeros((s.shape[0], 16), dtype=torch.float32)
        x[:, 0] = 2.0  # wood = 2
        logvar = torch.zeros_like(x)
        return x, logvar

    model.predict_state = mock_predict_state_has_wood

    planner_has_wood = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=16,
        cem_iterations=1,
        use_transition_model=True,
        transition_scoring=True,
        strict_inference=True,
        device=dev,
    )
    torch.manual_seed(777)
    np.random.seed(777)
    _, dbg_has_wood = planner_has_wood.plan(frame, belief)

    # When wood is learned to be present, table landmark attraction increases candidate scores
    assert dbg_has_wood["best_score"] > dbg_no_wood["best_score"], (
        f"Expected has_wood score ({dbg_has_wood['best_score']:.4f}) > "
        f"no_wood score ({dbg_no_wood['best_score']:.4f})"
    )
