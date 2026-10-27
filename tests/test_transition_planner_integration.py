import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


def test_transition_model_uncalled_when_disabled():
    """Verify that when use_transition_model=False (default), predict_transition is NEVER called."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    call_count = 0
    orig_predict_transition = model.predict_transition

    def counting_predict_transition(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_predict_transition(*args, **kwargs)

    model.predict_transition = counting_predict_transition

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=5,
        num_candidates=16,
        cem_iterations=2,
        use_transition_model=False,
        strict_inference=True,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    action, debug = planner.plan(test_frame, belief)

    assert call_count == 0, f"Expected predict_transition to be called 0 times, got {call_count}"
    assert debug["transition_model_used"] is False


def test_transition_model_called_per_candidate_step_when_enabled():
    """Verify that when use_transition_model=True, predict_transition is called for each candidate lookahead step."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    call_count = 0
    orig_predict_transition = model.predict_transition

    def counting_predict_transition(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_predict_transition(*args, **kwargs)

    model.predict_transition = counting_predict_transition

    horizon = 5
    cem_iterations = 2
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=horizon,
        num_candidates=16,
        cem_iterations=cem_iterations,
        use_transition_model=True,
        strict_inference=True,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    action, debug = planner.plan(test_frame, belief)

    # In CEM rollout, predict_transition is called for each horizon step across all iterations
    expected_calls = horizon * cem_iterations
    assert call_count == expected_calls, f"Expected {expected_calls} transition calls, got {call_count}"
    assert debug["transition_model_used"] is True
    assert "state_estimate" in debug
    assert len(debug["state_estimate"]) == 16
    assert "predicted_delta" in debug
    assert len(debug["predicted_delta"]) == 16
    assert "predicted_reward" in debug
    assert "predicted_done_probability" in debug
    assert 0.0 <= debug["predicted_done_probability"] <= 1.0
    assert "predicted_uncertainty" in debug
    assert debug["predicted_uncertainty"] >= 0.0


def test_shadow_mode_does_not_perturb_action_selection():
    """Verify shadow mode logs transition predictions but preserves exact action and scores."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Run A: Disabled
    torch.manual_seed(2050)
    np.random.seed(2050)
    planner_off = RecurrentMPCPlanner(model=model, horizon=4, num_candidates=8, cem_iterations=1, use_transition_model=False, strict_inference=True, device=dev)
    belief_a = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    act_off, dbg_off = planner_off.plan(frame, belief_a)

    # Run B: Shadow Enabled
    torch.manual_seed(2050)
    np.random.seed(2050)
    planner_on = RecurrentMPCPlanner(model=model, horizon=4, num_candidates=8, cem_iterations=1, use_transition_model=True, strict_inference=True, device=dev)
    belief_b = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    act_on, dbg_on = planner_on.plan(frame, belief_b)

    # In shadow mode, action and score must be bitwise identical
    assert act_off == act_on
    assert dbg_off["best_score"] == dbg_on["best_score"]
    assert dbg_off["reward_sum"] == dbg_on["reward_sum"]
    assert dbg_on["transition_model_used"] is True
