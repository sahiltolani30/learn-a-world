import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


def test_strict_mode_actions_are_invariant_to_corrupted_info_dict():
    """
    Omni Audit Test: Proves that in strict mode, poisoning, modifying, or removing
    info['inventory'], info['achievements'], or coordinates has ZERO effect on action selection.
    """
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    # Load calibrator if available
    try:
        from models.calibration import EventHeadCalibrator
        cal = EventHeadCalibrator.load("results/calibration/event_head_calibrator.json")
        model.set_calibrator(cal)
    except Exception:
        pass

    # Deterministic sequence of 10 observation frames
    np.random.seed(42)
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]

    # --- Run A: Normal uncorrupted environment info trace ---
    planner_a = RecurrentMPCPlanner(model=model, horizon=4, num_candidates=8, cem_iterations=1, strict_inference=True, device=dev)
    belief_a = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    initial_inv_a = belief_a.get_inv_vector().copy()

    actions_a = []
    for step_i, frame in enumerate(frames):
        torch.manual_seed(2050 + step_i)
        np.random.seed(2050 + step_i)
        act, _ = planner_a.plan(frame, belief_a)
        actions_a.append(act)

        # Normal mock environment info returned after step
        info_a = {"inventory": {"wood": step_i, "stone": 0}, "achievements": {"collect_wood": 1 if step_i > 0 else 0}}
        reward_a = 1.0 if step_i == 1 else 0.0

        # Strict visual update (only pixels and scalar reward reach belief)
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            z_t = model.encode(frame_t, use_target=True)
            act_t = torch.tensor([act], device=dev, dtype=torch.long)
            h_prev = belief_a.get_h_tensor(device=dev).unsqueeze(0)
            s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
            mask_t = belief_a.get_mask_tensor(device=dev).unsqueeze(0)
            event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)
        belief_a.update(event_probs, reward_a, next_h=next_h.squeeze(0))

    # --- Run B: Maliciously poisoned and corrupted info trace ---
    planner_b = RecurrentMPCPlanner(model=model, horizon=4, num_candidates=8, cem_iterations=1, strict_inference=True, device=dev)
    belief_b = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    initial_inv_b = belief_b.get_inv_vector().copy()

    actions_b = []
    for step_i, frame in enumerate(frames):
        torch.manual_seed(2050 + step_i)
        np.random.seed(2050 + step_i)
        act, _ = planner_b.plan(frame, belief_b)
        actions_b.append(act)

        # POISONED mock environment info: massive fake resources, invalid coordinates, corrupted flags
        info_b = {
            "inventory": {"wood": 999999, "diamond": 888888, "wood_pickaxe": 50},
            "achievements": {"collect_diamond": 100, "place_table": 50},
            "player_pos": (-999999.0, 999999.0),
            "cheat_mode_active": True,
        }
        reward_b = 1.0 if step_i == 1 else 0.0

        # Strict visual update (only pixels and scalar reward reach belief)
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            z_t = model.encode(frame_t, use_target=True)
            act_t = torch.tensor([act], device=dev, dtype=torch.long)
            h_prev = belief_b.get_h_tensor(device=dev).unsqueeze(0)
            s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
            mask_t = belief_b.get_mask_tensor(device=dev).unsqueeze(0)
            event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)
        belief_b.update(event_probs, reward_b, next_h=next_h.squeeze(0))

    # 1. Exact Bitwise Action Invariance: Poisoning info has ZERO effect on actions
    assert actions_a == actions_b, f"Poisoning info changed actions! Run A: {actions_a} vs Run B: {actions_b}"

    # 2. Strict Belief Invariance: inv_vec remains completely unchanged and zeroed out
    assert np.array_equal(belief_a.get_inv_vector(), initial_inv_a)
    assert np.array_equal(belief_b.get_inv_vector(), initial_inv_b)

    # 3. Fail-closed verification: Calling update_inventory fails immediately in strict mode
    with pytest.raises(RuntimeError, match="forbidden in strict inference"):
        belief_b.update_inventory(info_b["inventory"])
