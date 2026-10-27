import numpy as np
import pytest
import torch
import torch.nn as nn

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


class ExplodingHead(nn.Module):
    """A dummy head that raises an exception if ever called."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs):
        raise AssertionError(f"CRITICAL VIOLATION: {self.name} was called during MPC planning!")


class ZeroOutputHead(nn.Module):
    """A dummy head that outputs fixed zero tensors."""
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, s, a, *args, **kwargs):
        B = s.shape[0]
        return torch.zeros((B, self.out_dim), device=s.device)


def test_transition_heads_are_completely_unused_in_mpc_planning():
    """
    Step 3 Verification: Prove that RecurrentMPCPlanner in strict visual screening
    never queries state_delta_head or transition_head, and that action selection
    is bitwise identical whether these heads are active, zeroed out, or exploding.
    """
    dev = torch.device("cpu")
    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=256,
        num_actions=17,
        action_emb_dim=32,
        hidden_dim=256,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    )
    model.eval()

    # Load calibrator if available
    cal_path = "results/calibration/event_head_calibrator.json"
    try:
        from models.calibration import EventHeadCalibrator
        cal = EventHeadCalibrator.load(cal_path)
        model.set_calibrator(cal)
    except Exception:
        pass

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=5,
        num_candidates=16,
        cem_iterations=2,
        prior_mix=0.5,
        action_persistence=0.5,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256)

    # 1. Generate test visual observations
    np.random.seed(1234)
    test_frames = [
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(5)
    ]

    # --- Run A: Normal Model with Active Transition Heads ---
    actions_run_a = []
    debug_run_a = []
    for i, frame in enumerate(test_frames):
        torch.manual_seed(2050 + i)
        np.random.seed(2050 + i)
        belief.reset()
        act, dbg = planner.plan(frame, belief)
        actions_run_a.append(act)
        debug_run_a.append(dbg)

    # --- Run B: Exploding Heads (Throws error if forward() is called) ---
    model.state_delta_head = ExplodingHead("state_delta_head")
    model.transition_head = ExplodingHead("transition_head")

    actions_run_b = []
    debug_run_b = []
    for i, frame in enumerate(test_frames):
        torch.manual_seed(2050 + i)
        np.random.seed(2050 + i)
        belief.reset()
        # Should NOT raise AssertionError
        act, dbg = planner.plan(frame, belief)
        actions_run_b.append(act)
        debug_run_b.append(dbg)

    # --- Run C: Zero-Output Heads ---
    model.state_delta_head = ZeroOutputHead(out_dim=16)
    model.transition_head = ZeroOutputHead(out_dim=16)

    actions_run_c = []
    debug_run_c = []
    for i, frame in enumerate(test_frames):
        torch.manual_seed(2050 + i)
        np.random.seed(2050 + i)
        belief.reset()
        act, dbg = planner.plan(frame, belief)
        actions_run_c.append(act)
        debug_run_c.append(dbg)

    # Assert exact bitwise equivalence across all runs
    assert actions_run_a == actions_run_b, f"Actions differed between Run A and Run B: {actions_run_a} vs {actions_run_b}"
    assert actions_run_a == actions_run_c, f"Actions differed between Run A and Run C: {actions_run_a} vs {actions_run_c}"

    for i in range(len(test_frames)):
        assert debug_run_a[i]["best_score"] == debug_run_b[i]["best_score"]
        assert debug_run_a[i]["reward_sum"] == debug_run_b[i]["reward_sum"]
        assert debug_run_a[i]["terminal_value"] == debug_run_b[i]["terminal_value"]
        assert debug_run_a[i]["best_score"] == debug_run_c[i]["best_score"]
