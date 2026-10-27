"""
tests/test_adaptive_mpc.py
Unit tests and offline verification for adaptive frontier thresholding in MPC v3.
"""

import numpy as np
import pytest
import torch
from pathlib import Path

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner

CHECKPOINT_PATH = Path("checkpoints/jepa_crafter_v3_phase2.pt")
SHARD_PATH = Path("data/sequences/shard_structured_random_0000.npz")


def test_agent_belief_adaptive_thresholds():
    # In default strict mode, thresholds are uniform across all achievements
    belief_strict = AgentBeliefV3(frontier_threshold=0.25)
    assert np.allclose(belief_strict.thresholds, 0.25)

    # In privileged mode, legacy frontier indices are applied
    belief = AgentBeliefV3(frontier_threshold=0.25, common_threshold=0.50, strict_inference=False)
    idx_pick = CRAFTER_ACHIEVEMENTS.index("make_wood_pickaxe")
    idx_plant = CRAFTER_ACHIEVEMENTS.index("place_plant")
    
    assert belief.thresholds[idx_pick] == 0.25
    assert belief.thresholds[idx_plant] == 0.50
    
    # Simulate event probs: pickaxe = 0.30 (above 0.25), plant = 0.40 (below 0.50)
    probs = np.zeros(22, dtype=np.float32)
    probs[idx_pick] = 0.30
    probs[idx_plant] = 0.40
    
    belief.update(probs, reward=0.0)
    assert belief.m_bin[idx_pick] == 1.0, "Frontier event above 0.25 should unlock"
    assert belief.m_bin[idx_plant] == 0.0, "Common event below 0.50 should remain locked"


def test_planner_adaptive_threshold_offline():
    if not CHECKPOINT_PATH.exists() or not SHARD_PATH.exists():
        pytest.skip("Checkpoint or shard data missing")
        
    model = JEPAv3()
    model.load_checkpoint(str(CHECKPOINT_PATH), device="cpu")
    model.eval()
    
    shard = np.load(str(SHARD_PATH))
    obs = shard["obs"]
    actions = shard["actions"]
    events = shard["events"]
    idx_table = CRAFTER_ACHIEVEMENTS.index("place_table")
    step_t = np.where(events[:, idx_table] == 1)[0][0]
    
    belief = AgentBeliefV3(recurrent_dim=256, frontier_threshold=0.25, common_threshold=0.50)
    h = torch.zeros(1, 256)
    with torch.no_grad():
        for s_idx in range(max(0, step_t - 20), step_t + 1):
            frame = torch.tensor(obs[s_idx], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            z = model.encoder(frame)
            act = torch.tensor([actions[s_idx]])
            _, h = model.recurrent_step(z, act, h)
            
    obs_post = obs[step_t + 1]
    belief.h = h.squeeze(0).detach().numpy()
    belief.m_bin[idx_table] = 1.0
    belief.m_soft[idx_table] = 1.0
    belief.m_bin[CRAFTER_ACHIEVEMENTS.index("collect_wood")] = 1.0
    belief.m_soft[CRAFTER_ACHIEVEMENTS.index("collect_wood")] = 1.0
    
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=10,
        num_candidates=64,
        cem_iterations=3,
        frontier_threshold=0.25,
        common_threshold=0.50,
        device="cpu",
        standardize_scores=False,
    )
    
    idx_pick = CRAFTER_ACHIEVEMENTS.index("make_wood_pickaxe")
    assert planner.event_thresholds[0, idx_pick].item() == 0.25
    assert planner.denormalize_q is False
    assert planner.vital_weight == 0.5
    assert planner.death_risk_factor == 0.0
    assert planner.harvest_bonus == 0.15
    assert planner.action_persistence == 0.5


def test_mpc_continuation_discounting_execution():
    """Verify that continuation-discounted rollout executes end-to-end without errors."""
    model = JEPAv3()
    model.eval()
    
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=5,
        num_candidates=16,
        cem_iterations=1,
        frontier_threshold=0.25,
        common_threshold=0.50,
        denormalize_q=True,
        vital_weight=2.0,
        death_risk_factor=2.5,
        harvest_bonus=0.15,
        device="cpu",
        standardize_scores=False,
    )
    
    belief = AgentBeliefV3(frontier_threshold=0.25, common_threshold=0.50)
    dummy_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    action, debug = planner.plan(dummy_obs, belief)
    assert 0 <= action < 17
    assert "reward_sum" in debug
    assert "terminal_value" in debug
