import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from planning.semantic_memory import LongTermSemanticMemory, inventory_dict_to_vector


def test_agent_belief_strict_mode_defaults_and_guards():
    """Verify AgentBeliefV3 defaults to strict_inference=True, uses uniform thresholds, and guards inventory."""
    belief = AgentBeliefV3(recurrent_dim=256, frontier_threshold=0.25)
    assert belief.strict_inference is True
    # In strict mode, thresholds must be uniform across all 22 achievements (no hardcoded Crafter index subsets)
    assert np.allclose(belief.thresholds, 0.25)

    # Calling update_inventory in strict mode must fail closed with RuntimeError
    fake_inv = {"wood": 5, "stone": 2}
    with pytest.raises(RuntimeError, match="forbidden in strict inference"):
        belief.update_inventory(fake_inv)

    # Privileged diagnostic mode allows manual frontier indices and inventory updates
    diag_belief = AgentBeliefV3(recurrent_dim=256, frontier_threshold=0.25, common_threshold=0.50, strict_inference=False)
    assert diag_belief.strict_inference is False
    assert diag_belief.thresholds[0] == 0.25  # wood (frontier)
    assert diag_belief.thresholds[2] == 0.50  # sapling (common)
    diag_belief.update_inventory(fake_inv)
    assert diag_belief.get_inv_vector()[0] == 5.0


def test_mpc_planner_strict_mode_defaults_and_guards():
    """Verify RecurrentMPCPlanner enforces strict inference and uniform thresholds by default."""
    model = JEPAv3()
    model.eval()

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=5,
        num_candidates=16,
        frontier_threshold=0.25,
    )
    assert planner.strict_inference is True
    # Verify event_thresholds are uniform in strict mode
    expected_t = torch.full((1, 22), 0.25)
    assert torch.allclose(planner.event_thresholds, expected_t)

    # Attaching semantic memory in strict mode must fail closed
    dummy_memory = LongTermSemanticMemory(latent_dim=128)
    with pytest.raises(RuntimeError, match="Crafter semantic memory with hardcoded subgoals/recipes cannot be attached in strict inference"):
        RecurrentMPCPlanner(
            model=model,
            semantic_memory=dummy_memory,
            strict_inference=True,
        )


def test_semantic_memory_strict_guards():
    """Verify Crafter-specific semantic logic raises RuntimeError in strict inference."""
    # 1. inventory_dict_to_vector fails when given real dict in strict mode
    fake_inv = {"wood": 2}
    with pytest.raises(RuntimeError, match="Crafter inventory dictionaries are unavailable in strict inference"):
        inventory_dict_to_vector(fake_inv, strict_inference=True)

    # 2. deduce_active_subgoal fails in strict mode
    mem = LongTermSemanticMemory(latent_dim=128)
    curr_inv = torch.zeros(16)
    with pytest.raises(RuntimeError, match="Crafter semantic subgoals are unavailable in strict inference"):
        mem.deduce_active_subgoal(curr_inv, strict_inference=True)


def test_strict_mpc_plan_execution_with_zero_inventory_leak():
    """Verify planner runs strictly on visual observation with zero inventory/landmark leakage."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=8,
        cem_iterations=1,
        strict_inference=True,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    action, debug = planner.plan(test_frame, belief)

    assert isinstance(action, int)
    assert 0 <= action < 17
    assert "best_score" in debug
