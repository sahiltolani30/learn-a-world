import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.causal_memory import RelationHypothesisBank, TransitionPrototype
from planning.compositional_planner import CompositionalCausalPlanner
from planning.mpc_v3 import RecurrentMPCPlanner


def test_compositional_backward_chaining():
    """Verify that when resources are missing, planner backward-chains to harvest producer action."""
    bank = RelationHypothesisBank()

    # Target Prototype: Make Wood Pickaxe (Action 11)
    # Preconditions: wood >= 1.0 (index 0)
    # Return: 1.0
    proto_pickaxe = TransitionPrototype(
        prototype_id="proto_make_pickaxe",
        action=11,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[1.0] + [0.0] * 15,
        effect_delta=[-1.0, 0, 0, 0, 0, 0, 1.0] + [0.0] * 9,
        postcondition_key=[0.5] * 128,
        success_count=10,
        failure_count=0,
        mean_return=1.0,
        confidence=0.90,
    )
    bank.prototypes["proto_make_pickaxe"] = proto_pickaxe

    # Producer Prototype: Chop Tree (Action 7)
    # Yields: wood +1.0 (index 0)
    # Return: 0.90
    proto_tree = TransitionPrototype(
        prototype_id="proto_chop_tree",
        action=7,
        precondition_key=[0.8] + [0.1] * 127,
        precondition_state=[0.0] * 16,
        effect_delta=[1.0] + [0.0] * 15,
        postcondition_key=[0.5] * 128,
        success_count=15,
        failure_count=1,
        mean_return=0.90,
        confidence=0.93,
    )
    bank.prototypes["proto_chop_tree"] = proto_tree

    planner = CompositionalCausalPlanner(bank, min_prototype_confidence=0.5, sim_threshold=0.3)

    # Current state: Agent has 0 wood
    x_curr = np.zeros(16)
    z_curr = [1.0] + [0.0] * 127

    subgoal = planner.plan_subgoal(z_curr, x_curr)
    assert subgoal is not None
    # Because wood = 0, planner must NOT recommend craft action 11!
    # It must backward-chain to harvest action 7!
    assert subgoal["status"] == "HARVESTING_INPUT"
    assert subgoal["recommended_action"] == 7
    assert subgoal["missing_resource_dim"] == 0
    assert subgoal["deficit_amount"] == pytest.approx(1.0, abs=0.1)


def test_compositional_ready_to_execute():
    """Verify that when preconditions are met, planner recommends target breakthrough action."""
    bank = RelationHypothesisBank()
    proto_pickaxe = TransitionPrototype(
        prototype_id="proto_make_pickaxe",
        action=11,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[1.0] + [0.0] * 15,
        effect_delta=[-1.0, 0, 0, 0, 0, 0, 1.0] + [0.0] * 9,
        postcondition_key=[0.5] * 128,
        success_count=10,
        failure_count=0,
        mean_return=1.0,
        confidence=0.90,
    )
    bank.prototypes["proto_make_pickaxe"] = proto_pickaxe

    planner = CompositionalCausalPlanner(bank, min_prototype_confidence=0.5, sim_threshold=0.3)

    # Current state: Agent already has 2 wood!
    x_curr = np.zeros(16)
    x_curr[0] = 2.0  # wood = 2
    z_curr = [1.0] + [0.0] * 127

    subgoal = planner.plan_subgoal(z_curr, x_curr)
    assert subgoal is not None
    assert subgoal["status"] == "READY_TO_EXECUTE"
    assert subgoal["recommended_action"] == 11


def test_compositional_planner_mpc_integration():
    """Verify CompositionalCausalPlanner seamlessly guides RecurrentMPCPlanner in strict mode."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    bank = RelationHypothesisBank()
    # High-return tree chopping prototype
    proto_tree = TransitionPrototype(
        prototype_id="proto_chop_tree",
        action=7,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[0.0] * 16,
        effect_delta=[1.0] + [0.0] * 15,
        postcondition_key=[0.5] * 128,
        success_count=20,
        failure_count=0,
        mean_return=2.0,
        confidence=0.95,
    )
    bank.prototypes["proto_chop_tree"] = proto_tree

    comp_planner = CompositionalCausalPlanner(bank, min_prototype_confidence=0.5, sim_threshold=0.3, device=dev)

    mpc_planner = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=16,
        cem_iterations=1,
        strict_inference=True,
        compositional_planner=comp_planner,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    act, dbg = mpc_planner.plan(frame, belief)

    assert isinstance(act, int)
    assert 0 <= act < 17
    assert "best_score" in dbg
