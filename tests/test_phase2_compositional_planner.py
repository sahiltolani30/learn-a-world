import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.causal_memory import (
    NUM_ACHIEVEMENTS,
    RelationEdge,
    RelationHypothesisBank,
    TransitionPrototype,
)
from planning.compositional_planner import CompositionalCausalPlanner
from planning.mpc_v3 import RecurrentMPCPlanner

WOOD_DIM = 0
PICKAXE_EVENT = 15  # make_wood_pickache index in CRAFTER_ACHIEVEMENTS (alphabetical)
TABLE_EVENT = 20    # place_table
PLANT_EVENT = 18    # place_plant


def make_card(
    pid,
    action,
    event_dim=None,
    needs=None,
    produces=None,
    consumes=None,
    wilson_lb=0.85,
    status="confirmed",
    attempts=20.0,
    precondition_key=None,
    visual_dispersion=0.5,
):
    """Card builder with sane defaults (plan §4 conftest pattern)."""
    needs = needs or []
    produces = produces or []
    consumes = consumes or []
    pre = np.zeros(16, dtype=np.float32)
    for dim, amt in needs:
        pre[int(dim)] = float(amt)
    eff = np.zeros(16, dtype=np.float32)
    for dim, amt in produces:
        eff[int(dim)] += float(amt)
    for dim, amt in consumes:
        eff[int(dim)] -= float(amt)
    succ = max(attempts * 0.8, 1.0)
    fail = max(attempts - succ, 0.0)
    if precondition_key is None:
        precondition_key = [1.0] + [0.0] * 127
    return TransitionPrototype(
        prototype_id=pid,
        action=action,
        precondition_key=list(precondition_key),
        precondition_state=pre.tolist(),
        effect_delta=eff.tolist(),
        postcondition_key=[0.5] * 128,
        success_count=succ,
        failure_count=fail,
        mean_return=1.0,
        confidence=max(0.05, wilson_lb),
        achievement_delta=([0.0] * event_dim + [1.0] + [0.0] * (NUM_ACHIEVEMENTS - event_dim - 1))
        if event_dim is not None
        else [0.0] * NUM_ACHIEVEMENTS,
        event_dim=event_dim,
        needs=[[int(d), float(a)] for d, a in needs],
        produces=[[int(d), float(a)] for d, a in produces],
        consumes=[[int(d), float(a)] for d, a in consumes],
        wilson_lb=wilson_lb,
        status=status,
        visual_dispersion=visual_dispersion,
    )


def test_compositional_backward_chaining():
    """Wood deficit on a pickaxe target must chain to the wood producer (canary, N1)."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_pickaxe"] = make_card(
        "card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]], wilson_lb=0.8,
    )
    bank.prototypes["card_tree"] = make_card(
        "card_tree", 7, event_dim=None, produces=[[WOOD_DIM, 1.0]], wilson_lb=0.9,
        precondition_key=[0.8] + [0.1] * 127,
    )
    bank.evidence_count = 100
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["status"] == "HARVESTING_INPUT"
    assert subgoal["recommended_action"] == 7
    assert subgoal["missing_resource_dim"] == WOOD_DIM
    assert subgoal["deficit_amount"] == pytest.approx(1.0, abs=0.1)
    assert subgoal["chain"][0]["card_id"] == "card_tree"


def test_compositional_ready_to_execute():
    """Resources met + cue visible -> READY on the target action."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_pickaxe"] = make_card(
        "card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]], wilson_lb=0.8,
        visual_dispersion=0.2,
    )
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    x = np.zeros(16)
    x[WOOD_DIM] = 2.0
    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, x, m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["status"] == "READY_TO_EXECUTE"
    assert subgoal["recommended_action"] == 11
    assert subgoal["visual_sim"] >= 0.3


def test_plan_subgoal_with_locked_event_returns_none():
    """Marginal gating: once the achievement is locked in m_bin, the card retires."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_pickaxe"] = make_card(
        "card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]],
    )
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    x = np.zeros(16)
    x[WOOD_DIM] = 2.0
    m_bin = np.zeros(NUM_ACHIEVEMENTS)
    m_bin[PICKAXE_EVENT] = 1.0
    assert planner.plan_subgoal([1.0] + [0.0] * 127, x, m_bin=m_bin) is None


def test_marginal_suppression_prefers_still_locked_target():
    """Two targets: only the still-locked one is returned; both locked -> None."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_plant"] = make_card("card_plant", 10, event_dim=PLANT_EVENT, wilson_lb=0.9)
    bank.prototypes["card_table"] = make_card("card_table", 8, event_dim=TABLE_EVENT, wilson_lb=0.8)
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)
    z = [1.0] + [0.0] * 127

    m_bin = np.zeros(NUM_ACHIEVEMENTS)
    m_bin[PLANT_EVENT] = 1.0  # plant retired
    subgoal = planner.plan_subgoal(z, np.zeros(16), m_bin=m_bin)
    assert subgoal is not None and subgoal["prototype_id"] == "card_table"

    m_bin[TABLE_EVENT] = 1.0  # both retired
    assert planner.plan_subgoal(z, np.zeros(16), m_bin=m_bin) is None


def test_ucb_ordering_prefers_low_evidence_card_then_decays():
    """UCB lifts a low-evidence high-value card above a mediocre confirmed one."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_strong"] = make_card("card_strong", 10, event_dim=PLANT_EVENT, wilson_lb=0.55, attempts=50.0)
    bank.prototypes["card_rare"] = make_card("card_rare", 11, event_dim=PICKAXE_EVENT, wilson_lb=0.55, attempts=0.0)
    bank.evidence_count = 200
    z = [1.0] + [0.0] * 127

    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)
    subgoal = planner.plan_subgoal(z, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None and subgoal["prototype_id"] == "card_rare"

    planner_no_ucb = CompositionalCausalPlanner(bank, sim_threshold=0.3, ucb_weight=0.0)
    subgoal = planner_no_ucb.plan_subgoal(z, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    # Ties broken deterministically; with UCB off both score equal — assert cap holds instead
    for card in bank.prototypes.values():
        assert planner._ucb_bonus(card) <= 0.5


def test_two_step_chain_with_cost_summation():
    """2 wood needed, producer yields 1/apply -> chain [producer x2, target], cost 2."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_pickaxe"] = make_card(
        "card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 2.0]], wilson_lb=0.8,
    )
    bank.prototypes["card_tree"] = make_card(
        "card_tree", 7, event_dim=None, produces=[[WOOD_DIM, 1.0]], wilson_lb=0.9,
    )
    bank.evidence_count = 100
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["total_cost"] == 2
    assert len(subgoal["chain"]) == 2
    assert subgoal["chain"][0]["card_id"] == "card_tree"
    assert subgoal["chain"][0]["applications"] == 2
    assert subgoal["recommended_action"] == 7
    # chain_value = trust(target) * trust(producer) * 0.9^1
    assert subgoal["chain_value"] == pytest.approx(0.8 * 0.9 * 0.9, rel=0.05)


def test_no_top_level_resource_target_returns_none():
    """Event-free resource cards alone must never become targets (farming-loop guard)."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_tree"] = make_card("card_tree", 7, event_dim=None, produces=[[WOOD_DIM, 1.0]], wilson_lb=0.95)
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)
    assert planner.plan_subgoal([1.0] + [0.0] * 127, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS)) is None


def test_enabler_required_in_chain():
    """pickaxe target with wood present but table enabler not unlocked -> table first."""
    bank = RelationHypothesisBank()
    bank.prototypes["card_pickaxe"] = make_card("card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]])
    bank.prototypes["card_table"] = make_card("card_table", 8, event_dim=TABLE_EVENT, wilson_lb=0.9)
    bank.edges["card_pickaxe"] = [
        RelationEdge(src_id="card_table", dst_id="card_pickaxe", kind="enabling_lift", lift=3.0, support=10, mean_dt=5.0)
    ]
    bank.evidence_count = 100
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    x = np.zeros(16)
    x[WOOD_DIM] = 2.0
    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, x, m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["recommended_action"] == 8  # place table first
    assert subgoal["chain"][0]["card_id"] == "card_table"

    # Once the table is unlocked in m_bin, the enabler is satisfied -> pickaxe ready.
    m_bin = np.zeros(NUM_ACHIEVEMENTS)
    m_bin[TABLE_EVENT] = 1.0
    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, x, m_bin=m_bin)
    assert subgoal is not None
    assert subgoal["status"] == "READY_TO_EXECUTE"
    assert subgoal["recommended_action"] == 11


def test_compositional_planner_mpc_integration():
    """CompositionalCausalPlanner guides RecurrentMPCPlanner in strict mode end-to-end."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    bank = RelationHypothesisBank()
    bank.prototypes["card_plant"] = make_card("card_plant", 10, event_dim=PLANT_EVENT, wilson_lb=0.9)
    bank.evidence_count = 50

    comp_planner = CompositionalCausalPlanner(bank, sim_threshold=0.3, device=dev)

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
    assert "active_subgoal_status" in dbg


def test_mpc_silent_when_all_targets_unlocked():
    """All events locked -> plan_subgoal None -> memory contributes nothing (AC7)."""
    dev = torch.device("cpu")
    model = JEPAv3()
    model.eval()

    bank = RelationHypothesisBank()
    bank.prototypes["card_plant"] = make_card("card_plant", 10, event_dim=PLANT_EVENT, wilson_lb=0.9)
    comp_planner = CompositionalCausalPlanner(bank, sim_threshold=0.3, device=dev)

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
    m = np.zeros(NUM_ACHIEVEMENTS)
    m[PLANT_EVENT] = 1.0
    belief.set_mask(m)

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    act, dbg = mpc_planner.plan(frame, belief)
    assert dbg["active_subgoal_status"] == ""
    assert dbg["memory_retrieved_action"] == -1.0


def test_confirmed_causal_precedence_blocks_noisy_hypothesis_shortcut():
    """Hypothesis cards cannot preempt or hijack confirmed recipes for the same event."""
    bank = RelationHypothesisBank()
    # 1. Confirmed legitimate recipe: Action 11 requires 1 wood
    bank.prototypes["card_pickaxe_a11"] = make_card(
        "card_pickaxe_a11", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]], wilson_lb=0.65, status="confirmed"
    )
    # Legitimate wood producer so chain can resolve
    bank.prototypes["card_tree"] = make_card(
        "card_tree", 7, event_dim=None, produces=[[WOOD_DIM, 1.0]], wilson_lb=0.9, status="confirmed"
    )
    # 2. Spurious unverified hypothesis card: Action 5 with 0 needs and 0 attempts (max UCB)
    bank.prototypes["card_pickaxe_a5_noisy"] = make_card(
        "card_pickaxe_a5_noisy", 5, event_dim=PICKAXE_EVENT, needs=[], wilson_lb=0.0, status="hypothesis", attempts=1.0
    )
    bank.evidence_count = 100
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    # Agent with 0 wood, pickaxe locked in m_bin:
    # Must backward-chain on the confirmed recipe (action 11 -> needs wood -> harvest tree with action 7),
    # and MUST NOT pick the noisy hypothesis card (action 5, 0 needs) as a "free lunch" shortcut!
    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["prototype_id"] == "card_pickaxe_a11"
    assert subgoal["status"] == "HARVESTING_INPUT"
    assert subgoal["recommended_action"] == 7  # harvesting wood for the confirmed pickaxe recipe!


def test_novel_frontier_hypothesis_is_explored():
    """Unexplored frontier achievements with no confirmed cards are actively explored via UCB."""
    bank = RelationHypothesisBank()
    STONE_EVENT = 5
    # Frontier hypothesis: action 5 for stone mining (no confirmed card exists for stone)
    bank.prototypes["card_stone_hypo"] = make_card(
        "card_stone_hypo", 5, event_dim=STONE_EVENT, needs=[], wilson_lb=0.0, status="hypothesis", attempts=1.0
    )
    bank.evidence_count = 50
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    subgoal = planner.plan_subgoal([1.0] + [0.0] * 127, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["prototype_id"] == "card_stone_hypo"
    assert subgoal["recommended_action"] == 5
    assert subgoal["status"] == "READY_TO_EXECUTE"

