#!/usr/bin/env python3
"""
tests/test_dual_store_backward_compatibility.py — Unit tests for the Dual-Store
Causal Memory Architecture, Physical Consequence Gate, and Planning Cost Alignment.
"""

import numpy as np
import pytest
import torch

from dual_store.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from dual_store.causal_memory import (
    NUM_ACHIEVEMENTS,
    RelationHypothesisBank,
    TransitionPrototype,
    wilson_interval,
)
from dual_store.compositional_planner import CompositionalCausalPlanner


def make_test_card(
    pid,
    action,
    event_dim=None,
    needs=None,
    produces=None,
    consumes=None,
    wilson_lb=0.85,
    status="confirmed",
    attempts=20.0,
    failure_count=0.0,
    precondition_key=None,
    visual_dispersion=0.5,
):
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
    succ = max(attempts - failure_count, 0.0)
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
        failure_count=failure_count,
        mean_return=1.0,
        confidence=max(0.05, wilson_lb),
        achievement_delta=[0.0] * NUM_ACHIEVEMENTS,
        event_dim=event_dim,
        needs=[[int(d), float(a)] for d, a in needs],
        produces=[[int(d), float(a)] for d, a in produces],
        consumes=[[int(d), float(a)] for d, a in consumes],
        wilson_lb=wilson_lb,
        status=status,
        visual_dispersion=visual_dispersion,
    )


def test_physical_consequence_gate_blocks_hallucinated_cards():
    """EventHead spike with 0 delta and 0 reward must NOT spawn a hypothesis card."""
    bank = RelationHypothesisBank()
    steps = [
        {
            "obs_latent": [0.1] * 128,
            "grounded_state": [0.0] * 16,
            "action": 5,
            "reward": 0.0,  # Zero reward
            "done": False,
            "state_delta": [0.0] * 16,  # Zero delta
            "event_vec": [0] * 15 + [1] + [0] * 6,  # event 15 fired
            "mask_before_vec": [0] * 22,
            "confident_events": [15],
            "fallback_unlocked": [],
        }
    ]
    res = bank.learn_from_episode(steps, episode_id=1)
    assert res["created"] == 0
    assert len(bank.prototypes) == 0


def test_physical_consequence_gate_admits_grounded_cards():
    """EventHead unlock WITH state delta or reward creates a hypothesis card."""
    bank = RelationHypothesisBank()
    delta = [0.0] * 16
    delta[0] = 1.0  # Wood produced
    steps = [
        {
            "obs_latent": [0.1] * 128,
            "grounded_state": [0.0] * 16,
            "action": 7,
            "reward": 1.0,
            "done": False,
            "state_delta": delta,
            "event_vec": [0] * 6 + [1] + [0] * 15,  # event 6 (collect_wood) fired
            "mask_before_vec": [0] * 22,
            "confident_events": [6],
            "fallback_unlocked": [],
        }
    ]
    res = bank.learn_from_episode(steps, episode_id=1)
    assert res["created"] == 1
    assert len(bank.prototypes) == 1


def test_bayesian_falsification_pruning_removes_failed_hypotheses():
    """Online hypothesis with >= 3 failures and Wilson UB < 0.15 is pruned."""
    bank = RelationHypothesisBank()
    card = make_test_card(
        "card_noisy", 5, event_dim=15, status="hypothesis", attempts=3.0, failure_count=3.0, wilson_lb=0.0
    )
    card.created_from = "online"
    card.success_count = 0.0
    card.failure_count = 3.0
    bank.prototypes["card_noisy"] = card

    # Run an empty/unrelated step
    steps = [
        {
            "obs_latent": [0.0] * 128,
            "grounded_state": [0.0] * 16,
            "action": 0,
            "reward": 0.0,
            "done": False,
            "state_delta": [0.0] * 16,
            "event_vec": [0] * 22,
            "mask_before_vec": [0] * 22,
            "confident_events": [],
            "fallback_unlocked": [],
        }
    ]
    res = bank.learn_from_episode(steps, episode_id=2)
    assert bank.last_pruned_count == 1
    assert "card_noisy" not in bank.prototypes


def test_rare_discovery_protection_never_prunes_untested_breakthroughs():
    """Rare milestone with 1 success and 0 failures is NOT pruned."""
    bank = RelationHypothesisBank()
    card = make_test_card(
        "card_rare_diamond", 5, event_dim=10, status="hypothesis", attempts=1.0, failure_count=0.0, wilson_lb=0.05
    )
    card.created_from = "online"
    card.success_count = 1.0
    card.failure_count = 0.0
    bank.prototypes["card_rare_diamond"] = card

    steps = [
        {
            "obs_latent": [0.0] * 128,
            "grounded_state": [0.0] * 16,
            "action": 0,
            "reward": 0.0,
            "done": False,
            "state_delta": [0.0] * 16,
            "event_vec": [0] * 22,
            "mask_before_vec": [0] * 22,
            "confident_events": [],
            "fallback_unlocked": [],
        }
    ]
    res = bank.learn_from_episode(steps, episode_id=3)
    assert bank.last_pruned_count == 0
    assert "card_rare_diamond" in bank.prototypes


def test_hypothesis_trust_unfloored_after_failure():
    """Untested hypothesis gets trust floor (0.2); failed hypothesis drops to Wilson LB (0.0)."""
    bank = RelationHypothesisBank()
    planner = CompositionalCausalPlanner(bank, hypothesis_trust_floor=0.2)

    card_fresh = make_test_card("card_fresh", 5, event_dim=15, status="hypothesis", failure_count=0.0, wilson_lb=0.0)
    assert planner._trust(card_fresh) == 0.2

    card_failed = make_test_card("card_failed", 5, event_dim=15, status="hypothesis", failure_count=1.0, wilson_lb=0.0)
    assert planner._trust(card_failed) == 0.0


def test_first_link_visual_gating_allows_harvesting_search():
    """Multi-step chain gates on link 0 (tree) even when terminal table cue is not visible."""
    bank = RelationHypothesisBank()
    WOOD_DIM = 0
    PICKAXE_EVENT = 15
    bank.prototypes["card_pickaxe"] = make_test_card(
        "card_pickaxe", 11, event_dim=PICKAXE_EVENT, needs=[[WOOD_DIM, 1.0]], wilson_lb=0.8,
        precondition_key=[1.0] + [0.0] * 127, visual_dispersion=0.9
    )
    bank.prototypes["card_tree"] = make_test_card(
        "card_tree", 7, event_dim=None, produces=[[WOOD_DIM, 1.0]], wilson_lb=0.9,
        precondition_key=[0.0, 1.0] + [0.0] * 126, visual_dispersion=0.9
    )
    planner = CompositionalCausalPlanner(bank, sim_threshold=0.3)

    # Current observation is looking at a tree ([0, 1, ...]), NOT the table ([1, 0, ...]):
    obs = [0.0, 1.0] + [0.0] * 126
    subgoal = planner.plan_subgoal(obs, np.zeros(16), m_bin=np.zeros(NUM_ACHIEVEMENTS))
    assert subgoal is not None
    assert subgoal["status"] == "HARVESTING_INPUT"
    assert subgoal["recommended_action"] == 7
    assert subgoal["missing_resource_dim"] == WOOD_DIM


def test_untrained_readout_head_returns_safe_zeros():
    """JEPAv3 without trained state_readout_head in checkpoint returns zeros, not Gaussian noise."""
    model = JEPAv3()
    # Mock loaded keys without state_readout_head
    model.loaded_component_keys = {"encoder", "target_encoder"}
    s = torch.randn(1, 384)
    x_hat, uncertainty = model.predict_state(s)
    assert torch.allclose(x_hat, torch.zeros(1, 16))
    assert torch.allclose(uncertainty, torch.ones(1, 16))


def test_member_latents_compaction_on_save_and_load(tmp_path):
    """Member latents are bounded to at most K_max=8 on save and load to prevent bloat."""
    bank = RelationHypothesisBank()
    card = make_test_card("card_bloated", 5, event_dim=2)
    bank.prototypes["card_bloated"] = card
    # Give it 20 member latents
    bank._member_latents["card_bloated"] = [[float(i)] * 128 for i in range(20)]

    save_path = tmp_path / "compacted_bank.json"
    bank.save(save_path)

    # In memory, it was compacted to 8
    assert len(bank._member_latents["card_bloated"]) == 8
    assert bank._member_latents["card_bloated"][-1] == [19.0] * 128

    # Reloading also ensures bounded length <= 8
    fresh = RelationHypothesisBank()
    fresh.load(save_path)
    assert len(fresh._member_latents["card_bloated"]) == 8
    assert fresh._member_latents["card_bloated"][-1] == [19.0] * 128


def test_forward_state_projection_chains_cumulative_resources():
    """Verifies that multi-step backward chaining uses forward state projection to accumulate
    cumulative resources across enablers and consumers without any hardcoded thresholds."""
    from dual_store.causal_memory import RelationHypothesisBank, TransitionPrototype, RelationEdge
    from dual_store.compositional_planner import CompositionalCausalPlanner

    bank = RelationHypothesisBank()
    wood_c = TransitionPrototype(
        "card_wood", 5, [0.0]*128, [0.0]*16, [0.0]*16, [0.0]*128, 10.0, 1.0, 1.0, 1.0, 0.9,
        [0.0]*22, 6, [], [[0.0, 1.0]], [], 0.9, 0.8, 0.5, "confirmed", 10.0, 0.0, 1.0, 0.0, "offline"
    )
    bank.prototypes["card_wood"] = wood_c

    table_c = TransitionPrototype(
        "card_table", 8, [0.0]*128, [0.0]*16, [0.0]*16, [0.0]*128, 10.0, 1.0, 1.0, 1.0, 0.9,
        [0.0]*22, 20, [[0.0, 2.0]], [], [[0.0, 2.0]], 0.9, 0.8, 0.5, "confirmed", 10.0, 0.0, 1.0, 0.0, "offline"
    )
    bank.prototypes["card_table"] = table_c

    pick_c = TransitionPrototype(
        "card_pickaxe", 11, [0.0]*128, [0.0]*16, [0.0]*16, [0.0]*128, 10.0, 1.0, 1.0, 1.0, 0.9,
        [0.0]*22, 15, [[0.0, 1.0]], [], [[0.0, 1.0]], 0.9, 0.8, 0.5, "confirmed", 10.0, 0.0, 1.0, 0.0, "offline"
    )
    bank.prototypes["card_pickaxe"] = pick_c

    bank.refresh_edges()
    bank.edges["card_pickaxe"] = [RelationEdge("card_table", "card_pickaxe", "enabling_lift", None, 10.0, 5, 1.0)]

    planner = CompositionalCausalPlanner(bank, device="cpu")
    m_bin = np.zeros(22, dtype=np.float32)

    # 1. With 0 wood: must plan 2 wood (for table) + 1 wood (for pickaxe) = 3 total wood before table
    x_0 = np.zeros(16, dtype=np.float32)
    chain_0 = planner._resolve_chain(pick_c, x_0, m_bin, 0, set())
    assert chain_0 is not None
    wood_apps = sum(step["applications"] for step in chain_0 if step["card"].prototype_id == "card_wood")
    assert wood_apps == 3, f"Expected 3 cumulative wood applications, got {wood_apps}"
    # Verify table comes AFTER wood harvesting
    wood_idx = [i for i, s in enumerate(chain_0) if s["card"].prototype_id == "card_wood"]
    table_idx = [i for i, s in enumerate(chain_0) if s["card"].prototype_id == "card_table"][0]
    assert max(wood_idx) < table_idx, "All wood harvesting must precede table placement"

    # 2. With 2 wood: table consumes 2, so 1 more wood is needed for pickaxe
    x_2 = np.zeros(16, dtype=np.float32)
    x_2[0] = 2.0
    chain_2 = planner._resolve_chain(pick_c, x_2, m_bin, 0, set())
    assert chain_2 is not None
    wood_apps_2 = sum(step["applications"] for step in chain_2 if step["card"].prototype_id == "card_wood")
    assert wood_apps_2 == 1, f"Expected 1 wood application, got {wood_apps_2}"

    # 3. With 3 wood: both table and pickaxe are fully covered, 0 wood harvesting needed
    x_3 = np.zeros(16, dtype=np.float32)
    x_3[0] = 3.0
    chain_3 = planner._resolve_chain(pick_c, x_3, m_bin, 0, set())
    assert chain_3 is not None
    wood_apps_3 = sum(step["applications"] for step in chain_3 if step["card"].prototype_id == "card_wood")
    assert wood_apps_3 == 0, f"Expected 0 wood applications, got {wood_apps_3}"


