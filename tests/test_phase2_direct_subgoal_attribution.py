"""
Unit tests for Direct Active-Subgoal Credit Assignment and Bayesian Falsification Pruning.
Verifies:
1. When a planner actively executes a hypothesis card and encounters a near-miss (no delta, no reward),
   the failure count is charged directly to that active hypothesis card.
2. Unrelated cards of the same action do not absorb the failure penalty.
3. Once an unconfirmed online hypothesis card fails repeatedly (failure_count >= 3, Wilson UB < 0.20 or
   empirical rate <= 0.25), it is automatically pruned by Bayesian falsification.
4. When no active subgoal is specified, near-miss attribution falls back safely to visual matching.
"""

import numpy as np
import pytest
import torch

from planning.causal_memory import (
    RelationHypothesisBank,
    TransitionPrototype,
)


def _make_dummy_card(
    proto_id: str,
    action: int,
    event_dim: int,
    status: str = "hypothesis",
    created_from: str = "online",
    succ: float = 1.0,
    fail: float = 0.0,
) -> TransitionPrototype:
    return TransitionPrototype(
        prototype_id=proto_id,
        action=action,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[0.0] * 16,
        effect_delta=[0.0] * 16,
        postcondition_key=[1.0] + [0.0] * 127,
        success_count=succ,
        failure_count=fail,
        mean_return=0.0,
        mean_duration=1.0,
        confidence=0.5,
        achievement_delta=[0.0] * 22,
        event_dim=event_dim,
        needs=[],
        produces=[],
        consumes=[],
        unlock_prob=0.5,
        wilson_lb=0.05,
        visual_dispersion=0.1,
        status=status,
        success_offline=0.0,
        success_online=succ,
        failure_offline=0.0,
        failure_online=fail,
        created_from=created_from,
        schema_version="phase2_card_v2",
    )


def test_direct_subgoal_attribution_charges_active_card():
    bank = RelationHypothesisBank()
    
    # Target hypothesis card: action 5 claiming event 15
    hypo = _make_dummy_card("card_a5_e15_o0", action=5, event_dim=15, status="hypothesis", created_from="online", succ=1.0, fail=0.0)
    bank.prototypes["card_a5_e15_o0"] = hypo
    
    # Another action 5 card in the bank
    innocent = _make_dummy_card("card_a5_e6_c0", action=5, event_dim=6, status="confirmed", created_from="offline", succ=10.0, fail=0.0)
    bank.prototypes["card_a5_e6_c0"] = innocent

    # Step: Near-miss (0 reward, 0 delta) with active_subgoal_target = "card_a5_e15_o0"
    step = {
        "obs_latent": [1.0] + [0.0] * 127,
        "grounded_state": [0.0] * 16,
        "action": 5,
        "reward": 0.0,
        "done": False,
        "event_vec": [0] * 22,
        "mask_before_vec": [0] * 22,
        "confident_events": [],
        "fallback_unlocked": [],
        "state_delta": [0.0] * 16,
        "active_subgoal_target": "card_a5_e15_o0",
    }

    # Step 1: Failure 1
    bank.learn_from_episode([step], episode_id=1)
    assert hypo.failure_count == pytest.approx(1.0)
    assert hypo.failure_online == pytest.approx(1.0)
    assert innocent.failure_count == pytest.approx(0.0)
    assert "card_a5_e15_o0" in bank.prototypes

    # Step 2: Failure 2
    bank.learn_from_episode([step], episode_id=2)
    assert hypo.failure_count == pytest.approx(2.0)
    assert "card_a5_e15_o0" in bank.prototypes

    # Step 3: Failure 3 -> Empirical success rate 1/4 = 0.25 <= 0.25 -> Falsified!
    bank.learn_from_episode([step], episode_id=3)
    assert "card_a5_e15_o0" not in bank.prototypes, "Falsified hypothesis card must be pruned after 3 consecutive failures!"
    assert "card_a5_e6_c0" in bank.prototypes, "Confirmed innocent card must remain in bank!"


def test_near_miss_fallback_when_no_active_subgoal():
    bank = RelationHypothesisBank()
    card = _make_dummy_card("card_a5_e6_c0", action=5, event_dim=6, status="confirmed", created_from="offline", succ=10.0, fail=0.0)
    bank.prototypes["card_a5_e6_c0"] = card

    step = {
        "obs_latent": [1.0] + [0.0] * 127,
        "grounded_state": [0.0] * 16,
        "action": 5,
        "reward": 0.0,
        "done": False,
        "event_vec": [0] * 22,
        "mask_before_vec": [0] * 22,
        "confident_events": [],
        "fallback_unlocked": [],
        "state_delta": [0.0] * 16,
        "active_subgoal_target": None,
    }

    bank.learn_from_episode([step], episode_id=1)
    # Cosine match found the card and denied it
    assert card.failure_count == pytest.approx(1.0)
