"""
genesis_jepa.tests.test_gate2_causal_budgeting — Automated Verification Suite for Gate 2
========================================================================================
Verifies:
  1. Kernel Density Mode Estimation (KDME) extracts integer costs (|C_hat - C| < 0.05)
  2. Consumable Likelihood Ratio (Lambda >= 100.0) filters 100% of metabolic vitals drift
  3. Factorized Conjunctive Affordance Gate passes Counterfactual Deprivation Battery
  4. Upstream Deficit Budgeting inhibits premature table placement until 3 wood is in hand
"""
from __future__ import annotations

import numpy as np
import pytest

from genesis_jepa.modules.causal_budgeter import (
    AnonymousKernelCostInducer,
    ConsumableLikelihoodRatioGate,
    FactorizedConjunctiveAffordanceGate,
    UpstreamDeficitBudgeter,
)


def test_kernel_density_mode_estimation_extracts_exact_costs() -> None:
    """
    Verifies that KDME extracts exact integer costs from noisy continuous state deltas.
    """
    inducer = AnonymousKernelCostInducer()
    rng = np.random.default_rng(42)

    # Action 8: Table placement consumes 2 wood (noisy continuous readings)
    table_action = 8
    wood_dim = 0
    true_table_cost = 2.0

    for _ in range(50):
        # Observed delta is centered at -2.0 with sensory noise sigma = 0.08
        noisy_delta = rng.normal(loc=-true_table_cost, scale=0.08)
        delta_vec = np.zeros(4, dtype=np.float32)
        delta_vec[wood_dim] = noisy_delta
        inducer.observe(table_action, delta_vec)

    # Action 11: Pickaxe crafting consumes 1 wood
    pickaxe_action = 11
    true_pickaxe_cost = 1.0
    for _ in range(50):
        noisy_delta = rng.normal(loc=-true_pickaxe_cost, scale=0.06)
        delta_vec = np.zeros(4, dtype=np.float32)
        delta_vec[wood_dim] = noisy_delta
        inducer.observe(pickaxe_action, delta_vec)

    # Estimate mode costs
    est_table_cost = inducer.estimate_mode_cost(table_action, wood_dim)
    est_pickaxe_cost = inducer.estimate_mode_cost(pickaxe_action, wood_dim)

    assert est_table_cost is not None
    assert est_pickaxe_cost is not None

    assert abs(est_table_cost - true_table_cost) < 0.05, (
        f"KDME table cost estimation failed! Got {est_table_cost}, expected {true_table_cost}"
    )
    assert abs(est_pickaxe_cost - true_pickaxe_cost) < 0.05, (
        f"KDME pickaxe cost estimation failed! Got {est_pickaxe_cost}, expected {true_pickaxe_cost}"
    )


def test_consumable_likelihood_ratio_filters_vitals_drift() -> None:
    """
    Verifies that the Likelihood Ratio (Lambda >= 100.0) filters continuous metabolic drift.
    """
    gate = ConsumableLikelihoodRatioGate(lambda_threshold=100.0)
    rng = np.random.default_rng(1337)

    action_strike = 5
    action_noop = 0
    thirst_dim = 1
    wood_dim = 2

    # Simulate 100 steps
    for _ in range(100):
        # 1. Thirst drains at same rate whether striking tree or noop
        thirst_noop = rng.normal(loc=-0.05, scale=0.02)
        thirst_strike = rng.normal(loc=-0.05, scale=0.02)

        # 2. Wood changes only when striking tree
        wood_noop = 0.0
        wood_strike = rng.normal(loc=1.0, scale=0.1) if rng.random() < 0.3 else 0.0

        noop_vec = np.array([0.0, thirst_noop, wood_noop], dtype=np.float32)
        strike_vec = np.array([0.0, thirst_strike, wood_strike], dtype=np.float32)

        gate.observe(action_noop, noop_vec)
        gate.observe(action_strike, strike_vec)

    # Thirst must be REJECTED (Lambda ≈ 1.0 << 100.0)
    thirst_ratio = gate.compute_likelihood_ratio(action_strike, thirst_dim)
    assert not gate.is_true_consumable(action_strike, thirst_dim), (
        f"Thirst drift falsely accepted as consumable! Ratio={thirst_ratio:.2f}"
    )
    assert thirst_ratio < 5.0, f"Thirst ratio should be near 1.0, got {thirst_ratio:.2f}"

    # Wood must be ACCEPTED (Lambda >= 100.0)
    wood_ratio = gate.compute_likelihood_ratio(action_strike, wood_dim)
    assert gate.is_true_consumable(action_strike, wood_dim), (
        f"Wood delta failed consumable test! Ratio={wood_ratio:.2f}"
    )
    assert wood_ratio >= 100.0


def test_factorized_conjunctive_counterfactual_deprivation_battery() -> None:
    """
    Verifies Factorized Conjunctive Gating on the non-negotiable Counterfactual Deprivation Battery.
    """
    # Constraint 1 (Table Proximity): w = [10.0, 0.0], bias = 5.0
    # Constraint 2 (Wood Count >= 1): w = [0.0, 10.0], bias = 5.0
    gate = FactorizedConjunctiveAffordanceGate(
        constraints=[
            (np.array([10.0, 0.0]), 5.0),
            (np.array([0.0, 10.0]), 5.0),
        ]
    )

    # Test A: Material Deprivation — Table present, Wood = 0
    context_a = np.array([1.0, 0.0], dtype=np.float32)
    prob_a = gate.predict_affordance(context_a)
    assert prob_a < 0.01, f"Material Deprivation Failed! P={prob_a:.5f} >= 0.01"

    # Test B: Proximity Removal — Wood = 3, Table absent
    context_b = np.array([0.0, 3.0], dtype=np.float32)
    prob_b = gate.predict_affordance(context_b)
    assert prob_b < 0.01, f"Proximity Removal Failed! P={prob_b:.5f} >= 0.01"

    # Test C: Conjunction Positive — Table present, Wood = 1
    context_c = np.array([1.0, 1.0], dtype=np.float32)
    prob_c = gate.predict_affordance(context_c)
    assert prob_c >= 0.90, f"Conjunction Positive Failed! P={prob_c:.5f} < 0.90"


def test_upstream_deficit_budgeter_prevents_premature_placement() -> None:
    """
    Verifies that UpstreamDeficitBudgeter learns discrete costs from empirical data
    via KDME and correctly computes deficits, with gating disabled when unobserved.
    """
    inducer = AnonymousKernelCostInducer()
    budgeter = UpstreamDeficitBudgeter(cost_inducer=inducer)

    # 1. Unobserved state: gating is disabled (deficit = 0.0)
    assert budgeter.compute_budget_deficit(0.0) == 0.0
    assert budgeter.is_placement_permitted(0.0)

    # 2. Feed empirical observation data for catalyst action A (cost = 2.0) and target action B (cost = 1.0)
    rng = np.random.RandomState(42)
    # Action A: consumes 2.0 units of resource 0
    for _ in range(30):
        delta = np.array([-2.0 + rng.normal(0, 0.05)])
        inducer.observe(action_id=8, state_delta=delta)

    # Action B: consumes 1.0 unit of resource 0
    for _ in range(30):
        delta = np.array([-1.0 + rng.normal(0, 0.05)])
        inducer.observe(action_id=11, state_delta=delta)

    # Now verify deficit is computed purely from learned empirical modes (2.0 + 1.0 = 3.0)
    # With 0 material: Deficit = 3.0 -> False
    assert budgeter.compute_budget_deficit(0.0, target_action_id=11, catalyst_action_id=8) == 3.0
    assert not budgeter.is_placement_permitted(0.0, target_action_id=11, catalyst_action_id=8)

    # With 2 material: Deficit = 1.0 -> False
    assert budgeter.compute_budget_deficit(2.0, target_action_id=11, catalyst_action_id=8) == 1.0
    assert not budgeter.is_placement_permitted(2.0, target_action_id=11, catalyst_action_id=8)

    # With 3 material: Deficit = 0.0 -> True!
    assert budgeter.compute_budget_deficit(3.0, target_action_id=11, catalyst_action_id=8) == 0.0
    assert budgeter.is_placement_permitted(3.0, target_action_id=11, catalyst_action_id=8)
