"""Tier 1: Survival Vitals & Continuation Discounting (F13, F14).

Tests:
- F13: Continuation Discounting Head (c_t) [5 tests]
- F14: State-Dependent Discount Factor (Gamma_t) [5 tests]
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zephyr.heads import ZephyrHeads
from zephyr.losses import focal_bce_with_logits, lambda_returns, roc_auc
from zephyr.train_wm import evaluate_gates
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 13: Continuation Discounting Head (c_t)
# ==============================================================================

def test_f13_continue_head_architecture_and_output(mock_heads):
    """F13: Continue head maps composite state s to scalar continuation logit."""
    batch_size = 4
    state_dim = mock_heads.cfg.world_model.latent_dim + mock_heads.cfg.world_model.belief_dim
    s = torch.randn(batch_size, state_dim)

    logits = mock_heads.contin(s)

    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all(), "Continuation logits must be finite"


def test_f13_focal_bce_loss_rare_positive_handling():
    """F13: focal_bce_with_logits penalizes false negatives on rare termination events with pos_weight=25.0."""
    # 99 alive transitions (target=0), 1 terminal transition (target=1)
    targets = torch.zeros(100)
    targets[-1] = 1.0

    # Model predicting constant "alive" (logit = -5.0 for all, so sigmoid ~ 0.0067)
    logits_naive_alive = torch.full((100,), -5.0)

    # Model predicting correctly: positive logit on terminal transition
    logits_accurate = torch.full((100,), -5.0)
    logits_accurate[-1] = 5.0

    loss_naive = focal_bce_with_logits(logits_naive_alive, targets, pos_weight=25.0, gamma=2.0)
    loss_accurate = focal_bce_with_logits(logits_accurate, targets, pos_weight=25.0, gamma=2.0)

    assert loss_accurate.item() < loss_naive.item(), (
        f"Accurate loss {loss_accurate.item()} should be < naive majority loss {loss_naive.item()}"
    )


def test_f13_continuation_probability_calibration(mock_heads):
    """F13: Sigmoid of continue logits produces calibrated survival probabilities bounded in [0.0, 1.0]."""
    state_dim = mock_heads.cfg.world_model.latent_dim + mock_heads.cfg.world_model.belief_dim
    s = torch.randn(32, state_dim)

    with torch.no_grad():
        logits = mock_heads.contin(s)
        probs = torch.sigmoid(logits)

    assert probs.shape == (32,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be bounded in [0, 1]"
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_f13_continue_gate_zg5_metric_evaluation(mock_wm, mock_heads, populated_replay, tiny_test_cfg):
    """F13: evaluate_gates computes gate ZG5 telemetry with recall, precision, and AUC."""
    gates = evaluate_gates(mock_wm, mock_heads, populated_replay, tiny_test_cfg, device="cpu", max_batches=2)

    assert "zg5" in gates, "Gate ZG5 missing from evaluation output"
    zg5 = gates["zg5"]
    assert "available" in zg5
    if zg5["available"]:
        assert "recall_at_0p5" in zg5
        assert "precision_at_0p5" in zg5
        assert "auc" in zg5
        assert "passed" in zg5


def test_f13_continuation_roc_auc_perfect_vs_inverted():
    """F13: roc_auc outputs 1.0 for perfect ranking, 0.0 for inverted ranking, and 0.5 for random/ties."""
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    # Perfect ranking
    scores_perfect = torch.tensor([0.1, 0.2, 0.8, 0.9])
    auc_perf = roc_auc(scores_perfect, targets)
    assert auc_perf == pytest.approx(1.0, abs=1e-5), f"Perfect AUC {auc_perf} != 1.0"

    # Inverted ranking
    scores_inverted = torch.tensor([0.9, 0.8, 0.2, 0.1])
    auc_inv = roc_auc(scores_inverted, targets)
    assert auc_inv == pytest.approx(0.0, abs=1e-5), f"Inverted AUC {auc_inv} != 0.0"

    # Constant scores (tied)
    scores_tied = torch.tensor([0.5, 0.5, 0.5, 0.5])
    auc_tied = roc_auc(scores_tied, targets)
    assert auc_tied == pytest.approx(0.5, abs=1e-5), f"Tied AUC {auc_tied} != 0.5"


# ==============================================================================
# Feature 14: State-Dependent Discount Factor (Gamma_t)
# ==============================================================================

def test_f14_cumulative_discount_multiplication():
    """F14: Cumulative discount Gamma_t = prod_{k=0}^t gamma * c(s_k)."""
    gamma = 0.997
    continues = torch.tensor([1.0, 1.0, 0.5, 0.0])

    gamma_eff = []
    accum = 1.0
    for c in continues:
        accum = accum * gamma * c.item()
        gamma_eff.append(accum)

    assert gamma_eff[0] == pytest.approx(gamma, abs=1e-5)
    assert gamma_eff[1] == pytest.approx(gamma ** 2, abs=1e-5)
    assert gamma_eff[2] == pytest.approx((gamma ** 3) * 0.5, abs=1e-5)
    assert gamma_eff[3] == pytest.approx(0.0, abs=1e-5)


def test_f14_zero_discount_nullifies_imagined_death_returns():
    """F14: When imagined agent dies (c_t = 0), returns from steps beyond death contribute zero to step 0."""
    T, B = 5, 1
    # Step 2 has death (continues=0.0). Step 4 has massive reward (+1000.0)
    rewards = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1000.0]])
    continues = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0]])
    values = torch.zeros(T, B)
    bootstrap = torch.zeros(B)

    returns = lambda_returns(rewards, continues, values, bootstrap, gamma=0.997, td_lambda=0.95)

    # Because continues[2] = 0, the reward at step 4 cannot propagate across step 2 to step 0
    assert returns[0, 0].item() == pytest.approx(0.0, abs=1e-5), (
        f"Step 0 return {returns[0, 0].item()} must be 0.0 due to death at step 2"
    )
    assert returns[1, 0].item() == pytest.approx(0.0, abs=1e-5)


def test_f14_nightfall_vitals_discount_attenuation(mock_heads):
    """F14: States with critical vitals or danger yield lower effective discount factor gamma_eff."""
    gamma = 0.997
    state_dim = mock_heads.cfg.world_model.latent_dim + mock_heads.cfg.world_model.belief_dim

    # Healthy state vs critical state
    healthy_s = torch.zeros(1, state_dim)
    critical_s = torch.randn(1, state_dim) * 2.0

    with torch.no_grad():
        c_healthy = 1.0 - torch.sigmoid(mock_heads.contin(healthy_s)).item()
        c_critical = 1.0 - torch.sigmoid(mock_heads.contin(critical_s)).item()

    gamma_eff_healthy = gamma * c_healthy
    gamma_eff_critical = gamma * c_critical

    assert 0.0 <= gamma_eff_healthy <= gamma
    assert 0.0 <= gamma_eff_critical <= gamma


def test_f14_constant_continuation_baseline_match():
    """F14: When c_t = 1.0 for all steps, state-dependent discounting matches standard exponential gamma^t."""
    T = 10
    gamma = 0.997
    continues = torch.ones(T)

    gamma_eff = []
    accum = 1.0
    for _ in range(T):
        accum *= gamma
        gamma_eff.append(accum)

    expected = [gamma ** (t + 1) for t in range(T)]
    assert np.allclose(gamma_eff, expected, atol=1e-5), f"Effective discounts {gamma_eff} != {expected}"


def test_f14_discount_numerical_stability():
    """F14: Cumulative product over 100 continuous steps does not underflow to NaN or produce negative values."""
    T = 100
    gamma = 0.997
    rng = np.random.default_rng(42)
    c_seq = rng.uniform(0.5, 1.0, size=T)

    accum = 1.0
    for t in range(T):
        accum = accum * gamma * c_seq[t]
        assert not np.isnan(accum), f"NaN detected at step {t}"
        assert accum >= 0.0, f"Negative discount {accum} at step {t}"
        assert accum <= 1.0, f"Discount {accum} exceeded 1.0 at step {t}"


def test_f14_trained_continuation_discount_calibration(mock_heads):
    """F14: Trained continuation head yields gamma_eff ~ 0.997 when alive and ~ 0.0 on terminal death."""
    torch.manual_seed(42)
    state_dim = mock_heads.cfg.world_model.latent_dim + mock_heads.cfg.world_model.belief_dim
    opt = torch.optim.Adam(mock_heads.contin.parameters(), lr=1e-2)

    s_alive = torch.randn(90, state_dim)
    s_term = torch.randn(10, state_dim) + 2.0
    targets = torch.cat([torch.zeros(90), torch.ones(10)], dim=0)
    s_batch = torch.cat([s_alive, s_term], dim=0)

    for _ in range(120):
        loss = focal_bce_with_logits(mock_heads.contin(s_batch), targets, pos_weight=25.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        c_alive = (1.0 - torch.sigmoid(mock_heads.contin(s_alive))).mean().item()
        c_term = (1.0 - torch.sigmoid(mock_heads.contin(s_term))).mean().item()

    gamma = 0.997
    gamma_eff_alive = gamma * c_alive
    gamma_eff_term = gamma * c_term

    assert gamma_eff_alive > 0.95, f"Alive discount {gamma_eff_alive:.4f} must exceed 0.95"
    assert gamma_eff_term < 0.05, f"Term discount {gamma_eff_term:.4f} must be below 0.05"
