"""
Unit & Smoke Tests for Part 2 Honest Self-Distribution Diagnostic.
Runs on Mac CPU in under 1 second with synthetic episodes.
"""

import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from scripts.diagnose_q_calibration_self import (
    compute_discounted_returns,
    evaluate_self_q_calibration,
)


def test_discounted_returns_computation():
    """Smoke test: Discounted returns formula G_t = sum_k gamma^k r_{t+k}."""
    rewards = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    gamma = 0.9
    returns = compute_discounted_returns(rewards, gamma=gamma)

    assert len(returns) == 4
    # G_3 = 0.0
    assert pytest.approx(returns[3]) == 0.0
    # G_2 = 1.0 + 0.9 * 0 = 1.0
    assert pytest.approx(returns[2]) == 1.0
    # G_1 = 0.0 + 0.9 * 1.0 = 0.9
    assert pytest.approx(returns[1]) == 0.9
    # G_0 = 1.0 + 0.9 * 0.9 = 1.81
    assert pytest.approx(returns[0]) == 1.81


def test_evaluate_self_q_calibration_synthetic():
    """Smoke test: Q-calibration evaluation computes valid correlation and dispersion metrics."""
    model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=17)
    model.eval()

    # Synthetic episode with 10 steps
    T = 10
    synthetic_episode = {
        "obs": np.random.randint(0, 255, size=(T, 64, 64, 3), dtype=np.uint8),
        "actions": np.random.randint(0, 17, size=(T,), dtype=np.int64),
        "rewards": np.random.uniform(0.0, 1.0, size=(T,)).astype(np.float32),
        "dones": np.array([False] * (T - 1) + [True], dtype=bool),
    }

    metrics, q_pred, returns, q_std = evaluate_self_q_calibration(
        model=model,
        episodes=[synthetic_episode],
        gamma=0.99,
        device="cpu",
    )

    assert metrics["num_episodes"] == 1
    assert metrics["total_transitions"] == T
    assert -1.0 <= metrics["pearson_r"] <= 1.0
    assert -1.0 <= metrics["spearman_rho"] <= 1.0
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert len(q_pred) == T
    assert len(returns) == T
    assert len(q_std) == T
