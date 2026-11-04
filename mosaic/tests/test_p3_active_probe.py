"""Tests for MOSAIC Phase P3: Calibrated Uncertainty & Active Probing."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch

from mosaic.src.checkpoint import load_checkpoint, save_atomic
from mosaic.src.models.world_model import MosaicWorldModel
from mosaic.src.exploration.calibration import compute_calibration_curve, rank_correlation
from mosaic.src.exploration.active_probe import (
    DualValueHeads,
    ActiveProbeScorer,
    ActiveProbePolicy,
)


def test_calibration_curve_binned_and_correlation() -> None:
    # 1. Positively correlated uncertainties and errors
    rng = np.random.default_rng(42)
    true_signal = np.linspace(0.1, 1.0, 50)
    uncertainties = true_signal + rng.normal(0, 0.05, 50)
    errors = true_signal + rng.normal(0, 0.05, 50)

    report = compute_calibration_curve(uncertainties, errors, num_bins=5)
    assert report["calibration_passed"] is True
    assert report["rank_correlation"] > 0.5
    assert len(report["bins"]) == 5
    # Check monotonicity of bins
    assert report["monotonicity"] >= 0.6


def test_dual_value_heads_isolation() -> None:
    B = 4
    belief_dim = 128
    heads = DualValueHeads(belief_dim=belief_dim, hidden_dim=64)

    belief = torch.randn(B, belief_dim)
    v_task, v_info = heads(belief)
    assert v_task.shape == (B, 1)
    assert v_info.shape == (B, 1)

    # Backward pass on v_info only
    loss_info = v_info.pow(2).mean()
    loss_info.backward()

    # Verify task head parameters have zero gradient
    for name, p in heads.task_head.named_parameters():
        assert p.grad is None, f"Task head parameter {name} received unwanted gradient!"

    # Verify info head parameters received gradients
    has_grad = any(p.grad is not None for p in heads.info_head.parameters())
    assert has_grad, "Info head parameters failed to receive gradients!"


def test_active_probe_scorer_risk_penalty() -> None:
    wm = MosaicWorldModel(
        num_actions=2, latent_dim=32, action_dim=16, hidden_dim=32, stochastic_dim=16
    )
    scorer = ActiveProbeScorer(world_model=wm, beta_epistemic=1.0, rho_risk=10.0)

    belief = torch.randn(1, wm.belief.belief_dim)
    score_out = scorer.score_candidate_actions(belief)

    assert "scores" in score_out
    assert "disagreements" in score_out
    assert "death_risks" in score_out
    assert len(score_out["scores"]) == 2
    assert 0 <= score_out["best_action"] < 2


def test_active_probe_policy_contract() -> None:
    wm = MosaicWorldModel(
        num_actions=17, latent_dim=32, action_dim=16, hidden_dim=32, stochastic_dim=16
    )
    policy = ActiveProbePolicy(world_model=wm, mode="active_probe", seed=123)

    # Initial frame at step 0
    obs_0 = np.zeros((64, 64, 3), dtype=np.uint8)
    action_0 = policy.act(obs_0, 0.0, 0)
    assert 0 <= action_0 < 17
    assert len(policy.trace_history) == 1

    # Next frame at step 1
    obs_1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    action_1 = policy.act(obs_1, 1.0, 1)
    assert 0 <= action_1 < 17
    assert len(policy.trace_history) == 2


def test_active_probe_checkpoint_roundtrip() -> None:
    heads = DualValueHeads(belief_dim=64, hidden_dim=32)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "active_probe.pt"
        save_atomic(
            ckpt_path,
            {
                "format": "mosaic_runtime_v1",
                "phase": "p3_active_probe",
                "dual_values_state": heads.state_dict(),
            },
        )
        loaded = load_checkpoint(ckpt_path)
        assert loaded["format"] == "mosaic_runtime_v1"
        assert loaded["phase"] == "p3_active_probe"
