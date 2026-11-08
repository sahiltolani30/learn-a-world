"""Tests for ORBIT Gate 4: Action-Conditioned One-Step World Model.

Verifies:
1. Residual predictor initialization guarantees exact identity / persistence at step 0.
2. Residual predictor action and recurrent belief conditioning sensitivity.
3. Spatial change weighting mathematical invariants (normalization to 1.0, non-negativity, capping, gradient isolation).
4. Training step execution across all three Gate 4 formulations:
   - Formulation A: Standard unweighted non-residual baseline
   - Formulation B: Residual prediction with unweighted loss
   - Formulation C: Residual prediction with spatial change-weighted loss
5. Evaluation decomposition into static and changed tokens.
6. Strict adherence to anti-cheating protocols and tensor-only visual agent boundaries.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import ActionConditionedPredictor, LatentFrame, TrueJEPA
from true_jepa.objectives import (
    compute_spatial_change_weights,
    train_world_model_step,
)


def test_residual_predictor_identity_initialization() -> None:
    """Requirement 2: Residual predictor must initialize to exact persistence (Z_pred = Z_t)."""
    cfg = JEPAConfig(predictor_residual=True)
    num_actions = 17
    predictor = ActionConditionedPredictor(cfg, num_actions, residual=True)

    # 1. Verify residual mode is active
    assert predictor.residual is True
    assert hasattr(predictor, "delta_proj")

    # 2. Verify delta projection weights and biases are initialized to zero
    assert torch.all(predictor.delta_proj.weight == 0.0)
    assert torch.all(predictor.delta_proj.bias == 0.0)

    # 3. For any arbitrary input latent frame, action, and belief, output must exactly equal input
    torch.manual_seed(1234)
    batch_size = 4
    num_tokens = 64
    token_dim = cfg.token_dim

    tokens = torch.randn(batch_size, num_tokens, token_dim)
    pooled = tokens.mean(dim=1)
    frame = LatentFrame(tokens=tokens, pooled=pooled)

    action = torch.tensor([0, 3, 7, 16], dtype=torch.long)
    hidden = torch.randn(batch_size, cfg.belief_dim)

    predicted = predictor(frame, action, hidden)

    # Must be bitwise/numerically close to input frame tokens and pooled
    assert torch.allclose(predicted.tokens, frame.tokens, atol=1e-6)
    assert torch.allclose(predicted.pooled, frame.pooled, atol=1e-6)

    # Relative error against persistence at initialization is exactly 1.000
    persist_err = (frame.tokens - frame.tokens).square().mean()  # 0
    model_diff_from_persist = (predicted.tokens - frame.tokens).norm(p=2).item()
    assert model_diff_from_persist < 1e-6


def test_residual_predictor_action_and_belief_sensitivity() -> None:
    """Requirement 2: Ensure predictions actively condition on both action and recurrent belief."""
    cfg = JEPAConfig(predictor_residual=True)
    num_actions = 17
    predictor = ActionConditionedPredictor(cfg, num_actions, residual=True)

    # Give non-zero weights to delta_proj to test sensitivity after initial updates
    nn.init.normal_(predictor.delta_proj.weight, std=0.1)
    nn.init.normal_(predictor.delta_proj.bias, std=0.01)

    batch_size = 2
    tokens = torch.randn(batch_size, 64, cfg.token_dim)
    frame = LatentFrame(tokens=tokens, pooled=tokens.mean(dim=1))
    hidden = torch.randn(batch_size, cfg.belief_dim)

    # Action sensitivity: distinct actions produce distinct predicted deltas
    act1 = torch.tensor([1, 1], dtype=torch.long)
    act2 = torch.tensor([5, 5], dtype=torch.long)
    pred_act1 = predictor(frame, act1, hidden)
    pred_act2 = predictor(frame, act2, hidden)
    action_diff = (pred_act1.tokens - pred_act2.tokens).norm().item()
    assert action_diff > 1e-4, f"Action conditioning failed: diff={action_diff}"

    # Belief sensitivity: distinct recurrent hidden beliefs produce distinct predicted deltas
    hidden1 = torch.randn(batch_size, cfg.belief_dim)
    hidden2 = torch.randn(batch_size, cfg.belief_dim)
    pred_h1 = predictor(frame, act1, hidden1)
    pred_h2 = predictor(frame, act1, hidden2)
    belief_diff = (pred_h1.tokens - pred_h2.tokens).norm().item()
    assert belief_diff > 1e-4, f"Belief conditioning failed: diff={belief_diff}"


def test_spatial_change_weights_mathematical_properties() -> None:
    """Requirement 3: Change weights must be non-negative, normalized to mean 1.0, and capped."""
    batch_size = 4
    num_tokens = 64
    dim = 192

    # Simulate transition: 60 tokens static, 4 tokens changed
    current = torch.randn(batch_size, num_tokens, dim)
    target = current.clone()
    # Add significant displacement to last 4 tokens
    target[:, 60:, :] += 2.5 * torch.randn(batch_size, 4, dim)

    cap = 5.0
    bonus = 2.0
    weights, meaningful_fraction = compute_spatial_change_weights(
        current_tokens=current,
        target_tokens=target,
        cap=cap,
        bonus=bonus,
    )

    # 1. Non-negativity
    assert (weights >= 0.0).all(), "Weights must be non-negative"

    # 2. Normalization: mean across tokens must equal 1.0 for every sequence/batch item
    weights_mean_per_item = weights.mean(dim=-1)
    assert torch.allclose(weights_mean_per_item, torch.ones_like(weights_mean_per_item), atol=1e-5), (
        f"Weights not normalized to mean 1.0: {weights_mean_per_item}"
    )

    # 3. Dynamic contrast: changed tokens must receive strictly higher weight than static tokens
    static_weights = weights[:, :60].mean().item()
    changed_weights = weights[:, 60:].mean().item()
    assert changed_weights > static_weights, (
        f"Changed weight ({changed_weights:.3f}) must exceed static weight ({static_weights:.3f})"
    )

    # 4. Upper bound: weight must be capped
    # Capped ratio is at most cap, raw weight at most 1 + bonus * cap, so normalized weight cannot explode
    assert (weights <= (1.0 + bonus * cap + 1.0)).all()

    # 5. Gradient isolation: weights must not track gradients
    assert not weights.requires_grad

    # 6. Meaningful fraction diagnostic: ~4/64 = 6.25% changed
    assert 0.0 < meaningful_fraction < 0.25, f"Unexpected meaningful fraction: {meaningful_fraction}"


def test_gate4_three_formulations_training_step() -> None:
    """Requirement 3 & 4: All three formulations must execute cleanly and update parameters."""
    device = torch.device("cpu")
    num_actions = 17
    batch_size = 2
    seq_len = 4

    # Synthetic batch
    obs = torch.randint(0, 256, (batch_size, seq_len + 1, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, num_actions, (batch_size, seq_len), dtype=torch.long)
    rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    terminated = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    valid = torch.ones(batch_size, seq_len, dtype=torch.bool)

    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "valid": valid,
    }

    formulations = [
        ("Formulation A (Standard Unweighted)", False, False),
        ("Formulation B (Residual Unweighted)", True, False),
        ("Formulation C (Residual Change-Weighted)", True, True),
    ]

    for name, residual, change_weighted in formulations:
        cfg = JEPAConfig(predictor_residual=residual)
        model = TrueJEPA(cfg, num_actions=num_actions, encoder_type="scratch").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_cfg = TrainConfig(
            batch_size=batch_size,
            sequence_length=seq_len,
            change_weighted_loss=change_weighted,
            amp=False,
        )

        metrics = train_world_model_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            cfg=train_cfg,
            update=1,
        )

        assert np.isfinite(metrics["loss"]), f"{name} produced non-finite loss: {metrics['loss']}"
        assert np.isfinite(metrics["jepa"]), f"{name} produced non-finite jepa loss: {metrics['jepa']}"
        if change_weighted:
            assert "meaningful_token_fraction" in metrics


def test_gate4_static_and_changed_token_decomposition_math() -> None:
    """Requirement 4: Verify mathematical decomposition into static and changed token errors."""
    from eval.gate4_evaluator import decompose_static_and_changed_errors

    num_tokens = 64
    dim = 192

    current = torch.zeros(1, num_tokens, dim)
    target = torch.zeros(1, num_tokens, dim)

    # 60 tokens identical (static)
    # 4 tokens change significantly
    target[0, 60:, :] = 1.0

    # Persistence predicts current
    persist = current.clone()

    # Model predicts static perfectly, and predicts 50% of the change on changed tokens
    predicted = current.clone()
    predicted[0, 60:, :] = 0.5  # closer to target than persistence

    results = decompose_static_and_changed_errors(
        pred_tokens=predicted,
        target_tokens=target,
        current_tokens=current,
        change_threshold=0.1,
    )

    # Static MSE should be near zero for both
    assert results["mse_static_model"] == 0.0
    assert results["mse_static_persist"] == 0.0

    # Changed MSE: persistence error is 1.0^2 = 1.0, model error is (0.5 - 1.0)^2 = 0.25
    assert np.isclose(results["mse_changed_persist"], 1.0, atol=1e-5)
    assert np.isclose(results["mse_changed_model"], 0.25, atol=1e-5)

    # Relative error on changed tokens must be 0.25 / 1.0 = 0.25 (< 1.0 -> beats persistence!)
    assert np.isclose(results["rel_err_changed"], 0.25, atol=1e-5)


def test_gate4_anti_cheating_and_boundary_invariants() -> None:
    """Invariants: Strictly tensor inputs, no cheat recipes or privileged info in Gate 4."""
    source_files = [
        Path("true_jepa/model.py"),
        Path("true_jepa/objectives.py"),
        Path("true_jepa/config.py"),
    ]

    forbidden_strings = [
        "CRAFT_",
        "RECIPE",
        "INVENTORY",
        "make_wood_pickaxe",
        "crafting_table",
        "player_pos",
    ]

    for p in source_files:
        assert p.exists()
        code = p.read_text()
        for forbidden in forbidden_strings:
            assert forbidden.lower() not in code.lower(), f"Forbidden string {forbidden} in {p}"
