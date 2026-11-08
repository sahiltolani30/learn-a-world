"""Tests for ORBIT Gate 5: Temporal Memory and Multi-Step World Learning.

Verifies:
1. Multi-step training modes (one-step, autoregressive, mixed scheduled sampling).
2. Memory ablation behavior (verifying recurrent belief state actively modulates prediction).
3. Viewport exit-and-return probe mechanics (detecting cyclic/reversing actions and measuring memory retention).
4. Multi-step rollout stability across horizons H in {1, 2, 4, 8, 16}.
5. Scalar policy and value heads calibration.
6. Anti-cheating compliance and strict visual agent boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate5_evaluator import (
    evaluate_gate5_multi_step,
    evaluate_policy_value_calibration,
    evaluate_temporal_memory_ablation,
    evaluate_viewport_return_probe,
)
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import train_world_model_step


def test_gate5_training_rollout_modes() -> None:
    """Requirement 4: Verify one-step, autoregressive, and mixed scheduled rollout training modes."""
    device = torch.device("cpu")
    num_actions = 17
    batch_size = 2
    seq_len = 4

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

    modes = ["one_step", "autoregressive", "mixed"]

    for mode in modes:
        cfg = JEPAConfig(predictor_residual=True)
        model = TrueJEPA(cfg, num_actions=num_actions, encoder_type="scratch").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_cfg = TrainConfig(
            batch_size=batch_size,
            sequence_length=seq_len,
            change_weighted_loss=True,
            rollout_mode=mode,
            mixed_rollout_prob=0.5,
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

        assert np.isfinite(metrics["loss"]), f"Mode {mode} produced non-finite loss: {metrics['loss']}"
        assert np.isfinite(metrics["jepa"]), f"Mode {mode} produced non-finite jepa: {metrics['jepa']}"


def test_gate5_memory_ablation_behavior() -> None:
    """Requirement 3: Verify recurrent belief state causally alters predictions compared to h=0."""
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    # Give non-zero weights to delta projection so delta is active
    nn.init.normal_(model.predictor.delta_proj.weight, std=0.1)

    batch_size = 2
    tokens = torch.randn(batch_size, 64, cfg.token_dim)
    frame = LatentFrame(tokens=tokens, pooled=tokens.mean(dim=1))
    act = torch.tensor([1, 2], dtype=torch.long)

    # Condition B: Recurrent belief populated with history
    hidden_active = torch.randn(batch_size, cfg.belief_dim)
    pred_with_memory = model.predict(frame, act, hidden_active)

    # Condition A: Memory ablated (hidden = 0)
    hidden_zero = torch.zeros(batch_size, cfg.belief_dim)
    pred_without_memory = model.predict(frame, act, hidden_zero)

    diff = (pred_with_memory.tokens - pred_without_memory.tokens).norm().item()
    assert diff > 1e-4, f"Recurrent belief had zero effect on predictions: diff={diff}"


def test_gate5_viewport_probe_mechanics() -> None:
    """Requirement 3 & 7: Verify viewport exit and return detection on reversing action pairs."""
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    # Construct synthetic episode with a reversing action pair (Move North = 1, Move South = 2)
    T = 6
    obs = np.random.randint(0, 256, (T + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.array([0, 1, 2, 0, 3, 4], dtype=np.int64)  # Contains pairs (1,2) and (3,4)
    rewards = np.zeros(T, dtype=np.float32)

    synthetic_dataset = [{"obs": obs, "action": actions, "reward": rewards}]
    res = evaluate_viewport_return_probe(model, synthetic_dataset, torch.device("cpu"))

    assert res["total_return_events_detected"] >= 2, f"Failed to detect return events: {res}"
    assert np.isfinite(res["return_event_memory_mse"])
    assert np.isfinite(res["return_event_nomem_mse"])


def test_gate5_multi_step_horizons_1_to_16_evaluation() -> None:
    """Requirement 4: Verify stable evaluation across horizons 1, 2, 4, 8, 16."""
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    T = 20
    obs = np.random.randint(0, 256, (T + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.random.randint(0, 17, T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float32)

    synthetic_dataset = [{"obs": obs, "action": actions, "reward": rewards}]
    horizons = [1, 2, 4, 8, 16]

    res = evaluate_gate5_multi_step(model, synthetic_dataset, torch.device("cpu"), horizons=horizons)

    for h in horizons:
        str_h = str(h)
        assert str_h in res["teacher_forced_errors_by_horizon"]
        assert str_h in res["free_running_errors_by_horizon"]
        assert str_h in res["persistence_errors_by_horizon"]
        assert np.isfinite(res["free_running_errors_by_horizon"][str_h])


def test_gate5_policy_and_value_heads_calibration() -> None:
    """Requirement 5: Verify reward, continuation, value, and policy heads produce valid distributions."""
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    T = 15
    obs = np.random.randint(0, 256, (T + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.random.randint(0, 17, T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float32)

    synthetic_dataset = [{"obs": obs, "action": actions, "reward": rewards}]
    calib = evaluate_policy_value_calibration(model, synthetic_dataset, torch.device("cpu"))

    assert np.isfinite(calib["mean_reward_mae"])
    assert 0.0 <= calib["mean_continuation_prob"] <= 1.0
    assert calib["mean_policy_entropy"] > 0.0
    assert calib["policy_ready_for_training"] is True


def test_gate5_anti_cheating_and_isolation() -> None:
    """Requirement 6: Verify zero forbidden cheat tables or privileged state in Gate 5 code."""
    source_files = [
        Path("eval/gate5_evaluator.py"),
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
