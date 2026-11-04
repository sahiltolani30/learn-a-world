"""Unit tests for AEON Phase K multi-step closed-loop validation pipeline."""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.jepa_world_model import JepaWorldModel
from src.experiments.phase_k_multistep_validation import (
    collect_evaluation_trajectories,
    evaluate_multistep_closed_loop,
    run_phase_k_validation,
    compute_effective_rank,
)


def test_trajectory_collection_fresh_seeds():
    """Verify trajectory collection on fresh seeds respects window and episode boundaries."""
    seeds = [991, 992]
    eval_data = collect_evaluation_trajectories(
        seeds=seeds,
        steps_per_seed=20,
        context_length=4,
        max_horizon=8,
    )
    B = eval_data["num_samples"]
    assert B > 0, "Expected at least one valid evaluation window"
    assert eval_data["context_obs"].shape == (B, 4, 64, 64, 3)
    assert eval_data["prev_actions"].shape == (B, 4)
    assert eval_data["future_actions"].shape == (B, 8)
    for h in [1, 2, 4, 8]:
        assert h in eval_data["target_obs"]
        assert eval_data["target_obs"][h].shape == (B, 64, 64, 3)


def test_closed_loop_rollout_isolation_and_autoregression():
    """Verify that autoregressive_rollout strictly uses previous predicted states, not ground truth."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = JepaWorldModel(
        latent_dim=128,
        action_dim=64,
        context_dim=128,
        num_predictor_members=1,
        num_transformer_layers=1,
        num_heads=2,
        mlp_dim=128,
    ).to(device)
    model.eval()

    B = 2
    T_ctx = 4
    H = 4
    obs_ctx = torch.rand(B, T_ctx, 3, 64, 64, device=device)
    prev_act = torch.randint(0, 17, (B, T_ctx), device=device)
    fut_act = torch.randint(0, 17, (B, H), device=device)

    with torch.no_grad():
        z_t, h_t = model.encode_actual_history(obs_ctx, prev_act)
        rollout = model.autoregressive_rollout(z_t, h_t, fut_act, member_idx=0)

    preds = rollout["predictions_by_horizon"]
    assert len(preds) == H
    for k in range(1, H + 1):
        assert preds[k].shape == (B, 64, 128)
        assert torch.isfinite(preds[k]).all()

    # Verify manual 2-step closed loop matches autoregressive_rollout exactly
    with torch.no_grad():
        # Step 1
        a1_emb = model.action_embedding(fut_act[:, 0])
        z1_manual = model.predictor_ensemble.forward_member(0, z_t, h_t, a1_emb)
        u1_manual = model.online_encoder.pool(z1_manual)
        h1_manual = model.temporal_context.step(u1_manual, a1_emb, h_t)

        # Step 2 strictly using predicted z1_manual and h1_manual
        a2_emb = model.action_embedding(fut_act[:, 1])
        z2_manual = model.predictor_ensemble.forward_member(0, z1_manual, h1_manual, a2_emb)

    assert torch.allclose(preds[1], z1_manual, atol=1e-5), "Step 1 prediction does not match manual step"
    assert torch.allclose(preds[2], z2_manual, atol=1e-5), "Step 2 prediction failed closed-loop recurrence"


def test_multistep_evaluation_metrics():
    """Verify evaluation function computes all required multi-step horizons, baselines, and controls."""
    torch.manual_seed(123)
    device = torch.device("cpu")
    model = JepaWorldModel(
        latent_dim=128,
        action_dim=64,
        context_dim=128,
        num_predictor_members=1,
        num_transformer_layers=1,
        num_heads=2,
        mlp_dim=128,
    ).to(device)

    B = 4
    eval_data = {
        "context_obs": torch.randint(0, 256, (B, 4, 64, 64, 3), dtype=torch.uint8),
        "prev_actions": torch.randint(0, 17, (B, 4), dtype=torch.long),
        "future_actions": torch.randint(0, 17, (B, 8), dtype=torch.long),
        "target_obs": {
            1: torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8),
            2: torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8),
            4: torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8),
            8: torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8),
        },
    }

    res = evaluate_multistep_closed_loop(
        model=model,
        eval_data=eval_data,
        device=device,
        threshold=0.001,
        member_idx=0,
    )

    horizon_metrics = res["horizon_metrics"]
    growth_curve = res["growth_curve"]

    for h in [1, 2, 4, 8]:
        h_str = str(h)
        assert h_str in horizon_metrics
        assert "total_model_error" in horizon_metrics[h_str]
        assert "total_persistence_error" in horizon_metrics[h_str]
        assert "total_shuffled_error" in horizon_metrics[h_str]
        assert "total_action_advantage" in horizon_metrics[h_str]
        assert "high_change" in horizon_metrics[h_str]
        assert "dynamic_tokens" in horizon_metrics[h_str]
        assert "spatial_error_map_mean" in horizon_metrics[h_str]
        assert len(horizon_metrics[h_str]["spatial_error_map_mean"]) == 64
        assert horizon_metrics[h_str]["representation_health"]["is_finite"] is True

    assert "1" in growth_curve
    assert "8" in growth_curve


def test_missing_candidate_checkpoint_gate_stop():
    """Verify that when the candidate checkpoint is missing, Phase K halts safely without faking weights."""
    res = run_phase_k_validation(checkpoint_path="/nonexistent/checkpoint_finetuned_1000.pt")
    assert res["status"] == "STOPPED"
    assert res["decision_gate"] == "PRELIMINARY_GATE_STOP"
    assert "Refused to substitute random weights" in res["protocol_adherence"]
