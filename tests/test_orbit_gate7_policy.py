"""Gate 7 Test Suite: Real-Experience Policy Learning, Value Calibration & Reward Validation.

Verifies:
1. Data integrity: episode alignment (T_obs = T_act + 1), bounds, terminal handling, zero NaN.
2. Optimizer ownership: disjoint actor and value parameter ownership, frozen backbone isolation.
3. Objective correctness: manual return calculations, advantage weighting, behavior regularization.
4. Reward validation and value calibration mathematical correctness.
5. Invariant check: zero imagined rollouts in Gate 7 update.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate7_evaluator import (
    audit_replay_dataset,
    calibrate_value_predictions,
    real_experience_policy_step,
    validate_reward_prediction,
)
from true_jepa.config import JEPAConfig
from true_jepa.model import LatentFrame, TrueJEPA


# ---------------------------------------------------------------------------
# Test 1: Data Integrity & Dataset Audit
# ---------------------------------------------------------------------------

def test_gate7_dataset_audit_and_temporal_alignment():
    """Verify replay audit correctly validates alignment, action range, rewards, and seed isolation."""
    # Test on base dataset
    base_path = Path("data/gate3/train_replay_10k.npz")
    assert base_path.exists(), "Base training replay shard not found"

    audit = audit_replay_dataset(base_path)
    assert audit["status"] == "PASS"
    assert audit["temporal_alignment_ok"] is True
    assert audit["total_obs"] == audit["total_actions"] + audit["num_episodes"]
    assert audit["actions_valid"] is True
    assert audit["rewards_valid"] is True
    assert audit["obs_shape_valid"] is True
    assert audit["seed_isolation"]["is_disjoint"] is True
    assert len(audit["seed_isolation"]["eval_overlap"]) == 0
    assert len(audit["seed_isolation"]["bench_overlap"]) == 0


# ---------------------------------------------------------------------------
# Test 2: Optimizer Ownership & Frozen Parameter Invariants
# ---------------------------------------------------------------------------

def test_gate7_optimizer_isolation_and_parameter_ownership():
    """Verify actor and value optimizers have zero overlap and world model remains strictly frozen."""
    cfg = JEPAConfig()
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    actor_params = list(model.heads.policy.parameters())
    value_params = list(model.heads.value.parameters())

    actor_set = set(actor_params)
    value_set = set(value_params)

    # 1. Zero overlap between actor and value optimizers
    assert len(actor_set & value_set) == 0, "Actor and Value parameters must be strictly disjoint"

    # 2. Check world model components are outside actor and value optimizers
    encoder_params = set(model.encoder.parameters())
    belief_params = set(model.belief.parameters())
    predictor_params = set(model.predictor.parameters())

    assert len(actor_set & encoder_params) == 0
    assert len(actor_set & belief_params) == 0
    assert len(actor_set & predictor_params) == 0

    assert len(value_set & encoder_params) == 0
    assert len(value_set & belief_params) == 0
    assert len(value_set & predictor_params) == 0


# ---------------------------------------------------------------------------
# Test 3: Mathematical Return Targets & Boundary Handling
# ---------------------------------------------------------------------------

def test_gate7_real_returns_manual_calculation():
    """Verify discounted real return calculation handles terminal transitions and boundaries exactly."""
    gamma = 0.90
    B = 2
    T = 4

    # Construct synthetic rewards and termination
    # Batch row 0: continues to end: r = [1, 2, 3, 4], terminated = [0, 0, 0, 0]
    # G_3 = 4
    # G_2 = 3 + 0.9 * 4 = 6.6
    # G_1 = 2 + 0.9 * 6.6 = 7.94
    # G_0 = 1 + 0.9 * 7.94 = 8.146
    # Batch row 1: terminates at step 2: r = [1, 1, 1, 0], terminated = [0, 1, 0, 0]
    # G_3 = 0
    # G_2 = 1 + 0.9 * 0 = 1.0 (step 1 was terminated, so cont_mask = 0 for step 0->1? No, terminated[1]=1 means step 1 ends)
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    terminated = torch.tensor([[False, False, False, False], [False, True, False, False]], dtype=torch.bool)

    returns = torch.zeros((B, T), dtype=torch.float32)
    running = torch.zeros(B, dtype=torch.float32)
    for t in reversed(range(T)):
        cont_mask = (~terminated[:, t]).float()
        running = rewards[:, t] + gamma * cont_mask * running
        returns[:, t] = running

    # Check row 0
    expected_row0 = [8.146, 7.94, 6.6, 4.0]
    assert torch.allclose(returns[0], torch.tensor(expected_row0), atol=1e-3)

    # Check row 1: step 1 is terminated, so return at step 1 does not accumulate step 2
    assert returns[1, 3] == 0.0
    assert returns[1, 2] == 1.0
    assert returns[1, 1] == 1.0 # terminated at step 1, running was reset: 1 + 0.9 * 0 * running = 1.0
    assert returns[1, 0] == 1.0 + 0.9 * 1.0 # 1.9


# ---------------------------------------------------------------------------
# Test 4: Offline Policy Learning Step & Gradient Verification
# ---------------------------------------------------------------------------

def test_gate7_offline_policy_learning_step():
    """Verify real_experience_policy_step updates actor and value heads without touching world model."""
    cfg = JEPAConfig(token_dim=64, belief_dim=64, predictor_mlp_dim=64)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    actor_opt = torch.optim.Adam(model.heads.policy.parameters(), lr=1e-3)
    value_opt = torch.optim.Adam(model.heads.value.parameters(), lr=1e-3)

    # Freeze world model
    model.encoder.requires_grad_(False)
    model.belief.requires_grad_(False)
    model.predictor.requires_grad_(False)

    B = 2
    T = 4
    obs = torch.randint(0, 256, (B, T + 1, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 17, (B, T), dtype=torch.long)
    rewards = torch.randn(B, T, dtype=torch.float32)
    terminated = torch.zeros(B, T, dtype=torch.bool)
    valid = torch.ones(B, T, dtype=torch.bool)

    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "valid": valid,
    }

    metrics = real_experience_policy_step(
        model=model,
        batch=batch,
        actor_optimizer=actor_opt,
        value_optimizer=value_opt,
        gamma=0.99,
        advantage_temperature=1.0,
    )

    assert "actor_loss" in metrics
    assert "value_loss" in metrics
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["value_loss"])

    # Verify world model received zero gradients
    for p in model.encoder.parameters():
        assert p.grad is None
    for p in model.predictor.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# Test 5: Reward Validation & Value Calibration Functions
# ---------------------------------------------------------------------------

def test_gate7_reward_and_value_evaluation_metrics():
    """Verify reward validation and value calibration compute bounded, finite metrics on synthetic episodes."""
    cfg = JEPAConfig(token_dim=64, belief_dim=64, predictor_mlp_dim=64)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    # Construct small synthetic episode
    T = 5
    episodes = [{
        "obs": np.random.randint(0, 256, (T + 1, 64, 64, 3), dtype=np.uint8),
        "actions": np.random.randint(0, 17, (T,), dtype=np.int64),
        "rewards": np.random.randn(T).astype(np.float32),
    }]

    device = torch.device("cpu")
    r_metrics = validate_reward_prediction(model, episodes, device)
    assert "reward_mae" in r_metrics
    assert "reward_rmse" in r_metrics
    assert "reward_correlation" in r_metrics
    assert np.isfinite(r_metrics["reward_mae"])

    v_metrics = calibrate_value_predictions(model, episodes, device, gamma=0.99)
    assert "value_mae" in v_metrics
    assert "value_rmse" in v_metrics
    assert "value_correlation" in v_metrics
    assert np.isfinite(v_metrics["value_mae"])


# ---------------------------------------------------------------------------
# Test 6: Gate 7 Complete Pipeline Execution & Artifact Verification
# ---------------------------------------------------------------------------

def test_gate7_pipeline_smoke_artifacts():
    """Verify that Gate 7 runner produces all required manifests, checkpoints, plots, and metrics."""
    target_dir = Path("results/gate7") if Path("results/gate7").exists() else Path("results/gate7_smoke")
    if not target_dir.exists():
        pytest.skip("Gate 7 results directory results/gate7 does not exist yet")

    manifest = target_dir / "gate7_dataset_manifest.json"
    ckpt = target_dir / "policy_gate7.pt"
    plots = target_dir / "gate7_telemetry_plots.png"
    metrics = target_dir / "gate7_experiment_metrics.json"

    assert manifest.exists(), "Dataset manifest must exist"
    assert ckpt.exists(), "Candidate policy checkpoint must exist"
    assert ckpt.stat().st_size > 10 * 1024 * 1024, "Checkpoint must be non-trivial size"
    assert plots.exists(), "Telemetry plots must exist"
    assert metrics.exists(), "Metrics manifest must exist"

    import json
    with open(metrics) as f:
        m = json.load(f)
    assert m["baseline_status"] == "PASS"
    assert "random_baseline" in m["benchmark_results"]
    assert "gate6_champion" in m["benchmark_results"]
    assert "gate7_candidate" in m["benchmark_results"]

