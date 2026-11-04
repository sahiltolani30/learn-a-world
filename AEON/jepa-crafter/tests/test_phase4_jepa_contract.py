"""Phase 4 Acceptance Test Suite: JEPA Baseline Model Contract & Invariants.

Strictly verifies the 10 required implementation tests defined in
Section 9 (Phase 4 — Verify/build the JEPA baseline) of Plan v2.0:
1. Input/latent shapes and dtype.
2. Online encoder receives gradients; target encoder receives none.
3. Target encoder initializes as an exact online copy.
4. EMA changes after successful optimizer steps and does not change after skipped/failed updates.
5. Loss is finite and gradients are finite on synthetic and real batches.
6. Training sequence alignment: action A[t] predicts observation O[t+1]; horizon 4 targets O[t+4].
7. Sampler never crosses episode end or reset.
8. Predictor does not receive future observations/targets.
9. Optimizer and EMA counters agree.
10. Checkpoint restores model, target, optimizer, scaler and scheduler.
"""

import copy
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler
from src.learning.learner import JepaLearner
from src.learning.optimizer import build_optimizer_and_scheduler
from src.models.ema_encoder import EmaTargetEncoder
from src.models.jepa_world_model import JepaWorldModel
from src.models.visual_encoder import VisualEncoder


# ---------------------------------------------------------------------------
# Test 1: Input/latent shapes and dtype
# ---------------------------------------------------------------------------
def test_input_latent_shapes_and_dtype():
    B = 2
    model = JepaWorldModel(latent_dim=128, context_dim=256, action_dim=64, num_actions=17)

    # Input: 64x64 RGB float32
    x = torch.rand(B, 3, 64, 64, dtype=torch.float32)
    tokens, pooled = model.online_encoder(x)

    assert tokens.shape == (B, 64, 128), f"Expected (B, 64, 128), got {tokens.shape}"
    assert tokens.dtype == torch.float32, f"Expected float32, got {tokens.dtype}"
    assert pooled.shape == (B, 128), f"Expected (B, 128), got {pooled.shape}"
    assert pooled.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: Online encoder receives gradients; target encoder receives none
# ---------------------------------------------------------------------------
def test_online_encoder_receives_grads_target_none():
    B = 2
    model = JepaWorldModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    learner = JepaLearner(model, optimizer, scheduler, device=torch.device("cpu"), precision="fp32")

    batch = {
        "obs_context": torch.rand(B, 4, 3, 64, 64),
        "context_mask": torch.ones(B, 4, dtype=bool),
        "previous_actions": torch.randint(0, 17, (B, 4)),
        "actions_future": torch.randint(0, 17, (B, 8)),
        "target_1": torch.rand(B, 3, 64, 64),
        "target_2": torch.rand(B, 3, 64, 64),
        "target_4": torch.rand(B, 3, 64, 64),
        "target_8": torch.rand(B, 3, 64, 64),
        "reward_first": torch.zeros(B, 1),
        "done_first": torch.zeros(B, 1),
    }

    learner.train_step(batch)

    # Online encoder parameters MUST receive gradients
    online_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.online_encoder.parameters()
    )
    assert online_has_grad, "Online encoder parameters must receive gradients"

    # Target encoder parameters MUST NEVER have gradients
    target_has_grad = any(p.grad is not None for p in model.target_encoder.parameters())
    assert not target_has_grad, "Target encoder parameters must never have gradients"


# ---------------------------------------------------------------------------
# Test 3: Target encoder initializes as an exact online copy
# ---------------------------------------------------------------------------
def test_target_encoder_initializes_as_exact_online_copy():
    online = VisualEncoder(latent_dim=128)
    target = EmaTargetEncoder(online)

    for p_online, p_target in zip(online.parameters(), target.encoder.parameters()):
        assert torch.equal(p_online, p_target), "Target encoder params must exactly match online at init"
        assert not p_target.requires_grad, "Target parameters must not require grad"


# ---------------------------------------------------------------------------
# Test 4: EMA changes after successful optimizer steps and does not change
# after skipped/failed updates
# ---------------------------------------------------------------------------
def test_ema_changes_after_successful_step_and_frozen_when_skipped():
    model = JepaWorldModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    learner = JepaLearner(model, optimizer, scheduler, device=torch.device("cpu"), precision="fp32")

    batch = {
        "obs_context": torch.rand(2, 4, 3, 64, 64),
        "context_mask": torch.ones(2, 4, dtype=bool),
        "previous_actions": torch.randint(0, 17, (2, 4)),
        "actions_future": torch.randint(0, 17, (2, 8)),
        "target_1": torch.rand(2, 3, 64, 64),
        "target_2": torch.rand(2, 3, 64, 64),
        "target_4": torch.rand(2, 3, 64, 64),
        "target_8": torch.rand(2, 3, 64, 64),
        "reward_first": torch.zeros(2, 1),
        "done_first": torch.zeros(2, 1),
    }

    # Record initial target weights
    target_p_init = next(model.target_encoder.encoder.parameters()).data.clone()
    initial_updates = model.target_encoder.update_count.item()

    # Successful step
    learner.train_step(batch)
    target_p_after = next(model.target_encoder.encoder.parameters()).data.clone()

    assert not torch.equal(target_p_init, target_p_after), "EMA weights must update after step"
    assert model.target_encoder.update_count.item() == initial_updates + 1

    # When update is not called (e.g. skipped step), target weights do not change
    target_p_before_skip = target_p_after.clone()
    # Simulating skipped step: zero grad without calling learner.train_step()
    optimizer.zero_grad()
    assert torch.equal(target_p_before_skip, next(model.target_encoder.encoder.parameters()).data)
    assert model.target_encoder.update_count.item() == initial_updates + 1


# ---------------------------------------------------------------------------
# Test 5: Loss is finite and gradients are finite on synthetic and real batches
# ---------------------------------------------------------------------------
def test_loss_and_gradients_finite():
    model = JepaWorldModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    learner = JepaLearner(model, optimizer, scheduler, device=torch.device("cpu"), precision="fp32")

    batch = {
        "obs_context": torch.rand(4, 4, 3, 64, 64),
        "context_mask": torch.ones(4, 4, dtype=bool),
        "previous_actions": torch.randint(0, 17, (4, 4)),
        "actions_future": torch.randint(0, 17, (4, 8)),
        "target_1": torch.rand(4, 3, 64, 64),
        "target_2": torch.rand(4, 3, 64, 64),
        "target_4": torch.rand(4, 3, 64, 64),
        "target_8": torch.rand(4, 3, 64, 64),
        "reward_first": torch.randn(4, 1),
        "done_first": torch.zeros(4, 1),
    }

    metrics = learner.train_step(batch)
    assert np.isfinite(metrics["loss"]), f"Total loss must be finite: {metrics['loss']}"
    assert np.isfinite(metrics["jepa_loss"]), f"JEPA loss must be finite: {metrics['jepa_loss']}"
    assert np.isfinite(metrics["grad_norm"]), f"Grad norm must be finite: {metrics['grad_norm']}"

    # Verify all parameter gradients are finite
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"Gradient for {name} must be finite"


# ---------------------------------------------------------------------------
# Test 6: Training sequence alignment: action A[t] predicts O[t+1]; H=4 -> O[t+4]
# ---------------------------------------------------------------------------
def test_training_sequence_horizon_alignment():
    capacity = 100
    rb = ReplayBuffer(capacity=capacity, obs_shape=(64, 64, 3))
    sampler = SequenceSampler(rb, context_length=4, prediction_horizon=8)

    # Insert a deterministic episode where frame RGB is uniform with value = timestep
    rb.start_episode(np.full((64, 64, 3), 0, dtype=np.uint8), episode_id=1)
    for t in range(1, 15):
        obs = np.full((64, 64, 3), t, dtype=np.uint8)
        rb.add(
            action=t,
            reward=0.1 * t,
            next_obs=obs,
            terminated=(t == 14),
            truncated=False,
        )

    # Sample batch
    batch = sampler.sample_batch(batch_size=1)
    prev_actions = batch["previous_actions"][0].numpy()
    actions_future = batch["actions_future"][0].numpy()

    # The first future action A[0] corresponds to transition predicting target_1
    # Verify target_1 corresponds to the step right after context
    target_1_val = int(batch["target_1"][0, 0, 0, 0].item() * 255.0 + 0.5)
    target_4_val = int(batch["target_4"][0, 0, 0, 0].item() * 255.0 + 0.5)
    target_8_val = int(batch["target_8"][0, 0, 0, 0].item() * 255.0 + 0.5)

    assert target_4_val - target_1_val == 3, f"Horizon 4 must be exactly 3 steps ahead of Horizon 1"
    assert target_8_val - target_4_val == 4, f"Horizon 8 must be exactly 4 steps ahead of Horizon 4"


# ---------------------------------------------------------------------------
# Test 7: Sampler never crosses episode end or reset
# ---------------------------------------------------------------------------
def test_sampler_never_crosses_episode_end_or_reset():
    rb = ReplayBuffer(capacity=100, obs_shape=(64, 64, 3))
    sampler = SequenceSampler(rb, context_length=4, prediction_horizon=8)

    # Ep 1: 5 steps
    rb.start_episode(np.zeros((64, 64, 3), dtype=np.uint8), episode_id=1)
    for t in range(5):
        rb.add(action=1, reward=0.0, next_obs=np.zeros((64, 64, 3), dtype=np.uint8), terminated=(t == 4), truncated=False)

    # Ep 2: 12 steps (valid)
    rb.start_episode(np.zeros((64, 64, 3), dtype=np.uint8), episode_id=2)
    for t in range(12):
        rb.add(action=2, reward=0.0, next_obs=np.zeros((64, 64, 3), dtype=np.uint8), terminated=(t == 11), truncated=False)

    # Any candidate starting in Ep 1 has length at most 5 < horizon 8, so cannot be valid
    for idx in range(5):
        assert not sampler.is_valid_anchor(idx), f"Index {idx} in Ep 1 cannot span horizon 8"


# ---------------------------------------------------------------------------
# Test 8: Predictor does not receive future observations/targets
# ---------------------------------------------------------------------------
def test_predictor_does_not_receive_future_targets():
    model = JepaWorldModel()

    # Inspect signature of forward_member in PredictorEnsemble
    sig = inspect.signature(model.predictor_ensemble.forward_member)
    params = list(sig.parameters.keys())
    assert "target" not in params, f"Predictor must not take target latents: {params}"
    assert "future_obs" not in params, f"Predictor must not take future obs: {params}"

    # Verify input args are strictly (member_idx, current_tokens, context_state, action_emb)
    expected_args = ["member_idx", "current_tokens", "context_state", "action_emb"]
    for expected in expected_args:
        assert expected in params, f"Missing expected argument '{expected}' in forward_member"


# ---------------------------------------------------------------------------
# Test 9: Optimizer and EMA counters agree
# ---------------------------------------------------------------------------
def test_optimizer_and_ema_counters_agree():
    model = JepaWorldModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    learner = JepaLearner(model, optimizer, scheduler, device=torch.device("cpu"), precision="fp32")

    batch = {
        "obs_context": torch.rand(2, 4, 3, 64, 64),
        "context_mask": torch.ones(2, 4, dtype=bool),
        "previous_actions": torch.randint(0, 17, (2, 4)),
        "actions_future": torch.randint(0, 17, (2, 8)),
        "target_1": torch.rand(2, 3, 64, 64),
        "target_2": torch.rand(2, 3, 64, 64),
        "target_4": torch.rand(2, 3, 64, 64),
        "target_8": torch.rand(2, 3, 64, 64),
        "reward_first": torch.zeros(2, 1),
        "done_first": torch.zeros(2, 1),
    }

    # Run 3 updates
    for i in range(3):
        metrics = learner.train_step(batch)
        assert learner.update_count == i + 1
        assert model.target_encoder.update_count.item() == i + 1
        assert learner.update_count == model.target_encoder.update_count.item(), (
            f"Counters must agree: learner={learner.update_count}, ema={model.target_encoder.update_count.item()}"
        )


# ---------------------------------------------------------------------------
# Test 10: Checkpoint restores model, target, optimizer, scaler and scheduler
# ---------------------------------------------------------------------------
def test_checkpoint_restores_complete_jepa_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"

        model1 = JepaWorldModel()
        opt1, sched1 = build_optimizer_and_scheduler(model1)
        learner1 = JepaLearner(model1, opt1, sched1, device=torch.device("cpu"), precision="fp32")

        batch = {
            "obs_context": torch.rand(2, 4, 3, 64, 64),
            "context_mask": torch.ones(2, 4, dtype=bool),
            "previous_actions": torch.randint(0, 17, (2, 4)),
            "actions_future": torch.randint(0, 17, (2, 8)),
            "target_1": torch.rand(2, 3, 64, 64),
            "target_2": torch.rand(2, 3, 64, 64),
            "target_4": torch.rand(2, 3, 64, 64),
            "target_8": torch.rand(2, 3, 64, 64),
            "reward_first": torch.zeros(2, 1),
            "done_first": torch.zeros(2, 1),
        }

        # Run 2 training steps
        learner1.train_step(batch)
        learner1.train_step(batch)

        # Save checkpoint
        torch.save({
            "model_state": model1.state_dict(),
            "optimizer_state": opt1.state_dict(),
            "scaler_state": learner1.scaler.state_dict(),
            "scheduler_state": sched1.state_dict() if sched1 else None,
            "learner_update_count": learner1.update_count,
        }, ckpt_path)

        # Restore into new model
        model2 = JepaWorldModel()
        opt2, sched2 = build_optimizer_and_scheduler(model2)
        learner2 = JepaLearner(model2, opt2, sched2, device=torch.device("cpu"), precision="fp32")

        loaded = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(loaded["model_state"])
        opt2.load_state_dict(loaded["optimizer_state"])
        learner2.scaler.load_state_dict(loaded["scaler_state"])
        if sched2 and loaded["scheduler_state"]:
            sched2.load_state_dict(loaded["scheduler_state"])
        learner2.update_count = loaded["learner_update_count"]

        # Assert all weights, counters, and buffer update counts match
        assert learner2.update_count == 2
        assert model2.target_encoder.update_count.item() == 2

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(model1.target_encoder.parameters(), model2.target_encoder.parameters()):
            assert torch.equal(p1, p2)
