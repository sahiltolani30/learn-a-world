"""Tests for tensor shapes and forward/backward gradient flows across the JEPA model."""
import pytest
import torch
from src.models.jepa_world_model import JepaWorldModel
from src.learning.jepa_losses import JepaLoss
from src.learning.learner import JepaLearner
from src.learning.optimizer import build_optimizer_and_scheduler


def test_jepa_shapes_and_rollout():
    B = 2
    model = JepaWorldModel(latent_dim=128, context_dim=256, action_dim=64, num_actions=17)

    # 1. Visual Encoder shape check
    x = torch.rand(B, 3, 64, 64)
    tokens, pooled = model.online_encoder(x)
    assert tokens.shape == (B, 64, 128)
    assert pooled.shape == (B, 128)

    # 2. Context encoding shape check
    obs_context = torch.rand(B, 4, 3, 64, 64)
    prev_actions = torch.randint(0, 17, (B, 4))
    z_t, h_t = model.encode_actual_history(obs_context, prev_actions)
    assert z_t.shape == (B, 64, 128)
    assert h_t.shape == (B, 256)

    # 3. Autoregressive rollout shape check (8 steps)
    future_actions = torch.randint(0, 17, (B, 8))
    rollout = model.autoregressive_rollout(z_t, h_t, future_actions, member_idx=0)
    for k in [1, 2, 4, 8]:
        assert rollout["predictions_by_horizon"][k].shape == (B, 64, 128)
    assert rollout["auxiliary_first"].reward.shape == (B, 1)
    assert rollout["auxiliary_first"].done_probability.shape == (B, 1)


def test_jepa_backward_gradient_flow():
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

    metrics = learner.train_step(batch)
    assert "loss" in metrics
    assert "jepa_loss" in metrics
    assert metrics["loss"] > 0

    # Ensure online parameters received gradients
    online_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.online_encoder.parameters())
    assert online_has_grad, "Online encoder parameters must receive gradients"

    # Ensure target parameters have NO gradients
    target_has_grad = any(p.grad is not None for p in model.target_encoder.parameters())
    assert not target_has_grad, "Target encoder parameters must never have gradients"
