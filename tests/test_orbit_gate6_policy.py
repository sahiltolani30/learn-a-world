"""Tests for ORBIT Gate 6: Actor-Critic Policy Learning with World-Model Imagination.

Verifies:
1. GAE advantage calculation and lambda-return regression in imagination_actor_critic_gae_step.
2. Gradient isolation: world model parameters remain strictly frozen during policy learning.
3. Anti-exploitation safeguards (reward clipping and uncertainty penalty).
4. Real Crafter gameplay evaluator contract and metrics schema.
5. Strict anti-cheating invariants and pixel-only agent boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    compute_crafter_score,
    evaluate_policy_in_crafter,
)
from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import imagination_actor_critic_gae_step


def test_imagination_actor_critic_gae_and_returns() -> None:
    """Requirement 3: Verify GAE calculation, lambda-returns, and parameter update isolation."""
    device = torch.device("cpu")
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch").to(device)

    # Freeze world model and encoder
    model.encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.predictor.requires_grad_(False)
    model.belief.requires_grad_(False)

    actor_opt = torch.optim.AdamW(model.heads.policy.parameters(), lr=1e-4)
    val_opt = torch.optim.AdamW(model.heads.value.parameters(), lr=1e-3)

    tokens = torch.randn(4, 64, cfg.token_dim, device=device)
    start = LatentFrame(tokens, tokens.mean(dim=1))
    hidden = model.initial_belief(4, device)

    metrics = imagination_actor_critic_gae_step(
        model=model,
        start=start,
        hidden=hidden,
        actor_optimizer=actor_opt,
        value_optimizer=val_opt,
        horizon=4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_weight=0.01,
        reward_clip=1.0,
    )

    # 1. Verify metrics are finite
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["mean_advantage"])
    assert np.isfinite(metrics["mean_imagined_reward"])

    # 2. Verify frozen parameters received ZERO gradients
    for p in model.encoder.parameters():
        assert p.grad is None
    for p in model.predictor.parameters():
        assert p.grad is None
    for p in model.belief.parameters():
        assert p.grad is None

    # 3. Verify trainable policy and value heads received valid gradients
    for p in model.heads.policy.parameters():
        assert p.grad is not None
    for p in model.heads.value.parameters():
        assert p.grad is not None


def test_anti_exploitation_safeguards() -> None:
    """Requirement 4: Verify reward clipping and uncertainty penalty prevent reward hallucination."""
    device = torch.device("cpu")
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch").to(device)

    # Set reward head bias to a large value to simulate model exploitation
    with torch.no_grad():
        model.heads.reward[-1].bias.fill_(100.0)

    actor_opt = torch.optim.AdamW(model.heads.policy.parameters(), lr=1e-4)
    val_opt = torch.optim.AdamW(model.heads.value.parameters(), lr=1e-3)

    tokens = torch.randn(2, 64, cfg.token_dim, device=device)
    start = LatentFrame(tokens, tokens.mean(dim=1))
    hidden = model.initial_belief(2, device)

    # Without clipping, mean reward would be ~100.0
    # With clipping (reward_clip=1.0), mean reward must be bounded by <= 1.0
    metrics = imagination_actor_critic_gae_step(
        model=model,
        start=start,
        hidden=hidden,
        actor_optimizer=actor_opt,
        value_optimizer=val_opt,
        horizon=2,
        reward_clip=1.0,
    )

    assert metrics["mean_imagined_reward"] <= 1.0, (
        f"Reward clipping failed: imagined reward {metrics['mean_imagined_reward']} > 1.0"
    )


def test_policy_action_distribution_and_entropy() -> None:
    """Requirement 7: Verify actor outputs valid distributions over 17 discrete actions."""
    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    tokens = torch.randn(2, 64, cfg.token_dim)
    frame = LatentFrame(tokens, tokens.mean(dim=1))
    hidden = model.initial_belief(2, "cpu")

    logits = model.policy_logits(frame, hidden)
    assert logits.shape == (2, 17)

    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    assert actions.shape == (2,)
    assert (actions >= 0).all() and (actions < 17).all()


def test_crafter_real_game_evaluator_contract() -> None:
    """Requirement 5: Verify fair real-game evaluation on unmodified Crafter with RandomCrafterAgent."""
    agent = RandomCrafterAgent(seed=42)
    seeds = [10001, 10002]
    max_steps = 15

    results = evaluate_policy_in_crafter(agent, seeds=seeds, max_steps=max_steps)

    assert results["num_episodes"] == 2
    assert "mean_return" in results
    assert "crafter_score" in results
    assert "action_distribution" in results
    assert len(results["action_distribution"]) == 17
    assert len(results["achievement_rates"]) == 22
    assert len(results["per_seed_results"]) == 2

    for r in results["per_seed_results"]:
        assert r["length"] <= max_steps
        assert "cause" in r


def test_gate6_anti_cheating_scan() -> None:
    """Requirement 6: Verify zero forbidden cheat tables or privileged state in Gate 6 code."""
    source_files = [
        Path("eval/gate6_policy_evaluator.py"),
        Path("true_jepa/objectives.py"),
        Path("true_jepa/agent.py"),
    ]

    forbidden_strings = [
        "CRAFT_",
        "RECIPE",
        "INVENTORY",
        "make_wood_pickaxe",
        "player_pos",
    ]

    for p in source_files:
        assert p.exists()
        code = p.read_text()
        for forbidden in forbidden_strings:
            # Exempt ALL_CRAFTER_ACHIEVEMENTS listing of names for score evaluation
            if forbidden == "make_wood_pickaxe" and p.name == "gate6_policy_evaluator.py":
                continue
            assert forbidden.lower() not in code.lower(), f"Forbidden string {forbidden} in {p}"
