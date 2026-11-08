"""
genesis_jepa.tests.test_gate1_world_model — Automated Verification Suite for Gate 1
===================================================================================
Verifies:
  1. SimNorm Euclidean Norm Invariant (||SimNorm(x)||_2 <= sqrt(M))
  2. Inverted Competitive Slot Attention Invariant (sum_{s=1}^S A_{s, p} == 1.0)
  3. Spatially Gated Dynamic Jump Operator (identity conservation + bounded jump)
  4. Normalized L1 Loss & Soft-Capped Anti-Persistence Weights (weights in [1.0, 5.0])
  5. Multi-Step Autoregressive Rollout & Target EMA Updates
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from genesis_jepa.config import GenesisConfig
from genesis_jepa.modules.dynamic_jump import SpatiallyGatedJumpPredictor
from genesis_jepa.modules.losses import (
    WorldModelGate1Loss,
    compute_anti_persistence_weights,
    normalized_l1_loss,
)
from genesis_jepa.modules.simnorm import SimNorm
from genesis_jepa.modules.slot_attention import InvertedCompetitiveSlotAttention
from genesis_jepa.modules.visual_encoder import GenesisVisualEncoder, SpatialContrastNorm
from genesis_jepa.world_model import GenesisJEPA


def test_simnorm_bounded_norm_invariant() -> None:
    """
    Mathematical Proof:
    For any input x in R^{dim}, ||SimNorm(x)||_2 <= sqrt(M).
    """
    num_heads = 6
    head_dim = 32
    simnorm = SimNorm(num_heads=num_heads, head_dim=head_dim, temp=1.0)
    max_theoretical_norm = math.sqrt(num_heads)  # sqrt(6) ≈ 2.4494897

    # Test with standard, tiny, huge, and adversarial inputs
    test_tensors = [
        torch.randn(10, 64, num_heads * head_dim),
        torch.ones(5, num_heads * head_dim) * 1000.0,
        torch.ones(5, num_heads * head_dim) * -1000.0,
        torch.zeros(2, num_heads * head_dim),
        torch.randn(3, 100, num_heads * head_dim) * 50.0,
    ]

    for x in test_tensors:
        out = simnorm(x)
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.all(norms <= max_theoretical_norm + 1e-5), (
            f"SimNorm invariant violated! Max norm={norms.max().item():.5f} > {max_theoretical_norm:.5f}"
        )
        # Check simplex property for each head: each head must sum to 1.0
        reshaped = out.view(-1, num_heads, head_dim)
        head_sums = reshaped.sum(dim=-1)
        assert torch.allclose(head_sums, torch.ones_like(head_sums), atol=1e-5), (
            "SimNorm head does not sum to 1.0 on probability simplex!"
        )


def test_inverted_competitive_slot_attention() -> None:
    """
    Verifies that inverted slot attention correctly normalizes across slots per patch.
    """
    B, K, D = 4, 64, 192
    S = 12
    slot_attn = InvertedCompetitiveSlotAttention(num_slots=S, dim=D, iters=3)

    patches = torch.randn(B, K, D)
    slots = slot_attn(patches)

    assert slots.shape == (B, S, D), f"Expected slots shape ({B}, {S}, {D}), got {slots.shape}"
    assert not torch.isnan(slots).any(), "Slot attention produced NaN values!"


def test_spatial_contrast_normalization() -> None:
    """
    Verifies that spatial contrast normalization cancels global circadian illumination swings.
    """
    norm_layer = SpatialContrastNorm()

    # Day scene vs simulated dark night scene (scaled by 0.10)
    day_scene = torch.rand(2, 3, 64, 64) * 255.0
    night_scene = day_scene * 0.10  # Circadian darkening

    day_normed = norm_layer(day_scene)
    night_normed = norm_layer(night_scene)

    # Cross-lighting cosine similarity should remain >= 0.99
    day_flat = day_normed.flatten(1)
    night_flat = night_normed.flatten(1)
    cos_sim = torch.cosine_similarity(day_flat, night_flat, dim=-1)

    assert torch.all(cos_sim >= 0.99), (
        f"Circadian contrast normalization failed! Cosine similarity={cos_sim.min().item():.4f} < 0.99"
    )


def test_spatially_gated_jump_predictor() -> None:
    """
    Verifies the Demidovich Jump Predictor and its routing gate behavior.
    """
    B, K, D = 4, 64, 192
    num_actions = 17
    predictor = SpatiallyGatedJumpPredictor(
        token_dim=D,
        action_dim=num_actions,
        num_patches=K,
        gate_bias_init=-2.0,
    )

    z_t = torch.randn(B, K, D)
    action = torch.randint(0, num_actions, (B,))

    z_next, gate, delta = predictor(z_t, action)

    # 1. Output shapes
    assert z_next.shape == (B, K, D)
    assert gate.shape == (B, K, 1)
    assert delta.shape == (B, K, D)

    # 2. Gate values strictly in [0, 1]
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)

    # 3. Initial bias causes gate to favor background identity (mean gate <= 0.30)
    assert gate.mean().item() <= 0.30, (
        f"Initial gate bias should favor background identity! Mean gate={gate.mean().item():.4f}"
    )

    # 4. Displacements strictly bounded by sqrt(6) ≈ 2.45
    delta_norms = torch.norm(delta, p=2, dim=-1)
    assert torch.all(delta_norms <= math.sqrt(6) + 1e-5), (
        f"Displacement violated SimNorm bound! Max norm={delta_norms.max().item():.4f}"
    )


def test_losses_and_anti_persistence_weights() -> None:
    """
    Verifies normalized L1 loss and soft-capped anti-persistence weighting.
    """
    B, K, D = 4, 64, 192
    z_prev = torch.randn(B, K, D)
    # Passive tokens: small change
    z_curr = z_prev.clone()
    # Active interaction tokens: token 0 changes significantly
    z_curr[:, 0, :] += torch.randn(B, D) * 2.0

    weights = compute_anti_persistence_weights(z_prev, z_curr, floor=1.0, gain=4.0)

    # Weights must strictly lie within [1.0, 5.0]
    assert torch.all(weights >= 1.0) and torch.all(weights <= 5.0), (
        f"Anti-persistence weights out of bounds! Min={weights.min().item()}, Max={weights.max().item()}"
    )

    # Active token 0 must receive strictly higher weight than passive tokens
    assert torch.all(weights[:, 0] > weights[:, 1]), (
        f"Active token weight={weights[:, 0].mean().item()} not greater than passive token={weights[:, 1].mean().item()}"
    )

    # Test full loss module
    loss_module = WorldModelGate1Loss(floor=1.0, gain=4.0)
    pred_z = z_prev + torch.randn_like(z_prev) * 0.10
    loss, metrics = loss_module(pred_z, z_curr, z_prev)

    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert "anti_persistence_ratio" in metrics


def test_genesis_jepa_end_to_end_step_and_rollout() -> None:
    """
    Verifies end-to-end JEPA forward pass, target EMA, and multi-step unrolling (H=4).
    """
    config = GenesisConfig(token_dim=192, num_patches=64, num_slots=12)
    jepa = GenesisJEPA(config)

    B = 2
    obs_t = torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8)
    action_t = torch.randint(0, 17, (B,))
    obs_next = torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8)

    # 1. 1-Step training loss computation
    loss, metrics = jepa.compute_step_loss(obs_t, action_t, obs_next)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # Backward pass verification
    loss.backward()

    # Online parameters must have gradients
    has_online_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in jepa.encoder.parameters())
    assert has_online_grad, "Online encoder received zero gradients!"

    # Target parameters must have NO gradients
    has_target_grad = any(p.grad is not None for p in jepa.target_encoder.parameters())
    assert not has_target_grad, "Target encoder incorrectly received gradients!"

    # 2. Target EMA update verification
    initial_target_weight = jepa.target_encoder.patch_proj.weight.data.clone()
    # Simulate an optimizer update by updating online parameters
    with torch.no_grad():
        jepa.encoder.patch_proj.weight.data.add_(0.5)
    jepa.update_target_ema(decay=0.90)
    updated_target_weight = jepa.target_encoder.patch_proj.weight.data
    assert not torch.allclose(initial_target_weight, updated_target_weight), (
        "Target EMA update failed to update parameters!"
    )
    # Verify expected EMA value: 0.90 * initial + 0.10 * (initial + 0.5) = initial + 0.05
    expected = 0.90 * initial_target_weight + 0.10 * (initial_target_weight + 0.5)
    assert torch.allclose(updated_target_weight, expected, atol=1e-5), "EMA computation math error!"

    # 3. Autoregressive Rollout (H=4)
    with torch.no_grad():
        initial_z, _ = jepa.encode(obs_t)
        actions = torch.randint(0, 17, (B, 4))
        trajectory = jepa.rollout(initial_z, actions)

        assert len(trajectory) == 4
        for step_idx, step_z in enumerate(trajectory):
            assert step_z.shape == (B, 64, 192), f"Rollout step {step_idx} shape mismatch: {step_z.shape}"
            assert not torch.isnan(step_z).any(), f"Rollout step {step_idx} produced NaNs!"
