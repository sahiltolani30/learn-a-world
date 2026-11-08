"""Unit and integration test suite for ORBIT Gate 3 representation qualification.

Verifies:
1. Candidate A (from scratch) and Candidate B (frozen DINOv2) shape and contract parity.
2. DINOv2 frozen backbone parameter invariants, gradient isolation, and weight hash immutability.
3. DINOv2 full-frame nearest-neighbor preprocessing with zero crop and corner preservation.
4. Target encoder EMA scalar mathematics and requires_grad=False invariance.
5. Checkpoint save and reload equivalence for DINOv2 and Scratch encoders.
6. Mathematical diagnostics (active dimensions, effective rank, spatial diversity, drift).
7. Predictive utility probe against static persistence and shuffled-action controls.
"""
from __future__ import annotations

import copy
import hashlib
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.crafter_wrapper import CrafterEnv
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.dinov2_encoder import DINOv2SpatialAdapter, DINOv2VisualEncoder
from true_jepa.model import TrueJEPA, LatentFrame


def compute_effective_rank(tokens: torch.Tensor) -> float:
    """Computes Roy-Vetterli singular value effective rank over flattened tokens [M, D]."""
    # tokens shape: [B, N, D] -> [B*N, D]
    M = tokens.shape[0] * tokens.shape[1]
    D = tokens.shape[2]
    flat = tokens.detach().reshape(M, D).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    # Singular values
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    S_sum = S.sum()
    if S_sum <= 1e-12:
        return 1.0
    p = S / S_sum
    # Filter zeros for entropy
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def compute_active_dimensions(tokens: torch.Tensor, threshold: float = 1e-4) -> int:
    """Computes count of dimensions with channel variance above threshold."""
    M = tokens.shape[0] * tokens.shape[1]
    D = tokens.shape[2]
    flat = tokens.detach().reshape(M, D).float()
    var = flat.var(dim=0, unbiased=True)
    return int((var > threshold).sum().item())


def compute_spatial_diversity(tokens: torch.Tensor) -> float:
    """Computes average pairwise cosine similarity between spatial tokens within each frame."""
    B, N, D = tokens.shape
    normed = F.normalize(tokens.detach().float(), p=2, dim=-1)
    # Pairwise cosine: [B, N, N]
    sim_matrix = torch.bmm(normed, normed.transpose(1, 2))
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device).unsqueeze(0).expand(B, N, N)
    off_diag = sim_matrix[mask].reshape(B, N * (N - 1))
    return float(off_diag.mean().item())


# ---------------------------------------------------------------------------
# Test 1: Candidate A Scratch Encoder Contract
# ---------------------------------------------------------------------------

def test_gate3_scratch_encoder_contract():
    cfg = JEPAConfig(patch_size=8, token_dim=192)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")
    obs = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)

    latent = model.encode(obs)
    assert isinstance(latent, LatentFrame)
    assert latent.tokens.shape == (4, 64, 192)
    assert latent.pooled.shape == (4, 192)
    assert latent.tokens.dtype == torch.float32

    # Verify backward gradient flow
    loss = latent.tokens.sum() + latent.pooled.sum()
    loss.backward()
    for name, param in model.encoder.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        assert param.grad.abs().sum() > 0, f"Parameter {name} gradient is zero"


# ---------------------------------------------------------------------------
# Test 2: DINOv2 Integration, Gradient Isolation & Frozen Invariants
# ---------------------------------------------------------------------------

def test_gate3_dinov2_integration_and_invariants():
    """Verify DINOv2 loads actual weights, extracts tokens, isolates gradients, and keeps backbone invariant."""
    cfg = JEPAConfig(token_dim=192)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="dinov2")

    # 1. Verify backbone parameter count and frozen status
    backbone = model.encoder.backbone
    backbone_params = list(backbone.parameters())
    total_backbone_params = sum(p.numel() for p in backbone_params)
    # Measured parameter count of Meta AI dinov2_vits14 architecture is 22,056,576
    assert total_backbone_params == 22056576, f"Expected 22,056,576 params, got {total_backbone_params}"
    for name, p in backbone.named_parameters():
        assert not p.requires_grad, f"Backbone param {name} has requires_grad=True!"

    # 2. Record initial SHA-256 hash of backbone weights
    initial_hash = model.encoder.backbone_sha256()
    assert len(initial_hash) == 64

    # 3. Feed a real Crafter observation batch
    env = CrafterEnv(seed=3000, max_steps=10)
    obs_list = []
    obs, _ = env.reset(seed=3000)
    obs_list.append(obs)
    for _ in range(3):
        obs, _, _, _, _ = env.step(0)
        obs_list.append(obs)
    env.close()

    obs_batch = torch.from_numpy(np.stack(obs_list, axis=0)) # [4, 64, 64, 3] uint8
    latent = model.encode(obs_batch)

    # 4. Assert token shapes match Candidate A contract exactly
    assert latent.tokens.shape == (4, 64, 192)
    assert latent.pooled.shape == (4, 192)

    # 5. Run forward pass through predictor and compute loss
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    hidden = model.initial_belief(batch=4, device="cpu")
    predicted = model.predict(latent, actions, hidden)
    assert predicted.tokens.shape == (4, 64, 192)

    loss = predicted.tokens.sum() + predicted.pooled.sum()
    loss.backward()

    # 6. Verify gradient isolation: adapter & predictor get gradients, backbone gets NONE
    assert model.encoder.adapter.proj.weight.grad is not None
    assert model.encoder.adapter.proj.weight.grad.abs().sum() > 0
    assert model.predictor.belief_projection.weight.grad is not None

    for name, p in backbone.named_parameters():
        assert p.grad is None, f"Backbone parameter {name} leaked gradient!"

    # 7. Optimizer step on adapter and verify backbone hash is unchanged
    opt = torch.optim.AdamW(model.encoder.adapter.parameters(), lr=1e-3)
    opt.step()

    post_hash = model.encoder.backbone_sha256()
    assert post_hash == initial_hash, "Backbone SHA-256 hash changed after training step!"


# ---------------------------------------------------------------------------
# Test 3: Preprocessing Full FOV & Corner Preservation
# ---------------------------------------------------------------------------

def test_gate3_preprocessing_no_crop_boundary():
    """Verify that 64x64 -> 224x224 resize preserves full FOV and all 4 corners without cropping."""
    cfg = JEPAConfig(token_dim=192)
    encoder = DINOv2VisualEncoder(cfg)

    # Create synthetic image [1, 64, 64, 3] with unique corner values
    img = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    img[0, 0, 0] = torch.tensor([255, 0, 0])     # Top-Left: Red
    img[0, 0, 63] = torch.tensor([0, 255, 0])    # Top-Right: Green
    img[0, 63, 0] = torch.tensor([0, 0, 255])    # Bottom-Left: Blue
    img[0, 63, 63] = torch.tensor([255, 255, 0]) # Bottom-Right: Yellow

    preprocessed = encoder.preprocess(img)
    assert preprocessed.shape == (1, 3, 224, 224)

    # In nearest-neighbor interpolation, 64 -> 224 expands each pixel to 3.5 pixels.
    # Specifically, corner pixel (0,0) must cover pixel (0,0) in 224x224.
    # In normalized space, check that the corners reflect the distinct inputs
    tl_color = preprocessed[0, :, 0, 0]
    tr_color = preprocessed[0, :, 0, 223]
    bl_color = preprocessed[0, :, 223, 0]
    br_color = preprocessed[0, :, 223, 223]

    # All 4 corners must be distinct (proves no corner was cropped or padded away)
    assert not torch.allclose(tl_color, tr_color)
    assert not torch.allclose(tl_color, bl_color)
    assert not torch.allclose(tl_color, br_color)
    assert not torch.allclose(tr_color, br_color)


# ---------------------------------------------------------------------------
# Test 4: Target Encoder EMA Arithmetic
# ---------------------------------------------------------------------------

def test_gate3_target_encoder_ema_behavior():
    """Verify target encoder EMA updates accurately via tau formula and keeps requires_grad=False."""
    cfg = JEPAConfig(token_dim=192, ema_decay_start=0.996, ema_decay_end=0.999, ema_ramp_updates=1000)
    for enc_type in ["scratch", "dinov2"]:
        model = TrueJEPA(cfg, num_actions=17, encoder_type=enc_type)
        assert not any(p.requires_grad for p in model.target_encoder.parameters())

        if enc_type == "scratch":
            online_param = model.encoder.proj.weight
            target_param = model.target_encoder.proj.weight
        else:
            online_param = model.encoder.adapter.proj.weight
            target_param = model.target_encoder.adapter.proj.weight

        # Perturb online parameter
        with torch.no_grad():
            old_target = target_param.clone()
            online_param.add_(1.0)

        # Update at step 0 (tau = 0.996)
        model.update_target(update=0)
        expected = old_target * 0.996 + online_param * 0.004
        assert torch.allclose(target_param, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Checkpoint Save and Reload Roundtrip
# ---------------------------------------------------------------------------

def test_gate3_checkpoint_save_reload_roundtrip():
    """Verify that TrueJEPA with DINOv2 serializes and restores cleanly without corruption."""
    cfg = JEPAConfig(token_dim=192)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="dinov2")

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        model.save_checkpoint(tmp.name, update=42, optimizer={"test_opt": 1})
        loaded_model, meta = TrueJEPA.load_checkpoint(tmp.name)

        assert meta["update"] == 42
        assert loaded_model.encoder_type == "dinov2"
        assert meta["optimizer"] == {"test_opt": 1}

        # Verify weights match exactly
        obs = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
        with torch.no_grad():
            out1 = model.encode(obs)
            out2 = loaded_model.encode(obs)
            assert torch.allclose(out1.tokens, out2.tokens, atol=1e-5)
            assert torch.allclose(out1.pooled, out2.pooled, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: Mathematical Diagnostics on Controlled Synthetic Tensors
# ---------------------------------------------------------------------------

def test_gate3_mathematical_diagnostics_synthetic():
    """Verify that r_eff, D_active, and S_spatial detect collapse vs diversity accurately."""
    # 1. Complete dimensional and spatial collapse: all tokens are identical constant vector
    B, N, D = 4, 64, 192
    collapsed_tokens = torch.ones(B, N, D) * 2.5
    r_eff_c = compute_effective_rank(collapsed_tokens)
    d_act_c = compute_active_dimensions(collapsed_tokens)
    s_div_c = compute_spatial_diversity(collapsed_tokens)

    assert r_eff_c == pytest.approx(1.0, abs=0.1)
    assert d_act_c == 0
    assert s_div_c == pytest.approx(1.0, abs=1e-4)

    # 2. Rich orthogonal / standard normal tokens
    diverse_tokens = torch.randn(B, N, D)
    r_eff_d = compute_effective_rank(diverse_tokens)
    d_act_d = compute_active_dimensions(diverse_tokens)
    s_div_d = compute_spatial_diversity(diverse_tokens)

    assert r_eff_d > 10.0, f"Expected r_eff > 10, got {r_eff_d}"
    assert d_act_d > 120, f"Expected d_active > 120, got {d_act_d}"
    assert s_div_d < 0.2, f"Expected s_diversity < 0.2 for random, got {s_div_d}"


# ---------------------------------------------------------------------------
# Test 7: Predictive Utility Probe
# ---------------------------------------------------------------------------

def test_gate3_predictive_utility_probe():
    """Verify computation of relative prediction error vs persistence and shuffled actions."""
    cfg = JEPAConfig(token_dim=192)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="scratch")

    obs_t = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
    obs_next = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
    actions = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    hidden = model.initial_belief(batch=4, device="cpu")

    with torch.no_grad():
        current_frame = model.encode(obs_t)
        target_frame = model.target_encode(obs_next)
        pred_frame = model.predict(current_frame, actions, hidden)

        # Normalize tokens for fair distance
        pred_norm = F.normalize(pred_frame.tokens, p=2, dim=-1)
        target_norm = F.normalize(target_frame.tokens, p=2, dim=-1)
        current_norm = F.normalize(current_frame.tokens, p=2, dim=-1)

        err_model = F.mse_loss(pred_norm, target_norm).item()
        err_persist = F.mse_loss(current_norm, target_norm).item()
        rel_err = err_model / max(err_persist, 1e-8)

        # Verify relative error is a positive float
        assert isinstance(rel_err, float)
        assert rel_err > 0.0

        # Shuffled actions
        shuffled_actions = torch.tensor([2, 3, 4, 1], dtype=torch.long)
        pred_shuffled = model.predict(current_frame, shuffled_actions, hidden)
        pred_shuf_norm = F.normalize(pred_shuffled.tokens, p=2, dim=-1)
        err_shuffled = F.mse_loss(pred_shuf_norm, target_norm).item()
        assert isinstance(err_shuffled, float)
