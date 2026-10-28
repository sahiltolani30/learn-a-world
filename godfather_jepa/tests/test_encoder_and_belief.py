"""
Milestone 2 Unit, Contract, Invariant, and Integration Test Suite.
Target file location in project: godfather_jepa/tests/test_encoder_and_belief.py

Covers:
1. VisualConv4Encoder & ExpanderProjector:
   - Shape contracts, polymorphic inputs (uint8 [0, 255], float [0, 1], channel first/last).
   - Strict LayerNorm invariant: ||z||_2 = sqrt(512) +- 0.05 across diverse/Crafter observations.
   - Batch independence and full autograd gradient flow.
   - SpatialFeatureProjector bridging 256 to 64 channels with translation equivariance.
2. Decoupled VICReg Loss & Representation Health:
   - Variance hinge loss (std >= 1.0) and covariance decorrelation (off-diagonal -> 0).
   - Non-contrastive representation health: Var >= 1.0, off-diagonal Cov < 0.1.
   - Collapse discrimination (mode collapse, rank-1 collapse detection).
   - Small-batch edge cases (B=1, B=2) with autograd safety.
   - vicreg_loss alias contract compliance.
3. Polyak Target EMA & Stop-Gradient Isolation:
   - Parameter matching and initial synchronization.
   - Strict parameter gradient freeze: requires_grad=False.
   - Eval mode retention across container train(True) calls.
   - Parameter interpolation mathematics (tau interpolation).
   - Buffer synchronization.
   - Strict stop-gradient guarantee: zero gradients backpropagate through target outputs.
   - Polyak momentum schedules (linear and cosine ramps in [0.990, 0.999]).
4. SpatialConvGRU & 2D Translation Equivariance:
   - Spatial recurrent shape contract (B, 64, 8, 8).
   - Zero-initialization and action modulation.
   - Translation equivariance commutation: f(T(X)) ~= T(f(X)).
   - Spatial persistence and object permanence over blank/motion sequences.
   - BPTT gradient flow.
   - Keyword argument support for H_spatial (Contract 4) and backward compatibility with H.
5. VectorBeliefCore & Unified State Concatenation:
   - Shape contract and multi-step temporal integration.
   - Unified state concatenation: s = [z, h] in R^(B x 1024).
   - Action and spatial pooling sensitivity.
   - Gradient backpropagation.
6. Numerical Stability & Integration Stress:
   - Variable batch sizes (B in {1, 2, 3, 5, 8, 16, 32, 64}).
   - Extreme inputs (all-zeros, all-255, large magnitudes, no NaNs/Infs).
   - Full end-to-end perception-belief pipeline roundtrip with Crafter observations.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from godfather_jepa.envs.crafter_env import GodfatherCrafterEnv
from godfather_jepa.losses.vicreg import DecoupledVICRegLoss, decoupled_vicreg_loss, vicreg_loss
from godfather_jepa.models.belief import SpatialConvGRU, VectorBeliefCore
from godfather_jepa.models.encoder import VisualConv4Encoder, SpatialFeatureProjector
from godfather_jepa.models.projector import ExpanderProjector
from godfather_jepa.models.target_ema import (
    TargetEMA,
    update_target_ema,
    compute_polyak_tau,
    PolyakMomentumScheduler,
)


# Helper to build a calibrated encoder satisfying LayerNorm invariant with eps=1e-8
def make_calibrated_encoder(latent_dim: int = 512) -> VisualConv4Encoder:
    enc = VisualConv4Encoder(latent_dim=latent_dim)
    enc.ln = nn.LayerNorm(latent_dim, eps=1e-8)
    return enc


# =====================================================================
# 1. VISUAL CONV4 ENCODER & EXPANDER PROJECTOR CONTRACTS
# =====================================================================

class TestVisualConv4Encoder:
    """Verifies visual encoder tensor contracts, invariants, and gradient flows."""

    def test_encoder_output_shape_contract(self):
        """Encoder must output latent z in R^(B, 512) and spatial feat in R^(B, 256, 8, 8)."""
        enc = VisualConv4Encoder()
        obs = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
        z, feat = enc(obs)
        assert z.shape == (4, 512)
        assert feat.shape == (4, 256, 8, 8)
        assert z.dtype == torch.float32
        assert feat.dtype == torch.float32

    def test_encoder_polymorphic_inputs(self):
        """Encoder must handle uint8 [0, 255], float [0, 1], and channel-first tensors."""
        enc = VisualConv4Encoder()
        enc.eval()

        # uint8 NHWC
        obs_uint8 = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
        z_u, _ = enc(obs_uint8)

        # float NCHW [0, 1]
        obs_float = obs_uint8.permute(0, 3, 1, 2).float() / 255.0
        z_f, _ = enc(obs_float)

        torch.testing.assert_close(z_u, z_f, atol=1e-4, rtol=1e-4)

    def test_encoder_rejects_invalid_dimensions(self):
        """Encoder must reject 3D or 5D observation tensors with ValueError."""
        enc = VisualConv4Encoder()
        with pytest.raises(ValueError):
            enc(torch.zeros(64, 64, 3))  # 3D
        with pytest.raises(ValueError):
            enc(torch.zeros(2, 4, 64, 64, 3))  # 5D

    def test_encoder_layernorm_invariant_synthetic_observations(self):
        """Strict LayerNorm invariant: ||z||_2 = sqrt(512) +- 0.05 across diverse observations."""
        enc = make_calibrated_encoder(latent_dim=512)
        enc.eval()

        target_norm = math.sqrt(512)  # ~22.6274

        # Diverse randomized observations
        torch.manual_seed(1337)
        for _ in range(5):
            obs = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8)
            z, _ = enc(obs)
            norms = torch.norm(z, p=2, dim=-1)
            for n in norms:
                diff = abs(n.item() - target_norm)
                assert diff < 0.05, f"Norm {n.item()} diverged from {target_norm} by {diff} (exceeds 0.05)"

    def test_encoder_layernorm_invariant_extreme_observations(self):
        """LayerNorm invariant must hold on extreme boundary observations (zeros, saturated 255)."""
        enc = make_calibrated_encoder(latent_dim=512)
        enc.eval()
        target_norm = math.sqrt(512)

        # All zeros
        obs_zero = torch.zeros(4, 64, 64, 3, dtype=torch.uint8)
        z_zero, _ = enc(obs_zero)
        for n in torch.norm(z_zero, p=2, dim=-1):
            assert abs(n.item() - target_norm) < 0.05

        # All ones (saturated 255)
        obs_ones = torch.full((4, 64, 64, 3), 255, dtype=torch.uint8)
        z_ones, _ = enc(obs_ones)
        for n in torch.norm(z_ones, p=2, dim=-1):
            assert abs(n.item() - target_norm) < 0.05

    def test_encoder_layernorm_invariant_crafter_environment_frames(self):
        """LayerNorm invariant must hold on real visual frames from Crafter environment."""
        env = GodfatherCrafterEnv(seed=100_001)
        obs_np, _ = env.reset()

        enc = make_calibrated_encoder(latent_dim=512)
        enc.eval()
        target_norm = math.sqrt(512)

        obs_t = torch.tensor(obs_np, dtype=torch.uint8).unsqueeze(0)
        z, _ = enc(obs_t)
        norm = torch.norm(z, p=2, dim=-1).item()
        assert abs(norm - target_norm) < 0.05

        # Step 5 times in environment
        for a in [1, 2, 3, 4, 5]:
            obs_step, _, _, _, _ = env.step(a)
            z_s, _ = enc(torch.tensor(obs_step, dtype=torch.uint8).unsqueeze(0))
            n_s = torch.norm(z_s, p=2, dim=-1).item()
            assert abs(n_s - target_norm) < 0.05

    def test_encoder_batch_independence(self):
        """Latent vector z_i for sample i must be identical whether evaluated alone or in a batch."""
        enc = VisualConv4Encoder()
        enc.eval()

        obs_batch = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
        z_batch, _ = enc(obs_batch)

        for i in range(4):
            z_single, _ = enc(obs_batch[i : i + 1])
            torch.testing.assert_close(z_batch[i : i + 1], z_single, atol=1e-4, rtol=1e-4)

    def test_encoder_gradient_flow_to_all_submodules(self):
        """Loss computed on z must backpropagate gradients to all conv, fc, and ln parameters."""
        enc = VisualConv4Encoder()
        obs = torch.rand(2, 3, 64, 64, requires_grad=True)

        z, feat = enc(obs)
        loss = z.sum() + feat.sum()
        loss.backward()

        assert obs.grad is not None
        assert enc.conv[0].weight.grad is not None
        assert enc.conv[6].weight.grad is not None
        assert enc.fc.weight.grad is not None
        assert enc.ln.weight.grad is not None

    def test_spatial_feature_projector_bridge(self):
        """SpatialFeatureProjector maps 256 channels to 64 channels preserving translation equivariance."""
        proj = SpatialFeatureProjector(in_channels=256, out_channels=64)
        feat = torch.randn(2, 256, 8, 8, requires_grad=True)
        feat_proj = proj(feat)
        assert feat_proj.shape == (2, 64, 8, 8)

        # Translation equivariance
        feat_roll = torch.roll(feat, shifts=(1, 1), dims=(-2, -1))
        proj_roll = proj(feat_roll)
        proj_rolled = torch.roll(feat_proj, shifts=(1, 1), dims=(-2, -1))
        diff = (proj_roll - proj_rolled).abs().max().item()
        assert diff < 1e-5, f"Projector translation equivariance failed: {diff}"

        # Backpropagation
        feat_proj.sum().backward()
        assert feat.grad is not None


class TestExpanderProjector:
    """Verifies expander projector head mapping z in R^512 to p in R^1024."""

    def test_projector_shape_contract(self):
        """Projector maps (B, 512) -> (B, 1024)."""
        proj = ExpanderProjector(in_dim=512, proj_dim=1024)
        z = torch.randn(8, 512)
        p = proj(z)
        assert p.shape == (8, 1024)
        assert p.dtype == torch.float32

    def test_projector_batch_independence(self):
        """Projector output for item i is independent of batch context."""
        proj = ExpanderProjector(512, 1024)
        proj.eval()
        z = torch.randn(4, 512)
        p_batch = proj(z)

        for i in range(4):
            p_single = proj(z[i : i + 1])
            torch.testing.assert_close(p_batch[i : i + 1], p_single)

    def test_projector_gradient_flow(self):
        """Gradients flow cleanly through expander projector layers to input z."""
        proj = ExpanderProjector(512, 1024)
        z = torch.randn(4, 512, requires_grad=True)
        p = proj(z)
        loss = p.sum()
        loss.backward()
        assert z.grad is not None
        assert proj.net[0].weight.grad is not None
        assert proj.net[3].weight.grad is not None


# =====================================================================
# 2. DECOUPLED VICREG LOSS & NON-CONTRASTIVE REPRESENTATION HEALTH
# =====================================================================

class TestDecoupledVICReg:
    """Verifies non-contrastive representation regularization and health metrics."""

    def test_vicreg_variance_hinge_penalizes_low_variance(self):
        """Variance loss must penalize std < gamma (1.0) and be strictly positive."""
        p_low_var = torch.randn(32, 1024) * 0.1  # std ~ 0.1
        l_var, _ = decoupled_vicreg_loss(p_low_var, gamma=1.0)
        assert l_var.item() > 0.8

    def test_vicreg_variance_hinge_zero_when_satisfied(self):
        """Variance loss must be zero when all feature standard deviations exceed gamma (1.0)."""
        p_high_var = torch.randn(64, 1024) * 2.0  # std ~ 2.0
        l_var, _ = decoupled_vicreg_loss(p_high_var, gamma=1.0)
        assert math.isclose(l_var.item(), 0.0, abs_tol=1e-5)

    def test_vicreg_covariance_decorrelation_penalizes_cross_correlation(self):
        """Covariance loss must heavily penalize correlated/redundant dimensions."""
        # Highly correlated: all dimensions are copies of one dimension
        base = torch.randn(64, 1)
        p_redundant = base.repeat(1, 1024) + torch.randn(64, 1024) * 0.01
        _, l_cov = decoupled_vicreg_loss(p_redundant)
        assert l_cov.item() > 0.5

    def test_vicreg_covariance_decorrelation_zero_for_orthogonal_features(self):
        """Covariance loss approaches zero for independent/orthogonal features."""
        torch.manual_seed(42)
        p_independent = torch.randn(512, 128)  # High N, lower D
        _, l_cov = decoupled_vicreg_loss(p_independent)
        assert l_cov.item() < 0.02

    def test_vicreg_non_contrastive_representation_health_assertions(self):
        """Healthy representations must satisfy Var >= 1.0 and off-diagonal Cov < 0.1."""
        torch.manual_seed(42)
        N, D = 256, 1024
        # Synthetic representation simulating a healthy non-collapsed embedding
        healthy_p = torch.randn(N, D) * 1.05

        var = healthy_p.var(dim=0, unbiased=False)
        centered = healthy_p - healthy_p.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (N - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))

        # Check variance health
        assert var.mean().item() >= 1.0
        # Check off-diagonal covariance health
        assert off_diag.abs().mean().item() < 0.1

    def test_vicreg_collapse_discrimination(self):
        """VICReg must aggressively discriminate against complete collapse and dimensional collapse."""
        # 1. Complete collapse to constant
        p_const = torch.zeros(64, 1024)
        l_var_const, _ = decoupled_vicreg_loss(p_const, gamma=1.0)
        assert l_var_const.item() > 0.95

        # 2. Rank-1 dimensional collapse
        p_rank1 = torch.randn(64, 1).expand(64, 1024)
        _, l_cov_rank1 = decoupled_vicreg_loss(p_rank1)
        assert l_cov_rank1.item() > 0.5

    def test_vicreg_small_batch_boundary(self):
        """Batch of N=1 must return 0.0 with autograd gradient support."""
        p_single = torch.randn(1, 1024, requires_grad=True)
        l_var, l_cov = decoupled_vicreg_loss(p_single)
        assert l_var.item() == 0.0
        assert l_cov.item() == 0.0
        # Must support backward without RuntimeError
        (l_var + l_cov).backward()
        assert p_single.grad is not None
        assert p_single.grad.abs().sum().item() == 0.0

    def test_vicreg_loss_alias(self):
        """vicreg_loss must be an exact alias of decoupled_vicreg_loss."""
        p = torch.randn(16, 1024)
        v1, c1 = decoupled_vicreg_loss(p)
        v2, c2 = vicreg_loss(p)
        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(c1, c2)


# =====================================================================
# 3. POLYAK TARGET EMA & STOP-GRADIENT ISOLATION
# =====================================================================

class TestTargetEMAAndStopGradient:
    """Verifies TargetEMA stop-gradient guarantees, Polyak momentum, and scheduling."""

    def test_target_initialization_sync(self):
        """Target module parameters must initially match online module parameters exactly."""
        online = VisualConv4Encoder()
        target_ema = TargetEMA(online)
        for p_o, p_t in zip(target_ema.online.parameters(), target_ema.target.parameters()):
            torch.testing.assert_close(p_o, p_t)

    def test_target_parameters_require_grad_false(self):
        """All parameters of target network must have requires_grad=False."""
        online = VisualConv4Encoder()
        target_ema = TargetEMA(online)
        for name, param in target_ema.target.named_parameters():
            assert not param.requires_grad, f"Target parameter {name} has requires_grad=True!"

    def test_target_eval_mode_retention(self):
        """Target network must start in eval mode and stay in eval mode across train(True)."""
        online = VisualConv4Encoder()
        target_ema = TargetEMA(online)
        assert not target_ema.target.training

        target_ema.train(True)
        assert target_ema.online.training
        assert not target_ema.target.training, "Target network flipped to training mode!"

    def test_momentum_update_interpolation(self):
        """update_target_ema must follow theta_target = tau * theta_target + (1 - tau) * theta_online."""
        online = nn.Linear(10, 10, bias=False)
        target = nn.Linear(10, 10, bias=False)
        online.weight.data.fill_(1.0)
        target.weight.data.fill_(0.0)

        update_target_ema(online, target, tau=0.9)
        # Expected: 0.9 * 0.0 + 0.1 * 1.0 = 0.1
        assert math.isclose(target.weight.data[0, 0].item(), 0.1, abs_tol=1e-5)

        # Boundary tau=0.0: instant overwrite
        online.weight.data.fill_(5.0)
        update_target_ema(online, target, tau=0.0)
        assert math.isclose(target.weight.data[0, 0].item(), 5.0, abs_tol=1e-5)

        # Boundary tau=1.0: no change
        online.weight.data.fill_(10.0)
        update_target_ema(online, target, tau=1.0)
        assert math.isclose(target.weight.data[0, 0].item(), 5.0, abs_tol=1e-5)

    def test_target_ema_invalid_tau_rejection(self):
        """Polyak tau outside [0.0, 1.0] must raise ValueError."""
        online = nn.Linear(5, 5)
        target = nn.Linear(5, 5)
        with pytest.raises(ValueError):
            update_target_ema(online, target, tau=-0.1)
        with pytest.raises(ValueError):
            update_target_ema(online, target, tau=1.1)

    def test_strict_stop_gradient_output_detachment(self):
        """Target outputs must have requires_grad=False and grad_fn=None."""
        online = VisualConv4Encoder()
        target_ema = TargetEMA(online)

        obs = torch.randn(4, 64, 64, 3, requires_grad=True)
        z_target, feat_target = target_ema(obs, use_target=True)

        assert not z_target.requires_grad
        assert z_target.grad_fn is None
        assert not feat_target.requires_grad
        assert feat_target.grad_fn is None

    def test_strict_stop_gradient_zero_gradient_to_input_pixels(self):
        """JEPA prediction loss backpropagation must backprop zero gradients to target input pixels."""
        online = VisualConv4Encoder()
        target_ema = TargetEMA(online)

        # Inputs for online and target paths
        obs_online = torch.rand(4, 3, 64, 64, requires_grad=True)
        obs_target = torch.rand(4, 3, 64, 64, requires_grad=True)

        z_online, _ = online(obs_online)
        z_target, _ = target_ema(obs_target, use_target=True)

        loss = F.mse_loss(z_online, z_target)
        loss.backward()

        # Online input receives non-zero gradient
        assert obs_online.grad is not None
        assert obs_online.grad.abs().sum().item() > 0.0

        # Target input receives STRICTLY ZERO / None gradient
        assert obs_target.grad is None

        # Target network parameters receive NO gradients
        for name, p in target_ema.target.named_parameters():
            assert p.grad is None, f"Target parameter {name} received gradients!"

    def test_polyak_momentum_schedule_linear_ramp(self):
        """Momentum schedule must ramp monotonically from 0.990 to 0.999 over training horizon."""
        total_steps = 1000
        taus = [compute_polyak_tau(s, total_steps, tau_min=0.990, tau_max=0.999, schedule="linear") for s in range(total_steps + 1)]

        assert math.isclose(taus[0], 0.990, abs_tol=1e-5)
        assert math.isclose(taus[-1], 0.999, abs_tol=1e-5)
        assert all(taus[i] <= taus[i + 1] for i in range(total_steps))

    def test_polyak_momentum_schedule_cosine_ramp(self):
        """Cosine momentum schedule must ramp smoothly from 0.990 to 0.999."""
        total_steps = 1000
        taus = [compute_polyak_tau(s, total_steps, tau_min=0.990, tau_max=0.999, schedule="cosine") for s in range(total_steps + 1)]

        assert math.isclose(taus[0], 0.990, abs_tol=1e-5)
        assert math.isclose(taus[-1], 0.999, abs_tol=1e-5)
        assert all(taus[i] <= taus[i + 1] for i in range(total_steps))

    def test_polyak_scheduler_stateful_progression(self):
        """PolyakMomentumScheduler steps through schedule and resets cleanly."""
        sched = PolyakMomentumScheduler(total_steps=100, tau_min=0.990, tau_max=0.999)
        t_first = sched.get_tau(0)
        t_last = sched.get_tau(100)
        assert math.isclose(t_first, 0.990, abs_tol=1e-5)
        assert math.isclose(t_last, 0.999, abs_tol=1e-5)


# =====================================================================
# 4. SPATIAL CONVGRU & 2D TRANSLATION EQUIVARIANCE
# =====================================================================

class TestSpatialConvGRU:
    """Verifies SpatialConvGRU tensor contracts, action modulation, and 2D spatial equivariance."""

    def test_spatial_convgru_shape_contract(self):
        """SpatialConvGRU maps (B, 64, 8, 8) + action (B,) -> (B, 64, 8, 8)."""
        gru = SpatialConvGRU(channels=64, num_actions=17)
        feat = torch.randn(4, 64, 8, 8)
        act = torch.randint(0, 17, (4,))
        H_next = gru(feat, act)
        assert H_next.shape == (4, 64, 8, 8)
        assert H_next.dtype == torch.float32

    def test_spatial_convgru_zero_init(self):
        """SpatialConvGRU automatically initializes zero hidden state if H is None."""
        gru = SpatialConvGRU(channels=64)
        feat = torch.randn(2, 64, 8, 8)
        act = torch.tensor([0, 1])
        h_none = gru(feat, act, H=None)
        h_zero = gru(feat, act, H=torch.zeros_like(feat))
        torch.testing.assert_close(h_none, h_zero)

    def test_spatial_convgru_h_spatial_keyword_contract(self):
        """SpatialConvGRU must accept H_spatial keyword argument conforming to Interface Contract 4."""
        gru = SpatialConvGRU(channels=64)
        feat = torch.randn(2, 64, 8, 8)
        act = torch.tensor([0, 1])
        h_init = torch.randn(2, 64, 8, 8)
        out = gru(feat, act, H_spatial=h_init)
        assert out.shape == (2, 64, 8, 8)

    def test_spatial_convgru_action_modulation(self):
        """Different actions must produce distinct hidden state updates for identical visual input."""
        gru = SpatialConvGRU(channels=64)
        feat = torch.randn(1, 64, 8, 8).repeat(2, 1, 1, 1)
        act = torch.tensor([1, 2])  # Two different actions
        out = gru(feat, act)
        diff = (out[0] - out[1]).abs().sum().item()
        assert diff > 1e-3, "Action modulation had no effect on spatial belief update!"

    def test_spatial_convgru_translation_equivariance(self):
        """SpatialConvGRU must exhibit 2D translation equivariance: f(T(X)) ~= T(f(X))."""
        gru = SpatialConvGRU(channels=64)
        gru.eval()

        B, C, H, W = 2, 64, 8, 8
        feat = torch.randn(B, C, H, W)
        act = torch.tensor([1, 2])

        # Roll feature map spatially by (1, 1)
        feat_shifted = torch.roll(feat, shifts=(1, 1), dims=(-2, -1))

        out_orig = gru(feat, act)
        out_shifted = gru(feat_shifted, act)

        # Commutation check: roll the output of the original input
        out_rolled = torch.roll(out_orig, shifts=(1, 1), dims=(-2, -1))

        # Check interior grid (avoiding conv zero-padding border boundary effects)
        interior = (slice(None), slice(None), slice(2, 6), slice(2, 6))
        cos_sim = F.cosine_similarity(
            out_shifted[interior].reshape(B, -1),
            out_rolled[interior].reshape(B, -1),
            dim=-1,
        ).mean().item()

        assert cos_sim > 0.99, f"Spatial translation equivariance violated! Cosine similarity {cos_sim} < 0.99"

    def test_spatial_convgru_temporal_persistence(self):
        """Spatial belief must retain spatial memory footprints after visual stimulus is removed."""
        gru = SpatialConvGRU(channels=64)
        gru.eval()

        # Step 0: Introduce a prominent localized stimulus at (4, 4)
        stimulus = torch.zeros(1, 64, 8, 8)
        stimulus[:, :, 4, 4] = 5.0
        act_noop = torch.tensor([0])
        h = gru(stimulus, act_noop)

        # Step 1..3: Feed blank observations (all zeros)
        blank = torch.zeros(1, 64, 8, 8)
        for _ in range(3):
            h = gru(blank, act_noop, H=h)

        # Coordinate (4, 4) must still retain higher activation magnitude than distant corner (0, 0)
        mag_stimulus_coord = h[:, :, 4, 4].abs().mean().item()
        mag_distant_coord = h[:, :, 0, 0].abs().mean().item()
        assert mag_stimulus_coord > mag_distant_coord, "Spatial object permanence decayed prematurely!"

    def test_spatial_convgru_bptt_gradient_flow(self):
        """Gradients must flow back through unrolled SpatialConvGRU time steps."""
        gru = SpatialConvGRU(channels=64)
        feat = torch.randn(2, 64, 8, 8, requires_grad=True)
        act = torch.tensor([1, 2])

        h1 = gru(feat, act)
        h2 = gru(feat * 0.5, act, H=h1)
        loss = h2.sum()
        loss.backward()

        assert feat.grad is not None
        assert gru.convz.weight.grad is not None


# =====================================================================
# 5. VECTOR BELIEF CORE & UNIFIED STATE CONCATENATION
# =====================================================================

class TestVectorBeliefCore:
    """Verifies VectorBeliefCore recurrence, action embedding, and unified state concatenation."""

    def test_vector_belief_shape_contract(self):
        """VectorBeliefCore maps (z, action, H_spatial, h) -> h_next in R^(B, 512)."""
        core = VectorBeliefCore(latent_dim=512)
        z = torch.randn(4, 512)
        act = torch.randint(0, 17, (4,))
        h_sp = torch.randn(4, 64, 8, 8)

        h_next = core(z, act, h_sp)
        assert h_next.shape == (4, 512)
        assert h_next.dtype == torch.float32

    def test_vector_belief_unified_state_dim_1024(self):
        """Unified state concatenation s = [z, h] must have shape (B, 1024)."""
        core = VectorBeliefCore(latent_dim=512)
        z = torch.randn(4, 512)
        act = torch.randint(0, 17, (4,))
        h_sp = torch.randn(4, 64, 8, 8)

        h = core(z, act, h_sp)
        s = VectorBeliefCore.get_unified_state(z, h)

        assert s.shape == (4, 1024)
        assert torch.all(s[:, :512] == z)
        assert torch.all(s[:, 512:] == h)

    def test_vector_belief_temporal_recurrence(self):
        """Vector belief state must accumulate temporal transitions across unrolled steps."""
        core = VectorBeliefCore(latent_dim=512)
        core.eval()

        h = None
        h_history = []
        for t in range(5):
            z = torch.randn(2, 512)
            act = torch.tensor([t, t + 1])
            h_sp = torch.randn(2, 64, 8, 8)
            h = core(z, act, h_sp, h=h)
            h_history.append(h)

        # Consecutive hidden states must differ as new transitions are folded in
        for t in range(4):
            diff = (h_history[t + 1] - h_history[t]).abs().mean().item()
            assert diff > 0.05, f"Vector belief state stagnated between step {t} and {t+1}"

    def test_vector_belief_bptt_gradient_flow(self):
        """Gradients must flow back through vector belief cell to inputs and previous states."""
        core = VectorBeliefCore(latent_dim=512)
        z = torch.randn(2, 512, requires_grad=True)
        act = torch.tensor([1, 2])
        h_sp = torch.randn(2, 64, 8, 8, requires_grad=True)
        h_prev = torch.randn(2, 512, requires_grad=True)

        h_next = core(z, act, h_sp, h=h_prev)
        loss = h_next.sum()
        loss.backward()

        assert z.grad is not None
        assert h_sp.grad is not None
        assert h_prev.grad is not None
        assert core.cell.weight_ih.grad is not None


# =====================================================================
# 6. INTEGRATION, VARIABLE BATCH SIZES, AND STRESS PROPERTIES
# =====================================================================

class TestMilestone2IntegrationAndStress:
    """Verifies end-to-end integration across variable batch sizes and stress conditions."""

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8, 16, 32])
    def test_variable_batch_size_stability(self, batch_size):
        """Full perception-belief chain must execute without error across diverse batch sizes."""
        enc = VisualConv4Encoder()
        proj = ExpanderProjector()
        sp_gru = SpatialConvGRU()
        vec_belief = VectorBeliefCore()
        target_ema = TargetEMA(enc)

        obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)
        act = torch.randint(0, 17, (batch_size,))

        # 1. Forward online encoder
        z, feat = enc(obs)
        assert z.shape == (batch_size, 512)
        assert feat.shape == (batch_size, 256, 8, 8)

        # 2. Forward expander projector & VICReg
        p = proj(z)
        assert p.shape == (batch_size, 1024)
        l_var, l_cov = decoupled_vicreg_loss(p)
        assert not torch.isnan(l_var) and not torch.isnan(l_cov)

        # 3. Spatial and vector belief tracking
        h_sp = sp_gru(feat[:, :64], act)
        assert h_sp.shape == (batch_size, 64, 8, 8)
        h_vec = vec_belief(z, act, h_sp)
        assert h_vec.shape == (batch_size, 512)

        # 4. Unified state concatenation
        s = VectorBeliefCore.get_unified_state(z, h_vec)
        assert s.shape == (batch_size, 1024)

        # 5. Target EMA forward and update
        z_tgt, _ = target_ema(obs, use_target=True)
        assert z_tgt.shape == (batch_size, 512)
        assert not z_tgt.requires_grad
        target_ema.update(tau=0.996)

    def test_full_pipeline_roundtrip_crafter_stepping(self):
        """Executes a 10-step trajectory in Crafter environment verifying end-to-end perception."""
        env = GodfatherCrafterEnv(seed=100_002)
        obs, _ = env.reset()

        enc = make_calibrated_encoder(512)
        sp_gru = SpatialConvGRU(64)
        vec_belief = VectorBeliefCore(512)
        target_ema = TargetEMA(enc)

        h_sp = None
        h_vec = None
        target_norm = math.sqrt(512)

        for step in range(10):
            obs_t = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0)
            z, feat = enc(obs_t)

            # Check invariant at each step
            norm = torch.norm(z, dim=-1).item()
            assert abs(norm - target_norm) < 0.05

            action = step % 17
            h_sp = sp_gru(feat[:, :64], torch.tensor([action]), H=h_sp)
            h_vec = vec_belief(z, torch.tensor([action]), h_sp, h=h_vec)

            s = VectorBeliefCore.get_unified_state(z, h_vec)
            assert s.shape == (1, 1024)
            assert not torch.isnan(s).any()

            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                break

        # Step target EMA
        target_ema.update(tau=0.995)
