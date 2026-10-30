"""
tests/test_adversarial_m1.py

Adversarial Stress Test Suite for Milestone 1: Core JEPA Representation & Dynamics.
Conducted by Challenger 1 (Empirical Challenger).

Challenges Covered:
- Challenge 1: Numerical stability under extreme inputs (all zeros, huge values,
  subnormal numbers, NaNs/Infs handling, batch sizes B=1, B=2, B=128, unbatched,
  uint8/float64 dtypes, variable spatial resolutions, shape mismatches).
- Challenge 2: Representation collapse stress test: complete & near-complete collapse,
  massive variance penalty verification, non-NaN gradient verification under collapse,
  and verify_non_collapse precision triggers.
- Challenge 3: Long-horizon autoregressive rollout stability: H=20 (and H=50) rollout,
  latent norm explosion/vanishing checks, gradient backpropagation through H=20 steps,
  action transpose format invariance ((H, B) vs (B, H)), and epistemic disagreement
  trajectory analysis.
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import (
    ObservationEncoder,
    ResNetTinyEncoder,
    TargetEncoder,
    ActionEmbedding,
    ResidualMLPPredictor,
    LatentPredictor,
    VICRegLoss,
    verify_non_collapse,
    assert_non_collapse,
)


class TestChallenge1NumericalStability(unittest.TestCase):
    """
    Challenge 1: Numerical stability under extreme inputs.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        self.target_encoder = TargetEncoder(self.encoder, tau=0.99)
        self.predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32, num_heads=4)
        self.vicreg = VICRegLoss(sim_coeff=1.0, var_coeff=1.0, cov_coeff=0.04, gamma=1.0, epsilon=1e-4)

    def test_c1_1_encoder_all_zeros_all_ones(self):
        """Test encoder produces finite, normalized latents for all-zero and all-one images."""
        for name, val in [("all_zeros", 0.0), ("all_ones", 1.0)]:
            x = torch.full((4, 3, 64, 64), val, dtype=torch.float32)
            z = self.encoder(x)
            self.assertFalse(torch.isnan(z).any(), f"NaN detected for {name}")
            self.assertFalse(torch.isinf(z).any(), f"Inf detected for {name}")
            self.assertEqual(z.shape, (4, 256))
            # Verify LayerNorm properties: mean ≈ 0, std ≈ 1 (within epsilon=1e-5 tolerance of LayerNorm)
            mean_val = z.mean(dim=-1).abs().max().item()
            std_val = z.std(dim=-1, unbiased=False)
            self.assertLess(mean_val, 1e-4, f"LayerNorm mean too large: {mean_val}")
            self.assertTrue(torch.allclose(std_val, torch.ones_like(std_val), atol=1e-2),
                            f"LayerNorm std not ~1.0: {std_val}")

    def test_c1_2_encoder_extreme_pixel_magnitudes(self):
        """Test encoder handling of huge, tiny, and negative inputs."""
        extreme_inputs = [
            ("huge_pixels", torch.full((2, 3, 64, 64), 1e5)),
            ("tiny_subnormal", torch.full((2, 3, 64, 64), 1e-25)),
            ("negative_pixels", torch.full((2, 3, 64, 64), -1e4)),
        ]
        for name, x in extreme_inputs:
            z = self.encoder(x)
            self.assertFalse(torch.isnan(z).any(), f"NaN detected in {name}")
            self.assertFalse(torch.isinf(z).any(), f"Inf detected in {name}")
            # Even under extreme magnitude, LayerNorm must bound latents
            self.assertLess(z.abs().max().item(), 50.0, f"Latent output blew up for {name}")

    def test_c1_3_encoder_batch_size_scaling_and_consistency(self):
        """Test encoder across B in [1, 2, 7, 32, 128] and unbatched (3, 64, 64)."""
        single_img = torch.rand(3, 64, 64)
        z_unbatched = self.encoder(single_img)
        self.assertEqual(z_unbatched.shape, (256,))

        z_batched_1 = self.encoder(single_img.unsqueeze(0))
        self.assertEqual(z_batched_1.shape, (1, 256))
        self.assertTrue(torch.allclose(z_unbatched, z_batched_1.squeeze(0), atol=1e-5),
                        "Unbatched and B=1 batch output differ")

        for B in [1, 2, 7, 32, 128]:
            x = torch.rand(B, 3, 64, 64)
            z = self.encoder(x)
            self.assertEqual(z.shape, (B, 256))
            self.assertFalse(torch.isnan(z).any())
            self.assertFalse(torch.isinf(z).any())

    def test_c1_4_target_encoder_extreme_inputs_and_stop_gradient(self):
        """Test TargetEncoder preserves stop-gradient and eval mode under extreme inputs."""
        x = torch.full((3, 3, 64, 64), 1e4, requires_grad=True)
        z_target = self.target_encoder(x)
        self.assertFalse(z_target.requires_grad)
        self.assertIsNone(z_target.grad_fn)
        self.assertFalse(torch.isnan(z_target).any())

    def test_c1_5_predictor_extreme_latents_and_actions(self):
        """Test predictor under zeros, huge, and subnormal latents with boundary actions."""
        test_cases = [
            ("zero_latents", torch.zeros(4, 256)),
            ("huge_latents", torch.full((4, 256), 1e3)),
            ("tiny_latents", torch.full((4, 256), 1e-15)),
        ]
        actions = torch.tensor([0, 16, 5, 10], dtype=torch.long)
        for name, z in test_cases:
            mean_pred, all_preds = self.predictor(z, actions)
            self.assertFalse(torch.isnan(mean_pred).any(), f"NaN in mean_pred for {name}")
            self.assertFalse(torch.isinf(mean_pred).any(), f"Inf in mean_pred for {name}")
            self.assertFalse(torch.isnan(all_preds).any(), f"NaN in all_preds for {name}")
            self.assertFalse(torch.isinf(all_preds).any(), f"Inf in all_preds for {name}")
            disagreement = self.predictor.compute_epistemic_disagreement(all_preds)
            self.assertFalse(torch.isnan(disagreement).any(), f"NaN in disagreement for {name}")
            self.assertTrue((disagreement >= 0.0).all(), f"Negative disagreement for {name}")

    def test_c1_6_predictor_batch_sizes_and_formats(self):
        """Test predictor handles batch sizes B=1, B=2, B=128 and 1D/2D/one-hot actions."""
        for B in [1, 2, 128]:
            z = torch.randn(B, 256)
            a_1d = torch.randint(0, 17, (B,))
            a_2d = a_1d.unsqueeze(-1)
            a_onehot = F.one_hot(a_1d, num_classes=17).float()

            m1, all1 = self.predictor(z, a_1d)
            m2, all2 = self.predictor(z, a_2d)
            m3, all3 = self.predictor(z, a_onehot)

            self.assertEqual(m1.shape, (B, 256))
            self.assertEqual(all1.shape, (4, B, 256))
            self.assertTrue(torch.allclose(m1, m2, atol=1e-5), "1D and 2D action outputs differ")
            self.assertTrue(torch.allclose(m1, m3, atol=1e-4), "1D and one-hot action outputs differ")

    def test_c1_7_vicreg_batch_boundaries_b1_b2_b128(self):
        """Test VICRegLoss behavior on B=1, B=2, B=128."""
        # Case B=1: Sample variance is undefined. VICReg must return 0 var/cov and compute valid invariance without crash.
        z_p_1 = torch.randn(4, 1, 256, requires_grad=True)
        z_t_1 = torch.randn(1, 256)
        out_1 = self.vicreg(z_p_1, z_t_1)
        self.assertFalse(torch.isnan(out_1["loss"]))
        self.assertEqual(out_1["variance"].item(), 0.0)
        self.assertEqual(out_1["covariance"].item(), 0.0)
        out_1["loss"].backward()
        self.assertIsNotNone(z_p_1.grad)
        self.assertFalse(torch.isnan(z_p_1.grad).any())

        # Case B=2: Minimal valid batch for sample variance (B - 1 = 1)
        z_p_2 = torch.randn(4, 2, 256, requires_grad=True)
        z_t_2 = torch.randn(2, 256)
        out_2 = self.vicreg(z_p_2, z_t_2)
        self.assertFalse(torch.isnan(out_2["loss"]))
        self.assertGreater(out_2["variance"].item(), 0.0)
        out_2["loss"].backward()
        self.assertFalse(torch.isnan(z_p_2.grad).any())

        # Case B=128: Standard training batch
        z_p_128 = torch.randn(4, 128, 256, requires_grad=True)
        z_t_128 = torch.randn(128, 256)
        out_128 = self.vicreg(z_p_128, z_t_128)
        self.assertFalse(torch.isnan(out_128["loss"]))
        out_128["loss"].backward()
        self.assertFalse(torch.isnan(z_p_128.grad).any())

    def test_c1_8_vicreg_mismatched_dimensions_and_invalid_reduction(self):
        """Test VICRegLoss defensive rejection of mismatched shapes and invalid configs."""
        # Batch size mismatch
        with self.assertRaises(ValueError) as ctx_b:
            self.vicreg(torch.randn(4, 8, 256), torch.randn(4, 9, 256))
        self.assertIn("Batch size mismatch", str(ctx_b.exception))

        # Latent dimension mismatch
        with self.assertRaises(ValueError) as ctx_d:
            self.vicreg(torch.randn(4, 8, 256), torch.randn(4, 8, 128))
        self.assertIn("dimension mismatch", str(ctx_d.exception))

        # Invalid reduction mode
        with self.assertRaises(ValueError) as ctx_r:
            VICRegLoss(multi_head_reduction="unsupported_mode")
        self.assertIn("Unsupported multi_head_reduction", str(ctx_r.exception))

    def test_c1_9_encoder_variable_resolutions_and_dtypes(self):
        """Test encoder accepts uint8 and variable spatial resolutions (e.g. 84x84)."""
        # uint8 input: correctly cast and divided by 255.0
        x_uint8 = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
        z_uint8 = self.encoder(x_uint8)
        self.assertEqual(z_uint8.shape, (2, 256))
        self.assertEqual(z_uint8.dtype, torch.float32)

        # Atari 84x84 resolution
        x_atari = torch.randn(2, 3, 84, 84)
        z_atari = self.encoder(x_atari)
        self.assertEqual(z_atari.shape, (2, 256))
        self.assertFalse(torch.isnan(z_atari).any())

        # Defensive dtype caveat check: float64 raises RuntimeError because conv weights are float32
        x_f64 = torch.randn(2, 3, 64, 64, dtype=torch.float64)
        with self.assertRaises(RuntimeError) as ctx:
            self.encoder(x_f64)
        self.assertIn("Input type (double) and bias type (float) should be the same", str(ctx.exception))


class TestChallenge2RepresentationCollapse(unittest.TestCase):
    """
    Challenge 2: Representation collapse stress test.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.vicreg = VICRegLoss(sim_coeff=1.0, var_coeff=1.0, cov_coeff=0.04, gamma=1.0, epsilon=1e-4)

    def test_c2_1_complete_collapse_massive_variance_penalty(self):
        """Simulate identical constant representations across all batch samples."""
        B, D = 32, 256
        # Collapsed constant latents: every sample has vector [3.5, 3.5, ...]
        z_pred = torch.full((4, B, D), 3.5, requires_grad=True)
        z_target = torch.full((B, D), 3.5)

        loss_dict = self.vicreg(z_pred, z_target)

        # Invariance should be identically zero
        self.assertAlmostEqual(loss_dict["invariance"].item(), 0.0, places=6)

        # Theoretical variance penalty:
        # Var(Z_j) = 0. std = sqrt(0 + 1e-4) = 0.01.
        # Hinge penalty per dimension: max(0, 1.0 - 0.01) = 0.99.
        # Target also regularized -> 0.5 * (0.99 + 0.99) = 0.99.
        self.assertAlmostEqual(loss_dict["variance"].item(), 0.99, places=3)
        self.assertGreater(loss_dict["variance"].item(), 0.95, "Variance penalty must be massive under collapse")

        # Covariance should be zero (centered latents are identically zero)
        self.assertAlmostEqual(loss_dict["covariance"].item(), 0.0, places=6)

        # Verify gradient flow: strictly finite, no NaNs
        loss_dict["loss"].backward()
        self.assertIsNotNone(z_pred.grad)
        self.assertFalse(torch.isnan(z_pred.grad).any(), "NaN found in gradient under complete collapse")
        self.assertFalse(torch.isinf(z_pred.grad).any(), "Inf found in gradient under complete collapse")

    def test_c2_2_near_collapse_repulsive_gradient(self):
        """Simulate near-collapsed latents with tiny perturbations (sigma=1e-4)."""
        B, D = 64, 256
        center = torch.ones(1, D) * 2.0
        perturbation = torch.randn(B, D) * 1e-4
        z_pred_raw = (center + perturbation).unsqueeze(0).repeat(4, 1, 1).requires_grad_(True)
        z_target = center + torch.randn(B, D) * 1e-4

        loss_dict = self.vicreg(z_pred_raw, z_target)

        # Variance penalty must still be large (> 0.90)
        self.assertGreater(loss_dict["variance"].item(), 0.90)

        # Verify repulsive anti-collapse force on variance loss:
        # For a dimension with std < gamma, the gradient of variance hinge with respect to z_i
        # points in direction -(z_i - mean(z)). Therefore gradient descent (z <- z - lr * grad)
        # adds +(z_i - mean(z)), which expands the variance!
        loss_dict["variance"].backward()
        self.assertFalse(torch.isnan(z_pred_raw.grad).any())
        self.assertFalse(torch.isinf(z_pred_raw.grad).any())

        z_c = z_pred_raw[0] - z_pred_raw[0].mean(dim=0, keepdim=True)
        grad_v = z_pred_raw.grad[0]
        alignment = (grad_v * z_c).sum(dim=0)
        # Average alignment must be strictly negative (repulsive)
        self.assertLess(alignment.mean().item(), 0.0, "Variance gradient must act as repulsive anti-collapse force")

    def test_c2_3_partial_rank_1_collapse(self):
        """Simulate rank-1 collapse: all batch latents lie along a single 1D ray."""
        B, D = 32, 256
        direction = F.normalize(torch.randn(1, D), dim=-1)
        scalars = torch.randn(B, 1)  # Spread along single direction
        z_rank1 = scalars * direction

        # Check effective rank is approximately 1.0
        report = verify_non_collapse(z_rank1, threshold=0.1, strict=False)
        self.assertFalse(report["is_valid"])
        self.assertLess(report["effective_rank"], 2.0, f"Effective rank should be ~1.0, got {report['effective_rank']}")
        # 255 of the 256 dimensions have zero variance
        self.assertGreaterEqual(report["num_collapsed_dims"], 250)

    def test_c2_4_verify_non_collapse_triggers_cleanly(self):
        """Verify verify_non_collapse and assert_non_collapse trigger cleanly on collapse."""
        B, D = 32, 256

        # Subtest 1: Healthy representation (standard normal)
        z_healthy = torch.randn(B, D)
        report_healthy = verify_non_collapse(z_healthy, threshold=0.1, strict=True)
        self.assertTrue(report_healthy["is_valid"])
        self.assertGreater(report_healthy["min_variance"], 0.1)

        # Subtest 2: Artificially collapsed dimension
        z_bad = torch.randn(B, D)
        z_bad[:, 42] = 0.0  # Collapse dimension 42
        with self.assertRaises(AssertionError) as ctx:
            assert_non_collapse(z_bad, threshold=0.1)
        self.assertIn("collapse detected", str(ctx.exception))
        self.assertIn("1 feature dimension(s)", str(ctx.exception))

        # Subtest 3: Non-strict mode returns report without raising
        report_bad = verify_non_collapse(z_bad, threshold=0.1, strict=False)
        self.assertFalse(report_bad["is_valid"])
        self.assertEqual(report_bad["num_collapsed_dims"], 1)

        # Subtest 4: B < 2 raises ValueError
        z_b1 = torch.randn(1, D)
        with self.assertRaises(ValueError):
            verify_non_collapse(z_b1)


class TestChallenge3LongHorizonRollout(unittest.TestCase):
    """
    Challenge 3: Long-horizon autoregressive rollout stability (H=20, H=50).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32, num_heads=4)
        self.predictor.eval()

    def test_c3_1_rollout_horizon_20_latent_norm_bounds(self):
        """Rollout for H=20 steps and verify latent norm does not explode or vanish."""
        B = 16
        H = 20
        # Initialize z_0 with LayerNorm-like statistics (mean 0, std 1, norm approx sqrt(256)=16)
        z_0 = F.layer_norm(torch.randn(B, 256), (256,))
        initial_norm = z_0.norm(dim=-1)  # Shape (B,)

        # Test across multiple action regimes:
        # Regime A: Random actions
        # Regime B: Constant action 0
        # Regime C: Constant action 16
        # Regime D: Alternating actions
        regimes = {
            "random": torch.randint(0, 17, (H, B)),
            "constant_0": torch.zeros((H, B), dtype=torch.long),
            "constant_16": torch.full((H, B), 16, dtype=torch.long),
            "alternating": torch.tensor([[t % 17 for _ in range(B)] for t in range(H)], dtype=torch.long),
        }

        for regime_name, actions in regimes.items():
            traj_mean, traj_all = self.predictor.rollout(z_0, actions)
            self.assertEqual(traj_mean.shape, (H, B, 256))
            self.assertEqual(traj_all.shape, (4, H, B, 256))

            # Check for NaNs or Infs
            self.assertFalse(torch.isnan(traj_mean).any(), f"NaN in traj_mean for {regime_name}")
            self.assertFalse(torch.isinf(traj_mean).any(), f"Inf in traj_mean for {regime_name}")

            # Check norm evolution across horizon
            for step in range(H):
                step_norm = traj_mean[step].norm(dim=-1)  # Shape (B,)
                ratio = step_norm / initial_norm

                # Assert that ratio stays within [0.2, 5.0]
                min_ratio = ratio.min().item()
                max_ratio = ratio.max().item()
                self.assertGreater(
                    min_ratio, 0.2,
                    f"Latent norm vanished at step {step} in {regime_name}: min_ratio={min_ratio:.4f}"
                )
                self.assertLess(
                    max_ratio, 5.0,
                    f"Latent norm exploded at step {step} in {regime_name}: max_ratio={max_ratio:.4f}"
                )

    def test_c3_2_extended_rollout_horizon_50(self):
        """Stress-test extended horizon H=50 to verify long-term stability."""
        B = 8
        H = 50
        z_0 = F.layer_norm(torch.randn(B, 256), (256,))
        initial_norm = z_0.norm(dim=-1).mean().item()
        actions = torch.randint(0, 17, (H, B))

        traj_mean, traj_all = self.predictor.rollout(z_0, actions)
        self.assertFalse(torch.isnan(traj_mean).any())
        self.assertFalse(torch.isinf(traj_mean).any())

        final_norm = traj_mean[-1].norm(dim=-1).mean().item()
        ratio = final_norm / initial_norm
        # At H=50 steps, norm ratio should remain within [0.1, 8.0]
        self.assertGreater(ratio, 0.1, f"Norm vanished at H=50: ratio={ratio:.4f}")
        self.assertLess(ratio, 8.0, f"Norm exploded at H=50: ratio={ratio:.4f}")

    def test_c3_3_rollout_gradient_backpropagation_stability(self):
        """Verify backpropagation through H=20 unrolled steps has stable gradient flow."""
        B = 4
        H = 20
        predictor_train = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32, num_heads=4)
        z_0 = F.layer_norm(torch.randn(B, 256), (256,)).requires_grad_(True)
        actions = torch.randint(0, 17, (H, B))

        traj_mean, traj_all = predictor_train.rollout(z_0, actions)
        # Compute loss on terminal state
        terminal_loss = traj_mean[-1].sum()
        terminal_loss.backward()

        self.assertIsNotNone(z_0.grad)
        self.assertFalse(torch.isnan(z_0.grad).any(), "NaN in gradient through H=20 rollout")
        self.assertFalse(torch.isinf(z_0.grad).any(), "Inf in gradient through H=20 rollout")

        # Check gradient norm on z_0
        grad_norm = z_0.grad.norm().item()
        self.assertGreater(grad_norm, 1e-6, f"Vanishing gradient through H=20 rollout: {grad_norm}")
        self.assertLess(grad_norm, 1e4, f"Exploding gradient through H=20 rollout: {grad_norm}")

        # Check that predictor head parameters also received finite gradients
        for name, param in predictor_train.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient of {name}")

    def test_c3_4_epistemic_disagreement_across_horizon(self):
        """Verify epistemic disagreement across the 4 heads over H=20 steps."""
        B = 8
        H = 20
        z_0 = F.layer_norm(torch.randn(B, 256), (256,))
        actions = torch.randint(0, 17, (H, B))

        traj_mean, traj_all = self.predictor.rollout(z_0, actions)

        # traj_all has shape (4, H, B, 256)
        disagreements = []
        for t in range(H):
            all_preds_t = traj_all[:, t, :, :]  # (4, B, 256)
            disagree_t = self.predictor.compute_epistemic_disagreement(all_preds_t)  # (B,)
            self.assertFalse(torch.isnan(disagree_t).any())
            self.assertTrue((disagree_t >= 0.0).all())
            disagreements.append(disagree_t.mean().item())

        # Disagreements should remain finite and bounded
        max_disagree = max(disagreements)
        self.assertLess(max_disagree, 100.0, f"Disagreement exploded across horizon: {max_disagree}")

    def test_c3_5_rollout_action_tensor_format_invariance(self):
        """Verify rollout produces identical trajectories for (H, B) and (B, H) action formats."""
        B, H = 8, 12
        z_0 = torch.randn(B, 256)
        actions_hb = torch.randint(0, 17, (H, B))
        actions_bh = actions_hb.transpose(0, 1)

        mean_hb, all_hb = self.predictor.rollout(z_0, actions_hb)
        mean_bh, all_bh = self.predictor.rollout(z_0, actions_bh)

        self.assertTrue(torch.allclose(mean_hb, mean_bh, atol=1e-5), "Rollout differs between (H, B) and (B, H)")
        self.assertTrue(torch.allclose(all_hb, all_bh, atol=1e-5), "Rollout ensemble differs between (H, B) and (B, H)")


if __name__ == "__main__":
    unittest.main()
