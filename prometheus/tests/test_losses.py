"""
tests/test_losses.py

Unit and boundary test suite for VICRegLoss and non-collapse verification.
"""

import unittest
import torch
import torch.nn.functional as F
from src.models.losses import VICRegLoss, verify_non_collapse, assert_non_collapse


class TestVICRegLossTier1(unittest.TestCase):
    """Tier 1: Feature Correctness Tests."""

    def test_t1_1_invariance_loss_zero_on_identical(self):
        """Verify invariance loss is identically 0.0 when z_pred == z_target."""
        loss_fn = VICRegLoss(regularize_target=False)
        z = torch.randn(32, 256)
        out = loss_fn(z, z)
        self.assertAlmostEqual(out["invariance"].item(), 0.0, places=5)

    def test_t1_2_variance_hinge_inactive_when_spread(self):
        """Verify variance hinge produces 0.0 penalty when std >= gamma=1.0."""
        loss_fn = VICRegLoss(gamma=1.0, regularize_target=False)
        # Latents with standard deviation 2.0 > 1.0
        z = torch.randn(1000, 256) * 2.0
        out = loss_fn(z, z)
        self.assertAlmostEqual(out["pred_var"].item(), 0.0, places=3)

    def test_t1_3_covariance_decorrelation_zero_on_orthogonal(self):
        """Verify covariance loss is 0.0 when feature dimensions are orthogonal."""
        loss_fn = VICRegLoss(regularize_target=False)
        # Standard basis vectors have zero off-diagonal covariance
        eye = torch.eye(256)
        z = eye[:32]  # (32, 256)
        out = loss_fn(z, z)
        self.assertAlmostEqual(out["pred_cov"].item(), 0.0, places=2)

    def test_t1_4_multi_head_gradient_flow(self):
        """Verify gradients flow to all 4 heads when z_pred is (4, B, D)."""
        loss_fn = VICRegLoss(multi_head_reduction="mean_heads")
        z_p = torch.randn(4, 32, 256, requires_grad=True)
        z_t = torch.randn(32, 256)
        out = loss_fn(z_p, z_t)
        out["loss"].backward()
        self.assertIsNotNone(z_p.grad)
        self.assertEqual(z_p.grad.shape, (4, 32, 256))
        for k in range(4):
            self.assertGreater(z_p.grad[k].norm().item(), 0.0)

    def test_t1_5_loss_weighting_coefficients(self):
        """Verify total loss is exactly lambda*inv + mu*var + nu*cov."""
        loss_fn = VICRegLoss(sim_coeff=2.5, var_coeff=1.5, cov_coeff=0.1)
        z_p = torch.randn(32, 256)
        z_t = torch.randn(32, 256)
        out = loss_fn(z_p, z_t)
        expected = 2.5 * out["invariance"] + 1.5 * out["variance"] + 0.1 * out["covariance"]
        self.assertAlmostEqual(out["loss"].item(), expected.item(), places=4)

    def test_t1_6_multi_head_reduction_modes(self):
        """Verify mean_heads and average_first reduction options."""
        loss_mean = VICRegLoss(multi_head_reduction="mean_heads")
        loss_avg = VICRegLoss(multi_head_reduction="average_first")
        z_p = torch.randn(4, 16, 256)
        z_t = torch.randn(16, 256)
        out_mean = loss_mean(z_p, z_t)
        out_avg = loss_avg(z_p, z_t)
        self.assertIn("loss", out_mean)
        self.assertIn("loss", out_avg)

    def test_t1_7_invariance_normalization_modes_and_scale_balance(self):
        """Verify normalize_invariance=True default, exact MSE scaling, and O(1) balance."""
        loss_default = VICRegLoss()
        # 1. Verify True by default
        self.assertTrue(loss_default.normalize_invariance)

        # 2. Compare normalized vs unnormalized invariance computations
        loss_unnorm = VICRegLoss(normalize_invariance=False)
        B, D = 32, 256
        z_p = torch.randn(B, D)
        z_t = torch.randn(B, D)

        out_norm = loss_default(z_p, z_t)
        out_unnorm = loss_unnorm(z_p, z_t)

        expected_mse = torch.mean((z_p - z_t) ** 2)
        expected_sum_l2 = torch.sum((z_p - z_t) ** 2, dim=-1).mean()

        self.assertAlmostEqual(out_norm["invariance"].item(), expected_mse.item(), places=5)
        self.assertAlmostEqual(out_unnorm["invariance"].item(), expected_sum_l2.item(), places=3)

        # 3. Ratio between unnormalized and normalized is exactly D
        ratio = (out_unnorm["invariance"] / out_norm["invariance"]).item()
        self.assertAlmostEqual(ratio, float(D), delta=1e-3)

        # 4. Scale balance: normalized invariance is O(1) and balanced with variance hinge loss (gamma=1.0)
        self.assertLess(out_norm["invariance"].item(), 10.0)
        self.assertGreater(out_norm["invariance"].item(), 0.1)


class TestVICRegLossTier2(unittest.TestCase):
    """Tier 2: Boundary and Stress Tests."""

    def test_t2_1_collapsed_constant_no_nan_gradient(self):
        """Verify collapsed constant tensor produces finite non-NaN gradients."""
        loss_fn = VICRegLoss(epsilon=1e-4)
        z_p = torch.zeros(32, 256, requires_grad=True)
        z_t = torch.randn(32, 256)
        out = loss_fn(z_p, z_t)
        out["loss"].backward()
        self.assertFalse(torch.isnan(z_p.grad).any())
        self.assertAlmostEqual(out["pred_var"].item(), 1.0 - 0.01, places=3)

    def test_t2_2_batch_size_boundaries(self):
        """Verify behavior on extreme batch sizes B=1 and B=2."""
        loss_fn = VICRegLoss()
        # B=1 boundary
        z_1 = torch.randn(1, 256, requires_grad=True)
        out_1 = loss_fn(z_1, z_1)
        out_1["loss"].backward()
        self.assertFalse(torch.isnan(z_1.grad).any())

        # B=2 minimal degrees of freedom
        z_2 = torch.randn(2, 256, requires_grad=True)
        out_2 = loss_fn(z_2, z_2)
        out_2["loss"].backward()
        self.assertFalse(torch.isnan(z_2.grad).any())

    def test_t2_3_dimension_scaling(self):
        """Verify stability across latent dimensions D in [1, 2, 64, 256, 1024]."""
        loss_fn = VICRegLoss()
        for D in [1, 2, 64, 256, 1024]:
            z = torch.randn(16, D)
            out = loss_fn(z, z)
            self.assertFalse(torch.isnan(out["loss"]))

    def test_t2_4_stop_gradient_target_decoupling(self):
        """Verify target encoder weights receive zero gradients."""
        loss_fn = VICRegLoss(regularize_target=True)
        z_p = torch.randn(32, 256, requires_grad=True)
        z_t = torch.randn(32, 256, requires_grad=False)
        out = loss_fn(z_p, z_t)
        out["loss"].backward()
        self.assertIsNone(z_t.grad)

    def test_t2_5_verify_non_collapse_assertions(self):
        """Verify non-collapse assertion on healthy vs collapsed representations."""
        # 1. Healthy representations
        z_healthy = torch.randn(64, 256)
        res = verify_non_collapse(z_healthy, min_var_threshold=0.1, strict=True)
        self.assertTrue(res["is_valid"])
        self.assertGreater(res["min_variance"], 0.1)
        self.assertGreater(res["effective_rank"], 10.0)

        # 2. Partially collapsed representations (1 dead neuron)
        z_bad = torch.randn(64, 256)
        z_bad[:, 42] = 1.618  # Set column 42 to constant
        with self.assertRaises(AssertionError):
            verify_non_collapse(z_bad, min_var_threshold=0.1, strict=True)
        res_non_strict = verify_non_collapse(z_bad, min_var_threshold=0.1, strict=False)
        self.assertFalse(res_non_strict["is_valid"])
        self.assertEqual(res_non_strict["num_collapsed_dims"], 1)

        # 3. Assert helper with threshold keyword
        assert_non_collapse(z_healthy, threshold=0.1)
        with self.assertRaises(AssertionError):
            assert_non_collapse(z_bad, threshold=0.1)

    def test_t2_6_verify_non_collapse_small_batch_raises(self):
        """Verify B < 2 raises ValueError in verify_non_collapse."""
        z_small = torch.randn(1, 256)
        with self.assertRaises(ValueError):
            verify_non_collapse(z_small)


if __name__ == "__main__":
    unittest.main()
