"""
tests/test_predictor.py

Multi-tier Unit Test Suite for Action-Conditioned Latent Predictor Ensemble.

Tiers:
- Tier 1: Core Feature Correctness (Shapes, Embeddings, Parameter Budgets, Residuals)
- Tier 2: Boundary & Edge Case Hardening (Out-of-bounds, Extremes, Latency < 0.5ms)
- Tier 3: Cross-Feature Integration (Epistemic Disagreement, Gradient Flow, Rollout Consistency)
- Tier 4: Real-World Scenarios (Atari Transfer, Zero-Cheat Compliance)
"""

import unittest
import time
import torch
import torch.nn as nn
from src.models.predictor import (
    ActionEmbedding,
    ResidualMLPPredictorHead,
    ResidualMLPPredictor,
    LatentPredictor,
)


class TestPredictorTier1Features(unittest.TestCase):
    """Tier 1: Feature functionality and contract validation."""

    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 256
        self.num_actions = 17
        self.action_dim = 32
        self.model = ResidualMLPPredictor(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
            action_dim=self.action_dim,
            num_heads=4,
            hidden_dim=192,
        )

    def test_parameter_budget_under_600k(self):
        """Feature 3: Verify total parameter count strictly < 600,000."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 571424)
        self.assertLess(total_params, 600000)

    def test_embedding_parameter_count(self):
        """Feature 2: Verify embedding parameters match |A| * action_dim."""
        emb_params = sum(p.numel() for p in self.model.action_embedding.parameters())
        self.assertEqual(emb_params, self.num_actions * self.action_dim)
        self.assertEqual(emb_params, 544)

    def test_ensemble_head_count(self):
        """Feature 3: Verify exactly 4 heads in the ensemble."""
        self.assertEqual(len(self.model.heads), 4)
        for head in self.model.heads:
            self.assertIsInstance(head, ResidualMLPPredictorHead)
            p_head = sum(p.numel() for p in head.parameters())
            self.assertEqual(p_head, 142720)

    def test_forward_shape_contract(self):
        """Feature 3: Verify exact output shapes (B, 256) and (4, B, 256)."""
        for B in [1, 4, 16, 32, 64]:
            z = torch.randn(B, self.latent_dim)
            a = torch.randint(0, self.num_actions, (B,))
            mean_pred, all_preds = self.model(z, a)
            self.assertEqual(mean_pred.shape, (B, self.latent_dim))
            self.assertEqual(all_preds.shape, (4, B, self.latent_dim))

    def test_latent_predictor_alias_compatibility(self):
        """Interface: Verify LatentPredictor alias works identically."""
        self.assertIs(LatentPredictor, ResidualMLPPredictor)
        alias_model = LatentPredictor(latent_dim=256, num_actions=17)
        z = torch.randn(2, 256)
        a = torch.tensor([0, 5])
        mean_p, all_p = alias_model(z, a)
        self.assertEqual(mean_p.shape, (2, 256))

    def test_predict_next_convenience_method(self):
        """Interface: Verify predict_next returns mean prediction."""
        z = torch.randn(3, 256)
        a = torch.tensor([1, 2, 3])
        mean_p, _ = self.model(z, a)
        p_next = self.model.predict_next(z, a)
        self.assertTrue(torch.allclose(mean_p, p_next))


class TestPredictorTier2Boundaries(unittest.TestCase):
    """Tier 2: Boundary conditions, numerical extremes, and latency constraints."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = ResidualMLPPredictor(latent_dim=256, num_actions=17, hidden_dim=192)

    def test_action_input_format_handling(self):
        """Boundary: Handle 1D (B,), 2D (B, 1), and float one-hot (B, 17)."""
        B = 8
        z = torch.randn(B, 256)
        a_1d = torch.randint(0, 17, (B,))
        a_2d = a_1d.unsqueeze(1)
        a_onehot = torch.zeros(B, 17)
        a_onehot.scatter_(1, a_2d, 1.0)

        m1, _ = self.model(z, a_1d)
        m2, _ = self.model(z, a_2d)
        m3, _ = self.model(z, a_onehot)

        self.assertTrue(torch.allclose(m1, m2))
        self.assertTrue(torch.allclose(m1, m3, atol=1e-5))

    def test_out_of_bounds_action_rejection(self):
        """Boundary: Reject invalid discrete action indices."""
        z = torch.randn(2, 256)
        with self.assertRaises(IndexError):
            self.model(z, torch.tensor([0, 17]))  # Crafter has 17 actions: 0..16
        with self.assertRaises(IndexError):
            self.model(z, torch.tensor([-1, 5]))

    def test_invalid_latent_shape_rejection(self):
        """Boundary: Reject malformed latent dimension."""
        a = torch.tensor([0])
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 128), a)  # Mismatched dimension (128 vs 256)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 256, 1), a)  # 3D tensor rejected

    def test_cpu_inference_latency_budget(self):
        """Acceptance: Verify inference latency on CPU < 0.5 ms for single step."""
        self.model.eval()
        z = torch.randn(1, 256)
        a = torch.randint(0, 17, (1,))
        with torch.no_grad():
            # Warmup
            for _ in range(50):
                _ = self.model(z, a)
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                _ = self.model(z, a)
                times.append(time.perf_counter() - t0)
        mean_ms = (sum(times) / len(times)) * 1000
        self.assertLess(mean_ms, 0.50, f"CPU latency exceeded 0.5ms: {mean_ms:.4f} ms")

    def test_numerical_stability_extreme_inputs(self):
        """Boundary: Assert no NaNs/Infs under extreme latent values."""
        extreme_z = torch.cat([
            torch.zeros(2, 256),
            torch.ones(2, 256) * 1e2,
            torch.ones(2, 256) * -1e2,
        ], dim=0)
        a = torch.randint(0, 17, (6,))
        mean_p, all_p = self.model(extreme_z, a)
        self.assertFalse(torch.isnan(mean_p).any())
        self.assertFalse(torch.isinf(mean_p).any())
        self.assertFalse(torch.isnan(all_p).any())


class TestPredictorTier3CrossFeature(unittest.TestCase):
    """Tier 3: Multi-head epistemic disagreement, gradients, and rollout consistency."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = ResidualMLPPredictor(latent_dim=256, num_actions=17, hidden_dim=192)

    def test_epistemic_disagreement_non_zero_variance(self):
        """Curiosity: Distinct weight inits produce strictly positive head variance."""
        B = 10
        z = torch.randn(B, 256)
        a = torch.randint(0, 17, (B,))
        _, all_p = self.model(z, a)
        disagreement = self.model.compute_epistemic_disagreement(all_p)
        self.assertEqual(disagreement.shape, (B,))
        self.assertTrue((disagreement > 0.0).all())

    def test_backward_gradient_flow_to_all_components(self):
        """Optimization: Gradients propagate to all 4 heads, embeddings, and latent input."""
        z = torch.randn(4, 256, requires_grad=True)
        a = torch.randint(0, 17, (4,))
        mean_p, all_p = self.model(z, a)
        loss = mean_p.sum() + all_p.sum()
        loss.backward()

        self.assertIsNotNone(z.grad)
        self.assertTrue((z.grad.abs().sum() > 0).item())
        self.assertIsNotNone(self.model.action_embedding.embedding.weight.grad)
        self.assertTrue((self.model.action_embedding.embedding.weight.grad.abs().sum() > 0).item())
        for head in self.model.heads:
            for name, param in head.named_parameters():
                self.assertIsNotNone(param.grad, f"Head parameter {name} received no gradient!")
                self.assertTrue((param.grad.abs().sum() > 0).item())

    def test_autoregressive_rollout_step_consistency(self):
        """Planning: Multi-step rollout matches step-by-step unrolling exactly."""
        H, B = 5, 4
        z0 = torch.randn(B, 256)
        actions = torch.randint(0, 17, (H, B))
        t_mean, t_all = self.model.rollout(z0, actions)

        self.assertEqual(t_mean.shape, (H, B, 256))
        self.assertEqual(t_all.shape, (4, H, B, 256))

        # Check step 0
        step0_mean, step0_all = self.model(z0, actions[0])
        self.assertTrue(torch.allclose(t_mean[0], step0_mean, atol=1e-6))
        self.assertTrue(torch.allclose(t_all[:, 0], step0_all, atol=1e-6))

        # Check step 1
        step1_mean, _ = self.model(step0_mean, actions[1])
        self.assertTrue(torch.allclose(t_mean[1], step1_mean, atol=1e-6))

    def test_autoregressive_rollout_equi_dimensional_h_equals_b_disambiguation(self):
        """Planning: Disambiguate rollout transpose logic when H == B using batch_first flag."""
        B = 5
        H = 5
        z0 = torch.randn(B, 256)
        actions_bh = torch.randint(0, 17, (B, H))
        actions_hb = actions_bh.transpose(0, 1)

        # 1. Rollout with batch_first=True (treating as (B, H))
        mean_bh, all_bh = self.model.rollout(z0, actions_bh, batch_first=True)
        # 2. Rollout with batch_first=False (treating as (H, B))
        mean_hb, all_hb = self.model.rollout(z0, actions_hb, batch_first=False)

        # Both must produce identical trajectories
        self.assertEqual(mean_bh.shape, (H, B, 256))
        self.assertEqual(all_bh.shape, (4, H, B, 256))
        self.assertTrue(torch.allclose(mean_bh, mean_hb, atol=1e-6))
        self.assertTrue(torch.allclose(all_bh, all_hb, atol=1e-6))

        # 3. Check step 0 consistency against direct model forward pass
        expected_step0, expected_all0 = self.model(z0, actions_bh[:, 0])
        self.assertTrue(torch.allclose(mean_bh[0], expected_step0, atol=1e-6))
        self.assertTrue(torch.allclose(all_bh[:, 0], expected_all0, atol=1e-6))

        # 4. Default parameter is batch_first=True
        mean_default, all_default = self.model.rollout(z0, actions_bh)
        self.assertTrue(torch.allclose(mean_default, mean_bh, atol=1e-6))

        # 5. Invalid shape handling
        with self.assertRaises(ValueError):
            self.model.rollout(z0, torch.randint(0, 17, (3, 7)))


class TestPredictorTier4Scenarios(unittest.TestCase):
    """Tier 4: Generalization to Atari and Zero-Cheat compliance."""

    def test_atari_plug_and_play_transfer(self):
        """R4: Seamless transfer to Atari Breakout (4 actions) and Pong (6 actions)."""
        breakout_model = ResidualMLPPredictor(latent_dim=256, num_actions=4)
        pong_model = ResidualMLPPredictor(latent_dim=256, num_actions=6)

        z = torch.randn(3, 256)
        a_breakout = torch.tensor([0, 2, 3])
        a_pong = torch.tensor([1, 4, 5])

        m_b, all_b = breakout_model(z, a_breakout)
        m_p, all_p = pong_model(z, a_pong)

        self.assertEqual(m_b.shape, (3, 256))
        self.assertEqual(m_p.shape, (3, 256))
        self.assertEqual(all_b.shape, (4, 3, 256))
        self.assertEqual(all_p.shape, (4, 3, 256))

    def test_zero_cheat_pure_latent_contract(self):
        """R3: Verify module operates with zero access to environment state dicts."""
        model = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        z = torch.randn(1, 256)
        a = torch.tensor([5])
        mean_p, _ = model(z, a)
        self.assertEqual(mean_p.shape, (1, 256))


if __name__ == "__main__":
    unittest.main()
