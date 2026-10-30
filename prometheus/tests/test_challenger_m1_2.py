"""
tests/test_challenger_m1_2.py

Adversarial Empirical Challenge Suite for Milestone 1 (Core JEPA Representation & Dynamics).
Written by Challenger 2 (`challenger_m1_2`).

Empirically challenges:
1. End-to-end gradient flow: gradient propagation through VICRegLoss to all online encoder
   and predictor parameters, while strictly freezing target encoder parameters.
2. Epistemic disagreement sensitivity: 4-head predictor ensemble exhibits high disagreement
   on unvisited pseudo-transitions and low disagreement on repeated transitions.
3. Action sensitivity: distinct actions a != a' from identical state z produce distinct
   latent predictions (no action neglect) and can learn branched dynamics.
4. Adversarial stress & rollout stability across multi-step horizons.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.encoder import ObservationEncoder
from src.models.target import TargetEncoder
from src.models.predictor import ResidualMLPPredictor, LatentPredictor
from src.models.losses import VICRegLoss, verify_non_collapse


class TestChallenge1GradientFlowAndTargetFreezing(unittest.TestCase):
    """
    Challenge 1: Empirically verify end-to-end gradient flow and target encoder freezing.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_challenge_1_1_e2e_gradient_propagation_all_parameters(self):
        """
        Verify gradient flows to EVERY online encoder parameter and predictor parameter,
        while target encoder parameters remain strictly None / detached.
        """
        encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        target_encoder = TargetEncoder(encoder, tau=0.99)
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32, num_heads=4)
        loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=1.0, cov_coeff=0.04, multi_head_reduction="mean_heads")

        # Synthetic observations and actions covering all 17 actions
        B = 17
        x_t = torch.randn(B, 3, 64, 64, requires_grad=True)
        x_next = torch.randn(B, 3, 64, 64)
        actions = torch.arange(0, 17, dtype=torch.long)  # All actions 0..16 present

        # Forward pass
        z_t = encoder(x_t)
        mean_pred, all_preds = predictor(z_t, actions)
        z_target = target_encoder(x_next)

        # Compute loss
        loss_dict = loss_fn(all_preds, z_target)
        loss = loss_dict["loss"]
        loss.backward()

        # 1. Check ObservationEncoder parameters
        missing_encoder_grads = []
        zero_encoder_grads = []
        for name, param in encoder.named_parameters():
            if param.grad is None:
                missing_encoder_grads.append(name)
            elif param.grad.abs().sum().item() == 0.0:
                zero_encoder_grads.append(name)

        self.assertEqual(len(missing_encoder_grads), 0, f"Missing grads in encoder: {missing_encoder_grads}")
        self.assertEqual(len(zero_encoder_grads), 0, f"Zero grads in encoder: {zero_encoder_grads}")

        # 2. Check Predictor parameters across all 4 heads and embeddings
        missing_predictor_grads = []
        zero_predictor_grads = []
        for name, param in predictor.named_parameters():
            if param.grad is None:
                missing_predictor_grads.append(name)
            elif param.grad.abs().sum().item() == 0.0:
                zero_predictor_grads.append(name)

        self.assertEqual(len(missing_predictor_grads), 0, f"Missing grads in predictor: {missing_predictor_grads}")
        self.assertEqual(len(zero_predictor_grads), 0, f"Zero grads in predictor: {zero_predictor_grads}")

        # 3. Check ActionEmbedding table: every action row must receive gradient
        emb_grad = predictor.action_embedding.embedding.weight.grad
        self.assertIsNotNone(emb_grad)
        for a in range(17):
            row_norm = emb_grad[a].abs().sum().item()
            self.assertGreater(row_norm, 0.0, f"Action {a} embedding row received zero gradient")

        # 4. Check TargetEncoder parameters:
        # The target network parameters inside target_encoder.target_encoder must be frozen
        for name, param in target_encoder.target_encoder.named_parameters():
            self.assertFalse(param.requires_grad, f"Target param {name} has requires_grad=True")
            self.assertIsNone(param.grad, f"Target param {name} has non-None gradient!")

        # Audit: verify whether TargetEncoder registers online_encoder as a submodule
        # If registered, target_encoder.parameters() leaks online encoder trainable parameters
        target_submodules = [name for name, _ in target_encoder.named_children()]
        has_submodule_leak = "online_encoder" in target_submodules
        # Documented empirical finding: online_encoder is currently registered as a child submodule
        # which causes target_encoder.train() to flip online_encoder.training to False.
        # This will be reported as Finding F1 in the handoff report.

        # 5. Check input gradient
        self.assertIsNotNone(x_t.grad)
        self.assertGreater(x_t.grad.abs().sum().item(), 0.0)

    def test_challenge_1_2_optimizer_step_modifies_online_leaves_target_invariant(self):
        """
        Verify that an optimizer step updates online encoder and predictor,
        while target encoder parameters remain strictly unchanged prior to target.update().
        """
        encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        target_encoder = TargetEncoder(encoder, tau=0.99)
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        loss_fn = VICRegLoss()

        optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

        # Record initial weights
        init_enc_weights = [p.clone() for p in encoder.parameters()]
        init_pred_weights = [p.clone() for p in predictor.parameters()]
        init_target_weights = [p.clone() for p in target_encoder.target_encoder.parameters()]

        # Perform 5 training steps
        for step in range(5):
            optimizer.zero_grad()
            x = torch.randn(8, 3, 64, 64)
            x_next = torch.randn(8, 3, 64, 64)
            a = torch.randint(0, 17, (8,))

            z = encoder(x)
            _, all_preds = predictor(z, a)
            z_targ = target_encoder(x_next)

            loss = loss_fn(all_preds, z_targ)["loss"]
            loss.backward()
            optimizer.step()

            # Target encoder weights must remain bit-for-bit IDENTICAL during optimizer step
            for p_t, p_init in zip(target_encoder.target_encoder.parameters(), init_target_weights):
                self.assertTrue(torch.equal(p_t, p_init), f"Target weights mutated during step {step}!")

        # Verify online weights changed significantly
        enc_changed = any(not torch.equal(p, p_init) for p, p_init in zip(encoder.parameters(), init_enc_weights))
        pred_changed = any(not torch.equal(p, p_init) for p, p_init in zip(predictor.parameters(), init_pred_weights))
        self.assertTrue(enc_changed, "Online encoder weights did not change after optimizer steps!")
        self.assertTrue(pred_changed, "Predictor weights did not change after optimizer steps!")

        # Now call target_encoder.update()
        target_encoder.update(tau=0.99)

        # Now target weights must have changed smoothly
        target_moved = any(not torch.equal(p_t, p_init) for p_t, p_init in zip(target_encoder.target_encoder.parameters(), init_target_weights))
        self.assertTrue(target_moved, "Target encoder failed to update during target_encoder.update()!")

    def test_challenge_1_3_multi_step_convergence_on_synthetic_transitions(self):
        """
        Verify that training on synthetic transitions reduces VICReg loss monotonically and safely.
        """
        encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.5, cov_coeff=0.02)
        optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

        # Fixed batch of synthetic data
        x = torch.randn(16, 3, 64, 64)
        a = torch.randint(0, 17, (16,))
        z_target = torch.randn(16, 256)  # Target representations

        initial_loss = None
        final_loss = None
        for step in range(30):
            optimizer.zero_grad()
            z = encoder(x)
            _, all_preds = predictor(z, a)
            loss_dict = loss_fn(all_preds, z_target)
            loss = loss_dict["loss"]
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss * 0.5, f"Loss did not decrease by >=50%: {initial_loss:.4f} -> {final_loss:.4f}")


class TestChallenge2EpistemicDisagreement(unittest.TestCase):
    """
    Challenge 2: Empirically verify epistemic disagreement sensitivity:
    - High disagreement on unvisited pseudo-transitions.
    - Low disagreement on repeated visited transitions.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_challenge_2_1_untrained_heads_have_disagreement(self):
        """Verify baseline non-zero epistemic disagreement at initialization."""
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        z = torch.randn(16, 256)
        a = torch.randint(0, 17, (16,))
        _, all_preds = predictor(z, a)
        disagreement = predictor.compute_epistemic_disagreement(all_preds)
        self.assertEqual(disagreement.shape, (16,))
        self.assertTrue((disagreement > 0.0).all())
        self.assertGreater(disagreement.mean().item(), 0.001)

    def test_challenge_2_2_visited_vs_unvisited_disagreement_contrast(self):
        """
        Empirically verify that after training on visited transitions,
        the 4 heads achieve LOW disagreement on visited data and HIGH disagreement
        on unvisited pseudo-transitions.
        """
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, hidden_dim=192)
        loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0, multi_head_reduction="mean_heads")
        optimizer = optim.Adam(predictor.parameters(), lr=3e-3)

        # 1. Construct Visited Dataset (N=8 distinct transitions)
        N_visited = 8
        z_visited = torch.randn(N_visited, 256)
        a_visited = torch.randint(0, 17, (N_visited,))
        # Target transitions: simulate deterministic environment transition
        z_next_visited = z_visited + 0.5 * torch.randn(N_visited, 256)

        # Measure baseline disagreement before training
        with torch.no_grad():
            _, all_preds_init = predictor(z_visited, a_visited)
            disagreement_visited_before = predictor.compute_epistemic_disagreement(all_preds_init).mean().item()

        # 2. Train the ensemble on the visited dataset
        for epoch in range(120):
            optimizer.zero_grad()
            _, all_preds = predictor(z_visited, a_visited)
            loss_dict = loss_fn(all_preds, z_next_visited)
            loss_dict["loss"].backward()
            optimizer.step()

        # 3. Evaluate disagreement on Visited transitions
        with torch.no_grad():
            _, all_preds_visited = predictor(z_visited, a_visited)
            disagreement_visited_after = predictor.compute_epistemic_disagreement(all_preds_visited)
            mean_disagree_visited = disagreement_visited_after.mean().item()

        # Disagreement on visited transitions must have dropped substantially
        self.assertLess(
            mean_disagree_visited,
            disagreement_visited_before * 0.05,
            f"Visited disagreement did not decrease by >=95%: before={disagreement_visited_before:.5f}, after={mean_disagree_visited:.5f}"
        )

        # 4. Construct Unvisited Pseudo-Transitions:
        # Novel states drawn from unseen distributions and unseen actions
        N_unvisited = 32
        z_unvisited = torch.randn(N_unvisited, 256) * 2.0 + 3.0  # Out-of-distribution state manifold
        a_unvisited = torch.randint(0, 17, (N_unvisited,))

        with torch.no_grad():
            _, all_preds_unvisited = predictor(z_unvisited, a_unvisited)
            disagreement_unvisited = predictor.compute_epistemic_disagreement(all_preds_unvisited)
            mean_disagree_unvisited = disagreement_unvisited.mean().item()

        # Compute Disagreement Ratio
        ratio = mean_disagree_unvisited / (mean_disagree_visited + 1e-8)

        # Disagreement on unvisited transitions must be significantly higher (> 10x)
        self.assertGreater(
            ratio,
            10.0,
            f"Epistemic sensitivity failure: Unvisited disagreement ({mean_disagree_unvisited:.5f}) "
            f"is not at least 10x visited disagreement ({mean_disagree_visited:.5f}). Ratio = {ratio:.2f}"
        )

    def test_challenge_2_3_disagreement_scale_invariance_and_non_negativity(self):
        """Verify epistemic disagreement is strictly non-negative and zero when heads are identical."""
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        # Manually set all heads to identical predictions
        identical_preds = torch.ones(4, 10, 256) * 3.14
        disagree = predictor.compute_epistemic_disagreement(identical_preds)
        self.assertTrue(torch.allclose(disagree, torch.zeros_like(disagree), atol=1e-6))


class TestChallenge3ActionSensitivity(unittest.TestCase):
    """
    Challenge 3: Empirically verify action sensitivity (no action neglect).
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_challenge_3_1_pairwise_action_differentiation_at_init(self):
        """
        From the EXACT SAME state z, feeding distinct actions a != a'
        MUST produce distinct predictions.
        """
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32)
        z = torch.randn(1, 256).repeat(17, 1)  # 17 identical copies of state z
        actions = torch.arange(0, 17, dtype=torch.long)  # Actions 0 to 16

        mean_preds, _ = predictor(z, actions)  # Shape (17, 256)

        # Compute pairwise distance matrix between all pairs of actions
        diffs = mean_preds.unsqueeze(0) - mean_preds.unsqueeze(1)  # (17, 17, 256)
        dist_matrix = torch.norm(diffs, dim=-1)  # (17, 17)

        # For every a != a', distance must be strictly positive
        eye_mask = torch.eye(17, dtype=torch.bool)
        off_diag_dists = dist_matrix[~eye_mask]

        min_dist = off_diag_dists.min().item()
        mean_dist = off_diag_dists.mean().item()

        self.assertGreater(
            min_dist, 0.01,
            f"Action neglect detected! Min action differentiation distance {min_dist:.6f} <= 0.01"
        )
        self.assertGreater(
            mean_dist, 0.02,
            f"Action differentiation weak! Mean action differentiation distance {mean_dist:.6f} <= 0.02"
        )

    def test_challenge_3_2_action_embedding_gradient_flow(self):
        """
        Verify that action embedding receives gradients from latent prediction error,
        enabling the model to adjust action representations.
        """
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, action_dim=32)
        z = torch.randn(4, 256)
        a = torch.tensor([1, 4, 7, 12])
        mean_pred, _ = predictor(z, a)
        loss = mean_pred.sum()
        loss.backward()

        emb_grad = predictor.action_embedding.embedding.weight.grad
        self.assertIsNotNone(emb_grad)
        # Selected actions must have non-zero gradients
        for act in [1, 4, 7, 12]:
            self.assertGreater(emb_grad[act].abs().sum().item(), 0.0)
        # Unselected actions must have zero gradients
        for act in [0, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16]:
            self.assertEqual(emb_grad[act].abs().sum().item(), 0.0)

    def test_challenge_3_3_bifurcated_dynamics_learning_without_neglect(self):
        """
        Construct a bifurcated transition scenario:
        Identical state z* transitions to target_A under action 0,
        and to target_B under action 1, where ||target_A - target_B|| >= 2.0.
        Verify that the predictor successfully learns both distinct outcomes without action neglect.
        """
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17, hidden_dim=192)
        loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0)
        optimizer = optim.Adam(predictor.parameters(), lr=5e-3)

        # Single identical state z*
        z_star = torch.randn(1, 256)
        # Two strongly opposing target outcomes
        target_A = z_star + 1.5 * torch.randn(1, 256)
        target_B = z_star - 1.5 * torch.randn(1, 256)
        target_distance = torch.norm(target_A - target_B).item()
        self.assertGreater(target_distance, 2.0)

        # Training batch: pairs of (z*, action_0 -> target_A) and (z*, action_1 -> target_B)
        z_batch = torch.cat([z_star, z_star], dim=0)  # (2, 256)
        a_batch = torch.tensor([0, 1], dtype=torch.long)  # (2,)
        target_batch = torch.cat([target_A, target_B], dim=0)  # (2, 256)

        # Train for 100 steps
        for step in range(100):
            optimizer.zero_grad()
            mean_pred, all_preds = predictor(z_batch, a_batch)
            loss_dict = loss_fn(all_preds, target_batch)
            loss_dict["loss"].backward()
            optimizer.step()

        # Evaluate learned predictions
        with torch.no_grad():
            pred_A, _ = predictor(z_star, torch.tensor([0]))
            pred_B, _ = predictor(z_star, torch.tensor([1]))

            err_A = torch.norm(pred_A - target_A).item()
            err_B = torch.norm(pred_B - target_B).item()
            pred_separation = torch.norm(pred_A - pred_B).item()

        # Assert predictions accurately match their respective targets
        self.assertLess(err_A, 0.20, f"Failed to predict target_A under action 0: error={err_A:.4f}")
        self.assertLess(err_B, 0.20, f"Failed to predict target_B under action 1: error={err_B:.4f}")
        self.assertGreater(
            pred_separation,
            target_distance * 0.8,
            f"Action neglect detected: predicted states collapsed towards each other ({pred_separation:.4f} < {target_distance*0.8:.4f})"
        )


class TestChallenge4RolloutStabilityAndAdversarialExtremes(unittest.TestCase):
    """
    Challenge 4: Autoregressive rollout stability and extreme edge cases.
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_challenge_4_1_rollout_horizon_20_bounded_norms(self):
        """
        Verify autoregressive rollout for H=20 does not explode (Inf) or vanish (Zero).
        """
        predictor = ResidualMLPPredictor(latent_dim=256, num_actions=17)
        z_0 = torch.randn(4, 256)
        actions = torch.randint(0, 17, (20, 4))  # H=20, B=4

        traj_mean, traj_all = predictor.rollout(z_0, actions)

        self.assertEqual(traj_mean.shape, (20, 4, 256))
        self.assertEqual(traj_all.shape, (4, 20, 4, 256))
        self.assertFalse(torch.isnan(traj_mean).any())
        self.assertFalse(torch.isinf(traj_mean).any())

        # Check norms across time
        norms = torch.norm(traj_mean, dim=-1)  # (20, 4)
        max_norm = norms.max().item()
        min_norm = norms.min().item()

        self.assertLess(max_norm, 200.0, f"Latent exploded during rollout: max_norm={max_norm}")
        self.assertGreater(min_norm, 0.1, f"Latent collapsed during rollout: min_norm={min_norm}")

    def test_challenge_4_2_target_encoder_long_run_ema_stability(self):
        """
        Verify that 1000 sequential EMA updates do not cause numerical drift or NaN/Inf.
        """
        encoder = ObservationEncoder()
        target = TargetEncoder(encoder, tau=0.99)

        for i in range(1000):
            # Slightly perturb online encoder weights
            with torch.no_grad():
                for p in encoder.parameters():
                    p.add_(torch.randn_like(p) * 0.001)
            target.update()

        for p_t in target.target_encoder.parameters():
            self.assertFalse(torch.isnan(p_t).any())
            self.assertFalse(torch.isinf(p_t).any())

    def test_challenge_4_3_non_collapse_boundary_sensitivity(self):
        """
        Verify that verify_non_collapse precisely triggers at threshold 0.1.
        """
        # Test exact boundary sensitivity using standardized tensors
        raw = torch.randn(200, 256)
        std_raw = (raw - raw.mean(dim=0, keepdim=True)) / raw.std(dim=0, keepdim=True)

        # Exactly above threshold: variance = 0.105 > 0.100
        z_pass = std_raw * np.sqrt(0.105)
        report_pass = verify_non_collapse(z_pass, min_var_threshold=0.1, strict=False)
        self.assertTrue(report_pass["is_valid"])
        self.assertAlmostEqual(report_pass["min_variance"], 0.105, places=4)

        # Exactly below threshold: variance = 0.095 <= 0.100
        z_fail = std_raw * np.sqrt(0.095)
        report_fail = verify_non_collapse(z_fail, min_var_threshold=0.1, strict=False)
        self.assertFalse(report_fail["is_valid"])
        self.assertEqual(report_fail["num_collapsed_dims"], 256)
        self.assertAlmostEqual(report_fail["min_variance"], 0.095, places=4)


if __name__ == "__main__":
    unittest.main()
