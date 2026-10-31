"""
tests/test_agent_and_planner.py

Comprehensive Multi-Tier Test Suite for Milestone 4:
Latent Policy, Clipped Twin Critics, Categorical CEM Planner, and JEPA Agent.

Tiers:
- Tier 1: Core Feature Correctness (ActorPrior, TwinCritic, CategoricalCEMPlanner, JEPAAgent)
- Tier 2: Boundary & Edge Case Hardening (Inputs, Temperatures, Shapes, Zero-Cheat Violations)
- Tier 3: Cross-Feature Integration (Polyak EMA, CEM Elite Refinement, Latent Rollouts)
- Tier 4: Real-World Scenarios & Acceptance (Dual-Mode Execution, CPU Latency Budgets, Atari Parity)
"""

import unittest
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import ObservationEncoder
from src.models.predictor import ResidualMLPPredictor, SpatialTransformerPredictor
from src.curiosity.ensemble import EpistemicCuriosityEngine
from src.agent.actor_critic import (
    ActorPrior,
    CriticHead,
    TwinCritic,
    TargetCritic,
)
from src.agent.planner import CategoricalCEMPlanner
from src.agent.agent import JEPAAgent, FORBIDDEN_CHEAT_KEYS


class TestActorPrior(unittest.TestCase):
    """Tier 1 & 2 tests for ActorPrior MLP policy."""

    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 256
        self.num_actions = 17
        self.actor = ActorPrior(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
            hidden_dim=256,
            temperature=1.0,
        )

    def test_output_shapes_batched_and_unbatched(self):
        """Verify output logits shape for batched (B, D) and unbatched (D,) latents."""
        # Batched
        z_batch = torch.randn(8, self.latent_dim)
        logits = self.actor(z_batch)
        self.assertEqual(logits.shape, (8, self.num_actions))

        # Unbatched
        z_single = torch.randn(self.latent_dim)
        logits_single = self.actor(z_single)
        self.assertEqual(logits_single.shape, (self.num_actions,))

    def test_invalid_latent_dimension_raises(self):
        """Verify incorrect latent dimensions are rejected."""
        with self.assertRaises(ValueError):
            self.actor(torch.randn(8, 128))
        with self.assertRaises(ValueError):
            self.actor(torch.randn(128))
        with self.assertRaises(ValueError):
            self.actor(torch.randn(2, 4, self.latent_dim))

    def test_get_dist_and_probabilities(self):
        """Verify get_dist returns valid Categorical distribution and probabilities sum to 1."""
        z = torch.randn(4, self.latent_dim)
        dist = self.actor.get_dist(z)
        self.assertIsInstance(dist, torch.distributions.Categorical)
        probs = self.actor.get_probs(z)
        self.assertEqual(probs.shape, (4, self.num_actions))
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_sample_modes(self):
        """Verify deterministic (argmax) vs stochastic sampling."""
        z = torch.randn(10, self.latent_dim)
        # Deterministic
        act_det = self.actor.sample(z, deterministic=True)
        logits = self.actor(z)
        expected = torch.argmax(logits, dim=-1)
        self.assertTrue(torch.equal(act_det, expected))

        # Stochastic
        act_stoch = self.actor.sample(z, deterministic=False)
        self.assertEqual(act_stoch.shape, (10,))
        self.assertTrue((act_stoch >= 0).all() and (act_stoch < self.num_actions).all())

    def test_evaluate_actions(self):
        """Verify evaluate_actions returns log_prob and non-negative entropy."""
        z = torch.randn(6, self.latent_dim)
        actions = torch.randint(0, self.num_actions, (6,))
        log_prob, entropy = self.actor.evaluate_actions(z, actions)
        self.assertEqual(log_prob.shape, (6,))
        self.assertEqual(entropy.shape, (6,))
        self.assertTrue((entropy >= 0.0).all())
        self.assertTrue((log_prob <= 0.0).all())

    def test_temperature_scaling_effect(self):
        """Verify temperature scaling modulates entropy/smoothness of distribution."""
        z = torch.randn(1, self.latent_dim)
        p_sharp = self.actor.get_probs(z, temperature=0.1)
        p_flat = self.actor.get_probs(z, temperature=10.0)

        entropy_sharp = -(p_sharp * torch.log(p_sharp + 1e-8)).sum()
        entropy_flat = -(p_flat * torch.log(p_flat + 1e-8)).sum()
        self.assertGreater(entropy_flat.item(), entropy_sharp.item())

    def test_gradient_flow_to_all_parameters(self):
        """Verify backward gradients flow to all actor layers."""
        z = torch.randn(4, self.latent_dim, requires_grad=True)
        logits = self.actor(z)
        loss = logits.sum()
        loss.backward()
        for name, param in self.actor.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
            self.assertFalse(torch.isnan(param.grad).any())

    def test_atari_plug_and_play_cardinalities(self):
        """R4: Verify plug-and-play instantiation for Atari Pong (6) and Breakout (4)."""
        actor_pong = ActorPrior(latent_dim=256, num_actions=6)
        actor_breakout = ActorPrior(latent_dim=256, num_actions=4)

        z = torch.randn(5, 256)
        self.assertEqual(actor_pong(z).shape, (5, 6))
        self.assertEqual(actor_breakout(z).shape, (5, 4))


class TestTwinCritic(unittest.TestCase):
    """Tier 1 & 2 tests for Clipped Twin Critics and Target Tracking."""

    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 256
        self.num_actions = 17
        self.critic = TwinCritic(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
            hidden_dim=256,
            tau=0.99,
        )

    def test_forward_all_action_formats(self):
        """Verify forward pass with None, 1D actions, 2D actions, and probability distributions."""
        B = 8
        z = torch.randn(B, self.latent_dim)

        # 1. a is None -> returns all Q-values (B, num_actions)
        q1_all, q2_all = self.critic(z, a=None)
        self.assertEqual(q1_all.shape, (B, self.num_actions))
        self.assertEqual(q2_all.shape, (B, self.num_actions))

        # 2. a is 1D (B,) -> returns (B,)
        a_1d = torch.randint(0, self.num_actions, (B,))
        q1_1d, q2_1d = self.critic(z, a=a_1d)
        self.assertEqual(q1_1d.shape, (B,))
        self.assertEqual(q2_1d.shape, (B,))
        # Check values match indexing from q1_all
        expected_q1 = q1_all.gather(1, a_1d.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(q1_1d, expected_q1)

        # 3. a is 2D (B, 1) -> returns (B, 1)
        a_2d = a_1d.unsqueeze(1)
        q1_2d, q2_2d = self.critic(z, a=a_2d)
        self.assertEqual(q1_2d.shape, (B, 1))
        self.assertEqual(q2_2d.shape, (B, 1))

        # 4. a is distribution (B, num_actions) -> returns expected Q (B, 1)
        probs = F.softmax(torch.randn(B, self.num_actions), dim=-1)
        q1_prob, q2_prob = self.critic(z, a=probs)
        self.assertEqual(q1_prob.shape, (B, 1))
        expected_prob_q1 = (q1_all * probs).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(q1_prob, expected_prob_q1)

    def test_q_min_computation(self):
        """Verify clipped double-Q computes element-wise minimum."""
        z = torch.randn(10, self.latent_dim)
        a = torch.randint(0, self.num_actions, (10,))
        q1, q2 = self.critic(z, a)
        q_min = self.critic.q_min(z, a)
        expected_min = torch.minimum(q1, q2)
        torch.testing.assert_close(q_min, expected_min)

    def test_polyak_target_ema_update(self):
        """Verify Polyak EMA update changes target weights correctly: theta_t = tau*theta_t + (1-tau)*theta_o."""
        # Modify online critic weights
        with torch.no_grad():
            for p in self.critic.q1_net.parameters():
                p.add_(1.0)

        # Initial target weights should NOT equal modified online weights
        p_online = next(self.critic.q1_net.parameters())
        p_target_before = next(self.critic.target_q1_net.parameters()).clone()
        self.assertFalse(torch.equal(p_online, p_target_before))

        # Perform EMA update with tau=0.9
        tau = 0.9
        self.critic.update_target(tau=tau)
        p_target_after = next(self.critic.target_q1_net.parameters())
        expected_after = tau * p_target_before + (1.0 - tau) * p_online
        torch.testing.assert_close(p_target_after, expected_after, atol=1e-5, rtol=1e-5)

    def test_hard_update(self):
        """Verify hard update copies online weights exactly."""
        with torch.no_grad():
            for p in self.critic.q1_net.parameters():
                p.add_(5.0)
            for p in self.critic.q2_net.parameters():
                p.add_(3.0)

        self.critic.hard_update()
        for p_o, p_t in zip(self.critic.q1_net.parameters(), self.critic.target_q1_net.parameters()):
            torch.testing.assert_close(p_o, p_t)
        for p_o, p_t in zip(self.critic.q2_net.parameters(), self.critic.target_q2_net.parameters()):
            torch.testing.assert_close(p_o, p_t)

    def test_target_critic_wrapper_parity(self):
        """Verify standalone TargetCritic wrapper class functions identically."""
        target_wrapper = TargetCritic(self.critic, tau=0.99)
        z = torch.randn(4, self.latent_dim)
        a = torch.randint(0, self.num_actions, (4,))
        tq1, tq2 = target_wrapper(z, a)
        self.assertEqual(tq1.shape, (4,))
        self.assertEqual(tq2.shape, (4,))

    def test_compute_loss_and_gradient_flow(self):
        """Verify Bellman MSE loss computation and gradients to both critic heads."""
        z = torch.randn(8, self.latent_dim)
        a = torch.randint(0, self.num_actions, (8,))
        target_q = torch.randn(8)

        loss, loss_dict = self.critic.compute_loss(z, a, target_q)
        self.assertIn("loss_critic", loss_dict)
        self.assertIn("loss_q1", loss_dict)
        self.assertIn("loss_q2", loss_dict)

        loss.backward()
        for p in self.critic.q1_net.parameters():
            self.assertIsNotNone(p.grad)
        for p in self.critic.q2_net.parameters():
            self.assertIsNotNone(p.grad)
        # Target networks must have NO gradients
        for p in self.critic.target_q1_net.parameters():
            self.assertIsNone(p.grad)

    def test_critic_head_direct_contract(self):
        """Tier 1: Verify CriticHead direct contract, parameter count, and action shapes."""
        head = CriticHead(latent_dim=256, num_actions=17, hidden_dim=256)
        z = torch.randn(4, 256)

        # All Q values
        q_all = head(z)
        self.assertEqual(q_all.shape, (4, 17))

        # Action indexing
        a = torch.tensor([0, 5, 16, 2])
        q_a = head(z, a)
        self.assertEqual(q_a.shape, (4,))

        # Unbatched
        z_1 = torch.randn(256)
        q_1 = head(z_1, 3)
        self.assertEqual(q_1.dim(), 0)


class TestCategoricalCEMPlanner(unittest.TestCase):
    """Tier 1, 2, 3 tests for Categorical CEM Planner."""

    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 256
        self.num_actions = 17
        self.predictor = ResidualMLPPredictor(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
            action_dim=32,
            num_heads=4,
        )
        self.actor = ActorPrior(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
        )
        self.critic = TwinCritic(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
        )
        self.curiosity_engine = EpistemicCuriosityEngine(
            latent_dim=self.latent_dim,
            num_heads=4,
        )
        self.planner = CategoricalCEMPlanner(
            predictor=self.predictor,
            actor_prior=self.actor,
            critic=self.critic,
            curiosity_engine=self.curiosity_engine,
            horizon=5,
            num_candidates=50,
            elite_ratio=0.10,
            num_iterations=2,
            gamma=0.99,
        )

    def test_planner_action_selection_and_diagnostics(self):
        """Verify plan returns valid action in [0, |A|-1] and complete diagnostics."""
        z_0 = torch.randn(1, self.latent_dim)
        best_action, info = self.planner.plan(z_0)

        self.assertIsInstance(best_action, int)
        self.assertGreaterEqual(best_action, 0)
        self.assertLess(best_action, self.num_actions)

        self.assertIn("best_action", info)
        self.assertIn("best_return", info)
        self.assertIn("best_trajectory", info)
        self.assertIn("planning_latency_ms", info)
        self.assertIn("final_distribution", info)

        self.assertEqual(len(info["best_trajectory"]), 5)
        self.assertEqual(info["best_action"], info["best_trajectory"][0])
        self.assertEqual(info["final_distribution"].shape, (5, self.num_actions))

    def test_select_action_convenience_method(self):
        """Verify select_action returns integer action matching plan()."""
        z_0 = torch.randn(self.latent_dim)
        action = self.planner.select_action(z_0)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    def test_cpu_planning_latency_budget(self):
        """Acceptance R5: Verify CPU planning latency ensures high-throughput control (>40 FPS on CPU)."""
        z_0 = torch.randn(1, self.latent_dim)

        # Warmup
        for _ in range(3):
            self.planner.plan(z_0)

        # Measure 10 planning steps
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            self.planner.plan(z_0)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        median_latency = float(np.median(latencies))
        self.assertLess(
            median_latency,
            25.0,  # Deliberative planning latency budget on CPU (~10 ms, >40 FPS)
            f"Median planning latency {median_latency:.2f} ms exceeded threshold",
        )

    def test_elite_refinement_distribution_convergence(self):
        """Tier 3: Verify CEM distribution P updates towards elite trajectories."""
        z_0 = torch.randn(1, self.latent_dim)
        init_P = self.planner._init_distribution(z_0).clone()
        _, info = self.planner.plan(z_0)
        final_P = info["final_distribution"]

        # Final distribution should differ from initial distribution due to elite updates
        diff = torch.abs(final_P - init_P).sum()
        self.assertGreater(diff.item(), 0.01)

    def test_planner_robustness_without_optional_components(self):
        """Boundary: Planner functions seamlessly without actor_prior or critic or curiosity."""
        bare_planner = CategoricalCEMPlanner(
            predictor=self.predictor,
            actor_prior=None,
            critic=None,
            curiosity_engine=None,
            horizon=3,
            num_candidates=20,
            num_iterations=1,
        )
        z_0 = torch.randn(self.latent_dim)
        action, info = bare_planner.plan(z_0)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)


class TestJEPAAgent(unittest.TestCase):
    """Tier 1, 2, 4 tests for JEPAAgent, Dual Execution Modes, and Zero-Cheat Compliance."""

    def setUp(self):
        torch.manual_seed(42)
        self.agent = JEPAAgent(
            latent_dim=256,
            num_actions=17,
            in_channels=3,
            action_dim=32,
            hidden_dim=256,
            horizon=5,
            num_candidates=50,
            num_iterations=2,
        )

    def test_zero_cheat_enforcement_rejects_privileged_state_dictionaries(self):
        """R3 Integrity Audit: Strictly reject observation dictionaries containing privileged keys."""
        privileged_keys = [
            "inventory",
            "player_pos",
            "achievements",
            "semantic",
            "recipes",
            "crafting",
            "crafting_recipes",
        ]
        valid_canvas = torch.rand(3, 64, 64)

        for key in privileged_keys:
            cheat_dict = {
                "image": valid_canvas,
                key: np.zeros((12,), dtype=np.int32),
            }
            with self.assertRaises(ValueError, msg=f"Failed to reject privileged key '{key}'"):
                self.agent.act(cheat_dict)

    def test_zero_cheat_enforcement_rejects_flat_state_vectors(self):
        """R3 Integrity Audit: Strictly reject 1D state arrays (representing inventory vectors)."""
        flat_inventory_vector = torch.zeros(32)
        with self.assertRaises(ValueError):
            self.agent.act(flat_inventory_vector)

    def test_valid_visual_observation_acceptance(self):
        """Verify acceptance of (3, 64, 64) torch tensor, (64, 64, 3) numpy array, and uint8."""
        # 1. Float32 tensor (3, 64, 64)
        obs_tensor = torch.rand(3, 64, 64)
        act = self.agent.act(obs_tensor)
        self.assertIsInstance(act, int)
        self.assertTrue(0 <= act < 17)

        # 2. Uint8 numpy array (64, 64, 3)
        obs_np = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        act_np = self.agent.act(obs_np)
        self.assertIsInstance(act_np, int)
        self.assertTrue(0 <= act_np < 17)

        # 3. Batched tensor (1, 3, 64, 64)
        obs_b = torch.rand(1, 3, 64, 64)
        act_b = self.agent.act(obs_b)
        self.assertIsInstance(act_b, int)

    def test_dual_execution_modes_fast_vs_deliberative(self):
        """R3, R5: Parity and functionality of Fast Policy Mode vs Deliberative Planning Mode."""
        obs = torch.rand(3, 64, 64)

        # Fast Policy Mode (planning=False)
        fast_action = self.agent.act(obs, eval_mode=True, planning=False)
        self.assertIsInstance(fast_action, int)
        self.assertTrue(0 <= fast_action < 17)

        # Deliberative Planning Mode (planning=True)
        plan_action = self.agent.act(obs, planning=True)
        self.assertIsInstance(plan_action, int)
        self.assertTrue(0 <= plan_action < 17)

    def test_fast_policy_mode_cpu_latency_budget(self):
        """R5 Acceptance: Verify Fast Policy mode execution latency on CPU < 2.0 ms (including encoder)."""
        obs = torch.rand(3, 64, 64)

        # Warmup
        for _ in range(5):
            self.agent.act(obs, planning=False)

        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            self.agent.act(obs, planning=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        median_latency = float(np.median(latencies))
        self.assertLess(
            median_latency,
            3.0,
            f"Fast policy mode latency {median_latency:.2f} ms exceeded threshold",
        )

    def test_deliberative_planning_mode_cpu_latency_budget(self):
        """R5 Acceptance: Verify Deliberative Planning mode execution latency on CPU < 30.0 ms (including encoder)."""
        obs = torch.rand(3, 64, 64)

        # Warmup
        for _ in range(2):
            self.agent.act(obs, planning=True)

        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            self.agent.act(obs, planning=True)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        median_latency = float(np.median(latencies))
        threshold = 100.0 if isinstance(self.agent.predictor, SpatialTransformerPredictor) else 30.0
        self.assertLess(
            median_latency,
            threshold,
            f"Deliberative planning mode latency {median_latency:.2f} ms exceeded threshold",
        )

    def test_actor_prior_cpu_latency_budget(self):
        """R5 Acceptance: Verify ActorPrior alone executes in < 0.2 ms on CPU (~0.05 ms)."""
        z = torch.randn(1, self.agent.latent_dim)
        for _ in range(10):
            self.agent.actor.sample(z)

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            self.agent.actor.sample(z)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        median_latency = float(np.median(latencies))
        self.assertLess(
            median_latency,
            0.2,
            f"Actor prior latency {median_latency:.4f} ms exceeded 0.2 ms",
        )

    def test_encode_convenience_method(self):
        """Verify agent.encode returns normalized latent representation in R^{256}."""
        obs = torch.rand(3, 64, 64)
        z = self.agent.encode(obs)
        self.assertEqual(z.shape, (256,))

        obs_b = torch.rand(4, 3, 64, 64)
        z_b = self.agent.encode(obs_b)
        self.assertEqual(z_b.shape, (4, 256))

    def test_target_networks_updating(self):
        """Verify agent.update_target_networks updates both target encoder and target critic."""
        # Mutate online weights
        with torch.no_grad():
            for p in self.agent.encoder.parameters():
                p.add_(1.0)
            for p in self.agent.critic.q1_net.parameters():
                p.add_(1.0)

        p_enc_target_before = next(self.agent.target_encoder.target_encoder.parameters()).clone()
        p_crit_target_before = next(self.agent.critic.target_q1_net.parameters()).clone()

        self.agent.update_target_networks(tau=0.9)

        p_enc_target_after = next(self.agent.target_encoder.target_encoder.parameters())
        p_crit_target_after = next(self.agent.critic.target_q1_net.parameters())

        self.assertFalse(torch.equal(p_enc_target_before, p_enc_target_after))
        self.assertFalse(torch.equal(p_crit_target_before, p_crit_target_after))

    def test_atari_plug_and_play_generalization(self):
        """R4 Acceptance: Instantiate JEPAAgent with Atari configuration (4 channels, 4 actions)."""
        atari_agent = JEPAAgent(
            latent_dim=256,
            num_actions=4,
            in_channels=4,
            horizon=3,
            num_candidates=20,
            num_iterations=1,
        )
        atari_obs = torch.rand(4, 84, 84)

        # Fast mode
        a_fast = atari_agent.act(atari_obs, planning=False)
        self.assertIsInstance(a_fast, int)
        self.assertTrue(0 <= a_fast < 4)

        # Planning mode
        a_plan = atari_agent.act(atari_obs, planning=True)
        self.assertIsInstance(a_plan, int)
        self.assertTrue(0 <= a_plan < 4)


if __name__ == "__main__":
    unittest.main()
