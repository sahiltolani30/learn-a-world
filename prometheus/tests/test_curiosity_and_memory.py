"""
tests/test_curiosity_and_memory.py

Comprehensive Unit and Integration Test Suite for Milestone 3:
- Epistemic Curiosity Engine & RunningMeanStd Welford Normalization
- Latent Random Network Distillation (Latent RND)
- Two-Tier Persistent Replay Buffer
- CheckpointManager Atomic Persistence and Deserialization

Fulfills requirements:
- R2: Tabula rasa intrinsic curiosity, aleatoric noise rejection ("Noisy TV" trap elimination),
      and cross-episode persistent memory.
- R5: Atomic, preemption-resilient checkpointing compatible with local PC and Google Drive.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.curiosity import (
    RunningMeanStd,
    EpistemicCuriosityEngine,
    LatentRND,
)
from src.memory import (
    TwoTierReplayBuffer,
    CheckpointManager,
)
from src.models import (
    ObservationEncoder,
    LatentPredictor,
    TargetEncoder,
)


class TestRunningMeanStd(unittest.TestCase):
    """Tests for Online Welford Running Mean & Standard Deviation Normalizer."""

    def test_initial_values_and_device(self):
        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        self.assertAlmostEqual(rms.mean.item(), 0.0, places=5)
        self.assertAlmostEqual(rms.var.item(), 1.0, places=5)
        self.assertAlmostEqual(rms.count.item(), 0.0, places=5)
        self.assertAlmostEqual(rms.std.item(), 1.0, places=5)

    def test_welford_convergence_on_gaussian_stream(self):
        """Verify that streaming updates accurately recover true mean and variance."""
        np.random.seed(42)
        true_mean = 3.5
        true_std = 2.0
        true_var = true_std ** 2

        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        total_samples = 10_000
        batch_size = 50

        for _ in range(total_samples // batch_size):
            batch = np.random.normal(loc=true_mean, scale=true_std, size=batch_size)
            rms.update(batch)

        self.assertAlmostEqual(rms.mean.item(), true_mean, delta=0.15)
        self.assertAlmostEqual(rms.var.item(), true_var, delta=0.3)
        self.assertAlmostEqual(rms.count.item(), total_samples, delta=1.0)

    def test_parallel_welford_matches_batch_numpy(self):
        """Verify that online updates match direct batch statistics."""
        data_a = np.random.uniform(0.0, 10.0, size=(100,)).astype(np.float32)
        data_b = np.random.uniform(5.0, 15.0, size=(100,)).astype(np.float32)
        full_data = np.concatenate([data_a, data_b], axis=0)

        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        # Update first with A, then with B
        rms.update(data_a)
        rms.update(data_b)

        expected_mean = float(np.mean(full_data))
        expected_var = float(np.var(full_data))

        self.assertAlmostEqual(rms.mean.item(), expected_mean, delta=0.05)
        self.assertAlmostEqual(rms.var.item(), expected_var, delta=0.15)

    def test_numerical_stability_zero_variance_and_single_samples(self):
        """Verify zero variance (identical inputs) does not cause division by zero or NaN."""
        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        identical_batch = torch.ones(50) * 7.0
        rms.update(identical_batch)

        # Standard deviation should not crash or produce NaN
        std = rms.std.item()
        self.assertFalse(np.isnan(std))
        self.assertFalse(np.isinf(std))
        self.assertGreater(std, 0.0)

        # Normalization with zero variance should be finite
        normalized = rms.normalize(identical_batch)
        self.assertFalse(torch.isnan(normalized).any())
        self.assertFalse(torch.isinf(normalized).any())

    def test_normalize_and_denormalize_roundtrip(self):
        """Verify normalize -> denormalize recovers original data."""
        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rms.update(data)

        # Non-centered (scale only)
        norm_scale = rms.normalize(data, center=False)
        rec_scale = rms.denormalize(norm_scale, center=False)
        self.assertTrue(torch.allclose(data, rec_scale, atol=1e-4))

        # Centered
        norm_centered = rms.normalize(data, center=True)
        rec_centered = rms.denormalize(norm_centered, center=True)
        self.assertTrue(torch.allclose(data, rec_centered, atol=1e-4))

    def test_clipping_bounds(self):
        rms = RunningMeanStd(shape=(), clip_max=3.0)
        rms.update(torch.tensor([0.0, 1.0]))
        extreme_data = torch.tensor([1000.0])
        norm = rms.normalize(extreme_data)
        self.assertLessEqual(norm.item(), 3.0)

    def test_state_dict_serialization(self):
        rms1 = RunningMeanStd(shape=(), epsilon=1e-4)
        rms1.update(torch.tensor([2.0, 4.0, 6.0, 8.0]))

        state = rms1.state_dict()
        rms2 = RunningMeanStd(shape=(), epsilon=1e-4)
        rms2.load_state_dict(state)

        self.assertEqual(rms1.mean.item(), rms2.mean.item())
        self.assertEqual(rms1.var.item(), rms2.var.item())
        self.assertEqual(rms1.count.item(), rms2.count.item())


class TestEpistemicCuriosityEngine(unittest.TestCase):
    """Tests for EpistemicCuriosityEngine and Aleatoric Noise Trap Elimination."""

    def setUp(self):
        self.latent_dim = 256
        self.num_heads = 4
        self.engine = EpistemicCuriosityEngine(
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            beta=0.05,
            normalize_by_dim=True,
            clip_reward=10.0,
            use_welford=True,
        )

    def test_epistemic_disagreement_identical_predictions_zero_curiosity(self):
        """When all K heads make identical predictions, disagreement must be identically 0.0."""
        batch_size = 8
        consensus = torch.randn(batch_size, self.latent_dim)
        # Replicate consensus across all K=4 heads
        all_preds = consensus.unsqueeze(0).repeat(self.num_heads, 1, 1)  # (4, 8, 256)

        disagreement = self.engine.compute_epistemic_disagreement(all_preds)
        self.assertEqual(disagreement.shape, (batch_size,))
        self.assertTrue(torch.allclose(disagreement, torch.zeros(batch_size), atol=1e-7))

    def test_noisy_tv_aleatoric_trap_elimination(self):
        """
        Verify elimination of the Noisy TV trap:
        In stochastic environments, aleatoric noise produces identical conditional
        means across predictor heads (Var_k -> 0).
        Novel causal transitions cause head weights to diverge sharply (Var_k >> 0).
        """
        batch_size = 16
        # 1. Noisy TV (aleatoric noise): heads have converged to identical conditional expectation
        learned_mean = torch.randn(batch_size, self.latent_dim)
        noisy_tv_preds = learned_mean.unsqueeze(0).repeat(self.num_heads, 1, 1)

        # 2. Novel Causal Transition: heads produce divergent predictions
        torch.manual_seed(123)
        novel_preds = noisy_tv_preds.clone()
        for k in range(self.num_heads):
            # Divergent head predictions for novel state
            novel_preds[k] += torch.randn(batch_size, self.latent_dim) * (k + 1) * 0.8

        r_noisy_tv = self.engine.compute_epistemic_disagreement(noisy_tv_preds)
        r_novel = self.engine.compute_epistemic_disagreement(novel_preds)

        # Noisy TV reward must be zero
        self.assertTrue(torch.allclose(r_noisy_tv, torch.zeros(batch_size), atol=1e-7))
        # Novel transition reward must be strictly positive and substantial
        self.assertTrue((r_novel > 0.1).all())
        # Causal reward must vastly exceed noisy TV reward
        self.assertGreater(r_novel.mean().item(), 100 * r_noisy_tv.mean().item())

    def test_dimension_normalization_formula(self):
        """Verify normalize_by_dim=True divides by D, matching theoretical formulation."""
        engine_norm = EpistemicCuriosityEngine(
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            normalize_by_dim=True,
            use_welford=False,
        )
        engine_unnorm = EpistemicCuriosityEngine(
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            normalize_by_dim=False,
            use_welford=False,
        )

        all_preds = torch.randn(self.num_heads, 4, self.latent_dim)
        r_norm = engine_norm.compute_epistemic_disagreement(all_preds)
        r_unnorm = engine_unnorm.compute_epistemic_disagreement(all_preds)

        ratio = (r_unnorm / r_norm).mean().item()
        self.assertAlmostEqual(ratio, float(self.latent_dim), delta=0.01)

    def test_single_transition_handling(self):
        """Engine must handle (K, D) shape for single-step online inference."""
        all_preds = torch.randn(self.num_heads, self.latent_dim)
        r = self.engine.compute_epistemic_disagreement(all_preds)
        self.assertEqual(r.dim(), 0)  # Scalar tensor
        self.assertGreater(r.item(), 0.0)

    def test_invalid_head_count_rejection(self):
        """Must reject fewer than 2 heads with ValueError."""
        single_head_preds = torch.randn(1, 4, self.latent_dim)
        with self.assertRaises(ValueError):
            self.engine.compute_epistemic_disagreement(single_head_preds)

    def test_running_welford_normalization_integration(self):
        """Verify intrinsic reward scaling remains stationary under Welford updates."""
        self.engine.train()
        for _ in range(50):
            all_preds = torch.randn(self.num_heads, 16, self.latent_dim) * 2.0
            r_norm = self.engine.compute_intrinsic_reward(all_preds, update_stats=True)
            self.assertTrue((r_norm >= 0.0).all())
            self.assertTrue((r_norm <= self.engine.clip_reward).all())

        # Check that running stats updated
        self.assertGreater(self.engine.running_stats.count.item(), 100)
        self.assertGreater(self.engine.running_stats.var.item(), 0.0)

    def test_reward_blending_formula(self):
        """Verify r_total = r_ext + beta * r_int across tensors, arrays, and scalars."""
        # 1. Tensors
        ext_t = torch.tensor([1.0, 0.0, -1.0])
        int_t = torch.tensor([2.0, 4.0, 6.0])
        total_t = self.engine.blend_rewards(ext_t, int_t, beta=0.1)
        expected_t = torch.tensor([1.2, 0.4, -0.4])
        self.assertTrue(torch.allclose(total_t, expected_t, atol=1e-6))

        # 2. NumPy arrays
        ext_np = np.array([1.0, 2.0], dtype=np.float32)
        int_np = np.array([10.0, 20.0], dtype=np.float32)
        total_np = self.engine.blend_rewards(ext_np, int_np, beta=0.05)
        self.assertTrue(np.allclose(total_np, np.array([1.5, 3.0])))

        # 3. Python floats
        total_float = self.engine.blend_rewards(1.0, 4.0, beta=0.25)
        self.assertAlmostEqual(total_float, 2.0)

    def test_convenience_integration_with_latent_predictor(self):
        """Verify compute_from_predictor integrates directly with LatentPredictor."""
        predictor = LatentPredictor(latent_dim=self.latent_dim, num_actions=17, num_heads=self.num_heads)
        z = torch.randn(4, self.latent_dim)
        a = torch.randint(0, 17, (4,))

        mean_pred, int_reward = self.engine.compute_from_predictor(predictor, z, a)
        self.assertEqual(mean_pred.shape, (4, self.latent_dim))
        self.assertEqual(int_reward.shape, (4,))
        self.assertTrue((int_reward >= 0.0).all())


class TestLatentRND(unittest.TestCase):
    """Tests for Latent Random Network Distillation."""

    def setUp(self):
        self.latent_dim = 256
        self.rnd_dim = 128
        self.rnd = LatentRND(latent_dim=self.latent_dim, rnd_dim=self.rnd_dim)

    def test_target_frozen_requires_grad_false(self):
        """Target network parameters must have requires_grad=False."""
        for param in self.rnd.target_net.parameters():
            self.assertFalse(param.requires_grad)

    def test_predictor_trainable_requires_grad_true(self):
        """Predictor network parameters must have requires_grad=True."""
        predictor_params = list(self.rnd.predictor_parameters())
        self.assertGreater(len(predictor_params), 0)
        for param in predictor_params:
            self.assertTrue(param.requires_grad)

    def test_distillation_loss_decreases_with_training(self):
        """Predictor must learn to distill target output on repeated latents."""
        optimizer = torch.optim.Adam(self.rnd.predictor_parameters(), lr=1e-3)
        z_fixed = torch.randn(8, self.latent_dim)

        initial_loss = self.rnd.compute_loss(z_fixed).item()
        for _ in range(50):
            self.rnd.update_predictor(optimizer, z_fixed)

        final_loss = self.rnd.compute_loss(z_fixed).item()
        self.assertLess(final_loss, initial_loss)

    def test_novelty_score_higher_for_unseen_distribution(self):
        """Trained predictor must exhibit lower novelty on training distribution than unseen latents."""
        optimizer = torch.optim.Adam(self.rnd.predictor_parameters(), lr=2e-3)
        # Train extensively on distribution A (centered at +3.0)
        z_train = torch.randn(32, self.latent_dim) * 0.5 + 3.0
        for _ in range(100):
            self.rnd.update_predictor(optimizer, z_train)

        # Novelty on familiar distribution
        z_familiar = torch.randn(16, self.latent_dim) * 0.5 + 3.0
        novelty_familiar = self.rnd.compute_novelty(z_familiar).mean().item()

        # Novelty on completely unseen distribution (centered at -3.0)
        z_novel = torch.randn(16, self.latent_dim) * 0.5 - 3.0
        novelty_unseen = self.rnd.compute_novelty(z_novel).mean().item()

        self.assertGreater(novelty_unseen, novelty_familiar)


class TestTwoTierReplayBuffer(unittest.TestCase):
    """Tests for Two-Tier Prioritized Replay Buffer."""

    def test_tier1_fifo_insertion_and_overflow(self):
        """Verify Tier 1 circular rolling insertion and capacity limit."""
        tier1_cap = 10
        tier2_cap = 5
        buf = TwoTierReplayBuffer(tier1_capacity=tier1_cap, tier2_capacity=tier2_cap, seed=42)

        # Insert 15 items (exceeding capacity of 10)
        for i in range(15):
            obs = np.full((64, 64, 3), fill_value=i, dtype=np.uint8)
            next_obs = np.full((64, 64, 3), fill_value=i + 1, dtype=np.uint8)
            buf.add(obs, action=i % 17, reward=float(i), next_obs=next_obs, done=False, r_int=0.1)

        # Size must be capped at 10
        self.assertEqual(len(buf), tier1_cap)
        self.assertEqual(buf.t1_size, tier1_cap)

        # The pointer should have wrapped around to 5 (15 % 10)
        self.assertEqual(buf.t1_ptr, 5)

    def test_tier2_milestone_and_novelty_ingestion(self):
        """Verify that milestone flags and top 5% novelty transitions enter Tier 2."""
        buf = TwoTierReplayBuffer(tier1_capacity=100, tier2_capacity=20, novelty_percentile=0.95, seed=42)

        # Regular transitions with low curiosity
        for i in range(30):
            obs = np.zeros((64, 64, 3), dtype=np.uint8)
            buf.add(obs, action=0, reward=0.0, next_obs=obs, done=False, r_int=0.05, is_milestone=False)

        # Milestone transition
        obs_m = np.ones((64, 64, 3), dtype=np.uint8) * 100
        admitted = buf.add(obs_m, action=5, reward=1.0, next_obs=obs_m, done=False, r_int=0.05, is_milestone=True)
        self.assertTrue(admitted)
        self.assertEqual(buf.t2_size, 1)

        # High curiosity transition (> 95th percentile)
        obs_c = np.ones((64, 64, 3), dtype=np.uint8) * 200
        admitted_c = buf.add(obs_c, action=8, reward=0.0, next_obs=obs_c, done=False, r_int=10.0, epistemic_error=10.0)
        self.assertTrue(admitted_c)
        self.assertEqual(buf.t2_size, 2)

    def test_mixed_sampling_70_30_distribution(self):
        """Verify that sample(batch_size, tier2_ratio=0.3) draws exactly 70% Tier 1 and 30% Tier 2."""
        buf = TwoTierReplayBuffer(tier1_capacity=200, tier2_capacity=100, seed=123)

        # Populate Tier 1
        for i in range(100):
            obs = np.full((64, 64, 3), i, dtype=np.uint8)
            buf.add(obs, action=0, reward=0.0, next_obs=obs, done=False, r_int=0.01, is_milestone=False)

        # Populate Tier 2 with milestones
        for i in range(50):
            obs = np.full((64, 64, 3), 200 + i, dtype=np.uint8)
            buf.add(obs, action=1, reward=1.0, next_obs=obs, done=False, r_int=5.0, is_milestone=True)

        batch_size = 60
        batch = buf.sample(batch_size=batch_size, tier2_ratio=0.3)

        is_t2 = batch["is_tier2"]
        n_tier2 = is_t2.sum().item()
        n_tier1 = (~is_t2).sum().item()

        # Exactly 30% of 60 = 18 from Tier 2, 42 from Tier 1
        self.assertEqual(n_tier2, 18)
        self.assertEqual(n_tier1, 42)
        self.assertEqual(batch["obs"].shape, (batch_size, 3, 64, 64))

    def test_fallback_when_tier2_insufficient(self):
        """When Tier 2 has fewer transitions than 30%, buffer must safely draw available items."""
        buf = TwoTierReplayBuffer(tier1_capacity=100, tier2_capacity=20, seed=42)

        # Populate Tier 1 with 50 items
        for i in range(50):
            obs = np.zeros((64, 64, 3), dtype=np.uint8)
            buf.add(obs, action=0, reward=0.0, next_obs=obs, done=False, r_int=0.01)

        # Populate Tier 2 with only 2 items
        for i in range(2):
            obs = np.ones((64, 64, 3), dtype=np.uint8)
            buf.add(obs, action=1, reward=1.0, next_obs=obs, done=False, r_int=5.0, is_milestone=True)

        # Request batch size 20 (which would want 6 Tier 2 items)
        batch = buf.sample(batch_size=20, tier2_ratio=0.3)
        self.assertEqual(batch["obs"].shape, (20, 3, 64, 64))
        # It must draw at most 2 from Tier 2, and the remaining 18 from Tier 1
        self.assertEqual(batch["is_tier2"].sum().item(), 2)

    def test_ram_optimization_lazy_float32_decompression(self):
        """Verify storage is uint8 and output is float32 normalized in [0.0, 1.0]."""
        buf = TwoTierReplayBuffer(tier1_capacity=20, tier2_capacity=10, seed=42)

        # Pass observation in range [0, 255]
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        next_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        buf.add(obs, action=5, reward=1.0, next_obs=next_obs, done=False, r_int=0.5)

        # Internal storage must be uint8
        self.assertEqual(buf.t1_obs.dtype, np.uint8)
        self.assertEqual(buf.t1_next_obs.dtype, np.uint8)

        # Sampled batch must be torch.float32 normalized to [0.0, 1.0]
        batch = buf.sample(batch_size=1)
        self.assertEqual(batch["obs"].dtype, torch.float32)
        self.assertLessEqual(batch["obs"].max().item(), 1.0)
        self.assertGreaterEqual(batch["obs"].min().item(), 0.0)
        self.assertEqual(batch["obs"].shape, (1, 3, 64, 64))

    def test_flexible_input_observation_shapes(self):
        """Handles both (64, 64, 3) channels-last and (3, 64, 64) channels-first."""
        buf = TwoTierReplayBuffer(tier1_capacity=10, tier2_capacity=5)

        # Channels last
        obs_hwc = np.zeros((64, 64, 3), dtype=np.uint8)
        buf.add(obs_hwc, 0, 0.0, obs_hwc, False)

        # Channels first
        obs_chw = np.zeros((3, 64, 64), dtype=np.uint8)
        buf.add(obs_chw, 1, 0.0, obs_chw, False)

        batch = buf.sample(batch_size=2)
        self.assertEqual(batch["obs"].shape, (2, 3, 64, 64))

    def test_buffer_state_dict_roundtrip(self):
        """Verify full buffer serialization and restoration via state_dict."""
        buf1 = TwoTierReplayBuffer(tier1_capacity=50, tier2_capacity=20, seed=42)
        for i in range(15):
            obs = np.full((64, 64, 3), i, dtype=np.uint8)
            buf1.add(obs, action=i % 17, reward=float(i), next_obs=obs, done=False, r_int=float(i), is_milestone=(i % 5 == 0))

        state = buf1.state_dict()

        buf2 = TwoTierReplayBuffer(tier1_capacity=50, tier2_capacity=20, seed=42)
        buf2.load_state_dict(state)

        self.assertEqual(buf1.t1_size, buf2.t1_size)
        self.assertEqual(buf1.t2_size, buf2.t2_size)
        self.assertEqual(buf1.t1_ptr, buf2.t1_ptr)

        # Samples from both buffers should match given identical seeds
        buf1.rng = np.random.RandomState(999)
        buf2.rng = np.random.RandomState(999)
        b1 = buf1.sample(batch_size=4)
        b2 = buf2.sample(batch_size=4)
        self.assertTrue(torch.allclose(b1["obs"], b2["obs"]))
        self.assertTrue(torch.allclose(b1["actions"], b2["actions"]))


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager Atomic Persistence & Resumption."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(checkpoint_dir=self.temp_dir, keep_last_n=3)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_serialization_and_deserialization(self):
        """Verify atomic checkpoint saving and exact parameter restoration."""
        # Initialize modules
        encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        predictor = LatentPredictor(latent_dim=256, num_actions=17)
        target = TargetEncoder(encoder)
        buffer = TwoTierReplayBuffer(tier1_capacity=20, tier2_capacity=10)
        running_stats = RunningMeanStd(shape=())

        # Populate buffer and stats with data
        obs = np.ones((64, 64, 3), dtype=np.uint8) * 42
        buffer.add(obs, action=2, reward=5.0, next_obs=obs, done=False, r_int=1.23, is_milestone=True)
        running_stats.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        components = {
            "encoder": encoder,
            "predictor": predictor,
            "target": target,
            "buffer": buffer,
            "running_stats": running_stats,
        }
        metadata = {"step": 1000, "episode": 25, "achievements": ["collect_wood"]}

        checkpoint_file = os.path.join(self.temp_dir, "checkpoint_step_1000.pt")
        saved_path = self.manager.save(checkpoint_file, components, metadata=metadata, is_best=True)
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "best_checkpoint.pt")))

        # Create fresh models with different random weights
        fresh_encoder = ObservationEncoder(in_channels=3, latent_dim=256)
        fresh_predictor = LatentPredictor(latent_dim=256, num_actions=17)
        fresh_target = TargetEncoder(fresh_encoder)
        fresh_buffer = TwoTierReplayBuffer(tier1_capacity=20, tier2_capacity=10)
        fresh_running_stats = RunningMeanStd(shape=())

        fresh_components = {
            "encoder": fresh_encoder,
            "predictor": fresh_predictor,
            "target": fresh_target,
            "buffer": fresh_buffer,
            "running_stats": fresh_running_stats,
        }

        # Restore from checkpoint
        loaded_meta = self.manager.restore(saved_path, fresh_components)
        self.assertEqual(loaded_meta["step"], 1000)
        self.assertEqual(loaded_meta["episode"], 25)

        # Verify weights match with 0.0 error
        for p1, p2 in zip(encoder.parameters(), fresh_encoder.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        for p1, p2 in zip(predictor.parameters(), fresh_predictor.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Verify buffer and stats restored
        self.assertEqual(fresh_buffer.t1_size, 1)
        self.assertEqual(fresh_buffer.t2_size, 1)
        self.assertAlmostEqual(fresh_running_stats.mean.item(), running_stats.mean.item(), places=5)

    def test_checkpoint_pruning_keeps_last_n(self):
        """Verify older step checkpoints are pruned while preserving keep_last_n."""
        for step in [100, 200, 300, 400, 500]:
            filename = os.path.join(self.temp_dir, f"checkpoint_step_{step}.pt")
            self.manager.save(filename, components={"dummy": torch.tensor([step])})

        # Manager has keep_last_n=3, so only 300, 400, 500 should remain
        remaining = [os.path.basename(f) for f in os.listdir(self.temp_dir) if f.startswith("checkpoint_step_")]
        self.assertEqual(len(remaining), 3)
        self.assertIn("checkpoint_step_300.pt", remaining)
        self.assertIn("checkpoint_step_400.pt", remaining)
        self.assertIn("checkpoint_step_500.pt", remaining)
        self.assertNotIn("checkpoint_step_100.pt", remaining)
        self.assertNotIn("checkpoint_step_200.pt", remaining)

    def test_find_latest_checkpoint(self):
        """Verify find_latest_checkpoint identifies the highest step checkpoint."""
        for step in [50, 150, 300]:
            path = os.path.join(self.temp_dir, f"checkpoint_step_{step}.pt")
            self.manager.save(path, components={"val": torch.tensor(step)})

        latest = self.manager.find_latest_checkpoint(pattern="checkpoint_step_*.pt")
        self.assertIsNotNone(latest)
        self.assertTrue(latest.endswith("checkpoint_step_300.pt"))

    def test_checkpoint_load_nonexistent_raises(self):
        """Loading a non-existent checkpoint path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load(os.path.join(self.temp_dir, "nonexistent.pt"))

    def test_checkpoint_restore_strict_keyerror(self):
        """Restoring with strict=True must raise KeyError if a component is missing in the file."""
        path = os.path.join(self.temp_dir, "partial.pt")
        self.manager.save(path, components={"encoder": torch.tensor([1.0])})

        with self.assertRaises(KeyError):
            self.manager.restore(path, components={"predictor": LatentPredictor(256, 17)}, strict=True)


class TestCuriosityAndMemoryBoundaries(unittest.TestCase):
    """Additional edge cases and boundary conditions across curiosity and memory modules."""

    def test_multidimensional_running_stats(self):
        """Verify RunningMeanStd tracks multidimensional statistics correctly."""
        rms = RunningMeanStd(shape=(3,), epsilon=1e-4)
        data = torch.tensor([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]])
        rms.update(data)

        expected_mean = torch.tensor([2.0, 20.0, 200.0])
        self.assertTrue(torch.allclose(rms.mean, expected_mean, atol=1e-4))
        self.assertEqual(rms.shape, (3,))

    def test_reset_running_stats(self):
        """Verify reset() restores moments to initial defaults."""
        rms = RunningMeanStd(shape=(), epsilon=1e-4)
        rms.update(torch.tensor([10.0, 20.0, 30.0]))
        rms.reset()
        self.assertEqual(rms.mean.item(), 0.0)
        self.assertEqual(rms.var.item(), 1.0)
        self.assertEqual(rms.count.item(), 0.0)

    def test_epistemic_disagreement_backward_gradient_flow(self):
        """Verify gradients flow backward through epistemic disagreement to ensemble predictions."""
        engine = EpistemicCuriosityEngine(latent_dim=64, num_heads=4, use_welford=False)
        all_preds = torch.randn(4, 2, 64, requires_grad=True)

        disagreement = engine.compute_epistemic_disagreement(all_preds)
        loss = disagreement.sum()
        loss.backward()

        self.assertIsNotNone(all_preds.grad)
        self.assertTrue((all_preds.grad.abs().sum(dim=(1, 2)) > 0).all())

    def test_empty_buffer_sample_raises(self):
        """Sampling an empty buffer must raise ValueError."""
        buf = TwoTierReplayBuffer(tier1_capacity=10, tier2_capacity=5)
        with self.assertRaises(ValueError):
            buf.sample(batch_size=4)

    def test_insufficient_samples_raises(self):
        """Requesting more samples than available in buffer must raise ValueError."""
        buf = TwoTierReplayBuffer(tier1_capacity=10, tier2_capacity=5)
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        buf.add(obs, 0, 0.0, obs, False)
        with self.assertRaises(ValueError):
            buf.sample(batch_size=5)

    def test_tier2_priority_replacement_when_full(self):
        """When Tier 2 reaches capacity, lowest error transitions must be replaced by higher ones."""
        buf = TwoTierReplayBuffer(tier1_capacity=20, tier2_capacity=3, seed=42)
        obs = np.zeros((64, 64, 3), dtype=np.uint8)

        # Insert 3 transitions with novelty 1.0, 2.0, 3.0
        buf.add(obs, 0, 0.0, obs, False, is_milestone=True, epistemic_error=1.0)
        buf.add(obs, 1, 0.0, obs, False, is_milestone=True, epistemic_error=2.0)
        buf.add(obs, 2, 0.0, obs, False, is_milestone=True, epistemic_error=3.0)
        self.assertEqual(buf.t2_size, 3)

        # Now insert a much higher novelty transition (10.0)
        buf.add(obs, 3, 0.0, obs, False, is_milestone=True, epistemic_error=10.0)

        # Capacity must remain 3, but the lowest error (1.0) should have been replaced
        self.assertEqual(buf.t2_size, 3)
        self.assertNotIn(1.0, buf.t2_epistemic_errors[:3])
        self.assertIn(10.0, buf.t2_epistemic_errors[:3])


if __name__ == "__main__":
    unittest.main()
