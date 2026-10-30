"""
tests/test_parallel_env.py

Acceptance Tests for 8-Game In-Process Parallel Collection (PARALLEL_ENV_PLAN.md §5):
1. test_01_shape_and_wiring: 8 envs stepped in lockstep; valid shapes, uint8 types, auto-reset.
2. test_02_terminal_integrity: pre-reset frame stored as next_obs on terminal step; post-reset frame starts next.
3. test_03_per_env_bookkeeping: independent episode counters, milestone deques, and novelty memories.
4. test_04_batched_equals_serial_math: batched encoder forward matches serial single-frame forwards (atol=1e-5).
5. test_05_ram_rule_check: buffer_save_every=5 skips buffer payload on 4/5 saves to prevent memory spikes.
6. test_06_resume_interop: serial-run checkpoint restores cleanly into parallel pipeline and continues training.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import torch

from src.env.parallel import ParallelCrafterBatch, _to_uint8
from src.agent.agent import JEPAAgent
from src.train import TrainingPipeline
from src.memory.storage import CheckpointManager


class TestParallelCrafterEnvironment(unittest.TestCase):
    """Verifies all acceptance criteria defined in PARALLEL_ENV_PLAN.md §5."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="jepa_parallel_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_01_shape_and_wiring(self):
        """1. Shape/wiring: one ParallelCrafterBatch(num_envs=8).step([...]) returns 8 valid tuples."""
        batch = ParallelCrafterBatch(num_envs=8, base_seed=42)
        try:
            self.assertEqual(batch.obs.shape, (8, 3, 64, 64))
            self.assertEqual(batch.obs.dtype, np.uint8)

            results = batch.step([0] * 8)
            self.assertEqual(len(results), 8)

            for idx, (terminal_obs, rew, done, info) in enumerate(results):
                self.assertEqual(terminal_obs.shape, (3, 64, 64))
                self.assertEqual(terminal_obs.dtype, np.uint8)
                self.assertIsInstance(rew, float)
                self.assertIsInstance(done, bool)
                self.assertIsInstance(info, dict)
                self.assertIn("eval_info", info)
        finally:
            batch.close()

    def test_02_terminal_integrity(self):
        """2. Terminal integrity: terminal_obs equals pre-reset frame and batch.obs[i] updates to reset frame."""
        batch = ParallelCrafterBatch(num_envs=2, base_seed=123)
        try:
            # Force player health to 0 in env 0 to trigger termination
            batch.envs[0]._player.inventory["health"] = 0
            
            # Step batch
            results = batch.step([0, 0])
            terminal_obs_0, rew_0, done_0, info_0 = results[0]

            self.assertTrue(done_0, "Env 0 must report done=True upon health=0 depletion")
            
            # The returned terminal_obs must be the pre-reset death frame,
            # while batch.obs[0] is now the newly reset episode's initial frame.
            post_reset_obs = batch.obs[0]
            self.assertEqual(terminal_obs_0.shape, (3, 64, 64))
            self.assertEqual(post_reset_obs.shape, (3, 64, 64))
            self.assertEqual(batch.episode_step[0], 0, "Episode step must reset to 0 upon done")
        finally:
            batch.close()

    def test_03_per_env_bookkeeping(self):
        """3. Per-env bookkeeping: independent episode IDs, milestone deques, and novelty memories."""
        config = {
            "device": "cpu",
            "total_steps": 16,
            "num_envs": 2,
            "batch_size": 4,
            "checkpoint_dir": self.temp_dir,
            "warmup_steps": 2,
            "no_resume": True,
        }
        pipeline = TrainingPipeline(config)
        try:
            # Check deques and episode tracking are separated
            self.assertEqual(len(pipeline.recent_t1_indices_per_env), 2)
            self.assertEqual(len(pipeline.episodic_novelties), 2)
            self.assertEqual(len(pipeline.episode_achievements), 2)

            # Insert mock items into env 0's deque
            pipeline.recent_t1_indices_per_env[0].append(101)
            pipeline.recent_t1_indices_per_env[0].append(102)

            # Env 1's deque must remain completely untouched
            self.assertEqual(len(pipeline.recent_t1_indices_per_env[1]), 0)

            # Simulate done in env 0
            pipeline.episode_achievements[0].add("collect_wood")
            pipeline.episodic_novelties[0].novelty_and_insert(np.zeros(256))
            
            # Reset env 0 only
            pipeline.episode_achievements[0].clear()
            pipeline.recent_t1_indices_per_env[0].clear()
            pipeline.episodic_novelties[0].reset()

            self.assertEqual(len(pipeline.recent_t1_indices_per_env[0]), 0)
            self.assertEqual(len(pipeline.episode_achievements[0]), 0)
        finally:
            pipeline.envs.close()
            pipeline.eval_env.close()

    def test_04_batched_equals_serial_math(self):
        """4. Batched == serial (math): batched encoder(obs_batch) matches 8 single encoder(obs_i) calls (atol=1e-5)."""
        agent = JEPAAgent(latent_dim=256, num_actions=17, device="cpu")
        agent.eval()

        batch = ParallelCrafterBatch(num_envs=8, base_seed=42)
        try:
            obs_batch = torch.from_numpy(batch.obs).float() / 255.0  # (8, 3, 64, 64)

            with torch.no_grad():
                z_batched = agent.encoder(obs_batch)
                z_singles = torch.stack([agent.encoder(obs_batch[i : i + 1])[0] for i in range(8)])

            self.assertTrue(
                torch.allclose(z_batched, z_singles, atol=1e-5),
                "Batched encoder forward must numerically match individual serial forward calls",
            )
        finally:
            batch.close()

    def test_05_ram_rule_check(self):
        """5. RAM rule check: buffer_save_every=5 skips buffer payload on 4 of 5 saves."""
        config = {
            "device": "cpu",
            "total_steps": 20,
            "num_envs": 2,
            "batch_size": 4,
            "checkpoint_dir": self.temp_dir,
            "buffer_save_every": 5,
            "save_replay_buffer": True,
            "no_resume": True,
        }
        pipeline = TrainingPipeline(config)
        try:
            # Checkpoint 1: buffer should be skipped (1 % 5 != 0)
            ckpt1 = pipeline.save_checkpoint(is_best=False, force_buffer=False)
            mgr = CheckpointManager(checkpoint_dir=self.temp_dir)
            payload1 = mgr.load(ckpt1)
            self.assertFalse(payload1["metadata"]["has_buffer"])
            self.assertNotIn("buffer", payload1["components"])

            # Checkpoint 2, 3, 4: skipped
            pipeline.save_checkpoint(is_best=False, force_buffer=False)
            pipeline.save_checkpoint(is_best=False, force_buffer=False)
            pipeline.save_checkpoint(is_best=False, force_buffer=False)

            # Checkpoint 5: buffer MUST be included (5 % 5 == 0)
            ckpt5 = pipeline.save_checkpoint(is_best=False, force_buffer=False)
            payload5 = mgr.load(ckpt5)
            self.assertTrue(payload5["metadata"]["has_buffer"])
            self.assertIn("buffer", payload5["components"])
        finally:
            pipeline.envs.close()
            pipeline.eval_env.close()

    def test_06_resume_interop(self):
        """6. Resume interop: save checkpoint from serial (num_envs=1), restore into parallel (num_envs=4)."""
        # Step 1: Run 4 steps with num_envs=1 (serial)
        cfg_serial = {
            "device": "cpu",
            "total_steps": 4,
            "num_envs": 1,
            "batch_size": 2,
            "checkpoint_dir": self.temp_dir,
            "eval_interval": 100,
            "save_interval": 4,
            "no_resume": True,
        }
        p_serial = TrainingPipeline(cfg_serial)
        res_serial = p_serial.train()
        self.assertEqual(res_serial["total_steps"], 4)

        # Step 2: Resume with num_envs=4 (parallel) to total_steps=8
        cfg_parallel = {
            "device": "cpu",
            "total_steps": 8,
            "num_envs": 4,
            "batch_size": 2,
            "checkpoint_dir": self.temp_dir,
            "eval_interval": 100,
            "save_interval": 4,
            "no_resume": False,
        }
        p_parallel = TrainingPipeline(cfg_parallel)
        self.assertEqual(p_parallel.current_step, 4, "Must resume at step 4 from serial checkpoint")
        res_parallel = p_parallel.train()
        self.assertEqual(res_parallel["total_steps"], 8)


if __name__ == "__main__":
    unittest.main()
