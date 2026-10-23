"""
tests/test_rarity_archive.py — Unit & smoke tests for T2 Part 4 (Rarity + Archive Restarts).
Runs on Mac CPU in under 2 seconds.
"""

import os
import random
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.crafter_wrapper import CrafterEnv, patch_crafter_determinism
from models.curiosity import Plan2ExploreCuriosity
from models.jepa_v3 import JEPAv3
from models.rarity import RarityWeightScheduler
from planning.agent_v3 import AgentBeliefV3
from planning.rarity_mpc import CuriosityRarityMPCPlanner
from training.archive import ArchiveRestarter


class TestRarityWeightScheduler(unittest.TestCase):
    def test_weight_computation_formula(self):
        """Verify w_k = base / (count + 10)^decay formula and integer indexing."""
        scheduler = RarityWeightScheduler(
            num_achievements=22,
            base_weight=5.0,
            decay=0.5,
            epsilon=10.0,
        )
        weights = scheduler.get_weights()
        self.assertEqual(len(weights), 22)

        # Count 0: 5.0 / sqrt(10) ≈ 1.5811
        expected_w0 = 5.0 / np.sqrt(10.0)
        self.assertAlmostEqual(weights[0], expected_w0, places=4)

        # Update event on index 3
        event = np.zeros(22, dtype=np.uint8)
        event[3] = 90  # count becomes 90 -> count + 10 = 100 -> sqrt(100) = 10 -> 5.0 / 10 = 0.5
        scheduler.update(event)

        updated_weights = scheduler.get_weights()
        self.assertAlmostEqual(updated_weights[3], 0.5, places=4)
        self.assertAlmostEqual(updated_weights[0], expected_w0, places=4)

    def test_bonus_computation_tensor(self):
        """Verify tensor batch bonus calculation."""
        scheduler = RarityWeightScheduler(num_achievements=22, base_weight=5.0, decay=0.5)
        probs = torch.zeros(4, 22)
        probs[:, 0] = 1.0  # index 0 event with prob 1.0

        w0 = float(scheduler.get_weights()[0])
        bonuses = scheduler.compute_bonus_tensor(probs)
        self.assertEqual(bonuses.shape, (4,))
        for b in bonuses:
            self.assertAlmostEqual(b.item(), w0, places=4)


class TestArchiveRestarter(unittest.TestCase):
    def test_interesting_event_filter(self):
        """Verify only rare events (< threshold) trigger archive interesting."""
        restarter = ArchiveRestarter(rare_threshold=10, archive_size=5, stagnation_threshold=2)
        global_counts = np.array([50] * 22, dtype=np.int64)
        global_counts[7] = 2  # index 7 is rare (< 10)

        event_common = np.zeros(22, dtype=np.uint8)
        event_common[0] = 1
        self.assertFalse(restarter.is_interesting(event_common, global_counts))

        event_rare = np.zeros(22, dtype=np.uint8)
        event_rare[7] = 1
        self.assertTrue(restarter.is_interesting(event_rare, global_counts))
        self.assertEqual(restarter.get_rare_indices(event_rare, global_counts), [7])

    def test_add_and_sample(self):
        """Verify adding snapshots and sampling with belief state."""
        restarter = ArchiveRestarter(rare_threshold=10, archive_size=5, stagnation_threshold=2)
        fake_snap = {"state": "test_state"}
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros(22, dtype=np.uint8)
        mask[7] = 1
        h = np.ones(256, dtype=np.float32) * 0.42

        restarter.add(
            seed=6000,
            step=45,
            snapshot=fake_snap,
            obs=obs,
            unlocked_indices=[7],
            mask=mask,
            h=h,
        )
        self.assertEqual(len(restarter), 1)

        # Stagnation count
        restarter.record_episode_novelty(False)
        do_restart, reason = restarter.should_restart(stagnant_count=1)
        self.assertFalse(do_restart and reason == "stagnation")

        restarter.record_episode_novelty(False)
        do_restart, reason = restarter.should_restart(stagnant_count=2)
        self.assertTrue(do_restart)
        self.assertEqual(reason, "stagnation")

        sampled = restarter.sample()
        self.assertIsNotNone(sampled)
        self.assertEqual(sampled["seed"], 6000)
        self.assertEqual(sampled["step"], 45)
        np.testing.assert_array_equal(sampled["mask"], mask)
        np.testing.assert_array_almost_equal(sampled["h"], h)
        self.assertEqual(restarter.total_restarts, 1)
        self.assertEqual(restarter.episodes_since_novel, 0)

    def test_simulated_200_episode_restart_fraction(self):
        """Verification: simulated 200-episode run has restart fraction 40% +/- 5%."""
        restarter = ArchiveRestarter(
            rare_threshold=10,
            archive_size=50,
            stagnation_threshold=20,
            restart_prob=0.40,
        )
        # Populate archive with at least 1 state so restarts are possible
        fake_snap = {"state": "test_state"}
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        restarter.add(seed=6000, step=10, snapshot=fake_snap, obs=obs, unlocked_indices=[0])

        rng = random.Random(2)
        restarts = 0
        total_episodes = 200

        for ep in range(total_episodes):
            do_restart, _ = restarter.should_restart(stagnant_count=0, rng=rng)
            if do_restart:
                restarts += 1

        fraction = restarts / total_episodes
        # Must be 40% +/- 5% (between 0.35 and 0.45)
        self.assertTrue(
            0.35 <= fraction <= 0.45,
            f"Expected restart fraction in [0.35, 0.45], got {fraction:.3f} ({restarts}/{total_episodes})",
        )

    def test_rarity_aware_eviction_keeps_rarest_mask(self):
        """Verification: eviction test keeps the rarest mask and evicts the least rare entry."""
        restarter = ArchiveRestarter(
            rare_threshold=10,
            archive_size=3,
            stagnation_threshold=20,
        )
        global_counts = np.array([1000] * 22, dtype=np.int64)
        global_counts[7] = 2    # index 7 is rarest (count 2)
        global_counts[5] = 30   # index 5 is medium (count 30)
        global_counts[1] = 1000 # index 1 is common (count 1000)
        global_counts[0] = 500  # index 0 is common (count 500)

        fake_snap = {"state": "test"}
        obs = np.zeros((64, 64, 3), dtype=np.uint8)

        # Add 3 entries to fill archive
        # Entry 0: common (idx 1, count 1000)
        restarter.add(seed=6001, step=10, snapshot=fake_snap, obs=obs, unlocked_indices=[1], global_counts=global_counts)
        # Entry 1: medium (idx 5, count 30)
        restarter.add(seed=6002, step=10, snapshot=fake_snap, obs=obs, unlocked_indices=[5], global_counts=global_counts)
        # Entry 2: rarest (idx 7, count 2)
        restarter.add(seed=6003, step=10, snapshot=fake_snap, obs=obs, unlocked_indices=[7], global_counts=global_counts)

        self.assertEqual(len(restarter), 3)

        # Add 4th entry (idx 0, count 500) -> archive is full, must evict
        restarter.add(seed=6004, step=10, snapshot=fake_snap, obs=obs, unlocked_indices=[0], global_counts=global_counts)

        # Verify: entry carrying idx 7 (rarest count 2) is KEPT
        seeds_in_archive = {e["seed"] for e in restarter._archive}
        self.assertIn(6003, seeds_in_archive, "Sole carrier of rarest mask (seed 6003) was evicted!")
        # Verify: entry carrying idx 1 (least rare, count 1000) was evicted
        self.assertNotIn(6001, seeds_in_archive, "Commonest entry (seed 6001) was not evicted!")

    def test_restore_asserts_belief_mask_and_h_equal_snapshots(self):
        """Verification: restore test asserts belief mask/h equal the snapshot's."""
        restarter = ArchiveRestarter(rare_threshold=10, archive_size=5)
        fake_snap = {"env_step": 77}
        next_obs = np.full((64, 64, 3), 128, dtype=np.uint8)

        # Create known mask and h
        test_mask = np.zeros(22, dtype=np.uint8)
        test_mask[0] = 1
        test_mask[5] = 1
        test_mask[7] = 1

        test_h = np.linspace(-1.0, 1.0, 256, dtype=np.float32)

        restarter.add(
            seed=6050,
            step=77,
            snapshot=fake_snap,
            obs=next_obs,
            unlocked_indices=[7],
            mask=test_mask,
            h=test_h,
        )

        entry = restarter.sample()
        self.assertIsNotNone(entry)

        belief = AgentBeliefV3(recurrent_dim=256)
        # Apply restore logic:
        belief.set_mask(entry["mask"])
        belief.set_h(entry["h"])

        np.testing.assert_array_equal(belief.get_mask(), test_mask)
        np.testing.assert_array_almost_equal(belief.get_h(), test_h)


class TestCuriosityRarityPlannerSmoke(unittest.TestCase):
    def test_planner_step_and_ema_normalization(self):
        """Smoke test: CuriosityRarityMPCPlanner generates valid action and weighted shares."""
        patch_crafter_determinism()
        model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=17)
        curiosity = Plan2ExploreCuriosity(latent_dim=32, recurrent_dim=64, num_actions=17)
        scheduler = RarityWeightScheduler(num_achievements=22)

        planner = CuriosityRarityMPCPlanner(
            model=model,
            curiosity=curiosity,
            rarity_scheduler=scheduler,
            lambda_task=1.0,
            lambda_cur=0.5,
            lambda_rare=1.0,
            lambda_q=0.3,
            cap_q=0.5,
            horizon=2,
            num_candidates=4,
            cem_iterations=1,
            device="cpu",
        )
        belief = AgentBeliefV3(recurrent_dim=64)
        obs = np.zeros((64, 64, 3), dtype=np.uint8)

        action, debug = planner.plan(obs, belief)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 17)
        self.assertIn("total_score", debug)
        self.assertIn("curiosity_bonus_sum", debug)
        self.assertIn("rarity_bonus_sum", debug)
        self.assertIn("unc_bonus_sum", debug)
        self.assertIn("weighted_task_share", debug)
        self.assertIn("weighted_cur_share", debug)
        self.assertIn("weighted_rare_share", debug)
        self.assertIn("weighted_unc_share", debug)
        self.assertIn("ema_task", debug)
        self.assertIn("ema_q", debug)

        # Verify no uncertainty penalty in debug dict
        self.assertNotIn("uncertainty_penalty", debug)


if __name__ == "__main__":
    unittest.main()
