"""
tests/test_rarity_archive.py — Unit & smoke tests for T2 Part 4 (Rarity + Archive Restarts).
Runs on Mac CPU in under 2 seconds.
"""

import os
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
        """Verify adding snapshots and sampling."""
        restarter = ArchiveRestarter(rare_threshold=10, archive_size=5, stagnation_threshold=2)
        fake_snap = {"state": "test_state"}
        obs = np.zeros((64, 64, 3), dtype=np.uint8)

        restarter.add(seed=6000, step=45, snapshot=fake_snap, obs=obs, unlocked_indices=[7])
        self.assertEqual(len(restarter), 1)

        # Stagnation count
        restarter.record_episode_novelty(False)
        self.assertFalse(restarter.should_restart())  # 1 < 2
        restarter.record_episode_novelty(False)
        self.assertTrue(restarter.should_restart())   # 2 >= 2

        sampled = restarter.sample()
        self.assertIsNotNone(sampled)
        self.assertEqual(sampled["seed"], 6000)
        self.assertEqual(sampled["step"], 45)
        self.assertEqual(restarter.total_restarts, 1)
        self.assertEqual(restarter.episodes_since_novel, 0)


class TestCuriosityRarityPlannerSmoke(unittest.TestCase):
    def test_planner_step(self):
        """Smoke test: CuriosityRarityMPCPlanner generates valid action in [0, 16]."""
        patch_crafter_determinism()
        model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=17)
        curiosity = Plan2ExploreCuriosity(latent_dim=32, recurrent_dim=64, num_actions=17)
        scheduler = RarityWeightScheduler(num_achievements=22)

        planner = CuriosityRarityMPCPlanner(
            model=model,
            curiosity=curiosity,
            rarity_scheduler=scheduler,
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


if __name__ == "__main__":
    unittest.main()
