"""
tests/test_determinism.py — Unit tests for environment determinism and snapshot restore (T2 Part 3).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.crafter_wrapper import CrafterEnv, patch_crafter_determinism


class TestDeterminism(unittest.TestCase):
    def test_patch_crafter_determinism_idempotent(self):
        """Verify that calling patch_crafter_determinism multiple times is idempotent."""
        patch_crafter_determinism()
        patch_crafter_determinism()

    def test_crafter_determinism_smoke(self):
        """Verify that CrafterEnv is 100% deterministic between two fresh instances with same seed."""
        patch_crafter_determinism()
        seed = 4050
        env1 = CrafterEnv(seed=seed)
        env2 = CrafterEnv(seed=seed)
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        self.assertTrue(np.array_equal(o1, o2), "Reset observations diverged!")

        rng = np.random.RandomState(42)
        actions = [int(rng.randint(0, 17)) for _ in range(30)]

        for step, a in enumerate(actions):
            obs1, r1, term1, trunc1, _ = env1.step(a)
            obs2, r2, term2, trunc2, _ = env2.step(a)
            self.assertTrue(np.array_equal(obs1, obs2), f"Observation diverged at step {step}!")
            self.assertEqual(r1, r2, f"Reward diverged at step {step}!")
            self.assertEqual(term1, term2, f"Terminated diverged at step {step}!")
            self.assertEqual(trunc1, trunc2, f"Truncated diverged at step {step}!")
            if term1 or trunc1:
                break

    def test_crafter_snapshot_restore_smoke(self):
        """Verify that CrafterEnv snapshot and restore produces byte-identical rollouts."""
        patch_crafter_determinism()
        seed = 4051
        env = CrafterEnv(seed=seed)
        env.reset()

        # Step forward 20 steps
        for _ in range(20):
            env.step(1)

        # Take snapshot
        snapshot = env.snapshot_state()

        rng = np.random.RandomState(123)
        actions = [int(rng.randint(0, 17)) for _ in range(30)]

        # Rollout path A
        obs_a = []
        rewards_a = []
        for a in actions:
            o, r, term, trunc, _ = env.step(a)
            obs_a.append(o)
            rewards_a.append(r)
            if term or trunc:
                break

        # Restore snapshot and rollout path B
        env.restore_state(snapshot)
        obs_b = []
        rewards_b = []
        for a in actions:
            o, r, term, trunc, _ = env.step(a)
            obs_b.append(o)
            rewards_b.append(r)
            if term or trunc:
                break

        self.assertEqual(len(obs_a), len(obs_b), f"Lengths diverged: {len(obs_a)} vs {len(obs_b)}")
        for i, (oa, ob) in enumerate(zip(obs_a, obs_b)):
            self.assertTrue(np.array_equal(oa, ob), f"Observation mismatch at step {i} after restore!")
        self.assertEqual(rewards_a, rewards_b, "Rewards diverged after restore!")


if __name__ == "__main__":
    unittest.main()
