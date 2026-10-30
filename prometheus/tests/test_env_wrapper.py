"""
tests/test_env_wrapper.py

Comprehensive Test Suite for Milestone 2: Zero-Cheat Environment Abstraction & Wrappers.
Covers:
- Feature 6: Zero-Cheat Crafter Wrapper (Tier 1 Features & Tier 2 Boundaries)
- Feature 7: Generalizable Env Interface & Atari ALE Adapter (Tier 1 Features & Tier 2 Boundaries)
- Cross-Interface Parity and Encoder Plug-and-Play (Tier 3 & Tier 4)
- Adversarial Zero-Cheat Leakage Audit

Authoritative Specifications: PROJECT.md, TEST_INFRA.md, ORIGINAL_REQUEST.md (R3, R4)
"""

import unittest
import numpy as np
import torch
import gymnasium as gym

from src.env import (
    ZeroCheatWrapper,
    FrameStackWrapper,
    make_crafter_env,
    AtariEnvWrapper,
    make_atari_env,
    ResetObs,
    FORBIDDEN_CHEAT_KEYS,
    MockAtariEnv,
)
from src.models import ObservationEncoder


class TestZeroCheatCrafterTier1Features(unittest.TestCase):
    """Tier 1: Core Features of ZeroCheatWrapper."""

    def setUp(self):
        self.env = make_crafter_env(seed=123)

    def tearDown(self):
        self.env.close()

    def test_01_crafter_env_instantiation_and_spaces(self):
        """Feature 6: Verify Crafter 17 discrete actions and normalized observation space."""
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 17)
        self.assertEqual(self.env.num_actions, 17)
        self.assertEqual(len(self.env.action_names), 17)
        self.assertIn("noop", self.env.action_names)
        self.assertIn("do", self.env.action_names)
        self.assertIn("place_table", self.env.action_names)

        # Observation space contract
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (3, 64, 64))
        self.assertEqual(self.env.observation_space.dtype, np.float32)

    def test_02_reset_observation_shape_dtype_and_normalization(self):
        """Feature 6: Verify reset observation is a (3, 64, 64) float32 tensor in [0.0, 1.0]."""
        obs = self.env.reset()
        self.assertTrue(torch.is_tensor(obs), "Reset observation must be a torch.Tensor")
        self.assertEqual(obs.shape, (3, 64, 64))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertGreaterEqual(obs.min().item(), 0.0)
        self.assertLessEqual(obs.max().item(), 1.0)

    def test_03_zero_cheat_info_sanitization(self):
        """Feature 6 / R3: Strict zero-cheat guarantee - all privileged keys scrubbed from info."""
        obs = self.env.reset()
        # Step action 5 ('do') repeatedly to trigger potential achievements/inventory changes
        for _ in range(5):
            obs, reward, terminated, truncated, info = self.env.step(5)
            # Assert NONE of the forbidden cheat keys exist in info
            for forbidden in FORBIDDEN_CHEAT_KEYS:
                self.assertNotIn(
                    forbidden,
                    info,
                    f"Zero-cheat violation: privileged key '{forbidden}' found in info!",
                )
                if "eval_info" in info:
                    self.assertNotIn(
                        forbidden,
                        info["eval_info"],
                        f"Zero-cheat violation: privileged key '{forbidden}' found in eval_info!",
                    )

    def test_04_step_contract_gymnasium_5tuple(self):
        """Feature 6: Verify step returns standard (obs, reward, terminated, truncated, info) 5-tuple."""
        self.env.reset()
        result = self.env.step(0)  # noop
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5, "Step must return exactly 5 elements")

        obs, reward, terminated, truncated, info = result
        self.assertTrue(torch.is_tensor(obs))
        self.assertEqual(obs.shape, (3, 64, 64))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_05_safe_logging_metrics(self):
        """Feature 6: Verify safe logging-only metrics are tracked for evaluation plotting."""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertIn("discount", info)
        self.assertIn("achievement_count", info)
        self.assertIn("unlocked_achievements", info)
        self.assertIsInstance(info["achievement_count"], int)
        self.assertIsInstance(info["unlocked_achievements"], list)
        self.assertIn("eval_info", info)


class TestZeroCheatCrafterTier2Boundaries(unittest.TestCase):
    """Tier 2: Boundary and Adversarial Tests for ZeroCheatWrapper."""

    def setUp(self):
        self.env = make_crafter_env(seed=456)

    def tearDown(self):
        self.env.close()

    def test_06_all_17_discrete_actions_executable(self):
        """Boundary: Verify each of the 17 discrete actions can be stepped without error."""
        self.env.reset()
        for a in range(17):
            obs, rew, term, trunc, info = self.env.step(a)
            self.assertEqual(obs.shape, (3, 64, 64))
            self.assertEqual(obs.dtype, torch.float32)
            self.assertGreaterEqual(obs.min().item(), 0.0)
            self.assertLessEqual(obs.max().item(), 1.0)

    def test_07_invalid_action_rejection(self):
        """Boundary: Reject out-of-bounds discrete actions and invalid action types."""
        self.env.reset()
        invalid_actions = [-1, 17, 100, -99, "invalid", 3.14]
        for inv_a in invalid_actions:
            with self.assertRaises(ValueError, msg=f"Action {inv_a} should have been rejected"):
                self.env.step(inv_a)

    def test_08_dual_reset_unpacking_semantics(self):
        """Boundary: ResetObs must support both direct tensor usage and Gymnasium (obs, info) unpacking."""
        # 1. Direct assignment
        obs_direct = self.env.reset()
        self.assertTrue(torch.is_tensor(obs_direct))
        self.assertEqual(obs_direct.shape, (3, 64, 64))

        # 2. Dual unpacking
        obs_unpacked, info_unpacked = self.env.reset()
        self.assertTrue(torch.is_tensor(obs_unpacked))
        self.assertEqual(obs_unpacked.shape, (3, 64, 64))
        self.assertIsInstance(info_unpacked, dict)
        self.assertIn("discount", info_unpacked)

        # 3. Explicit return_info=True
        obs_explicit, info_explicit = self.env.reset(return_info=True)
        self.assertTrue(torch.is_tensor(obs_explicit))
        self.assertIsInstance(info_explicit, dict)

    def test_09_multiple_resets_and_seed_consistency(self):
        """Boundary: Consecutive resets clear achievement tracking and reset step counts."""
        for s in [10, 20, 30]:
            obs = self.env.reset(seed=s)
            self.assertEqual(obs.shape, (3, 64, 64))
            self.assertEqual(self.env.last_info["achievement_count"], 0)
            self.assertEqual(len(self.env.last_info["unlocked_achievements"]), 0)

    def test_10_visual_hud_pixel_presence(self):
        """Feature 6 / R3: Confirm HUD status pixels (rows 52-63) contain visual content."""
        obs = self.env.reset()
        # Rows 52-63 contain vital bars and inventory item count badges
        hud_region = obs[:, 52:64, :]
        self.assertEqual(hud_region.shape, (3, 12, 64))
        # HUD pixels must not be empty/constant
        hud_variance = hud_region.var().item()
        self.assertGreater(
            hud_variance,
            1e-4,
            "HUD region must contain visual variance representing vitals and item counts",
        )


class TestFrameStackWrapperTier1Features(unittest.TestCase):
    """Tier 1: FrameStackWrapper Features."""

    def test_11_framestack_k1_identity(self):
        """Feature 6: FrameStack with k=1 preserves original shape (3, 64, 64)."""
        env = make_crafter_env(seed=42, frame_stack=1)
        obs = env.reset()
        self.assertEqual(obs.shape, (3, 64, 64))
        obs2, _, _, _, _ = env.step(0)
        self.assertEqual(obs2.shape, (3, 64, 64))
        env.close()

    def test_12_framestack_k3_channel_concatenation(self):
        """Feature 6: FrameStack with k=3 stacks frames along channels to (9, 64, 64)."""
        env = make_crafter_env(seed=42, frame_stack=3)
        self.assertEqual(env.observation_space.shape, (9, 64, 64))
        obs = env.reset()
        self.assertEqual(obs.shape, (9, 64, 64))
        self.assertEqual(obs.dtype, torch.float32)

        obs2, rew, term, trunc, info = env.step(1)
        self.assertEqual(obs2.shape, (9, 64, 64))
        env.close()

    def test_13_framestack_temporal_buffer_progression(self):
        """Feature 6: FrameStack shifts frames chronologically: frame_t becomes frame_{t-1}."""
        env = make_crafter_env(seed=42, frame_stack=3)
        obs0 = env.reset()
        # At reset, all 3 frame slots contain identical initial frame
        self.assertTrue(torch.allclose(obs0[0:3], obs0[3:6]))
        self.assertTrue(torch.allclose(obs0[3:6], obs0[6:9]))

        # Take a step
        obs1, _, _, _, _ = env.step(1)
        # Newest frame is at obs1[6:9], previous is at obs1[3:6], oldest at obs1[0:3]
        self.assertTrue(torch.allclose(obs1[0:3], obs0[3:6]))
        self.assertTrue(torch.allclose(obs1[3:6], obs0[6:9]))
        env.close()

    def test_14_framestack_reset_initialization(self):
        """Feature 6: Reset on FrameStack refills the buffer with k identical copies."""
        env = make_crafter_env(seed=42, frame_stack=4)
        env.reset()
        for _ in range(5):
            env.step(0)
        obs = env.reset()
        self.assertEqual(obs.shape, (12, 64, 64))
        for i in range(1, 4):
            self.assertTrue(torch.allclose(obs[0:3], obs[i * 3 : (i + 1) * 3]))
        env.close()

    def test_15_framestack_invalid_k_rejection(self):
        """Boundary: Invalid k < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            make_crafter_env(seed=42, frame_stack=0)
        with self.assertRaises(ValueError):
            make_crafter_env(seed=42, frame_stack=-2)


class TestAtariGeneralizationTier1Features(unittest.TestCase):
    """Tier 1: Atari ALE Wrapper Features and Interface Standardization."""

    def test_16_atari_breakout_discrete_4_actions(self):
        """Feature 7 / R4: Atari Breakout standardizes to 4 actions and (3, 84, 84) obs."""
        env = make_atari_env("Breakout", seed=42)
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 4)
        self.assertEqual(env.num_actions, 4)
        self.assertEqual(env.observation_space.shape, (3, 84, 84))

        obs = env.reset()
        self.assertEqual(obs.shape, (3, 84, 84))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertGreaterEqual(obs.min().item(), 0.0)
        self.assertLessEqual(obs.max().item(), 1.0)
        env.close()

    def test_17_atari_pong_discrete_6_actions(self):
        """Feature 7 / R4: Atari Pong standardizes to 6 actions and (3, 84, 84) obs."""
        env = make_atari_env("Pong", seed=42)
        self.assertEqual(env.action_space.n, 6)
        self.assertEqual(env.num_actions, 6)
        self.assertEqual(env.observation_space.shape, (3, 84, 84))

        obs = env.reset()
        self.assertEqual(obs.shape, (3, 84, 84))
        env.close()

    def test_18_atari_grayscale_option(self):
        """Feature 7: Grayscale option standardizes to 1 channel (1, 84, 84)."""
        env = make_atari_env("Breakout", seed=42, grayscale=True)
        self.assertEqual(env.observation_space.shape, (1, 84, 84))
        obs = env.reset()
        self.assertEqual(obs.shape, (1, 84, 84))
        env.close()

    def test_19_atari_framestack_4_standard(self):
        """Feature 7 / R4: Standard DeepMind Atari 4-frame grayscale stack (4, 84, 84)."""
        env = make_atari_env("Breakout", seed=42, frame_stack=4, grayscale=True)
        self.assertEqual(env.observation_space.shape, (4, 84, 84))
        obs = env.reset()
        self.assertEqual(obs.shape, (4, 84, 84))
        obs2, rew, term, trunc, info = env.step(1)
        self.assertEqual(obs2.shape, (4, 84, 84))
        env.close()

    def test_20_atari_step_contract_and_clean_info(self):
        """Feature 7: Verify Atari step returns 5-tuple matching Crafter contract."""
        env = make_atari_env("Breakout", seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(torch.is_tensor(obs))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        for forbidden in FORBIDDEN_CHEAT_KEYS:
            self.assertNotIn(forbidden, info)
        env.close()


class TestEnvGeneralizationParityTier2CrossInterface(unittest.TestCase):
    """Tier 2 & 3: Cross-Interface Parity and Model Integration."""

    def test_21_crafter_atari_interface_parity(self):
        """Feature 7 / R4: Verify Crafter and Atari environments share identical interface."""
        c_env = make_crafter_env(seed=1)
        a_env = make_atari_env("Breakout", seed=1)

        # Attribute parity
        for attr in ["action_space", "observation_space", "num_actions", "action_names"]:
            self.assertTrue(hasattr(c_env, attr), f"Crafter missing {attr}")
            self.assertTrue(hasattr(a_env, attr), f"Atari missing {attr}")

        # Reset return parity
        c_obs = c_env.reset()
        a_obs = a_env.reset()
        self.assertTrue(torch.is_tensor(c_obs))
        self.assertTrue(torch.is_tensor(a_obs))
        self.assertEqual(c_obs.dtype, torch.float32)
        self.assertEqual(a_obs.dtype, torch.float32)

        # Step return parity
        c_res = c_env.step(0)
        a_res = a_env.step(0)
        self.assertEqual(len(c_res), 5)
        self.assertEqual(len(a_res), 5)
        self.assertEqual(len(c_res), len(a_res))

        c_env.close()
        a_env.close()

    def test_22_crafter_atari_encoder_plug_and_play(self):
        """Feature 1 / R4: Plug-and-play visual encoder integration across Crafter and Atari."""
        # Crafter: 3x64x64
        crafter_env = make_crafter_env(seed=42)
        c_obs = crafter_env.reset()
        c_enc = ObservationEncoder(in_channels=3, latent_dim=256)
        c_z = c_enc(c_obs)
        self.assertEqual(c_z.shape, (256,))

        # Atari: 3x84x84
        atari_env = make_atari_env("Breakout", seed=42)
        a_obs = atari_env.reset()
        a_enc = ObservationEncoder(in_channels=3, latent_dim=256)
        a_z = a_enc(a_obs)
        self.assertEqual(a_z.shape, (256,))

        # Atari 4-frame grayscale: 4x84x84
        atari_4f_env = make_atari_env("Breakout", seed=42, frame_stack=4, grayscale=True)
        a_4f_obs = atari_4f_env.reset()
        a_4f_enc = ObservationEncoder(in_channels=4, latent_dim=256)
        a_4f_z = a_4f_enc(a_4f_obs)
        self.assertEqual(a_4f_z.shape, (256,))

        crafter_env.close()
        atari_env.close()
        atari_4f_env.close()

    def test_23_factory_functions_kwargs_forwarding(self):
        """Boundary: Factory functions forward keyword arguments properly."""
        env = make_crafter_env(seed=999, area=(64, 64), length=500)
        self.assertIsInstance(env, ZeroCheatWrapper)
        obs = env.reset()
        self.assertEqual(obs.shape, (3, 64, 64))
        env.close()

    def test_24_mock_atari_fallback_robustness(self):
        """Boundary: Synthetic MockAtariEnv exhibits complete interface parity."""
        mock = MockAtariEnv("pong", seed=42)
        wrapped = AtariEnvWrapper(mock, image_size=(84, 84))
        self.assertEqual(wrapped.action_space.n, 6)
        obs = wrapped.reset()
        self.assertEqual(obs.shape, (3, 84, 84))
        obs2, rew, term, trunc, info = wrapped.step(0)
        self.assertEqual(obs2.shape, (3, 84, 84))
        self.assertIsInstance(rew, float)
        self.assertIsInstance(term, bool)
        wrapped.close()

    def test_25_adversarial_cheat_leakage_audit(self):
        """R3 Integrity Audit: 100-step trajectory audit verifying zero cheat leakage."""
        env = make_crafter_env(seed=777)
        obs = env.reset()
        rng = np.random.RandomState(777)

        for step_idx in range(100):
            action = int(rng.randint(0, 17))
            obs, reward, terminated, truncated, info = env.step(action)

            # 1. Verify observation is purely visual RGB pixels
            self.assertEqual(obs.shape, (3, 64, 64))
            self.assertEqual(obs.dtype, torch.float32)
            self.assertTrue(obs.min() >= 0.0 and obs.max() <= 1.0)

            # 2. Verify all forbidden keys are completely absent from info
            for forbidden in FORBIDDEN_CHEAT_KEYS:
                self.assertNotIn(
                    forbidden,
                    info,
                    f"Leakage detected at step {step_idx}: key '{forbidden}' found in info!",
                )
                if "eval_info" in info:
                    self.assertNotIn(
                        forbidden,
                        info["eval_info"],
                        f"Leakage detected at step {step_idx}: key '{forbidden}' in eval_info!",
                    )

            if terminated or truncated:
                obs = env.reset()

        env.close()


if __name__ == "__main__":
    unittest.main()
