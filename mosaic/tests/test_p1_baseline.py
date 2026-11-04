"""P1 baseline adapter tests that do not start a long training run."""

from __future__ import annotations

import inspect

import numpy as np

from mosaic.src.agent import RandomVisualPolicy
from mosaic.src.env_boundary import GymPixelEnv


def test_gym_facade_uses_empty_info_only() -> None:
    environment = GymPixelEnv(seed=4, max_episode_steps=2, permutation_seed=8)
    try:
        observation, reset_info = environment.reset(seed=4)
        assert observation.dtype == np.uint8
        assert reset_info == {}
        next_observation, reward, terminated, truncated, info = environment.step(0)
        assert next_observation.shape == observation.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info == {}
    finally:
        environment.close()


def test_random_baseline_uses_exact_policy_contract() -> None:
    policy = RandomVisualPolicy(17, seed=9)
    assert list(inspect.signature(policy.act).parameters) == [
        "obs_uint8",
        "reward_prev",
        "step",
    ]
    action = policy.act(np.zeros((64, 64, 3), dtype=np.uint8), 0.0, 0)
    assert 0 <= action < 17
