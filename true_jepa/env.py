"""Minimal pixel/reward-only Crafter adapter.

The adapter deliberately discards the environment's info object at the API
boundary.  This makes it impossible for collection or action selection to read
inventory, coordinates, semantic maps, or achievement arrays accidentally.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class PixelCrafter:
    """Crafter environment exposing only observations, rewards, and done flags."""

    def __init__(self, seed: int, max_steps: int = 1000) -> None:
        import crafter

        self._crafter = crafter
        self.max_steps = int(max_steps)
        self._seed = int(seed)
        self._steps = 0
        self._env = crafter.Env(seed=self._seed)
        self.num_actions = int(self._env.action_space.n)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = int(seed)
            self._env = self._crafter.Env(seed=self._seed)
        self._steps = 0
        obs = self._env.reset()
        return self._validate(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool]:
        obs, reward, terminated, _info = self._env.step(int(action))
        self._steps += 1
        truncated = self._steps >= self.max_steps and not bool(terminated)
        return self._validate(obs), float(reward), bool(terminated), bool(truncated)

    @staticmethod
    def _validate(obs: np.ndarray) -> np.ndarray:
        value = np.asarray(obs, dtype=np.uint8)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB observation, got {value.shape}")
        return np.ascontiguousarray(value)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()
