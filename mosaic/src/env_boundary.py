"""Crafter boundary that exposes pixels, scalar rewards, and opaque tokens only."""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np

from .api import validate_action, validate_observation


class OpaqueActionPermutation:
    """Run-local token permutation held entirely outside the policy."""

    def __init__(self, num_actions: int, seed: Optional[int] = None) -> None:
        if int(num_actions) < 2:
            raise ValueError("num_actions must be at least two")
        self.num_actions = int(num_actions)
        self._token_to_environment = np.random.default_rng(seed).permutation(
            self.num_actions
        ).astype(np.int64)

    def token_to_environment(self, token: int) -> int:
        """Translate an opaque token at the environment boundary."""

        checked = validate_action(token, self.num_actions)
        return int(self._token_to_environment[checked])

    def provenance(self) -> dict[str, Any]:
        """Return consistency metadata without exposing the permutation itself."""

        digest = hashlib.sha256(self._token_to_environment.tobytes()).hexdigest()
        return {"num_actions": self.num_actions, "mapping_sha256": digest}


class CrafterPixelEnv:
    """Minimal Crafter adapter with no learner-visible metadata."""

    def __init__(
        self,
        seed: Optional[int] = None,
        max_episode_steps: int = 1000,
        permutation_seed: Optional[int] = None,
    ) -> None:
        import crafter

        self._crafter = crafter
        self._seed = seed
        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self._env = crafter.Env(seed=seed)
        self.observation_shape = tuple(self._env.observation_space.shape)
        self.num_actions = int(self._env.action_space.n)
        mapping_seed = seed if permutation_seed is None else permutation_seed
        self.action_mapping = OpaqueActionPermutation(self.num_actions, mapping_seed)
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset and return only the sanitized RGB observation."""

        if seed is not None:
            self._seed = int(seed)
            self._env = self._crafter.Env(seed=self._seed)
        raw = self._env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        self.step_count = 0
        return validate_observation(raw, self.observation_shape)

    def step(self, action_token: int) -> Tuple[np.ndarray, float, bool, bool]:
        """Execute an opaque token and return `(pixels, reward, terminated, truncated)`."""

        token = validate_action(action_token, self.num_actions)
        raw = self._env.step(self.action_mapping.token_to_environment(token))
        if len(raw) != 4:
            raise RuntimeError(f"unexpected Crafter step result length: {len(raw)}")
        next_obs, reward_raw, done_raw, _metadata = raw
        del _metadata
        self.step_count += 1
        terminated = bool(done_raw)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)
        return (
            validate_observation(next_obs, self.observation_shape),
            float(reward_raw),
            terminated,
            truncated,
        )

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


class GymPixelEnv(gym.Env):
    """Gymnasium facade for PPO while keeping `info` empty and non-semantic."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: Optional[int] = None,
        max_episode_steps: int = 1000,
        permutation_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._pixel = CrafterPixelEnv(
            seed=seed,
            max_episode_steps=max_episode_steps,
            permutation_seed=permutation_seed,
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self._pixel.observation_shape,
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(self._pixel.num_actions)
        self.num_actions = self._pixel.num_actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            self._pixel._seed = int(seed)
        observation = self._pixel.reset(seed=seed)
        return observation, {}

    def step(self, action: int):
        observation, reward, terminated, truncated = self._pixel.step(action)
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self._pixel.close()
