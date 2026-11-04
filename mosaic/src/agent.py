"""Visual policy adapters that satisfy the exact MOSAIC action contract."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .api import validate_action, validate_observation, validate_reward, validate_step


class RandomVisualPolicy:
    """Uniform control baseline with the same public interface as MOSAIC."""

    def __init__(self, num_actions: int, seed: Optional[int] = None) -> None:
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, obs_uint8: np.ndarray, reward_prev: float, step: int) -> int:
        """Return an opaque token without reading privileged information."""

        validate_observation(obs_uint8)
        validate_reward(reward_prev)
        validate_step(step)
        return int(self.rng.integers(0, self.num_actions))


class SB3VisualPolicy:
    """Wrap a Stable-Baselines3 policy behind the three-argument API.

    The adapter owns recurrent inference state and episode-start flags. The
    wrapped model sees only the current RGB observation; no environment info
    is passed to it.
    """

    def __init__(self, model: Any, num_actions: int, deterministic: bool = True) -> None:
        self.model = model
        self.num_actions = int(num_actions)
        self.deterministic = bool(deterministic)
        self._state: Any = None

    def reset(self) -> None:
        self._state = None

    def act(self, obs_uint8: np.ndarray, reward_prev: float, step: int) -> int:
        """Predict one opaque token using only the public visual contract."""

        observation = validate_observation(obs_uint8)
        validate_reward(reward_prev)
        current_step = validate_step(step)
        if current_step == 0:
            self.reset()

        try:
            action, self._state = self.model.predict(
                observation,
                state=self._state,
                episode_start=np.asarray([current_step == 0], dtype=bool),
                deterministic=self.deterministic,
            )
        except TypeError:
            action, self._state = self.model.predict(
                observation,
                deterministic=self.deterministic,
            )
        selected = int(np.asarray(action).reshape(-1)[0])
        return validate_action(selected, self.num_actions)
