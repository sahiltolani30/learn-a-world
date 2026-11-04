"""Strict visual-only policy contracts for MOSAIC P0."""

from __future__ import annotations

import numbers
from typing import Protocol, Tuple

import numpy as np


DEFAULT_OBS_SHAPE: Tuple[int, int, int] = (64, 64, 3)


class VisualPolicy(Protocol):
    """The only single-environment policy surface exposed to collection."""

    def act(self, obs_uint8: np.ndarray, reward_prev: float, step: int) -> int:
        ...


def validate_observation(
    obs_uint8: np.ndarray,
    shape: Tuple[int, int, int] = DEFAULT_OBS_SHAPE,
) -> np.ndarray:
    """Validate and return a contiguous uint8 HWC observation."""

    value = np.asarray(obs_uint8)
    expected = tuple(int(item) for item in shape)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(
            f"expected uint8 observation {expected}, got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def validate_reward(reward_prev: float) -> float:
    """Validate the single scalar reward permitted at action time."""

    if isinstance(reward_prev, (dict, list, tuple, np.ndarray)):
        raise TypeError("reward_prev must be a scalar")
    value = float(reward_prev)
    if not np.isfinite(value):
        raise ValueError("reward_prev must be finite")
    return value


def validate_step(step: int) -> int:
    """Validate the public non-negative episode step."""

    if isinstance(step, (bool, np.bool_)) or not isinstance(step, numbers.Integral):
        raise TypeError("step must be an integer")
    value = int(step)
    if value < 0:
        raise ValueError("step must be non-negative")
    return value


def validate_action(action: int, num_actions: int) -> int:
    """Validate an opaque action token without assigning it semantic meaning."""

    if isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral):
        raise TypeError("action must be an integer token")
    value = int(action)
    if value < 0 or value >= int(num_actions):
        raise ValueError(f"action token {value} is outside [0, {int(num_actions)})")
    return value
