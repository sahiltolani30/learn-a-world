"""Fast in-process lockstep collection for independent pixel-only environments."""
from __future__ import annotations

from typing import Callable, List, Protocol, Tuple

import numpy as np


class BatchEnvironment(Protocol):
    num_actions: int

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool]: ...

    def close(self) -> None: ...


class PixelBatch:
    """Step N environments in lockstep while keeping terminal frames intact."""

    def __init__(self, factory: Callable[[], BatchEnvironment], num_envs: int) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.factory = factory
        self.num_envs = int(num_envs)
        self.envs: List[BatchEnvironment] = [factory() for _ in range(self.num_envs)]
        self.num_actions = int(self.envs[0].num_actions)
        self.obs = np.stack([self._validate(env.reset()) for env in self.envs])
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_ids = np.arange(self.num_envs, dtype=np.int64)

    @staticmethod
    def _validate(obs: np.ndarray) -> np.ndarray:
        value = np.asarray(obs, dtype=np.uint8)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB observation, got {value.shape}")
        return np.ascontiguousarray(value)

    def step(self, actions: np.ndarray) -> List[Tuple[np.ndarray, float, bool, bool]]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got {actions.shape}")
        results: List[Tuple[np.ndarray, float, bool, bool]] = []
        for index, (env, action) in enumerate(zip(self.envs, actions.tolist())):
            terminal_obs, reward, terminated, truncated = env.step(int(action))
            terminal_obs = self._validate(terminal_obs)
            done = bool(terminated or truncated)
            results.append((terminal_obs, float(reward), bool(terminated), bool(truncated)))
            if done:
                env.close()
                self.envs[index] = self.factory()
                self.obs[index] = self._validate(self.envs[index].reset())
                self.episode_ids[index] += self.num_envs
                self.episode_steps[index] = 0
            else:
                self.obs[index] = terminal_obs
                self.episode_steps[index] += 1
        return results

    def reset(self) -> np.ndarray:
        for index, env in enumerate(self.envs):
            env.close()
            self.envs[index] = self.factory()
        self.obs = np.stack([self._validate(env.reset()) for env in self.envs])
        self.episode_steps.fill(0)
        return self.obs.copy()

    def close(self) -> None:
        for env in self.envs:
            env.close()
