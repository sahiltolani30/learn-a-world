"""
Gymnasium-compatible wrapper for the Crafter environment.
Provides standard Gymnasium spaces, reset/step return signatures, and seeding.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import crafter

class CrafterEnv(gym.Env):
    """
    Gymnasium wrapper around crafter.Env.
    Observation: Box(0, 255, (64, 64, 3), uint8)
    Action: Discrete(17)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        seed: Optional[int] = None,
        render_size: int = 64,
        max_steps: int = 1000,
        **kwargs: Any,
    ):
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps
        self._env = crafter.Env(seed=seed)
        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._env = crafter.Env(seed=seed)
        obs = self._env.reset()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = False
        info_copy = {
            "inventory": dict(info.get("inventory", {})),
            "achievements": dict(info.get("achievements", {})),
            "discount": float(info.get("discount", 1.0)),
            "semantic": np.copy(info["semantic"]) if "semantic" in info and info["semantic"] is not None else None,
            "player_pos": list(info.get("player_pos", [])) if "player_pos" in info and info["player_pos"] is not None else None,
            "reward": float(reward),
        }
        return obs, float(reward), terminated, truncated, info_copy

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self):
        return None
