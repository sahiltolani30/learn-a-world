"""
src/env/wrapper.py

Zero-Cheat Crafter Environment Wrapper and Frame Stacking Utilities.
Authoritative Specification: PROJECT.md, ORIGINAL_REQUEST.md (R3, R4)

Guarantees:
1. Zero-Cheat Compliance:
   - Strips all privileged internal game state keys from `info` ('inventory', 'achievements',
     'semantic', 'player_pos', internal crafting tables/recipes).
   - Segregates safe logging metrics ('unlocked_achievements', 'achievement_count') strictly
     for evaluator/tensorboard logging.
   - Observation is purely the raw visual RGB canvas (C, H, W) normalized to [0.0, 1.0] float32,
     where HUD vitals and inventory items are visible solely as visual pixels in rows 52-63.
2. Interface Standardization:
   - reset() returns observation tensor (C, H, W) in [0.0, 1.0] float32 (and supports dual (obs, info) unpacking).
   - step(action) returns (obs, reward, terminated, truncated, info) matching Gymnasium 5-tuple contract.
3. FrameStackWrapper:
   - Stacks k consecutive frames along channel dimension (k * C, H, W) for velocity/POMDP resolution.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# ABI & compatibility shim for NumPy 2.x and SciPy internal array_api_compat
try:
    if hasattr(np, "_core") and hasattr(np._core, "_multiarray_umath"):
        if not hasattr(np._core._multiarray_umath, "_blas_supports_fpe"):
            setattr(np._core._multiarray_umath, "_blas_supports_fpe", lambda *args, **kwargs: False)
except Exception:
    pass

import torch
import gymnasium as gym
from gymnasium import spaces
import crafter


FORBIDDEN_CHEAT_KEYS: Set[str] = {
    "inventory",
    "achievements",
    "semantic",
    "player_pos",
    "recipes",
    "crafting",
    "crafting_recipes",
}


class ResetObs(torch.Tensor):
    """
    Subclass of torch.Tensor returned by reset() that satisfies both:
    1. Direct tensor usage: obs = env.reset(); obs.shape -> (C, H, W), encoder(obs)
    2. Gymnasium-style unpacking: obs, info = env.reset()
    """

    @staticmethod
    def __new__(cls, data: torch.Tensor, info: Optional[Dict[str, Any]] = None):
        res = torch.as_tensor(data).as_subclass(cls)
        res._info = info if info is not None else {}
        return res

    def __iter__(self):
        # Enables dual-style unpacking: obs, info = env.reset()
        yield self
        yield self._info

    @property
    def info(self) -> Dict[str, Any]:
        return self._info


class ZeroCheatWrapper(gym.Env):
    """
    Zero-Cheat Wrapper for crafter.Env().
    
    Strictly prevents the agent from accessing internal environment state:
    - No inventory dict access
    - No achievement ground truth dict access
    - No semantic 2D tile map access
    - No player coordinate access
    
    The agent learns affordances and world dynamics solely through visual pixels
    rendered in the 64x64 RGB canvas.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env: Optional[crafter.Env] = None, seed: Optional[int] = None, **crafter_kwargs):
        super().__init__()
        if env is not None:
            self.env = env
        else:
            self.env = crafter.Env(seed=seed, **crafter_kwargs)

        # Crafter has 17 discrete actions
        self.action_space = spaces.Discrete(17)
        # Visual RGB observation canvas normalized to [0.0, 1.0] float32
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32
        )

        self.action_names: List[str] = list(self.env.action_names)
        self.num_actions: int = len(self.action_names)
        self._unlocked_achievements: Set[str] = set()
        self.last_info: Dict[str, Any] = {}

    def _sanitize_info(self, raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrubs all privileged cheat keys from info dictionary and isolates
        safe evaluator-only logging metrics.
        """
        # Track newly unlocked achievements for evaluator logging
        if "achievements" in raw_info and isinstance(raw_info["achievements"], dict):
            newly_unlocked = {
                name for name, count in raw_info["achievements"].items() if count > 0
            }
            self._unlocked_achievements.update(newly_unlocked)

        unlocked_list = sorted(list(self._unlocked_achievements))
        count = len(unlocked_list)

        clean_info: Dict[str, Any] = {
            "discount": float(raw_info.get("discount", 1.0)),
            # Safe logging-only metrics (for evaluation plots/tensorboard; NEVER fed to agent policy)
            "achievement_count": count,
            "unlocked_achievements": unlocked_list,
            "eval_info": {
                "achievement_count": count,
                "unlocked_achievements": unlocked_list,
            },
        }

        # Rigorous zero-cheat verification: ensure no privileged keys leak
        for forbidden in FORBIDDEN_CHEAT_KEYS:
            if forbidden in clean_info:
                del clean_info[forbidden]
            if "eval_info" in clean_info and forbidden in clean_info["eval_info"]:
                del clean_info["eval_info"][forbidden]

        return clean_info

    def _process_obs(self, raw_obs: np.ndarray) -> torch.Tensor:
        """
        Converts raw Crafter uint8 RGB canvas (64, 64, 3) to (3, 64, 64) float32 in [0.0, 1.0].
        Asserts zero cheat guarantee: observation is purely the raw visual RGB canvas.
        """
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)

        if raw_obs.shape != (64, 64, 3):
            raise ValueError(f"Expected Crafter observation shape (64, 64, 3), got {raw_obs.shape}")

        # Permute (H, W, C) -> (C, H, W) and normalize [0, 255] -> [0.0, 1.0] float32
        obs = torch.from_numpy(raw_obs).permute(2, 0, 1).float() / 255.0

        # Zero-cheat verification assertions
        assert obs.shape == (3, 64, 64), f"Invalid normalized obs shape: {obs.shape}"
        assert obs.dtype == torch.float32, f"Invalid normalized obs dtype: {obs.dtype}"
        assert obs.min() >= 0.0 and obs.max() <= 1.0, f"Observation out of [0, 1] range: [{obs.min()}, {obs.max()}]"

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Resets Crafter environment.
        
        Returns:
            obs: Observation tensor (3, 64, 64) normalized to [0.0, 1.0] float32.
                 Also unpackable as (obs, info) for Gymnasium compatibility.
        """
        if seed is not None:
            self.env._seed = seed

        raw_obs = self.env.reset()
        obs = self._process_obs(raw_obs)
        self._unlocked_achievements.clear()

        clean_info = {
            "discount": 1.0,
            "achievement_count": 0,
            "unlocked_achievements": [],
            "eval_info": {
                "achievement_count": 0,
                "unlocked_achievements": [],
            },
        }
        self.last_info = clean_info

        if return_info:
            return obs, clean_info
        return ResetObs(obs, clean_info)

    def step(
        self, action: Union[int, np.integer, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Steps Crafter environment with discrete action a in [0, 16].
        
        Returns:
            obs: Tensor (3, 64, 64) float32 in [0.0, 1.0]
            reward: float
            terminated: bool (player death)
            truncated: bool (episode step limit reached)
            info: Dict scrubbed of all privileged state
        """
        # Auto-reset if step is called before reset
        if getattr(self.env, "_step", None) is None:
            self.reset()

        # Validate discrete action
        if isinstance(action, (float, np.floating)):
            raise ValueError(f"Action must be an integer, got float {action}")

        if isinstance(action, torch.Tensor):
            if action.is_floating_point():
                raise ValueError(f"Action must be integer tensor, got float tensor {action}")
            if action.numel() != 1:
                raise ValueError(f"Action tensor must be scalar or 1-element, got shape {action.shape}")
            action = int(action.item())
        elif isinstance(action, (int, np.integer)):
            action = int(action)
        else:
            raise ValueError(f"Action must be an integer, got {type(action)}")

        if action < 0 or action >= self.num_actions:
            raise ValueError(
                f"Action {action} out of bounds for Crafter Discrete({self.num_actions})"
            )

        raw_obs, reward, done, raw_info = self.env.step(action)
        obs = self._process_obs(raw_obs)

        # Crafter: dead = player.health <= 0, over = step >= length
        dead = bool(raw_info.get("discount", 1.0) == 0.0)
        terminated = dead
        truncated = bool(done and not dead)

        clean_info = self._sanitize_info(raw_info)
        self.last_info = clean_info

        return obs, float(reward), terminated, truncated, clean_info

    def render(self):
        return self.env.render() if hasattr(self.env, "render") else None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class FrameStackWrapper(gym.Wrapper):
    """
    Frame stacking wrapper for POMDP velocity and temporal dynamics resolution.
    Stacks k consecutive observation frames along channel dimension (k * C, H, W).
    """

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        if k < 1:
            raise ValueError(f"frame_stack k must be >= 1, got {k}")
        self.k: int = k
        self.frames: deque = deque(maxlen=k)

        orig_shape = env.observation_space.shape
        c, h, w = orig_shape[0], orig_shape[1], orig_shape[2]
        self.c: int = c
        self.h: int = h
        self.w: int = w

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k * c, h, w),
            dtype=np.float32,
        )

    def _get_stacked_obs(self) -> torch.Tensor:
        assert len(self.frames) == self.k, f"Expected {self.k} frames, got {len(self.frames)}"
        return torch.cat(list(self.frames), dim=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        res = self.env.reset(seed=seed, options=options)
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs = res
            info = getattr(obs, "_info", {}) if hasattr(obs, "_info") else {}

        # Fill frame buffer with k copies of the initial observation
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.clone() if isinstance(obs, torch.Tensor) else torch.tensor(obs))

        stacked = self._get_stacked_obs()
        if return_info:
            return stacked, info
        return ResetObs(stacked, info)

    def step(
        self, action: Any
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs.clone() if isinstance(obs, torch.Tensor) else torch.tensor(obs))
        stacked = self._get_stacked_obs()
        return stacked, reward, terminated, truncated, info


def make_crafter_env(
    seed: Optional[int] = None,
    frame_stack: int = 1,
    **kwargs,
) -> gym.Env:
    """
    Factory function to construct a standardized Zero-Cheat Crafter environment.
    
    Args:
        seed: Random seed for world generation.
        frame_stack: Number of frames to stack (1 = no stacking, 3 or 4 for temporal dynamics).
        **kwargs: Additional parameters forwarded to crafter.Env (e.g. area=(64, 64), length=10000).
        
    Returns:
        Zero-cheat wrapped environment producing normalized float32 observation tensors.
    """
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")

    env = crafter.Env(seed=seed, **kwargs)
    wrapped = ZeroCheatWrapper(env=env, seed=seed)

    if frame_stack > 1:
        wrapped = FrameStackWrapper(wrapped, k=frame_stack)

    return wrapped
