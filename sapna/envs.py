"""
sapna/envs.py

ParallelCrafterBatch and ZeroCheatWrapper for SAPNA.
Copied from src/env/parallel.py and src/env/wrapper.py with action_repeat support per SAPNA_PLAN.md §7.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import crafter

# ABI & compatibility shim for NumPy 2.x and SciPy internal array_api_compat
try:
    if hasattr(np, "_core") and hasattr(np._core, "_multiarray_umath"):
        if not hasattr(np._core._multiarray_umath, "_blas_supports_fpe"):
            setattr(np._core._multiarray_umath, "_blas_supports_fpe", lambda *args, **kwargs: False)
except Exception:
    pass


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

        assert obs.shape == (3, 64, 64), f"Invalid normalized obs shape: {obs.shape}"
        assert obs.dtype == torch.float32, f"Invalid normalized obs dtype: {obs.dtype}"
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if seed is not None:
            self.env.reset(seed=seed)
            raw_obs = self.env.reset()
        else:
            raw_obs = self.env.reset()

        self._unlocked_achievements.clear()
        obs = self._process_obs(raw_obs)

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
        self, action: Any
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        if getattr(self.env, "_step", None) is None:
            self.reset()

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
    """
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")

    env = crafter.Env(seed=seed, **kwargs)
    wrapped = ZeroCheatWrapper(env=env, seed=seed)

    if frame_stack > 1:
        wrapped = FrameStackWrapper(wrapped, k=frame_stack)

    return wrapped


def _to_uint8(obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Standardize an environment observation into uint8 (3, 64, 64) numpy array.
    Supports PyTorch tensors and NumPy arrays in [0, 1] float or [0, 255] uint8.
    """
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    elif isinstance(obs, np.ndarray):
        arr = obs.copy()
    else:
        arr = np.asarray(obs)

    # Scale floats in [0, 1] to [0, 255] uint8
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.05 and arr.min() >= -0.05:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Ensure shape (C, H, W)
    if arr.ndim == 3 and arr.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))

    return arr


class ParallelCrafterBatch:
    """
    N Crafter environments stepped in lockstep inside the main process.
    All neural-network calls happen OUTSIDE this class, batched by the training loop.
    
    Guarantees:
    - Deterministic per-environment seeding (seed = base_seed + 1000 * i).
    - Auto-reset on episode completion with zero state leaks.
    - Zero terminal-frame corruption: returns pre-reset terminal observation for buffer insertion,
      while internally initializing self.obs[i] with the post-reset frame for the next step.
    - Action repeat support per SAPNA_PLAN.md §7.
    """

    def __init__(self, num_envs: int = 8, base_seed: int = 42, action_repeat: int = 2):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")

        self.num_envs = num_envs
        self.base_seed = base_seed
        self.action_repeat = action_repeat

        # env i gets seed = base_seed + 1000 * i (deterministic, documented)
        self.envs: List[gym.Env] = [
            make_crafter_env(seed=base_seed + 1000 * i) for i in range(num_envs)
        ]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # Observations cross the boundary as uint8 (N, 3, 64, 64) numpy array
        self.obs = np.stack([_to_uint8(self.envs[i].reset()) for i in range(num_envs)])
        self.episode_step = [0] * num_envs
        self.episode_id = [i for i in range(num_envs)]

    def step_single(
        self, i: int, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Un-repeated single environment step for slot i.
        Returns: (terminal_obs, reward, done, info)
        """
        env = self.envs[i]
        nxt, rew, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        terminal_obs = _to_uint8(nxt)

        if done:
            reset_obs = env.reset()
            self.obs[i] = _to_uint8(reset_obs)
        else:
            self.obs[i] = terminal_obs

        return terminal_obs, float(rew), done, info

    def step_with_repeat(
        self, i: int, a: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Per-slot step with action_repeat loop per SAPNA_PLAN.md §7.
        Manages episode_step and episode_id at the repeated transition level.
        """
        cur_step = self.episode_step[i]
        cur_ep = self.episode_id[i]
        r_sum, done, obs, info = 0.0, False, None, None

        for _ in range(self.action_repeat):
            obs, r, done, info = self.step_single(i, a)
            r_sum += float(r)
            if done:
                break

        if done:
            self.episode_step[i] = 0
            self.episode_id[i] += self.num_envs
        else:
            self.episode_step[i] = cur_step + 1

        info["_step_in_ep"] = cur_step
        info["_episode_id"] = cur_ep

        return obs, r_sum, done, info

    def step(
        self, actions: List[int]
    ) -> List[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Step all envs in lockstep with action_repeat. Auto-resets finished envs.
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions for batch step, got {len(actions)}"
            )

        results = []
        for i, a in enumerate(actions):
            results.append(self.step_with_repeat(i, a))

        return results

    def reset(self) -> np.ndarray:
        """
        Reset all N environments to initial states.
        """
        self.obs = np.stack([_to_uint8(self.envs[i].reset()) for i in range(self.num_envs)])
        self.episode_step = [0] * self.num_envs
        return self.obs

    def close(self) -> None:
        """
        Close all N underlying environments.
        """
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
