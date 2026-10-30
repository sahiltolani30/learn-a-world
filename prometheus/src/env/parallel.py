"""
src/env/parallel.py

In-Process Lockstep Batched Crafter Environment Collection (PARALLEL_ENV_PLAN.md).
Owns N Crafter environment instances inside the main process, stepping them in lockstep
while allowing all neural-network perception and action selection calls to be batched
into single tensor operations on GPU/CPU.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import gymnasium as gym

from src.env.wrapper import make_crafter_env, ZeroCheatWrapper


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
    """

    def __init__(self, num_envs: int = 8, base_seed: int = 42):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.num_envs = num_envs
        self.base_seed = base_seed

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

    def step(
        self, actions: List[int]
    ) -> List[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Step all envs in lockstep. Auto-resets finished envs.
        
        Args:
            actions: List of N discrete action integers.
            
        Returns:
            results: List of N tuples: (terminal_obs, reward, done, info).
            The terminal_obs is the pre-reset observation of the step (safe for buffer storage),
            while self.obs[i] is updated to the new episode's initial reset observation if done.
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions for batch step, got {len(actions)}"
            )

        results = []
        for i, env in enumerate(self.envs):
            nxt, rew, term, trunc, info = env.step(actions[i])
            done = bool(term or trunc)
            terminal_obs = _to_uint8(nxt)

            if done:
                reset_obs = env.reset()  # auto-reset; done=True marks the transition terminal
                self.obs[i] = _to_uint8(reset_obs)
                self.episode_step[i] = 0
            else:
                self.obs[i] = terminal_obs
                self.episode_step[i] += 1

            results.append((terminal_obs, float(rew), done, info))

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
