"""Crafter environment adapter with opaque action mapping and leak protection."""
from typing import Tuple, Optional, Any
import numpy as np
import crafter

from src.env.opaque_actions import OpaqueActionMapping
from src.env.evaluator_tap import EvaluatorTap
from src.env.learner_observation_filter import LearnerObservationFilter
from src.utils.safety_checks import assert_observation_contract, assert_action_token_contract


class CrafterAdapter:
    """Adapts Crafter to the strict learner contract:
    - Normalizes Gym/Gymnasium step/reset signatures.
    - Translates anonymous action tokens (0..16) via an OpaqueActionMapping.
    - Discards all info dicts before returning to the learner.
    - Connects an isolated EvaluatorTap for metrics only.
    - Emits only: (obs, reward, terminated, truncated).
    """

    def __init__(
        self,
        env_id: str = "CrafterReward-v1",
        reward_mode: str = "scalar",
        seed: Optional[int] = None,
        opaque_mapping: Optional[OpaqueActionMapping] = None,
        evaluator_tap: Optional[EvaluatorTap] = None,
        max_episode_steps: int = 10000,
    ):
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        
        # Instantiate raw environment
        # Crafter provides crafter.Env(seed=...)
        # We can pass seed to Crafter Env or crafter.Env()
        self._env = crafter.Env(seed=seed)
        
        # Startup contract checks
        assert self._env.action_space.n == 17, f"Expected 17 actions, got {self._env.action_space.n}"
        
        # Action mapping
        if opaque_mapping is None:
            self.action_mapping = OpaqueActionMapping(num_actions=17, seed=seed)
        else:
            self.action_mapping = opaque_mapping

        self.evaluator_tap = evaluator_tap
        self._step_count = 0
        self._current_obs: Optional[np.ndarray] = None

    @property
    def opaque_mapping(self) -> OpaqueActionMapping:
        return self.action_mapping

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return sanitized uint8 [64, 64, 3] RGB observation."""
        if seed is not None:
            self._env.reset(seed=seed) if hasattr(self._env, "reset") and "seed" in self._env.reset.__code__.co_varnames else None
            # If crafter.Env doesn't take seed in reset:
            if hasattr(self._env, "_world") and hasattr(self._env._world, "reset"):
                pass  # crafter sets seed on construction or reset
            self._env = crafter.Env(seed=seed)
            
        raw_obs = self._env.reset()
        if isinstance(raw_obs, tuple):
            # Gym 0.26+ returns (obs, info)
            raw_obs, _ = raw_obs
            
        obs = LearnerObservationFilter.sanitize_observation(raw_obs)
        self._step_count = 0
        self._current_obs = obs
        
        if self.evaluator_tap is not None:
            self.evaluator_tap.reset_episode()
            
        return obs

    def step(self, action_token: int) -> Tuple[np.ndarray, float, bool, bool]:
        """Execute opaque action token.
        
        Returns:
            next_obs: uint8 [64, 64, 3]
            reward: float32 scalar (forced 0.0 if reward_mode == 'none' or CrafterNoReward)
            terminated: bool
            truncated: bool
        """
        assert_action_token_contract(action_token, num_actions=17)
        env_action = self.action_mapping.token_to_env_action(action_token)
        
        step_result = self._env.step(env_action)
        self._step_count += 1
        
        # Normalize 4-tuple vs 5-tuple
        if len(step_result) == 4:
            next_obs_raw, reward_raw, done_raw, info = step_result
            if self._step_count >= self.max_episode_steps:
                terminated = False
                truncated = True
            else:
                terminated = bool(done_raw)
                truncated = False
        elif len(step_result) == 5:
            next_obs_raw, reward_raw, terminated, truncated, info = step_result
        else:
            raise RuntimeError(f"Unexpected step result length: {len(step_result)}")

        # Adjust reward if reward_mode is none or env is CrafterNoReward
        if self.reward_mode == "none" or "NoReward" in self.env_id:
            reward = 0.0
        else:
            reward = float(reward_raw)

        # Send info to evaluator tap ONLY
        if self.evaluator_tap is not None:
            self.evaluator_tap.record_step(info, reward=float(reward_raw), done=(terminated or truncated))

        # Sanitize and strip all info before returning to learner
        next_obs, reward, terminated, truncated = LearnerObservationFilter.sanitize_step(
            next_obs_raw, reward, terminated, truncated
        )
        self._current_obs = next_obs
        
        return next_obs, reward, terminated, truncated

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
