"""Observation and transition filter enforcing strict information boundaries.

Prevents any info, inventory, achievement, or semantic metadata from entering
the learning agent, replay buffer, world model, or planner.
"""
from typing import Tuple, Dict, Any
import numpy as np
from src.utils.safety_checks import assert_observation_contract, assert_action_token_contract


class LearnerObservationFilter:
    """Strictly validates and filters raw environment outputs into permitted learner types only."""

    FORBIDDEN_KEYS = {
        "info", "inventory", "achievements", "semantic", "action_names",
        "player_pos", "player_facing", "objects", "map", "health", "food",
        "drink", "energy", "crafter_state"
    }

    @classmethod
    def sanitize_observation(cls, obs: Any) -> np.ndarray:
        """Ensure observation is strictly a uint8 numpy array with shape [64, 64, 3]."""
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.uint8)
        else:
            if obs.dtype != np.uint8:
                obs = obs.astype(np.uint8)
                
        assert_observation_contract(obs)
        return obs

    @classmethod
    def sanitize_step(
        cls,
        obs_raw: Any,
        reward_raw: Any,
        terminated_raw: Any,
        truncated_raw: Any,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """Validate and return strictly (obs_uint8, reward_float, terminated_bool, truncated_bool).
        
        Guarantees that no dictionary, no extra metadata, and no hidden state leaves this function.
        """
        obs = cls.sanitize_observation(obs_raw)
        
        # Scalar reward only
        if isinstance(reward_raw, (dict, list, tuple)):
            raise ValueError(f"Forbidden reward type: {type(reward_raw)}. Reward must be a single scalar.")
        reward = float(reward_raw)
        
        terminated = bool(terminated_raw)
        truncated = bool(truncated_raw)
        
        return obs, reward, terminated, truncated

    @classmethod
    def inspect_against_leakage(cls, obj: Any, visited: set = None) -> None:
        """Recursively inspect an object to verify that none of the forbidden keys or leaks exist."""
        if visited is None:
            visited = set()
            
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() in cls.FORBIDDEN_KEYS:
                    raise AssertionError(f"Information leak detected! Forbidden key '{k}' found in learner data structure.")
                cls.inspect_against_leakage(v, visited)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                cls.inspect_against_leakage(item, visited)
        elif hasattr(obj, "__dict__"):
            cls.inspect_against_leakage(vars(obj), visited)
