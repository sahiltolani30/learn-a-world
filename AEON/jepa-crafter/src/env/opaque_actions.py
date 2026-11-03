"""Opaque action mapping for environment interaction.

Enforces the boundary: the model only ever sees anonymous tokens 0..16.
The adapter translates these tokens to environment actions via a run-seeded permutation.
"""
from typing import List, Dict, Any, Optional
import numpy as np


class OpaqueActionMapping:
    """Manages an anonymous permutation between agent tokens (0..16) and environment action indices (0..16).
    
    The mapping is bijective, fixed for an entire run/checkpoint, and contains no semantic names.
    """
    
    def __init__(self, num_actions: int = 17, seed: Optional[int] = None, permutation: Optional[List[int]] = None):
        self.num_actions = num_actions
        if permutation is not None:
            if sorted(permutation) != list(range(num_actions)):
                raise ValueError(f"Invalid permutation: must be a permutation of 0..{num_actions - 1}")
            self._token_to_env = list(permutation)
        else:
            rng = np.random.RandomState(seed)
            self._token_to_env = list(rng.permutation(num_actions))
            
        self._env_to_token = [0] * num_actions
        for token, env_action in enumerate(self._token_to_env):
            self._env_to_token[env_action] = token

    def token_to_env_action(self, token: int) -> int:
        """Map an agent's opaque token to the environment's action index."""
        if not (0 <= token < self.num_actions):
            raise ValueError(f"Invalid action token {token}, must be in [0, {self.num_actions - 1}]")
        return self._token_to_env[token]

    def env_action_to_token(self, env_action: int) -> int:
        """Map an environment action index back to opaque token (used for evaluation/diagnostics only)."""
        if not (0 <= env_action < self.num_actions):
            raise ValueError(f"Invalid env action {env_action}, must be in [0, {self.num_actions - 1}]")
        return self._env_to_token[env_action]

    @property
    def permutation(self) -> List[int]:
        return list(self._token_to_env)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "permutation": self._token_to_env,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpaqueActionMapping":
        return cls(
            num_actions=data["num_actions"],
            permutation=data["permutation"],
        )
