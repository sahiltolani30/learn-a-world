"""
Dynamics Perturbation Wrapper for Phase 4 Test-Time Adaptation.
Simulates secret test-time physics shifts without the agent being explicitly informed:
1. Action Permutation / Control Inversion: secret remapping of action vectors (e.g. inverted controls).
2. Stochastic Slip / Friction Shift: transitions slip into orthogonal directions with probability p.
3. Sticky Actions: previous action repeats stochastically.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym

class DynamicsPerturbationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that perturbs the underlying environment dynamics at test time.
    """

    def __init__(
        self,
        env: gym.Env,
        perturbation_type: str = "invert_controls",
        slip_probability: float = 0.5,
        action_map: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.perturbation_type = perturbation_type
        self.slip_probability = slip_probability
        self.last_action: Optional[int] = None
        self.rng = np.random.default_rng(seed)

        # Default control inversion for 4-action grid environments (0: UP <-> 1: DOWN, 2: LEFT <-> 3: RIGHT)
        if action_map is not None:
            self.action_map = action_map
        elif perturbation_type in ("invert_controls", "action_permutation"):
            num_actions = getattr(env.action_space, "n", 4)
            if num_actions == 4:
                self.action_map = {0: 1, 1: 0, 2: 3, 3: 2}  # Invert UP/DOWN, LEFT/RIGHT
            elif num_actions == 5:  # Push-T (0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
                self.action_map = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}
            else:
                # Circular shift by 1 for arbitrary action space (e.g. Crafter 17 actions)
                self.action_map = {a: (a + 1) % num_actions for a in range(num_actions)}
        else:
            self.action_map = {}

    def set_perturbation(self, perturbation_type: str, slip_probability: Optional[float] = None) -> None:
        """Dynamically toggle or change perturbation mode at test time."""
        self.perturbation_type = perturbation_type
        if slip_probability is not None:
            self.slip_probability = slip_probability

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_action = None
        obs, info = self.env.reset(seed=seed, options=options)
        info["dynamics_perturbed"] = bool(self.perturbation_type != "none")
        info["perturbation_type"] = self.perturbation_type
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        effective_action = int(action)

        if self.perturbation_type in ("invert_controls", "action_permutation"):
            effective_action = self.action_map.get(effective_action, effective_action)

        elif self.perturbation_type == "stochastic_slip":
            if self.rng.random() < self.slip_probability:
                num_actions = getattr(self.env.action_space, "n", 4)
                effective_action = int(self.rng.integers(0, num_actions))

        elif self.perturbation_type == "sticky_actions":
            if self.last_action is not None and self.rng.random() < self.slip_probability:
                effective_action = self.last_action

        self.last_action = int(action)
        obs, reward, terminated, truncated, info = self.env.step(effective_action)

        info["applied_action"] = effective_action
        info["original_action"] = int(action)
        info["dynamics_perturbed"] = bool(self.perturbation_type != "none")

        return obs, reward, terminated, truncated, info
