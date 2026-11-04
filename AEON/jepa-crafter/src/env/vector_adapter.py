"""Vectorized multi-environment adapter for stepping N Crafter games in parallel."""
from typing import List, Tuple, Optional, Callable
import numpy as np

from src.env.crafter_adapter import CrafterAdapter
from src.env.opaque_actions import OpaqueActionMapping
from src.env.evaluator_tap import EvaluatorTap
from src.env.learner_observation_filter import LearnerObservationFilter


class VectorCrafterAdapter:
    """Manages N independent Crafter environments running in lockstep on CPU.
    
    Guarantees:
    - All N environments share the exact same OpaqueActionMapping for the run.
    - Each environment operates on its own independent random seed.
    - Outputs are strictly filtered: (obs, reward, terminated, truncated).
    - Auto-resets individual environments upon episode completion.
    """

    def __init__(
        self,
        num_envs: int = 8,
        env_id: str = "CrafterReward-v1",
        reward_mode: str = "scalar",
        seed_start: int = 1000,
        opaque_mapping: Optional[OpaqueActionMapping] = None,
        evaluator_tap: Optional[EvaluatorTap] = None,
        max_episode_steps: int = 10000,
    ):
        self.num_envs = max(1, int(num_envs))
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.seed_counter = seed_start
        self.evaluator_tap = evaluator_tap
        self.max_episode_steps = max_episode_steps

        # Shared opaque action mapping
        if opaque_mapping is None:
            self.action_mapping = OpaqueActionMapping(num_actions=17, seed=seed_start)
        else:
            self.action_mapping = opaque_mapping

        # Create N individual adapters
        self.adapters: List[CrafterAdapter] = []
        for i in range(self.num_envs):
            ad = CrafterAdapter(
                env_id=self.env_id,
                reward_mode=self.reward_mode,
                seed=self.seed_counter,
                opaque_mapping=self.action_mapping,
                evaluator_tap=self.evaluator_tap,
                max_episode_steps=self.max_episode_steps,
                worker_id=i,
            )
            self.adapters.append(ad)
            self.seed_counter += 1

        self.current_obs = np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)

    @property
    def opaque_mapping(self) -> OpaqueActionMapping:
        return self.action_mapping

    def reset(self) -> np.ndarray:
        """Reset all N environments and return batched observations [num_envs, 64, 64, 3]."""
        for i, ad in enumerate(self.adapters):
            obs = ad.reset(seed=self.seed_counter)
            self.seed_counter += 1
            self.current_obs[i] = obs
        return self.current_obs.copy()

    def step(self, action_tokens: List[int]) -> List[Tuple[np.ndarray, float, bool, bool]]:
        """Step all N environments with their respective action tokens.
        
        Returns:
            results: list of N tuples (next_obs, reward, terminated, truncated)
        """
        assert len(action_tokens) == self.num_envs, f"Expected {self.num_envs} actions, got {len(action_tokens)}"
        results = []

        for i, (ad, token) in enumerate(zip(self.adapters, action_tokens)):
            next_obs, reward, terminated, truncated = ad.step(int(token))
            results.append((next_obs, reward, terminated, truncated))

            if terminated or truncated:
                # Auto-reset environment upon completion
                fresh_obs = ad.reset(seed=self.seed_counter)
                self.seed_counter += 1
                self.current_obs[i] = fresh_obs
            else:
                self.current_obs[i] = next_obs

        return results

    def close(self) -> None:
        for ad in self.adapters:
            ad.close()
