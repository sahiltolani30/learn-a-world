"""One-way evaluator tap for benchmark metrics.

Isolates all environment info (including achievements, inventory, status)
strictly to an external metrics path. This information is completely quarantined
and NEVER reaches the learner, replay buffer, world model, or planner.
"""
from typing import Dict, Any, List, Optional
import copy


class EvaluatorTap:
    """One-way sink for environment info, recording achievement stats and evaluation logs."""

    def __init__(self):
        self.episode_count = 0
        self.all_episode_achievements: List[Dict[str, int]] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Isolated worker accumulators indexed by env_id
        self._worker_rewards: Dict[int, float] = {}
        self._worker_lengths: Dict[int, int] = {}
        self._worker_achievements: Dict[int, Dict[str, int]] = {}

    def record_step(self, info: Optional[Dict[str, Any]], reward: float, done: bool, env_id: int = 0) -> None:
        """Capture one step's info and reward for worker env_id. Fully isolates parallel environments."""
        self._worker_rewards[env_id] = self._worker_rewards.get(env_id, 0.0) + float(reward)
        self._worker_lengths[env_id] = self._worker_lengths.get(env_id, 0) + 1

        if env_id not in self._worker_achievements:
            self._worker_achievements[env_id] = {}

        if info is not None:
            # Check for Crafter achievement dict
            if "achievements" in info and isinstance(info["achievements"], dict):
                for ach_name, count in info["achievements"].items():
                    if count > 0:
                        self._worker_achievements[env_id][ach_name] = max(
                            self._worker_achievements[env_id].get(ach_name, 0), int(count)
                        )

        if done:
            self.episode_count += 1
            self.all_episode_achievements.append(copy.deepcopy(self._worker_achievements[env_id]))
            self.episode_rewards.append(self._worker_rewards[env_id])
            self.episode_lengths.append(self._worker_lengths[env_id])

            self._worker_achievements[env_id] = {}
            self._worker_rewards[env_id] = 0.0
            self._worker_lengths[env_id] = 0

    def reset_episode(self, env_id: int = 0) -> None:
        """Reset internal accumulator on explicit reset for worker env_id."""
        self._worker_achievements[env_id] = {}
        self._worker_rewards[env_id] = 0.0
        self._worker_lengths[env_id] = 0

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of evaluated episodes."""
        return {
            "num_episodes": self.episode_count,
            "mean_reward": float(sum(self.episode_rewards) / max(1, len(self.episode_rewards))),
            "mean_length": float(sum(self.episode_lengths) / max(1, len(self.episode_lengths))),
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "achievement_histories": copy.deepcopy(self.all_episode_achievements),
        }
