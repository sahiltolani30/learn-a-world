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
        self.current_episode_achievements: Dict[str, int] = {}
        self.all_episode_achievements: List[Dict[str, int]] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        self._current_reward = 0.0
        self._current_length = 0

    def record_step(self, info: Optional[Dict[str, Any]], reward: float, done: bool) -> None:
        """Capture one step's info and reward. Deepcopies or extracts only evaluator primitives."""
        self._current_reward += float(reward)
        self._current_length += 1
        
        if info is not None:
            # Check for Crafter achievement dict
            if "achievements" in info and isinstance(info["achievements"], dict):
                for ach_name, count in info["achievements"].items():
                    if count > 0:
                        self.current_episode_achievements[ach_name] = max(
                            self.current_episode_achievements.get(ach_name, 0), int(count)
                        )

        if done:
            self.episode_count += 1
            self.all_episode_achievements.append(copy.deepcopy(self.current_episode_achievements))
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            
            self.current_episode_achievements = {}
            self._current_reward = 0.0
            self._current_length = 0

    def reset_episode(self) -> None:
        """Reset internal accumulator on explicit reset."""
        self.current_episode_achievements = {}
        self._current_reward = 0.0
        self._current_length = 0

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of evaluated episodes."""
        return {
            "num_episodes": self.episode_count,
            "mean_reward": float(sum(self.episode_rewards) / max(1, len(self.episode_rewards))),
            "mean_length": float(sum(self.episode_lengths) / max(1, len(self.episode_lengths))),
            "achievement_histories": copy.deepcopy(self.all_episode_achievements),
        }
