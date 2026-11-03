"""Records per-episode achievement events and exports official achievement rates."""
import csv
import os
from typing import Dict, List, Any, Optional
from src.evaluation.crafter_score import CRAFTER_ACHIEVEMENTS, compute_crafter_score


class AchievementRecorder:
    """Aggregates binary achievement success across episodes (at most once per episode)."""

    def __init__(self):
        self.num_episodes = 0
        self.achievement_counts: Dict[str, int] = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def record_episode(self, episode_achievements: Dict[str, int], total_reward: float = 0.0, length: int = 0) -> None:
        """Record the outcome of one completed episode."""
        self.num_episodes += 1
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)

        for ach in CRAFTER_ACHIEVEMENTS:
            if episode_achievements.get(ach, 0) > 0:
                self.achievement_counts[ach] += 1

    def get_success_rates_pct(self) -> Dict[str, float]:
        """Return percentage of episodes (0 to 100) where each achievement was unlocked."""
        if self.num_episodes == 0:
            return {ach: 0.0 for ach in CRAFTER_ACHIEVEMENTS}

        return {
            ach: (self.achievement_counts[ach] / float(self.num_episodes)) * 100.0
            for ach in CRAFTER_ACHIEVEMENTS
        }

    def get_summary(self) -> Dict[str, Any]:
        """Compute full summary including rates, aggregate Crafter score, mean reward, and mean length."""
        rates = self.get_success_rates_pct()
        score = compute_crafter_score(rates)
        mean_rew = float(sum(self.episode_rewards) / max(1, len(self.episode_rewards)))
        mean_len = float(sum(self.episode_lengths) / max(1, len(self.episode_lengths)))

        return {
            "num_episodes": self.num_episodes,
            "crafter_score": score,
            "mean_reward": mean_rew,
            "mean_length": mean_len,
            "achievement_rates": rates,
        }

    def export_csv(self, file_path: str, seed: Optional[int] = None, run_name: str = "run") -> None:
        """Export achievement rates and score as a CSV row."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        summary = self.get_summary()
        rates = summary["achievement_rates"]

        headers = ["run_name", "seed", "num_episodes", "crafter_score", "mean_reward", "mean_length"] + CRAFTER_ACHIEVEMENTS
        row = {
            "run_name": run_name,
            "seed": seed if seed is not None else 0,
            "num_episodes": self.num_episodes,
            "crafter_score": f"{summary['crafter_score']:.4f}",
            "mean_reward": f"{summary['mean_reward']:.4f}",
            "mean_length": f"{summary['mean_length']:.2f}",
        }
        for ach in CRAFTER_ACHIEVEMENTS:
            row[ach] = f"{rates[ach]:.2f}"

        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
