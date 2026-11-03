"""Episode storage and serialization for held-out validation and offline datasets."""
import os
from typing import List, Dict, Any, Optional
import numpy as np


class Episode:
    """A single complete trajectory."""

    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []

    def add(self, obs: np.ndarray, action: int, reward: float, terminated: bool, truncated: bool) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def set_final_observation(self, final_obs: np.ndarray) -> None:
        self.observations.append(final_obs)

    def __len__(self) -> int:
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "observations": np.stack(self.observations, axis=0),
            "actions": np.array(self.actions, dtype=np.uint8),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "terminated": np.array(self.terminated, dtype=bool),
            "truncated": np.array(self.truncated, dtype=bool),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        ep = cls(episode_id=int(data["episode_id"]))
        obs_arr = data["observations"]
        for o in obs_arr[:-1]:
            ep.observations.append(o)
        ep.actions = list(data["actions"])
        ep.rewards = list(data["rewards"])
        ep.terminated = list(data["terminated"])
        ep.truncated = list(data["truncated"])
        ep.observations.append(obs_arr[-1])
        return ep


class EpisodeStore:
    """Manages collections of full episodes (e.g., held-out validation set)."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        self.episodes: List[Episode] = []
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def save(self, file_path: str) -> None:
        """Save all stored episodes into a single compressed npz archive."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_to_save = {}
        for i, ep in enumerate(self.episodes):
            d = ep.to_dict()
            for k, v in d.items():
                data_to_save[f"ep_{i}_{k}"] = v
        data_to_save["num_episodes"] = len(self.episodes)
        np.savez_compressed(file_path, **data_to_save)

    @classmethod
    def load(cls, file_path: str) -> "EpisodeStore":
        """Load stored episodes from npz archive."""
        data = np.load(file_path)
        num_episodes = int(data["num_episodes"])
        store = cls()
        for i in range(num_episodes):
            ep_dict = {
                "episode_id": data[f"ep_{i}_episode_id"],
                "observations": data[f"ep_{i}_observations"],
                "actions": data[f"ep_{i}_actions"],
                "rewards": data[f"ep_{i}_rewards"],
                "terminated": data[f"ep_{i}_terminated"],
                "truncated": data[f"ep_{i}_truncated"],
            }
            store.add_episode(Episode.from_dict(ep_dict))
        return store
