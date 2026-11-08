"""Boundary-safe replay containing only the observable interaction stream."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PixelEpisode:
    """One episode; no environment info, inventory, coordinates, or achievements.

    ``obs`` contains the initial frame plus one successor frame per action, so
    its length is exactly ``actions + 1``.  The final frame is retained even
    when the final action terminates the episode.
    """

    obs: np.ndarray             # uint8 [T+1,H,W,C]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    seed: int
    source: str

    def __post_init__(self) -> None:
        steps = int(self.actions.shape[0])
        if steps < 1:
            raise ValueError("an episode must contain at least one action")
        if self.obs.ndim != 4 or self.obs.dtype != np.uint8:
            raise ValueError("obs must be uint8 [T,H,W,C]")
        if self.obs.shape[0] != steps + 1:
            raise ValueError("obs must contain one more frame than actions")
        for name, value in (
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        ):
            if value.shape != (steps,):
                raise ValueError(f"{name} must have shape ({steps},)")


class PixelReplay:
    """FIFO episode replay with windows that never cross a reset."""

    def __init__(self, capacity_frames: Optional[int] = None) -> None:
        self.capacity_frames = capacity_frames
        self._episodes: List[PixelEpisode] = []
        self._frames = 0

    def add(self, episode: PixelEpisode) -> None:
        episode_frames = int(episode.obs.shape[0])
        if self.capacity_frames is not None and episode_frames > self.capacity_frames:
            raise ValueError(
                f"episode length ({episode_frames}) exceeds replay capacity ({self.capacity_frames})"
            )
        self._episodes.append(episode)
        self._frames += episode_frames
        while (
            self.capacity_frames is not None
            and self._frames > self.capacity_frames
            and self._episodes
        ):
            old = self._episodes.pop(0)
            self._frames -= int(old.obs.shape[0])

    def __len__(self) -> int:
        return self._frames

    @property
    def episodes(self) -> int:
        return len(self._episodes)

    @property
    def records(self) -> List[PixelEpisode]:
        return list(self._episodes)

    def sample(
        self, batch_size: int, sequence_length: int, rng: np.random.Generator
    ) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise RuntimeError("cannot sample an empty replay")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        example = self._episodes[0].obs
        obs = np.zeros((batch_size, sequence_length + 1) + example.shape[1:], dtype=np.uint8)
        actions = np.zeros((batch_size, sequence_length), dtype=np.int64)
        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        terminated = np.zeros((batch_size, sequence_length), dtype=bool)
        truncated = np.zeros((batch_size, sequence_length), dtype=bool)
        valid = np.zeros((batch_size, sequence_length), dtype=bool)

        for row in range(batch_size):
            episode = self._episodes[int(rng.integers(0, len(self._episodes)))]
            length = int(episode.actions.shape[0])
            use = min(length, sequence_length)
            start = 0 if length <= sequence_length else int(rng.integers(0, length - sequence_length + 1))
            obs[row, : use + 1] = episode.obs[start : start + use + 1]
            if use < sequence_length:
                obs[row, use + 1 :] = episode.obs[-1]
            actions[row, :use] = episode.actions[start : start + use]
            rewards[row, :use] = episode.rewards[start : start + use]
            terminated[row, :use] = episode.terminated[start : start + use]
            truncated[row, :use] = episode.truncated[start : start + use]
            valid[row, :use] = True

        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "valid": torch.from_numpy(valid),
        }

    def sample_with_prefix(
        self, batch_size: int, sequence_length: int, rng: np.random.Generator
    ) -> Dict[str, torch.Tensor]:
        """Sample sequence windows along with full episode prefixes from step 0 (R05 finding).

        This enables exact hidden belief reconstruction using current model weights,
        eliminating the mid-episode false reset trap.
        """
        if not self._episodes:
            raise RuntimeError("cannot sample an empty replay")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        example = self._episodes[0].obs
        obs = np.zeros((batch_size, sequence_length + 1) + example.shape[1:], dtype=np.uint8)
        actions = np.zeros((batch_size, sequence_length), dtype=np.int64)
        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        terminated = np.zeros((batch_size, sequence_length), dtype=bool)
        truncated = np.zeros((batch_size, sequence_length), dtype=bool)
        valid = np.zeros((batch_size, sequence_length), dtype=bool)

        sampled_episodes = []
        starts = []

        for row in range(batch_size):
            episode = self._episodes[int(rng.integers(0, len(self._episodes)))]
            length = int(episode.actions.shape[0])
            use = min(length, sequence_length)
            start = 0 if length <= sequence_length else int(rng.integers(0, length - sequence_length + 1))
            sampled_episodes.append(episode)
            starts.append(start)

            obs[row, : use + 1] = episode.obs[start : start + use + 1]
            if use < sequence_length:
                obs[row, use + 1 :] = episode.obs[-1]
            actions[row, :use] = episode.actions[start : start + use]
            rewards[row, :use] = episode.rewards[start : start + use]
            terminated[row, :use] = episode.terminated[start : start + use]
            truncated[row, :use] = episode.truncated[start : start + use]
            valid[row, :use] = True

        max_prefix = max(starts)
        prefix_obs = np.zeros((batch_size, max_prefix + 1) + example.shape[1:], dtype=np.uint8)
        prefix_actions = np.zeros((batch_size, max_prefix), dtype=np.int64)
        prefix_rewards = np.zeros((batch_size, max_prefix), dtype=np.float32)

        for row, (ep, start) in enumerate(zip(sampled_episodes, starts)):
            if start > 0:
                prefix_obs[row, : start + 1] = ep.obs[: start + 1]
                prefix_actions[row, :start] = ep.actions[:start]
                prefix_rewards[row, :start] = ep.rewards[:start]

        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "valid": torch.from_numpy(valid),
            "prefix_obs": torch.from_numpy(prefix_obs),
            "prefix_actions": torch.from_numpy(prefix_actions),
            "prefix_rewards": torch.from_numpy(prefix_rewards),
            "prefix_lengths": torch.tensor(starts, dtype=torch.long),
        }

    def save(self, path: str) -> None:
        if not self._episodes:
            raise RuntimeError("refusing to save empty replay")
        obs_lengths = np.asarray([episode.obs.shape[0] for episode in self._episodes], dtype=np.int64)
        action_lengths = obs_lengths - 1
        meta = json.dumps({
            "format": "true_jepa_replay_v1",
            "episodes": len(self._episodes),
            "capacity_frames": self.capacity_frames,
        })
        np.savez_compressed(
            path,
            obs=np.concatenate([episode.obs for episode in self._episodes]),
            actions=np.concatenate([episode.actions for episode in self._episodes]),
            rewards=np.concatenate([episode.rewards for episode in self._episodes]),
            terminated=np.concatenate([episode.terminated for episode in self._episodes]),
            truncated=np.concatenate([episode.truncated for episode in self._episodes]),
            obs_lengths=obs_lengths,
            action_lengths=action_lengths,
            seeds=np.asarray([episode.seed for episode in self._episodes], dtype=np.int64),
            sources=np.asarray([episode.source for episode in self._episodes], dtype="<U32"),
            meta=np.asarray(meta),
        )

    @classmethod
    def load(cls, path: str) -> "PixelReplay":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != "true_jepa_replay_v1":
                raise ValueError("unexpected replay format")
            capacity_frames = meta.get("capacity_frames")
            replay = cls(capacity_frames=capacity_frames)
            obs_offset = 0
            action_offset = 0
            for index, (obs_length, action_length) in enumerate(
                zip(data["obs_lengths"].tolist(), data["action_lengths"].tolist())
            ):
                obs_end = obs_offset + int(obs_length)
                action_end = action_offset + int(action_length)
                replay.add(
                    PixelEpisode(
                        obs=data["obs"][obs_offset:obs_end].astype(np.uint8),
                        actions=data["actions"][action_offset:action_end].astype(np.int64),
                        rewards=data["rewards"][action_offset:action_end].astype(np.float32),
                        terminated=data["terminated"][action_offset:action_end].astype(bool),
                        truncated=data["truncated"][action_offset:action_end].astype(bool),
                        seed=int(data["seeds"][index]),
                        source=str(data["sources"][index]),
                    )
                )
                obs_offset = obs_end
                action_offset = action_end
            return replay
