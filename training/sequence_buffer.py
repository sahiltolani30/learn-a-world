"""
Sequence Replay Buffer for Learn-a-World v2.
Stores entire episodes for multi-step latent unrolling and n-step return learning:
- obs: uint8 [T, 64, 64, 3]
- actions: int64 [T]
- rewards: float32 [T]
- dones: bool [T]
- events: uint8 [T, 22] (1 where achievement i was newly unlocked at step t)
- mask_before: uint8 [T, 22] (unlocked status before step t)
- source: str (tag identifying collection policy)

Features:
- Sharded chunked persistence (.npz) + dataset_index.json
- Fast PyTorch sequence sampling in 'uniform' and 'event_balanced' modes
- Train/Validation episode split for diagnostic evaluation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

ACHIEVEMENT_TO_IDX: Dict[str, int] = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22

def save_achievement_index(out_dir: Union[str, Path]) -> Path:
    """Saves standard achievement name to index mapping next to dataset."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    index_file = p / "achievement_index.json"
    with open(index_file, "w") as f:
        json.dump(ACHIEVEMENT_TO_IDX, f, indent=2)
    return index_file

class EpisodeBuffer:
    """
    Episodic sequence buffer storing complete trajectories with per-step
    achievement events and unlocked masks for TD-MPC2 / Dreamer-style training.
    """

    def __init__(self, capacity_transitions: int = 150_000):
        self.capacity_transitions = capacity_transitions
        self.episodes: List[Dict[str, Any]] = []
        self._total_transitions = 0
        self._cached_windows: Optional[List[Tuple[int, int]]] = None
        self._cached_event_windows: Optional[List[Tuple[int, int]]] = None
        self._cached_seq_len: Optional[int] = None

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return self._total_transitions

    def _invalidate_window_cache(self) -> None:
        self._cached_windows = None
        self._cached_event_windows = None
        self._cached_seq_len = None

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        events: np.ndarray,
        mask_before: np.ndarray,
        source: str = "structured_random",
    ) -> None:
        """
        Adds a complete episode to the buffer.
        obs: [T, 64, 64, 3] uint8
        actions: [T] int64
        rewards: [T] float32
        dones: [T] bool
        events: [T, 22] uint8
        mask_before: [T, 22] uint8
        """
        T = len(actions)
        if T == 0:
            return

        assert obs.shape[0] == T, f"Obs length {obs.shape[0]} != actions length {T}"
        assert rewards.shape[0] == T
        assert dones.shape[0] == T
        assert events.shape == (T, NUM_ACHIEVEMENTS), f"Events shape {events.shape} != {(T, NUM_ACHIEVEMENTS)}"
        assert mask_before.shape == (T, NUM_ACHIEVEMENTS)

        episode = {
            "obs": obs.astype(np.uint8, copy=False),
            "actions": actions.astype(np.int64, copy=False),
            "rewards": rewards.astype(np.float32, copy=False),
            "dones": dones.astype(np.bool_, copy=False),
            "events": events.astype(np.uint8, copy=False),
            "mask_before": mask_before.astype(np.uint8, copy=False),
            "source": str(source),
        }

        self.episodes.append(episode)
        self._total_transitions += T
        self._invalidate_window_cache()

        # Evict oldest episodes if exceeding capacity
        while self._total_transitions > self.capacity_transitions and len(self.episodes) > 1:
            evicted = self.episodes.pop(0)
            self._total_transitions -= len(evicted["actions"])

    def _build_windows(self, seq_len: int) -> None:
        """Indexes all valid sequence start positions across episodes."""
        if self._cached_seq_len == seq_len and self._cached_windows is not None:
            return

        windows: List[Tuple[int, int]] = []
        event_windows: List[Tuple[int, int]] = []
        event_weights: List[float] = []

        # Generic per-event-type counts across entire buffer (no hardcoded names)
        num_event_types = self.episodes[0]["events"].shape[-1] if self.episodes else NUM_ACHIEVEMENTS
        event_type_counts = np.zeros(num_event_types, dtype=np.float64)
        for ep in self.episodes:
            event_type_counts += np.sum(ep["events"] > 0, axis=0)

        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["actions"])
            if T < seq_len:
                continue
            events = ep["events"]
            for start_t in range(T - seq_len + 1):
                win = (ep_idx, start_t)
                windows.append(win)
                w_events = events[start_t : start_t + seq_len]
                active_types = np.where(np.any(w_events > 0, axis=0))[0]
                if len(active_types) > 0:
                    event_windows.append(win)
                    # Generic inverse-frequency weighting: rarer event types get higher weight
                    w_weight = float(np.sum(1.0 / np.maximum(event_type_counts[active_types], 1.0)))
                    event_weights.append(w_weight)

        self._cached_windows = windows
        self._cached_event_windows = event_windows
        if event_weights:
            total_w = sum(event_weights)
            self._cached_event_probs = np.array([w / total_w for w in event_weights], dtype=np.float64)
            # Normalize to guarantee exact sum of 1.0
            self._cached_event_probs /= np.sum(self._cached_event_probs)
        else:
            self._cached_event_probs = None
        self._cached_seq_len = seq_len

    def sample_sequences(
        self,
        batch_size: int = 32,
        seq_len: int = 16,
        mode: str = "uniform",
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of contiguous sequence windows.
        mode:
          - 'uniform': all valid windows sampled uniformly.
          - 'event_balanced': 50% from event windows (weighted inversely by event-type rarity,
            no hardcoded names), 50% uniform.
        Returns:
          obs: [B, seq_len, 3, 64, 64] float32 in [0, 1]
          actions: [B, seq_len] int64
          rewards: [B, seq_len] float32
          dones: [B, seq_len] bool
          events: [B, seq_len, 22] float32
          mask_before: [B, seq_len, 22] float32
        """
        self._build_windows(seq_len)
        if not self._cached_windows:
            raise ValueError(f"No valid sequence windows found of length {seq_len} in buffer.")

        chosen_windows: List[Tuple[int, int]] = []
        if mode == "event_balanced" and self._cached_event_windows and self._cached_event_probs is not None:
            n_event = min(batch_size // 2, len(self._cached_event_windows))
            n_uniform = batch_size - n_event

            event_indices = np.random.choice(
                len(self._cached_event_windows),
                size=n_event,
                replace=True,
                p=self._cached_event_probs,
            )
            for idx in event_indices:
                chosen_windows.append(self._cached_event_windows[idx])

            uniform_indices = np.random.choice(len(self._cached_windows), size=n_uniform, replace=True)
            for idx in uniform_indices:
                chosen_windows.append(self._cached_windows[idx])
        else:
            indices = np.random.choice(len(self._cached_windows), size=batch_size, replace=True)
            for idx in indices:
                chosen_windows.append(self._cached_windows[idx])

        # Collate batches
        b_obs = np.empty((batch_size, seq_len, 64, 64, 3), dtype=np.uint8)
        b_actions = np.empty((batch_size, seq_len), dtype=np.int64)
        b_rewards = np.empty((batch_size, seq_len), dtype=np.float32)
        b_dones = np.empty((batch_size, seq_len), dtype=np.bool_)
        b_events = np.empty((batch_size, seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)
        b_mask = np.empty((batch_size, seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)

        for i, (ep_idx, start_t) in enumerate(chosen_windows):
            ep = self.episodes[ep_idx]
            end_t = start_t + seq_len
            b_obs[i] = ep["obs"][start_t:end_t]
            b_actions[i] = ep["actions"][start_t:end_t]
            b_rewards[i] = ep["rewards"][start_t:end_t]
            b_dones[i] = ep["dones"][start_t:end_t]
            b_events[i] = ep["events"][start_t:end_t]
            b_mask[i] = ep["mask_before"][start_t:end_t]

        # Convert obs from [B, T, H, W, C] uint8 to [B, T, C, H, W] float32 in [0, 1]
        obs_tensor = torch.from_numpy(b_obs).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
        actions_tensor = torch.from_numpy(b_actions).long()
        rewards_tensor = torch.from_numpy(b_rewards).float()
        dones_tensor = torch.from_numpy(b_dones).bool()
        events_tensor = torch.from_numpy(b_events).float()
        mask_tensor = torch.from_numpy(b_mask).float()

        if device is not None:
            dev = torch.device(device)
            obs_tensor = obs_tensor.to(dev)
            actions_tensor = actions_tensor.to(dev)
            rewards_tensor = rewards_tensor.to(dev)
            dones_tensor = dones_tensor.to(dev)
            events_tensor = events_tensor.to(dev)
            mask_tensor = mask_tensor.to(dev)

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "dones": dones_tensor,
            "events": events_tensor,
            "mask_before": mask_tensor,
        }

    def save_shard(self, path: Union[str, Path]) -> Path:
        """Saves current buffer contents to a single compressed .npz archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if len(self.episodes) == 0:
            np.savez_compressed(
                p,
                obs=np.empty((0, 64, 64, 3), dtype=np.uint8),
                actions=np.empty((0,), dtype=np.int64),
                rewards=np.empty((0,), dtype=np.float32),
                dones=np.empty((0,), dtype=np.bool_),
                events=np.empty((0, NUM_ACHIEVEMENTS), dtype=np.uint8),
                mask_before=np.empty((0, NUM_ACHIEVEMENTS), dtype=np.uint8),
                episode_lengths=np.empty((0,), dtype=np.int64),
                sources=np.array([], dtype=object),
            )
            return p

        cat_obs = np.concatenate([ep["obs"] for ep in self.episodes], axis=0)
        cat_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
        cat_rewards = np.concatenate([ep["rewards"] for ep in self.episodes], axis=0)
        cat_dones = np.concatenate([ep["dones"] for ep in self.episodes], axis=0)
        cat_events = np.concatenate([ep["events"] for ep in self.episodes], axis=0)
        cat_mask = np.concatenate([ep["mask_before"] for ep in self.episodes], axis=0)
        ep_lens = np.array([len(ep["actions"]) for ep in self.episodes], dtype=np.int64)
        sources = np.array([ep["source"] for ep in self.episodes], dtype=object)

        np.savez_compressed(
            p,
            obs=cat_obs,
            actions=cat_actions,
            rewards=cat_rewards,
            dones=cat_dones,
            events=cat_events,
            mask_before=cat_mask,
            episode_lengths=ep_lens,
            sources=sources,
        )
        return p

    def load_shard(self, path: Union[str, Path]) -> None:
        """Loads and unpacks episodes from a .npz archive."""
        p = Path(path)
        data = np.load(p, allow_pickle=True)

        ep_lens = data["episode_lengths"]
        if len(ep_lens) == 0:
            return

        cat_obs = data["obs"]
        cat_actions = data["actions"]
        cat_rewards = data["rewards"]
        cat_dones = data["dones"]
        cat_events = data["events"]
        cat_mask = data["mask_before"]
        sources = data["sources"]

        offset = 0
        for i, length in enumerate(ep_lens):
            length = int(length)
            self.add_episode(
                obs=cat_obs[offset : offset + length],
                actions=cat_actions[offset : offset + length],
                rewards=cat_rewards[offset : offset + length],
                dones=cat_dones[offset : offset + length],
                events=cat_events[offset : offset + length],
                mask_before=cat_mask[offset : offset + length],
                source=str(sources[i]) if i < len(sources) else "unknown",
            )
            offset += length

    def split_train_val(self, val_ratio: float = 0.10) -> Tuple["EpisodeBuffer", "EpisodeBuffer"]:
        """
        Splits buffer by held-out last val_ratio of episodes.
        Ensures temporal/session validation consistency.
        """
        n_eps = len(self.episodes)
        n_val = max(1, int(round(n_eps * val_ratio))) if n_eps > 1 else 0
        n_train = n_eps - n_val

        train_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)
        val_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)

        for ep in self.episodes[:n_train]:
            train_buf.add_episode(
                obs=ep["obs"],
                actions=ep["actions"],
                rewards=ep["rewards"],
                dones=ep["dones"],
                events=ep["events"],
                mask_before=ep["mask_before"],
                source=ep["source"],
            )

        for ep in self.episodes[n_train:]:
            val_buf.add_episode(
                obs=ep["obs"],
                actions=ep["actions"],
                rewards=ep["rewards"],
                dones=ep["dones"],
                events=ep["events"],
                mask_before=ep["mask_before"],
                source=ep["source"],
            )

        return train_buf, val_buf

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Calculates per-achievement positive counts and source breakdown."""
        ach_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
        source_counts: Dict[str, int] = {}

        for ep in self.episodes:
            src = ep["source"]
            source_counts[src] = source_counts.get(src, 0) + len(ep["actions"])
            ep_event_sums = ep["events"].sum(axis=0)
            for i, ach in enumerate(CRAFTER_ACHIEVEMENTS):
                ach_counts[ach] += int(ep_event_sums[i])

        return {
            "total_episodes": len(self.episodes),
            "total_transitions": self._total_transitions,
            "per_achievement_positive_counts": ach_counts,
            "per_source_counts": source_counts,
        }

    def save_dataset_index(self, out_dir: Union[str, Path], shard_names: Optional[List[str]] = None) -> Path:
        """Writes dataset_index.json summary."""
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        stats = self.get_dataset_statistics()
        if shard_names is not None:
            stats["shards"] = shard_names
        index_file = p / "dataset_index.json"
        with open(index_file, "w") as f:
            json.dump(stats, f, indent=2)
        save_achievement_index(p)
        return index_file
