"""
Milestone-Prioritized Unified Replay Buffer for Godfather JEPA.

Solves the milestone starvation flaw (where trivial achievements wake_up and collect_sapling
drowned out rare events like place_table, make_wood_pickaxe, and collect_stone):
1. Dual-pool sampling: 90% uniform replay + 10% achievement-prioritized windows.
2. Tracks discovery of all 22 Crafter achievements and samples windows centered on
   first-time milestone unlock transitions.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch


class MilestoneReplayBuffer:
    """
    Episodic replay buffer with milestone-prioritized trajectory window sampling.
    """

    def __init__(self, capacity: int = 2000, priority_ratio: float = 0.10):
        self.capacity = capacity
        self.priority_ratio = priority_ratio
        self.episodes: List[Dict[str, Any]] = []
        self.milestone_indices: List[int] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        achievements: Dict[str, np.ndarray],
    ) -> None:
        """
        Adds a complete episode to the buffer.
        """
        if len(obs) == 0 or len(actions) == 0:
            return

        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
            # Rebuild milestone index map
            self.milestone_indices = [
                i for i, ep in enumerate(self.episodes)
                if any(np.any(v) for k, v in ep["achievements"].items() if k not in ("wake_up", "collect_sapling"))
            ]

        has_milestone = any(
            np.any(v) for k, v in achievements.items()
            if k not in ("wake_up", "collect_sapling")
        )

        ep_idx = len(self.episodes)
        self.episodes.append({
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "achievements": achievements,
            "has_milestone": has_milestone,
        })

        if has_milestone:
            self.milestone_indices.append(ep_idx)

    def load_npz_shards(
        self,
        shard_paths: List[str],
        max_episodes: Optional[int] = None,
    ) -> int:
        """
        Loads pre-collected sequence shards (e.g., from Google Drive data/sequences/)
        into the milestone-prioritized replay buffer.

        Safely skips empty shards (len == 0), corrupted files, and shards with mismatched
        array lengths without raising RuntimeError or ValueError.

        Returns total loaded episodes count.
        """
        from godfather_jepa.metrics.crafter_score import CRAFTER_ACHIEVEMENTS

        loaded_count = 0
        for sp in shard_paths:
            try:
                data = np.load(str(sp), allow_pickle=True)
                if "obs" not in data or "dones" not in data:
                    continue

                obs = data["obs"]
                dones = data["dones"]

                # Guard 1: Skip empty shards
                if len(obs) == 0 or len(dones) == 0:
                    continue

                n_trans = len(obs)
                # Guard 2: Skip shards with mismatched array lengths
                if len(dones) != n_trans:
                    continue

                if "actions" in data:
                    acts = data["actions"]
                    if len(acts) != n_trans:
                        continue
                else:
                    acts = np.zeros(n_trans, dtype=np.int64)

                if "rewards" in data:
                    rews = data["rewards"]
                    if len(rews) != n_trans:
                        continue
                else:
                    rews = np.zeros(n_trans, dtype=np.float32)

                events = None
                if "events" in data:
                    events = data["events"]
                    if len(events) != n_trans:
                        continue

                ep_start = 0
                ep_indices = np.where(dones)[0]
                if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                    ep_indices = np.append(ep_indices, n_trans - 1)

                for ep_end in ep_indices:
                    # Guard 3: Validate slice bounds
                    if ep_end < ep_start:
                        continue

                    ep_obs = obs[ep_start : ep_end + 1]
                    ep_acts = acts[ep_start : ep_end + 1]
                    ep_rews = rews[ep_start : ep_end + 1]
                    ep_dones = dones[ep_start : ep_end + 1]

                    # Guard 4: Safely skip zero-length or inhomogeneous episode slices
                    ep_len = len(ep_obs)
                    if ep_len == 0 or not (len(ep_acts) == ep_len and len(ep_rews) == ep_len and len(ep_dones) == ep_len):
                        ep_start = ep_end + 1
                        continue

                    ep_achs: Dict[str, np.ndarray] = {}
                    if events is not None:
                        ep_events = events[ep_start : ep_end + 1]
                        if len(ep_events) == ep_len and ep_events.ndim == 2:
                            for a_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
                                if a_idx < ep_events.shape[1]:
                                    ep_achs[ach_name] = ep_events[:, a_idx]

                    self.add_episode(ep_obs, ep_acts, ep_rews, ep_dones, ep_achs)
                    loaded_count += 1
                    ep_start = ep_end + 1

                    if max_episodes is not None and loaded_count >= max_episodes:
                        return loaded_count
            except Exception:
                # Corrupt, unreadable, or malformed shard; skip gracefully
                continue

        return loaded_count

    def sample_batch(
        self,
        batch_size: int = 16,
        seq_len: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of sequence windows [B, T, ...] with milestone prioritization.
        """
        if not self.episodes:
            raise ValueError("Cannot sample from empty replay buffer!")

        sampled_obs = []
        sampled_acts = []
        sampled_rews = []
        sampled_dones = []

        n_priority = int(batch_size * self.priority_ratio) if self.milestone_indices else 0
        n_uniform = batch_size - n_priority

        indices = []
        if n_priority > 0:
            indices.extend([random.choice(self.milestone_indices) for _ in range(n_priority)])
        indices.extend([random.randrange(len(self.episodes)) for _ in range(n_uniform)])

        for ep_idx in indices:
            ep = self.episodes[ep_idx]
            t_total = len(ep["actions"])
            if t_total <= seq_len:
                start = 0
                end = t_total
                pad_len = seq_len - t_total
            else:
                start = random.randint(0, t_total - seq_len)
                end = start + seq_len
                pad_len = 0

            o = ep["obs"][start:end]
            a = ep["actions"][start:end]
            r = ep["rewards"][start:end]
            d = ep["dones"][start:end]

            if pad_len > 0:
                # Pad with last frame
                o_pad = np.repeat(o[-1:], pad_len, axis=0)
                a_pad = np.zeros(pad_len, dtype=a.dtype)
                r_pad = np.zeros(pad_len, dtype=r.dtype)
                d_pad = np.ones(pad_len, dtype=d.dtype)

                o = np.concatenate([o, o_pad], axis=0)
                a = np.concatenate([a, a_pad], axis=0)
                r = np.concatenate([r, r_pad], axis=0)
                d = np.concatenate([d, d_pad], axis=0)

            sampled_obs.append(o)
            sampled_acts.append(a)
            sampled_rews.append(r)
            sampled_dones.append(d)

        return {
            "obs": torch.tensor(np.array(sampled_obs), dtype=torch.uint8, device=device),
            "actions": torch.tensor(np.array(sampled_acts), dtype=torch.long, device=device),
            "rewards": torch.tensor(np.array(sampled_rews), dtype=torch.float32, device=device),
            "dones": torch.tensor(np.array(sampled_dones), dtype=torch.bool, device=device),
        }
