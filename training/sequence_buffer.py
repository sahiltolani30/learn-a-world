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
# Optional diagnostic state adapter width. It is deliberately a dimension, not
# a list of Crafter names; other environments can use another width.
DEFAULT_STATE_DELTA_DIM = 16

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
        self._cached_event_probs: Optional[np.ndarray] = None
        self._cached_frontier_windows: Optional[List[Tuple[int, int]]] = None
        self._cached_frontier_probs: Optional[np.ndarray] = None
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
        self._cached_event_probs = None
        self._cached_frontier_windows = None
        self._cached_frontier_probs = None
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
        state_delta: Optional[np.ndarray] = None,
        state_delta_valid: Optional[np.ndarray] = None,
        inventory_deltas: Optional[np.ndarray] = None,
        state_current: Optional[np.ndarray] = None,
        state_current_valid: Optional[np.ndarray] = None,
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

        if state_delta is None and inventory_deltas is not None:
            state_delta = inventory_deltas
        if state_delta is None:
            state_delta = np.zeros((T, DEFAULT_STATE_DELTA_DIM), dtype=np.float32)
            state_delta_valid = np.zeros(T, dtype=np.bool_)
        else:
            state_delta = np.asarray(state_delta, dtype=np.float32)
            assert state_delta.shape[0] == T, f"State-delta length {state_delta.shape[0]} != {T}"
            if state_delta.ndim != 2:
                raise ValueError(f"state_delta must be [T, D], got {state_delta.shape}")
            if state_delta_valid is None:
                state_delta_valid = np.ones(T, dtype=np.bool_)
        state_delta_valid = np.asarray(state_delta_valid, dtype=np.bool_)
        assert state_delta_valid.shape == (T,)

        if state_current is None:
            state_current = np.zeros((T, state_delta.shape[-1]), dtype=np.float32)
            if T > 1 and np.any(state_delta_valid):
                state_current[1:] = np.cumsum(state_delta[:-1], axis=0)
        else:
            state_current = np.asarray(state_current, dtype=np.float32)
            assert state_current.shape[0] == T

        if state_current_valid is None:
            if not np.any(state_delta_valid):
                state_current_valid = np.zeros(T, dtype=np.bool_)
            else:
                cum_valid = np.cumprod(state_delta_valid.astype(int)).astype(bool)
                state_current_valid = np.ones(T, dtype=np.bool_)
                state_current_valid[1:] = cum_valid[:-1]
        else:
            state_current_valid = np.asarray(state_current_valid, dtype=np.bool_)
            assert state_current_valid.shape == (T,)

        episode = {
            "obs": obs.astype(np.uint8, copy=False),
            "actions": actions.astype(np.int64, copy=False),
            "rewards": rewards.astype(np.float32, copy=False),
            "dones": dones.astype(np.bool_, copy=False),
            "events": events.astype(np.uint8, copy=False),
            "mask_before": mask_before.astype(np.uint8, copy=False),
            "state_delta": state_delta.astype(np.float32, copy=False),
            "state_delta_valid": state_delta_valid,
            "state_current": state_current.astype(np.float32, copy=False),
            "state_current_valid": state_current_valid,
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
        frontier_windows: List[Tuple[int, int]] = []
        frontier_weights: List[float] = []
        labeled_windows: List[Tuple[int, int]] = []

        # Frontier achievement index set: tools, mining, combat, crafting
        frontier_indices_set = {0, 1, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20}

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
            state_valid = ep.get("state_delta_valid")
            for start_t in range(T - seq_len + 1):
                win = (ep_idx, start_t)
                windows.append(win)
                if state_valid is not None and np.any(state_valid[start_t : start_t + seq_len]):
                    labeled_windows.append(win)
                w_events = events[start_t : start_t + seq_len]
                active_types = np.where(np.any(w_events > 0, axis=0))[0]
                if len(active_types) > 0:
                    event_windows.append(win)
                    # Generic inverse-frequency weighting: rarer event types get higher weight
                    w_weight = float(np.sum(1.0 / np.maximum(event_type_counts[active_types], 1.0)))
                    event_weights.append(w_weight)

                    # Frontier windows: subset with rare crafting, tool, or mining events
                    active_frontier = [t for t in active_types if t in frontier_indices_set]
                    if active_frontier:
                        frontier_windows.append(win)
                        f_weight = float(np.sum(1.0 / np.maximum(event_type_counts[active_frontier], 1.0)))
                        frontier_weights.append(f_weight)

        self._cached_windows = windows
        self._cached_event_windows = event_windows
        self._cached_frontier_windows = frontier_windows
        self._cached_labeled_windows = labeled_windows

        if event_weights:
            total_w = sum(event_weights)
            self._cached_event_probs = np.array([w / total_w for w in event_weights], dtype=np.float64)
            self._cached_event_probs /= np.sum(self._cached_event_probs)
        else:
            self._cached_event_probs = None

        if frontier_weights:
            total_fw = sum(frontier_weights)
            self._cached_frontier_probs = np.array([w / total_fw for w in frontier_weights], dtype=np.float64)
            self._cached_frontier_probs /= np.sum(self._cached_frontier_probs)
        else:
            self._cached_frontier_probs = None

        self._cached_seq_len = seq_len

    def sample_sequences(
        self,
        batch_size: int = 32,
        seq_len: int = 16,
        mode: str = "uniform",
        labeled_ratio: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of contiguous sequence windows.
        mode:
          - 'uniform': all valid windows sampled uniformly.
          - 'event_balanced': 50% from event windows (weighted inversely by event-type rarity,
            no hardcoded names), 50% uniform.
          - 'frontier_stratified': 25% frontier, 25% event, 50% uniform.
        labeled_ratio:
          - Fraction of batch (e.g. 0.5) guaranteed to be sampled from windows containing
            valid state-delta labels (if any labeled windows exist in buffer).
        Returns:
          obs: [B, seq_len, 3, 64, 64] float32 in [0, 1]
          actions: [B, seq_len] int64
          rewards: [B, seq_len] float32
          dones: [B, seq_len] bool
          events: [B, seq_len, 22] float32
          mask_before: [B, seq_len, 22] float32
          state_delta: [B, seq_len, D] float32
          state_delta_valid: [B, seq_len] bool
        """
        self._build_windows(seq_len)
        if not self._cached_windows:
            raise ValueError(f"No valid sequence windows found of length {seq_len} in buffer.")

        chosen_windows: List[Tuple[int, int]] = []

        # 1. Sample labeled windows if requested and available
        n_labeled = 0
        if labeled_ratio > 0.0 and self._cached_labeled_windows:
            target_labeled = max(1, int(round(batch_size * labeled_ratio)))
            n_labeled = min(target_labeled, len(self._cached_labeled_windows))
            labeled_indices = np.random.choice(len(self._cached_labeled_windows), size=n_labeled, replace=True)
            for idx in labeled_indices:
                chosen_windows.append(self._cached_labeled_windows[idx])

        n_remaining = batch_size - n_labeled

        # 2. Sample remaining slots according to mode
        if n_remaining > 0:
            if mode == "frontier_stratified" and self._cached_frontier_windows and self._cached_frontier_probs is not None:
                n_frontier = min(n_remaining // 4, len(self._cached_frontier_windows))
                n_event = min(n_remaining // 4, len(self._cached_event_windows)) if self._cached_event_windows else 0
                n_uniform = n_remaining - n_frontier - n_event

                if n_frontier > 0:
                    f_indices = np.random.choice(
                        len(self._cached_frontier_windows),
                        size=n_frontier,
                        replace=True,
                        p=self._cached_frontier_probs,
                    )
                    for idx in f_indices:
                        chosen_windows.append(self._cached_frontier_windows[idx])

                if n_event > 0 and self._cached_event_probs is not None:
                    e_indices = np.random.choice(
                        len(self._cached_event_windows),
                        size=n_event,
                        replace=True,
                        p=self._cached_event_probs,
                    )
                    for idx in e_indices:
                        chosen_windows.append(self._cached_event_windows[idx])

                if n_uniform > 0:
                    u_indices = np.random.choice(len(self._cached_windows), size=n_uniform, replace=True)
                    for idx in u_indices:
                        chosen_windows.append(self._cached_windows[idx])

            elif mode == "event_balanced" and self._cached_event_windows and self._cached_event_probs is not None:
                n_event = min(n_remaining // 2, len(self._cached_event_windows))
                n_uniform = n_remaining - n_event

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
                indices = np.random.choice(len(self._cached_windows), size=n_remaining, replace=True)
                for idx in indices:
                    chosen_windows.append(self._cached_windows[idx])

        # Shuffle so labeled and unlabeled sequences are interspersed
        if n_labeled > 0 and len(chosen_windows) > 1:
            np.random.shuffle(chosen_windows)

        # Collate batches
        b_obs = np.empty((batch_size, seq_len, 64, 64, 3), dtype=np.uint8)
        b_actions = np.empty((batch_size, seq_len), dtype=np.int64)
        b_rewards = np.empty((batch_size, seq_len), dtype=np.float32)
        b_dones = np.empty((batch_size, seq_len), dtype=np.bool_)
        b_events = np.empty((batch_size, seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)
        b_mask = np.empty((batch_size, seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)
        state_dim = int(self.episodes[0].get("state_delta", np.zeros((1, DEFAULT_STATE_DELTA_DIM))).shape[-1])
        b_state_delta = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
        b_state_valid = np.zeros((batch_size, seq_len), dtype=np.bool_)
        b_state_current = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
        b_state_current_valid = np.zeros((batch_size, seq_len), dtype=np.bool_)

        for i, (ep_idx, start_t) in enumerate(chosen_windows):
            ep = self.episodes[ep_idx]
            end_t = start_t + seq_len
            b_obs[i] = ep["obs"][start_t:end_t]
            b_actions[i] = ep["actions"][start_t:end_t]
            b_rewards[i] = ep["rewards"][start_t:end_t]
            b_dones[i] = ep["dones"][start_t:end_t]
            b_events[i] = ep["events"][start_t:end_t]
            b_mask[i] = ep["mask_before"][start_t:end_t]
            ep_delta = ep.get("state_delta")
            if ep_delta is not None:
                if ep_delta.shape[-1] != state_dim:
                    raise ValueError("All episodes in a buffer must use the same state_delta width")
                b_state_delta[i] = ep_delta[start_t:end_t]
                b_state_valid[i] = ep.get(
                    "state_delta_valid", np.ones(len(ep["actions"]), dtype=np.bool_)
                )[start_t:end_t]
            ep_current = ep.get("state_current")
            if ep_current is not None:
                b_state_current[i] = ep_current[start_t:end_t]
                b_state_current_valid[i] = ep.get(
                    "state_current_valid", np.ones(len(ep["actions"]), dtype=np.bool_)
                )[start_t:end_t]

        # Convert obs from [B, T, H, W, C] uint8 to [B, T, C, H, W] float32 in [0, 1]
        obs_tensor = torch.from_numpy(b_obs).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
        actions_tensor = torch.from_numpy(b_actions).long()
        rewards_tensor = torch.from_numpy(b_rewards).float()
        dones_tensor = torch.from_numpy(b_dones).bool()
        events_tensor = torch.from_numpy(b_events).float()
        mask_tensor = torch.from_numpy(b_mask).float()
        state_delta_tensor = torch.from_numpy(b_state_delta).float()
        state_valid_tensor = torch.from_numpy(b_state_valid).bool()
        state_current_tensor = torch.from_numpy(b_state_current).float()
        state_current_valid_tensor = torch.from_numpy(b_state_current_valid).bool()

        if device is not None:
            dev = torch.device(device)
            obs_tensor = obs_tensor.to(dev)
            actions_tensor = actions_tensor.to(dev)
            rewards_tensor = rewards_tensor.to(dev)
            dones_tensor = dones_tensor.to(dev)
            events_tensor = events_tensor.to(dev)
            mask_tensor = mask_tensor.to(dev)
            state_delta_tensor = state_delta_tensor.to(dev)
            state_valid_tensor = state_valid_tensor.to(dev)
            state_current_tensor = state_current_tensor.to(dev)
            state_current_valid_tensor = state_current_valid_tensor.to(dev)

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "dones": dones_tensor,
            "events": events_tensor,
            "mask_before": mask_tensor,
            "state_delta": state_delta_tensor,
            "state_delta_valid": state_valid_tensor,
            "state_current": state_current_tensor,
            "state_current_valid": state_current_valid_tensor,
        }

    # Backward compatibility alias
    sample_batch = sample_sequences

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
                state_delta=np.empty((0, DEFAULT_STATE_DELTA_DIM), dtype=np.float32),
                state_delta_valid=np.empty((0,), dtype=np.bool_),
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
        state_dims = {ep["state_delta"].shape[-1] for ep in self.episodes}
        if len(state_dims) != 1:
            raise ValueError(f"Inconsistent state_delta widths in buffer: {state_dims}")
        cat_state_delta = np.concatenate([ep["state_delta"] for ep in self.episodes], axis=0)
        cat_state_valid = np.concatenate([
            ep.get("state_delta_valid", np.ones(len(ep["actions"]), dtype=np.bool_))
            for ep in self.episodes
        ], axis=0)
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
            state_delta=cat_state_delta,
            state_delta_valid=cat_state_valid,
            episode_lengths=ep_lens,
            sources=sources,
        )
        return p

    def load_shard(self, path: Union[str, Path]) -> None:
        """Loads and unpacks episodes from a .npz archive."""
        p = Path(path)
        data = np.load(p, allow_pickle=True)

        if "episode_lengths" in data:
            ep_lens = data["episode_lengths"]
        elif "dones" in data:
            dones = data["dones"]
            done_indices = np.where(dones)[0]
            if len(done_indices) == 0:
                ep_lens = np.array([len(dones)], dtype=np.int64)
            else:
                lens = []
                prev = 0
                for d_idx in done_indices:
                    lens.append(int(d_idx + 1 - prev))
                    prev = int(d_idx + 1)
                if prev < len(dones):
                    lens.append(int(len(dones) - prev))
                ep_lens = np.array(lens, dtype=np.int64)
        else:
            total_t = len(data["actions"]) if "actions" in data else len(data["obs"])
            ep_lens = np.array([total_t], dtype=np.int64)

        if len(ep_lens) == 0:
            return

        cat_obs = data["obs"]
        cat_actions = data["actions"]
        cat_rewards = data["rewards"]
        cat_dones = data["dones"]
        cat_events = data["events"]
        cat_mask = data["mask_before"] if "mask_before" in data else np.zeros_like(cat_events)
        cat_state_delta = data["state_delta"] if "state_delta" in data else None
        cat_state_valid = data["state_delta_valid"] if "state_delta_valid" in data else None
        cat_state_current = data["state_current"] if "state_current" in data else None
        cat_state_current_valid = data["state_current_valid"] if "state_current_valid" in data else None
        sources = data["sources"] if "sources" in data else np.array(["unknown"] * len(ep_lens), dtype=object)

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
                state_delta=(cat_state_delta[offset : offset + length] if cat_state_delta is not None else None),
                state_delta_valid=(cat_state_valid[offset : offset + length] if cat_state_valid is not None else None),
                state_current=(cat_state_current[offset : offset + length] if cat_state_current is not None else None),
                state_current_valid=(cat_state_current_valid[offset : offset + length] if cat_state_current_valid is not None else None),
                source=str(sources[i]) if i < len(sources) else "unknown",
            )
            offset += length

    def split_train_val(
        self,
        val_ratio: float = 0.05,
        seed: int = 0,
    ) -> Tuple["EpisodeBuffer", "EpisodeBuffer"]:
        """
        Splits by held-out episodes in-place with zero array copying.

        Performs stratified splitting between labeled and unlabeled episodes so that
        validation is guaranteed to contain labeled episodes if any exist in the dataset.
        """
        rng = np.random.RandomState(seed)
        n_eps = len(self.episodes)
        if n_eps <= 1:
            train_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)
            val_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)
            train_buf.episodes = list(self.episodes)
            train_buf._total_transitions = self._total_transitions
            self.episodes = []
            self._total_transitions = 0
            self._invalidate_window_cache()
            return train_buf, val_buf

        labeled_indices = [
            i for i, ep in enumerate(self.episodes)
            if np.any(ep.get("state_delta_valid", False))
        ]
        unlabeled_indices = [
            i for i, ep in enumerate(self.episodes)
            if not np.any(ep.get("state_delta_valid", False))
        ]

        val_indices = set()
        # Stratified split on labeled episodes
        if labeled_indices:
            perm_labeled = rng.permutation(labeled_indices)
            if len(labeled_indices) >= 2:
                # Ensure at least 1 in val and at least 1 in train (at least 2 in val if >= 5)
                min_val = 2 if len(labeled_indices) >= 5 else 1
                n_val_labeled = min(
                    len(labeled_indices) - 1,
                    max(min_val, int(round(len(labeled_indices) * val_ratio))),
                )
            else:
                n_val_labeled = 1 if val_ratio >= 1.0 else 0
            val_indices.update(int(i) for i in perm_labeled[:n_val_labeled])

        # Stratified split on unlabeled episodes
        if unlabeled_indices:
            perm_unlabeled = rng.permutation(unlabeled_indices)
            n_val_unlabeled = max(0, int(round(len(unlabeled_indices) * val_ratio)))
            if val_ratio > 0.0 and len(val_indices) == 0 and len(unlabeled_indices) > 1:
                n_val_unlabeled = max(1, n_val_unlabeled)
            val_indices.update(int(i) for i in perm_unlabeled[:n_val_unlabeled])

        train_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)
        val_buf = EpisodeBuffer(capacity_transitions=self.capacity_transitions)

        # Move episode references directly without copying arrays.
        train_buf.episodes = [ep for i, ep in enumerate(self.episodes) if i not in val_indices]
        train_buf._total_transitions = sum(len(ep["actions"]) for ep in train_buf.episodes)

        val_buf.episodes = [ep for i, ep in enumerate(self.episodes) if i in val_indices]
        val_buf._total_transitions = sum(len(ep["actions"]) for ep in val_buf.episodes)

        # Clear original buffer to release any duplicate references
        self.episodes = []
        self._total_transitions = 0
        self._invalidate_window_cache()

        return train_buf, val_buf

    def get_labeled_statistics(self) -> Dict[str, Any]:
        """Calculates precise transition and episode counts for labeled state deltas."""
        labeled_eps = 0
        labeled_transitions = 0
        for ep in self.episodes:
            valid = ep.get("state_delta_valid")
            if valid is not None:
                n_v = int(np.sum(valid))
                if n_v > 0:
                    labeled_eps += 1
                    labeled_transitions += n_v
        total_transitions = self._total_transitions
        density = (labeled_transitions / total_transitions) if total_transitions > 0 else 0.0
        return {
            "total_episodes": len(self.episodes),
            "labeled_episodes": labeled_eps,
            "total_transitions": total_transitions,
            "labeled_transitions": labeled_transitions,
            "labeled_density_pct": round(density * 100.0, 4),
        }

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Calculates per-achievement positive counts and source breakdown."""
        ach_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
        source_counts: Dict[str, int] = {}
        state_delta_valid_count = 0

        for ep in self.episodes:
            src = ep["source"]
            source_counts[src] = source_counts.get(src, 0) + len(ep["actions"])
            ep_event_sums = ep["events"].sum(axis=0)
            state_delta_valid_count += int(np.sum(ep.get("state_delta_valid", np.zeros(len(ep["actions"]), dtype=np.bool_))))
            for i, ach in enumerate(CRAFTER_ACHIEVEMENTS):
                ach_counts[ach] += int(ep_event_sums[i])

        return {
            "total_episodes": len(self.episodes),
            "total_transitions": self._total_transitions,
            "per_achievement_positive_counts": ach_counts,
            "per_source_counts": source_counts,
            "state_delta_valid_transitions": state_delta_valid_count,
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
