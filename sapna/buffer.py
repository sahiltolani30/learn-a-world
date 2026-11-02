"""
sapna/buffer.py

Replay buffer for SAPNA per SAPNA_PLAN.md §6.
Pre-allocated numpy arrays, ring buffer, per-episode index tracking, 70/30 sequence sampling
with contiguous window validation, and fail-safe checkpoint roundtrip assertions.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class SequenceBatch(dict):
    """Dictionary subclass supporting attribute-style access (e.g. batch.obs)."""
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'SequenceBatch' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self[name] = value


class SAPNAReplayBuffer:
    """
    Ring buffer storing up to 100,000 transitions.
    Maintains a per-episode index per §6 to support interleaved parallel environments
    and contiguous sequence sampling (T=16 or fallback T=8) with 70% uniform / 30% milestone priority.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        # Pre-allocated arrays per §6
        self.obs = np.zeros((capacity, 3, 64, 64), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int8)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.episode_id = np.zeros(capacity, dtype=np.int64)
        self.step_in_ep = np.zeros(capacity, dtype=np.int64)
        self.milestone = np.zeros(capacity, dtype=bool)
        self.r_int_norm = np.zeros(capacity, dtype=np.float32)

        # Per-episode index: episode_id -> list of buffer slot indices per §6
        self.episode_indices = defaultdict(list)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: int,
        reward: float,
        done: bool,
        episode_id: int,
        step_in_ep: int,
        milestone: bool,
        r_int_norm: float = 0.0,
    ) -> None:
        """
        Insert a transition into the buffer at current pointer position.
        """
        if isinstance(obs, torch.Tensor):
            arr = obs.detach().cpu().numpy()
        else:
            arr = np.asarray(obs)

        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.05 and arr.min() >= -0.05:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))

        idx = self.ptr

        # Ring buffer overwrite maintenance: remove overwritten slot from old episode index
        if self.size == self.capacity:
            old_ep = self.episode_id[idx]
            if old_ep in self.episode_indices:
                try:
                    self.episode_indices[old_ep].remove(idx)
                    if not self.episode_indices[old_ep]:
                        del self.episode_indices[old_ep]
                except ValueError:
                    pass

        self.obs[idx] = arr
        self.action[idx] = np.int8(action)
        self.reward[idx] = np.float32(reward)
        self.done[idx] = bool(done)
        self.episode_id[idx] = np.int64(episode_id)
        self.step_in_ep[idx] = np.int64(step_in_ep)
        self.milestone[idx] = bool(milestone)
        self.r_int_norm[idx] = np.float32(r_int_norm)

        self.episode_indices[int(episode_id)].append(idx)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def count_valid_windows(self, T: int = 16) -> int:
        """Counts valid sequence windows of length T across all active episodes."""
        return sum(max(0, len(idxs) - T + 1) for idxs in self.episode_indices.values())

    def _get_valid_starts(self, T: int = 16) -> Tuple[List[int], List[int]]:
        """
        Returns (all_valid_starts, milestone_valid_starts).
        For acceptance test T6 compatibility.
        """
        all_starts = []
        ms_starts = []
        for ep, idxs in self.episode_indices.items():
            if len(idxs) >= T:
                for k in range(len(idxs) - T + 1):
                    start_idx = idxs[k]
                    all_starts.append(start_idx)
                    window_idxs = idxs[k : k + T]
                    if any(self.milestone[i] for i in window_idxs):
                        ms_starts.append(start_idx)
        return all_starts, ms_starts

    @property
    def valid_windows(self) -> int:
        """Returns the number of valid sequence windows of length T=16."""
        return self.count_valid_windows(T=16)

    def sample_sequences(self, B: int = 8, T: int = 16) -> SequenceBatch:
        """
        Sample B sequences of length T with 70% uniform / 30% milestone priority.
        If fewer than 64 valid windows of length T exist, falls back to T=8 per §6.
        """
        actual_T = T
        if self.count_valid_windows(actual_T) < 64 and actual_T > 8:
            actual_T = 8

        eligible_eps = [ep for ep, idxs in self.episode_indices.items() if len(idxs) >= actual_T]
        if not eligible_eps:
            raise RuntimeError(f"Cannot sample sequences: 0 valid windows of length {actual_T} in buffer.")

        # Find eligible milestone episodes
        ms_eps = [ep for ep in eligible_eps if any(self.milestone[i] for i in self.episode_indices[ep])]

        sampled_windows = []
        for _ in range(B):
            use_ms = (np.random.rand() < 0.3) and (len(ms_eps) > 0)
            if use_ms:
                ep = int(np.random.choice(ms_eps))
                idxs = self.episode_indices[ep]
                ms_pos = [k for k, buf_idx in enumerate(idxs) if self.milestone[buf_idx]]
                chosen_ms = int(np.random.choice(ms_pos))
                min_start = max(0, chosen_ms - actual_T + 1)
                max_start = min(len(idxs) - actual_T, chosen_ms)
                start_k = int(np.random.randint(min_start, max_start + 1))
            else:
                ep = int(np.random.choice(eligible_eps))
                idxs = self.episode_indices[ep]
                start_k = int(np.random.randint(0, len(idxs) - actual_T + 1))

            sampled_windows.append(idxs[start_k : start_k + actual_T])

        # Extract sequence slices (B, actual_T, ...)
        batch_obs = np.empty((B, actual_T, 3, 64, 64), dtype=np.float32)
        batch_act = np.empty((B, actual_T), dtype=np.int64)
        batch_rew = np.empty((B, actual_T), dtype=np.float32)
        batch_done = np.empty((B, actual_T), dtype=bool)
        batch_ep = np.empty((B, actual_T), dtype=np.int64)
        batch_step = np.empty((B, actual_T), dtype=np.int64)
        batch_ms = np.empty((B, actual_T), dtype=bool)
        batch_rint = np.empty((B, actual_T), dtype=np.float32)

        for b, window in enumerate(sampled_windows):
            batch_obs[b] = self.obs[window].astype(np.float32) / 255.0
            batch_act[b] = self.action[window]
            batch_rew[b] = self.reward[window]
            batch_done[b] = self.done[window]
            batch_ep[b] = self.episode_id[window]
            batch_step[b] = self.step_in_ep[window]
            batch_ms[b] = self.milestone[window]
            batch_rint[b] = self.r_int_norm[window]

        return SequenceBatch({
            "obs": torch.from_numpy(batch_obs),
            "action": torch.from_numpy(batch_act),
            "reward": torch.from_numpy(batch_rew),
            "done": torch.from_numpy(batch_done),
            "episode_id": torch.from_numpy(batch_ep),
            "step_in_ep": torch.from_numpy(batch_step),
            "milestone": torch.from_numpy(batch_ms),
            "r_int_norm": torch.from_numpy(batch_rint),
            "seq_len_actual": actual_T,
        })

    def save_state(self) -> Dict[str, np.ndarray]:
        """Serialize all buffer arrays and pointers for checkpointing per §6."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "ptr": self.ptr,
            "obs": self.obs[:self.size].copy(),
            "action": self.action[:self.size].copy(),
            "reward": self.reward[:self.size].copy(),
            "done": self.done[:self.size].copy(),
            "episode_id": self.episode_id[:self.size].copy(),
            "step_in_ep": self.step_in_ep[:self.size].copy(),
            "milestone": self.milestone[:self.size].copy(),
            "r_int_norm": self.r_int_norm[:self.size].copy(),
        }

    def load_state(self, state: Dict[str, Union[int, np.ndarray]]) -> None:
        """Restore buffer state and rebuild per-episode index per §6."""
        expected_size = int(state["size"])
        target_capacity = max(self.capacity, int(state.get("capacity", expected_size)))
        if target_capacity > len(self.obs):
            # Grow arrays safely to target_capacity when expanding at resume
            self.obs = np.zeros((target_capacity, 3, 64, 64), dtype=np.uint8)
            self.action = np.zeros(target_capacity, dtype=np.int64)
            self.reward = np.zeros(target_capacity, dtype=np.float32)
            self.done = np.zeros(target_capacity, dtype=np.float32)
            self.episode_id = np.zeros(target_capacity, dtype=np.int64)
            self.step_in_ep = np.zeros(target_capacity, dtype=np.int64)
            self.milestone = np.zeros(target_capacity, dtype=bool)
            self.r_int_norm = np.zeros(target_capacity, dtype=np.float32)

        self.capacity = target_capacity
        self.size = expected_size
        self.ptr = int(state["ptr"])

        self.obs[:expected_size] = state["obs"]
        self.action[:expected_size] = state["action"]
        self.reward[:expected_size] = state["reward"]
        self.done[:expected_size] = state["done"]
        self.episode_id[:expected_size] = state["episode_id"]
        self.step_in_ep[:expected_size] = state["step_in_ep"]
        self.milestone[:expected_size] = state["milestone"]
        self.r_int_norm[:expected_size] = state["r_int_norm"]

        # Rebuild per-episode index
        self.episode_indices = defaultdict(list)
        for idx in range(expected_size):
            self.episode_indices[int(self.episode_id[idx])].append(idx)

        # Assert len(buffer) == expected
        if len(self) != expected_size:
            raise RuntimeError(
                f"SAPNA buffer roundtrip failed: len(buffer)={len(self)} != expected={expected_size}"
            )

        # Roundtrip-verify 3 random transitions byte-identical
        if expected_size > 0:
            num_samples = min(3, expected_size)
            rng = np.random.RandomState(42)
            check_indices = rng.choice(expected_size, size=num_samples, replace=False)
            for idx in check_indices:
                if not np.array_equal(self.obs[idx], state["obs"][idx]):
                    raise RuntimeError("SAPNA buffer roundtrip failed: obs mismatch — aborting to protect the diary")
                if self.action[idx] != state["action"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: action mismatch — aborting to protect the diary")
                if self.reward[idx] != state["reward"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: reward mismatch — aborting to protect the diary")
                if self.done[idx] != state["done"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: done mismatch — aborting to protect the diary")
                if self.episode_id[idx] != state["episode_id"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: episode_id mismatch — aborting to protect the diary")
                if self.step_in_ep[idx] != state["step_in_ep"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: step_in_ep mismatch — aborting to protect the diary")
                if self.milestone[idx] != state["milestone"][idx]:
                    raise RuntimeError("SAPNA buffer roundtrip failed: milestone mismatch — aborting to protect the diary")
