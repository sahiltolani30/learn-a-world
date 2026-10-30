"""
src/memory/buffer.py

Two-Tier Persistent Replay Buffer for Crafter & Atari JEPA World Model
with Milestone-Prioritized Sampling and n-step Trajectory Support (ROOT_CAUSE_FIX_PLAN.md):
- Tier 1 (Rolling FIFO, default capacity 50,000-100,000): stores recent trajectory transitions.
- Tier 2 (Curated Milestone Archive, default capacity 20,000): permanently captures
  milestone discoveries and high-novelty breakthroughs.
- Milestone Prioritization (C2): samples milestones with elevated priority in T1 (25%) and T2 (80%),
  with retroactive 30-step trailing context window flagging.
- Sequence Metadata & n-step Sampling (C3): stores episode_id and step_in_ep to enable
  multi-step (5-step) discounted returns and bootstrap states along trajectories.
- RAM Optimization: stores observations as compact uint8 (C, H, W), decompressing
  to normalized float32 [0.0, 1.0] only upon batch sampling.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from collections import deque
import numpy as np
import torch


def _to_uint8_chw(obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Standardize observation tensor/array into uint8 (C, H, W) format.
    
    Handles:
    - (H, W, C) uint8 e.g. Crafter raw frames (64, 64, 3) -> (3, 64, 64)
    - (C, H, W) uint8 or float in [0, 1]
    - PyTorch tensors and NumPy arrays
    """
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    elif isinstance(obs, np.ndarray):
        arr = obs.copy()
    else:
        arr = np.asarray(obs)

    # Scale floats in [0, 1] to [0, 255] uint8
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.05 and arr.min() >= -0.05:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Standardize to (C, H, W)
    if arr.ndim == 3:
        # Check if channels are last: e.g. (64, 64, 3) or (64, 64, 1)
        if arr.shape[2] in (1, 3, 4, 9) and arr.shape[0] not in (1, 3, 4, 9):
            arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        # (H, W) -> (1, H, W)
        arr = np.expand_dims(arr, axis=0)

    return np.ascontiguousarray(arr, dtype=np.uint8)


class TwoTierReplayBuffer:
    """
    Two-Tier Prioritized Replay Buffer.
    
    - Tier 1: Rolling FIFO buffer tracking recent policy transitions.
    - Tier 2: Curated milestone archive storing top 5% epistemic curiosity transitions
              and major game milestones (achievements, survival breakthroughs).
    - Mixed Sampler: 70% Tier 1 + 30% Tier 2 to eliminate catastrophic forgetting.
    - RAM Efficient: compact uint8 storage, lazy float32 decompression at batch retrieval.
    """

    def __init__(
        self,
        tier1_capacity: int = 100_000,
        tier2_capacity: int = 20_000,
        novelty_percentile: float = 0.95,
        default_tier2_ratio: float = 0.3,
        milestone_t1_priority: float = 0.25,
        milestone_t2_priority: float = 0.80,
        seed: Optional[int] = None,
    ):
        """
        Initialize TwoTierReplayBuffer.
        
        Args:
            tier1_capacity: Max capacity for rolling FIFO buffer (Tier 1).
            tier2_capacity: Max capacity for curated milestone archive (Tier 2).
            novelty_percentile: Epistemic novelty quantile threshold for Tier 2 (default top 5% = 0.95).
            default_tier2_ratio: Default ratio of Tier 2 samples in mixed batch (default 0.3 = 30%).
            milestone_t1_priority: Sampling priority for milestone transitions in Tier 1.
            milestone_t2_priority: Sampling priority for milestone transitions in Tier 2.
            seed: Optional random seed for reproducible sampling.
        """
        if tier1_capacity <= 0:
            raise ValueError(f"tier1_capacity must be positive, got {tier1_capacity}")
        if tier2_capacity <= 0:
            raise ValueError(f"tier2_capacity must be positive, got {tier2_capacity}")

        self.tier1_capacity = tier1_capacity
        self.tier2_capacity = tier2_capacity
        self.novelty_percentile = novelty_percentile
        self.default_tier2_ratio = default_tier2_ratio
        self.milestone_t1_priority = milestone_t1_priority
        self.milestone_t2_priority = milestone_t2_priority
        self.rng = np.random.RandomState(seed)

        # Lazy allocation placeholders for Tier 1 preallocated storage
        self._t1_allocated = False
        self._obs_shape: Optional[Tuple[int, ...]] = None

        self.t1_obs: Optional[np.ndarray] = None
        self.t1_next_obs: Optional[np.ndarray] = None
        self.t1_actions: Optional[np.ndarray] = None
        self.t1_rewards: Optional[np.ndarray] = None
        self.t1_dones: Optional[np.ndarray] = None
        self.t1_int_rewards: Optional[np.ndarray] = None
        self.t1_epistemic_errors: Optional[np.ndarray] = None
        self.t1_is_milestone: Optional[np.ndarray] = None
        self.t1_episode_id: Optional[np.ndarray] = None
        self.t1_step_in_ep: Optional[np.ndarray] = None

        self.t1_ptr: int = 0
        self.t1_size: int = 0

        # Tier 2 Curated Milestone Archive storage
        self._t2_allocated = False
        self.t2_obs: Optional[np.ndarray] = None
        self.t2_next_obs: Optional[np.ndarray] = None
        self.t2_actions: Optional[np.ndarray] = None
        self.t2_rewards: Optional[np.ndarray] = None
        self.t2_dones: Optional[np.ndarray] = None
        self.t2_int_rewards: Optional[np.ndarray] = None
        self.t2_epistemic_errors: Optional[np.ndarray] = None
        self.t2_is_milestone: Optional[np.ndarray] = None
        self.t2_episode_id: Optional[np.ndarray] = None
        self.t2_step_in_ep: Optional[np.ndarray] = None

        self.t2_size: int = 0

        # Rolling window for calculating epistemic novelty percentiles
        self._recent_novelties: deque = deque(maxlen=2000)

    def _allocate_tier1(self, obs_shape: Tuple[int, ...]) -> None:
        """Preallocate contiguous memory for Tier 1 buffer."""
        self._obs_shape = obs_shape
        self.t1_obs = np.empty((self.tier1_capacity, *obs_shape), dtype=np.uint8)
        self.t1_next_obs = np.empty((self.tier1_capacity, *obs_shape), dtype=np.uint8)
        self.t1_actions = np.empty(self.tier1_capacity, dtype=np.int64)
        self.t1_rewards = np.empty(self.tier1_capacity, dtype=np.float32)
        self.t1_dones = np.empty(self.tier1_capacity, dtype=np.bool_)
        self.t1_int_rewards = np.empty(self.tier1_capacity, dtype=np.float32)
        self.t1_epistemic_errors = np.empty(self.tier1_capacity, dtype=np.float32)
        self.t1_is_milestone = np.empty(self.tier1_capacity, dtype=np.bool_)
        self.t1_episode_id = np.empty(self.tier1_capacity, dtype=np.int64)
        self.t1_step_in_ep = np.empty(self.tier1_capacity, dtype=np.int64)
        self._t1_allocated = True

    def _allocate_tier2(self, obs_shape: Tuple[int, ...]) -> None:
        """Preallocate contiguous memory for Tier 2 buffer."""
        self.t2_obs = np.empty((self.tier2_capacity, *obs_shape), dtype=np.uint8)
        self.t2_next_obs = np.empty((self.tier2_capacity, *obs_shape), dtype=np.uint8)
        self.t2_actions = np.empty(self.tier2_capacity, dtype=np.int64)
        self.t2_rewards = np.empty(self.tier2_capacity, dtype=np.float32)
        self.t2_dones = np.empty(self.tier2_capacity, dtype=np.bool_)
        self.t2_int_rewards = np.empty(self.tier2_capacity, dtype=np.float32)
        self.t2_epistemic_errors = np.empty(self.tier2_capacity, dtype=np.float32)
        self.t2_is_milestone = np.empty(self.tier2_capacity, dtype=np.bool_)
        self.t2_episode_id = np.empty(self.tier2_capacity, dtype=np.int64)
        self.t2_step_in_ep = np.empty(self.tier2_capacity, dtype=np.int64)
        self._t2_allocated = True

    def flag_milestone_at(self, index: int) -> None:
        """Mark the T1 slot at `index` as milestone (backward credit window)."""
        if 0 <= index < self.tier1_capacity and self.t1_is_milestone is not None:
            self.t1_is_milestone[index] = True

    @property
    def size(self) -> int:
        """Total transitions stored across Tier 1 and Tier 2."""
        return self.t1_size + self.t2_size

    def __len__(self) -> int:
        return self.t1_size

    def get_novelty_threshold(self) -> float:
        """
        Calculate the current epistemic novelty threshold for Tier 2 admission.
        Returns the (novelty_percentile * 100)-th percentile of recent transitions.
        """
        if len(self._recent_novelties) < 20:
            return float("inf")

        arr = np.array(self._recent_novelties)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # If there is negligible spread across recent transitions, no items qualify on novelty alone
        if (max_val - min_val) < 1e-4:
            return float("inf")

        pct_val = float(np.percentile(arr, self.novelty_percentile * 100))
        # Ensure threshold is strictly above baseline minimum
        return max(pct_val, min_val + 1e-4)

    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[int, np.integer, torch.Tensor],
        reward: Union[float, np.floating, torch.Tensor],
        next_obs: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.bool_, torch.Tensor],
        r_int: Union[float, np.floating, torch.Tensor] = 0.0,
        is_milestone: bool = False,
        epistemic_error: Optional[float] = None,
        episode_id: int = -1,
        step_in_ep: int = -1,
    ) -> bool:
        """
        Insert transition into TwoTierReplayBuffer.
        
        Always inserts into Tier 1 rolling buffer.
        Inserts into Tier 2 if transition is marked as a milestone breakthrough
        OR its epistemic curiosity exceeds the top 5% novelty percentile.
        
        Args:
            obs: Observation tensor/array.
            action: Discrete action taken.
            reward: Task external reward.
            next_obs: Next observation tensor/array.
            done: Episode termination flag.
            r_int: Epistemic intrinsic reward.
            is_milestone: True if transition unlocked achievement or broke survival record.
            epistemic_error: Raw predictor ensemble disagreement (defaults to r_int if None).
            
        Returns:
            admitted_to_tier2: True if transition was captured in Tier 2 Archive.
        """
        # Convert observations to compact uint8 (C, H, W)
        obs_u8 = _to_uint8_chw(obs)
        next_obs_u8 = _to_uint8_chw(next_obs)

        if not self._t1_allocated:
            self._allocate_tier1(obs_u8.shape)
        if not self._t2_allocated:
            self._allocate_tier2(obs_u8.shape)

        # Cast scalar fields
        act = int(action.item() if isinstance(action, torch.Tensor) else action)
        rew = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
        d = bool(done.item() if isinstance(done, torch.Tensor) else done)
        r_i = float(r_int.item() if isinstance(r_int, torch.Tensor) else r_int)
        e_err = float(epistemic_error) if epistemic_error is not None else r_i

        # 1. Insert into Tier 1 (FIFO circular buffer)
        idx = self.t1_ptr
        self.t1_obs[idx] = obs_u8
        self.t1_next_obs[idx] = next_obs_u8
        self.t1_actions[idx] = act
        self.t1_rewards[idx] = rew
        self.t1_dones[idx] = d
        self.t1_int_rewards[idx] = r_i
        self.t1_epistemic_errors[idx] = e_err
        self.t1_is_milestone[idx] = is_milestone
        self.t1_episode_id[idx] = int(episode_id)
        self.t1_step_in_ep[idx] = int(step_in_ep)

        self.t1_ptr = (self.t1_ptr + 1) % self.tier1_capacity
        self.t1_size = min(self.t1_size + 1, self.tier1_capacity)

        # Track recent epistemic curiosity distribution
        self._recent_novelties.append(e_err)

        # 2. Evaluate Tier 2 Admission Criteria
        admit_to_tier2 = False
        threshold = self.get_novelty_threshold()
        if is_milestone or (threshold < float("inf") and e_err >= threshold):
            admit_to_tier2 = True
            self._insert_tier2(
                obs=obs_u8,
                next_obs=next_obs_u8,
                action=act,
                reward=rew,
                done=d,
                r_int=r_i,
                epistemic_error=e_err,
                is_milestone=is_milestone,
                episode_id=int(episode_id),
                step_in_ep=int(step_in_ep),
            )

        return admit_to_tier2

    def _insert_tier2(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        r_int: float,
        epistemic_error: float,
        is_milestone: bool,
        episode_id: int = -1,
        step_in_ep: int = -1,
    ) -> None:
        """Insert or replace transition in Tier 2 Milestone Archive."""
        if self.t2_size < self.tier2_capacity:
            idx = self.t2_size
            self.t2_size += 1
        else:
            # Buffer is full: soft priority replacement.
            # Milestones are preserved over non-milestones;
            # among non-milestones, replace the one with the lowest epistemic error.
            non_milestone_indices = np.where(~self.t2_is_milestone[:self.tier2_capacity])[0]
            if len(non_milestone_indices) > 0:
                # Find non-milestone with lowest error
                min_sub_idx = np.argmin(self.t2_epistemic_errors[non_milestone_indices])
                replace_idx = non_milestone_indices[min_sub_idx]
                # Only replace if new item is a milestone or has higher error
                if not is_milestone and epistemic_error <= self.t2_epistemic_errors[replace_idx]:
                    return  # Reject: existing item has higher priority
                idx = replace_idx
            else:
                # All entries are milestones; replace lowest error milestone if new one is higher
                min_idx = np.argmin(self.t2_epistemic_errors[:self.tier2_capacity])
                if epistemic_error <= self.t2_epistemic_errors[min_idx]:
                    return
                idx = min_idx

        self.t2_obs[idx] = obs
        self.t2_next_obs[idx] = next_obs
        self.t2_actions[idx] = action
        self.t2_rewards[idx] = reward
        self.t2_dones[idx] = done
        self.t2_int_rewards[idx] = r_int
        self.t2_epistemic_errors[idx] = epistemic_error
        self.t2_is_milestone[idx] = is_milestone
        self.t2_episode_id[idx] = int(episode_id)
        self.t2_step_in_ep[idx] = int(step_in_ep)

    def sample(
        self,
        batch_size: int = 64,
        tier2_ratio: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a mixed batch of transitions: (1 - tier2_ratio) from Tier 1,
        and tier2_ratio from Tier 2.
        
        Args:
            batch_size: Total transitions to sample.
            tier2_ratio: Fraction of batch from Tier 2 (defaults to self.default_tier2_ratio, 0.3).
            device: Target torch device ('cpu' or 'cuda').
            
        Returns:
            Dictionary with tensors:
            - 'obs': (B, C, H, W) float32 in [0.0, 1.0]
            - 'actions': (B,) int64
            - 'rewards': (B,) float32
            - 'next_obs': (B, C, H, W) float32 in [0.0, 1.0]
            - 'dones': (B,) bool
            - 'int_rewards': (B,) float32
            - 'epistemic_errors': (B,) float32
            - 'is_tier2': (B,) bool mask
        """
        if self.t1_size == 0 and self.t2_size == 0:
            raise ValueError("Cannot sample from empty TwoTierReplayBuffer.")

        ratio = tier2_ratio if tier2_ratio is not None else self.default_tier2_ratio
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"tier2_ratio must be in [0.0, 1.0], got {ratio}")

        # Compute desired counts
        target_n2 = int(round(batch_size * ratio))

        # Adjust for available items in Tier 2
        actual_n2 = min(target_n2, self.t2_size)
        actual_n1 = batch_size - actual_n2

        # If Tier 1 doesn't have enough samples, pull more from Tier 2 if possible
        if actual_n1 > self.t1_size:
            shortfall = actual_n1 - self.t1_size
            can_borrow = min(shortfall, self.t2_size - actual_n2)
            actual_n2 += can_borrow
            actual_n1 = batch_size - actual_n2

        if actual_n1 > self.t1_size or (actual_n1 + actual_n2) < batch_size:
            total_avail = self.t1_size + self.t2_size
            raise ValueError(
                f"Requested batch_size={batch_size}, but buffer contains only "
                f"{total_avail} total transitions (Tier 1: {self.t1_size}, Tier 2: {self.t2_size})."
            )

        # Sample indices
        if actual_n1 > 0:
            ms1 = np.nonzero(self.t1_is_milestone[:self.t1_size])[0]
            if len(ms1) > 0 and self.milestone_t1_priority > 0.0:
                rand_vals = self.rng.random_sample(size=actual_n1)
                pick_ms = rand_vals < self.milestone_t1_priority
                t1_indices = np.empty(actual_n1, dtype=np.int64)
                n_ms = int(np.sum(pick_ms))
                if n_ms > 0:
                    t1_indices[pick_ms] = ms1[self.rng.randint(0, len(ms1), size=n_ms)]
                if actual_n1 - n_ms > 0:
                    t1_indices[~pick_ms] = self.rng.randint(0, self.t1_size, size=actual_n1 - n_ms)
            else:
                t1_indices = self.rng.randint(0, self.t1_size, size=actual_n1)
        else:
            t1_indices = np.array([], dtype=np.int64)

        if actual_n2 > 0:
            ms2 = np.nonzero(self.t2_is_milestone[:self.t2_size])[0]
            if len(ms2) > 0 and self.milestone_t2_priority > 0.0:
                rand_vals = self.rng.random_sample(size=actual_n2)
                pick_ms = rand_vals < self.milestone_t2_priority
                t2_indices = np.empty(actual_n2, dtype=np.int64)
                n_ms = int(np.sum(pick_ms))
                if n_ms > 0:
                    t2_indices[pick_ms] = ms2[self.rng.randint(0, len(ms2), size=n_ms)]
                if actual_n2 - n_ms > 0:
                    t2_indices[~pick_ms] = self.rng.randint(0, self.t2_size, size=actual_n2 - n_ms)
            else:
                t2_indices = self.rng.randint(0, self.t2_size, size=actual_n2)
        else:
            t2_indices = np.array([], dtype=np.int64)

        # Gather data slices
        if actual_n1 > 0 and actual_n2 > 0:
            b_obs = np.concatenate([self.t1_obs[t1_indices], self.t2_obs[t2_indices]], axis=0)
            b_next_obs = np.concatenate([self.t1_next_obs[t1_indices], self.t2_next_obs[t2_indices]], axis=0)
            b_actions = np.concatenate([self.t1_actions[t1_indices], self.t2_actions[t2_indices]], axis=0)
            b_rewards = np.concatenate([self.t1_rewards[t1_indices], self.t2_rewards[t2_indices]], axis=0)
            b_dones = np.concatenate([self.t1_dones[t1_indices], self.t2_dones[t2_indices]], axis=0)
            b_int_rewards = np.concatenate([self.t1_int_rewards[t1_indices], self.t2_int_rewards[t2_indices]], axis=0)
            b_errors = np.concatenate([self.t1_epistemic_errors[t1_indices], self.t2_epistemic_errors[t2_indices]], axis=0)
            b_is_t2 = np.concatenate([np.zeros(actual_n1, dtype=np.bool_), np.ones(actual_n2, dtype=np.bool_)], axis=0)
            b_is_ms = np.concatenate([self.t1_is_milestone[t1_indices], self.t2_is_milestone[t2_indices]], axis=0)
        elif actual_n1 > 0:
            b_obs = self.t1_obs[t1_indices]
            b_next_obs = self.t1_next_obs[t1_indices]
            b_actions = self.t1_actions[t1_indices]
            b_rewards = self.t1_rewards[t1_indices]
            b_dones = self.t1_dones[t1_indices]
            b_int_rewards = self.t1_int_rewards[t1_indices]
            b_errors = self.t1_epistemic_errors[t1_indices]
            b_is_t2 = np.zeros(actual_n1, dtype=np.bool_)
            b_is_ms = self.t1_is_milestone[t1_indices]
        else:
            b_obs = self.t2_obs[t2_indices]
            b_next_obs = self.t2_next_obs[t2_indices]
            b_actions = self.t2_actions[t2_indices]
            b_rewards = self.t2_rewards[t2_indices]
            b_dones = self.t2_dones[t2_indices]
            b_int_rewards = self.t2_int_rewards[t2_indices]
            b_errors = self.t2_epistemic_errors[t2_indices]
            b_is_t2 = np.ones(actual_n2, dtype=np.bool_)
            b_is_ms = self.t2_is_milestone[t2_indices]

        # RAM-optimized lazy conversion to float32 [0.0, 1.0] upon sampling
        dev = torch.device(device)
        return {
            "obs": torch.from_numpy(b_obs).to(device=dev, dtype=torch.float32) / 255.0,
            "actions": torch.from_numpy(b_actions).to(device=dev, dtype=torch.int64),
            "rewards": torch.from_numpy(b_rewards).to(device=dev, dtype=torch.float32),
            "next_obs": torch.from_numpy(b_next_obs).to(device=dev, dtype=torch.float32) / 255.0,
            "dones": torch.from_numpy(b_dones).to(device=dev, dtype=torch.bool),
            "int_rewards": torch.from_numpy(b_int_rewards).to(device=dev, dtype=torch.float32),
            "epistemic_errors": torch.from_numpy(b_errors).to(device=dev, dtype=torch.float32),
            "is_tier2": torch.from_numpy(b_is_t2).to(device=dev, dtype=torch.bool),
            "is_milestone": torch.from_numpy(b_is_ms).to(device=dev, dtype=torch.bool),
        }

    def sample_nstep(
        self,
        batch_size: int = 64,
        n: int = 5,
        gamma: float = 0.99,
        tier2_ratio: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a mixed batch of transitions with n-step discounted returns and bootstrap states.

        For Tier 1 entries with valid sequence metadata (episode_id >= 0), walks up to n steps forward
        along the same episode trajectory, accumulating discounted rewards and identifying the bootstrap observation.
        Falls back cleanly to 1-step for Tier 2 transitions or invalid/interrupted trajectories.
        """
        if self.t1_size == 0 and self.t2_size == 0:
            raise ValueError("Cannot sample from empty TwoTierReplayBuffer.")

        ratio = tier2_ratio if tier2_ratio is not None else self.default_tier2_ratio
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"tier2_ratio must be in [0.0, 1.0], got {ratio}")

        target_n2 = int(round(batch_size * ratio))
        actual_n2 = min(target_n2, self.t2_size)
        actual_n1 = batch_size - actual_n2

        if actual_n1 > self.t1_size:
            shortfall = actual_n1 - self.t1_size
            can_borrow = min(shortfall, self.t2_size - actual_n2)
            actual_n2 += can_borrow
            actual_n1 = batch_size - actual_n2

        if actual_n1 > self.t1_size or (actual_n1 + actual_n2) < batch_size:
            total_avail = self.t1_size + self.t2_size
            raise ValueError(
                f"Requested batch_size={batch_size}, but buffer contains only "
                f"{total_avail} total transitions (Tier 1: {self.t1_size}, Tier 2: {self.t2_size})."
            )

        # 1. Sample indices respecting C2 milestone priorities
        if actual_n1 > 0:
            ms1 = np.nonzero(self.t1_is_milestone[:self.t1_size])[0]
            if len(ms1) > 0 and self.milestone_t1_priority > 0.0:
                rand_vals = self.rng.random_sample(size=actual_n1)
                pick_ms = rand_vals < self.milestone_t1_priority
                t1_indices = np.empty(actual_n1, dtype=np.int64)
                n_ms = int(np.sum(pick_ms))
                if n_ms > 0:
                    t1_indices[pick_ms] = ms1[self.rng.randint(0, len(ms1), size=n_ms)]
                if actual_n1 - n_ms > 0:
                    t1_indices[~pick_ms] = self.rng.randint(0, self.t1_size, size=actual_n1 - n_ms)
            else:
                t1_indices = self.rng.randint(0, self.t1_size, size=actual_n1)
        else:
            t1_indices = np.array([], dtype=np.int64)

        if actual_n2 > 0:
            ms2 = np.nonzero(self.t2_is_milestone[:self.t2_size])[0]
            if len(ms2) > 0 and self.milestone_t2_priority > 0.0:
                rand_vals = self.rng.random_sample(size=actual_n2)
                pick_ms = rand_vals < self.milestone_t2_priority
                t2_indices = np.empty(actual_n2, dtype=np.int64)
                n_ms = int(np.sum(pick_ms))
                if n_ms > 0:
                    t2_indices[pick_ms] = ms2[self.rng.randint(0, len(ms2), size=n_ms)]
                if actual_n2 - n_ms > 0:
                    t2_indices[~pick_ms] = self.rng.randint(0, self.t2_size, size=actual_n2 - n_ms)
            else:
                t2_indices = self.rng.randint(0, self.t2_size, size=actual_n2)
        else:
            t2_indices = np.array([], dtype=np.int64)

        # Base arrays
        if actual_n1 > 0 and actual_n2 > 0:
            b_obs = np.concatenate([self.t1_obs[t1_indices], self.t2_obs[t2_indices]], axis=0)
            b_next_obs = np.concatenate([self.t1_next_obs[t1_indices], self.t2_next_obs[t2_indices]], axis=0)
            b_actions = np.concatenate([self.t1_actions[t1_indices], self.t2_actions[t2_indices]], axis=0)
            b_rewards = np.concatenate([self.t1_rewards[t1_indices], self.t2_rewards[t2_indices]], axis=0)
            b_dones = np.concatenate([self.t1_dones[t1_indices], self.t2_dones[t2_indices]], axis=0)
            b_int_rewards = np.concatenate([self.t1_int_rewards[t1_indices], self.t2_int_rewards[t2_indices]], axis=0)
            b_errors = np.concatenate([self.t1_epistemic_errors[t1_indices], self.t2_epistemic_errors[t2_indices]], axis=0)
            b_is_t2 = np.concatenate([np.zeros(actual_n1, dtype=np.bool_), np.ones(actual_n2, dtype=np.bool_)], axis=0)
            b_is_ms = np.concatenate([self.t1_is_milestone[t1_indices], self.t2_is_milestone[t2_indices]], axis=0)
        elif actual_n1 > 0:
            b_obs = self.t1_obs[t1_indices]
            b_next_obs = self.t1_next_obs[t1_indices]
            b_actions = self.t1_actions[t1_indices]
            b_rewards = self.t1_rewards[t1_indices]
            b_dones = self.t1_dones[t1_indices]
            b_int_rewards = self.t1_int_rewards[t1_indices]
            b_errors = self.t1_epistemic_errors[t1_indices]
            b_is_t2 = np.zeros(actual_n1, dtype=np.bool_)
            b_is_ms = self.t1_is_milestone[t1_indices]
        else:
            b_obs = self.t2_obs[t2_indices]
            b_next_obs = self.t2_next_obs[t2_indices]
            b_actions = self.t2_actions[t2_indices]
            b_rewards = self.t2_rewards[t2_indices]
            b_dones = self.t2_dones[t2_indices]
            b_int_rewards = self.t2_int_rewards[t2_indices]
            b_errors = self.t2_epistemic_errors[t2_indices]
            b_is_t2 = np.ones(actual_n2, dtype=np.bool_)
            b_is_ms = self.t2_is_milestone[t2_indices]

        # 2. Compute n-step returns and bootstrap states
        nstep_returns = np.empty(batch_size, dtype=np.float32)
        boot_obs = np.empty_like(b_next_obs)
        boot_gamma = np.empty(batch_size, dtype=np.float32)
        boot_valid = np.empty(batch_size, dtype=np.bool_)

        cap = self.tier1_capacity

        # Process Tier 1 items
        for batch_i in range(actual_n1):
            idx = int(t1_indices[batch_i])
            base_ep = int(self.t1_episode_id[idx]) if self.t1_episode_id is not None else -1
            base_step = int(self.t1_step_in_ep[idx]) if self.t1_step_in_ep is not None else -1

            # If no sequence metadata available, 1-step fallback
            if base_ep < 0 or base_step < 0:
                nstep_returns[batch_i] = self.t1_rewards[idx]
                boot_obs[batch_i] = self.t1_next_obs[idx]
                boot_gamma[batch_i] = gamma
                boot_valid[batch_i] = not self.t1_dones[idx]
                continue

            # Start chain at k=0
            R = float(self.t1_rewards[idx])
            last_valid_slot = idx
            num_terms = 1
            is_terminal = bool(self.t1_dones[idx])

            if not is_terminal:
                for k in range(1, n):
                    if k >= self.t1_size:
                        break
                    cand_slot = (idx + k) % cap
                    # Chain must belong to same episode and sequential step
                    if (
                        self.t1_episode_id[cand_slot] != base_ep
                        or self.t1_step_in_ep[cand_slot] != base_step + k
                    ):
                        break
                    R += (gamma ** k) * float(self.t1_rewards[cand_slot])
                    last_valid_slot = cand_slot
                    num_terms += 1
                    if self.t1_dones[cand_slot]:
                        is_terminal = True
                        break

            nstep_returns[batch_i] = R
            boot_obs[batch_i] = self.t1_next_obs[last_valid_slot]
            boot_gamma[batch_i] = float(gamma ** num_terms)
            boot_valid[batch_i] = not is_terminal

        # Process Tier 2 items (1-step fallback)
        for t2_i in range(actual_n2):
            batch_i = actual_n1 + t2_i
            idx = int(t2_indices[t2_i])
            nstep_returns[batch_i] = self.t2_rewards[idx]
            boot_obs[batch_i] = self.t2_next_obs[idx]
            boot_gamma[batch_i] = gamma
            boot_valid[batch_i] = not self.t2_dones[idx]

        dev = torch.device(device)
        return {
            "obs": torch.from_numpy(b_obs).to(device=dev, dtype=torch.float32) / 255.0,
            "actions": torch.from_numpy(b_actions).to(device=dev, dtype=torch.int64),
            "rewards": torch.from_numpy(b_rewards).to(device=dev, dtype=torch.float32),
            "next_obs": torch.from_numpy(b_next_obs).to(device=dev, dtype=torch.float32) / 255.0,
            "dones": torch.from_numpy(b_dones).to(device=dev, dtype=torch.bool),
            "int_rewards": torch.from_numpy(b_int_rewards).to(device=dev, dtype=torch.float32),
            "epistemic_errors": torch.from_numpy(b_errors).to(device=dev, dtype=torch.float32),
            "is_tier2": torch.from_numpy(b_is_t2).to(device=dev, dtype=torch.bool),
            "is_milestone": torch.from_numpy(b_is_ms).to(device=dev, dtype=torch.bool),
            "nstep_return": torch.from_numpy(nstep_returns).to(device=dev, dtype=torch.float32),
            "boot_obs": torch.from_numpy(boot_obs).to(device=dev, dtype=torch.float32) / 255.0,
            "boot_gamma": torch.from_numpy(boot_gamma).to(device=dev, dtype=torch.float32),
            "boot_valid": torch.from_numpy(boot_valid).to(device=dev, dtype=torch.bool),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer state for checkpoint persistence."""
        return {
            "tier1_capacity": self.tier1_capacity,
            "tier2_capacity": self.tier2_capacity,
            "novelty_percentile": self.novelty_percentile,
            "default_tier2_ratio": self.default_tier2_ratio,
            "t1_ptr": self.t1_ptr,
            "t1_size": self.t1_size,
            "t2_size": self.t2_size,
            "obs_shape": self._obs_shape,
            "milestone_t1_priority": self.milestone_t1_priority,
            "milestone_t2_priority": self.milestone_t2_priority,
            "t1_obs": self.t1_obs[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_next_obs": self.t1_next_obs[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_actions": self.t1_actions[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_rewards": self.t1_rewards[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_dones": self.t1_dones[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_int_rewards": self.t1_int_rewards[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_epistemic_errors": self.t1_epistemic_errors[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_is_milestone": self.t1_is_milestone[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_episode_id": self.t1_episode_id[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t1_step_in_ep": self.t1_step_in_ep[:self.t1_size] if self._t1_allocated and self.t1_size > 0 else None,
            "t2_obs": self.t2_obs[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_next_obs": self.t2_next_obs[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_actions": self.t2_actions[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_rewards": self.t2_rewards[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_dones": self.t2_dones[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_int_rewards": self.t2_int_rewards[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_epistemic_errors": self.t2_epistemic_errors[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_is_milestone": self.t2_is_milestone[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_episode_id": self.t2_episode_id[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "t2_step_in_ep": self.t2_step_in_ep[:self.t2_size] if self._t2_allocated and self.t2_size > 0 else None,
            "recent_novelties": list(self._recent_novelties),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay buffer state from serialized checkpoint."""
        loaded_cap = state.get("tier1_capacity", self.tier1_capacity)
        self.tier1_capacity = min(loaded_cap, 50_000)
        self.tier2_capacity = state.get("tier2_capacity", self.tier2_capacity)
        self.novelty_percentile = state.get("novelty_percentile", self.novelty_percentile)
        self.default_tier2_ratio = state.get("default_tier2_ratio", self.default_tier2_ratio)
        self.milestone_t1_priority = state.get("milestone_t1_priority", self.milestone_t1_priority)
        self.milestone_t2_priority = state.get("milestone_t2_priority", self.milestone_t2_priority)
        self.t1_ptr = state.get("t1_ptr", 0)
        self.t1_size = state.get("t1_size", 0)
        self.t2_size = state.get("t2_size", 0)

        obs_shape = state.get("obs_shape")
        if obs_shape is not None:
            self._allocate_tier1(tuple(obs_shape))
            self._allocate_tier2(tuple(obs_shape))

            if self.t1_size > 0 and "t1_obs" in state:
                t1_o = state.pop("t1_obs")
                if t1_o is not None:
                    self.t1_obs[:self.t1_size] = t1_o
                t1_no = state.pop("t1_next_obs")
                if t1_no is not None:
                    self.t1_next_obs[:self.t1_size] = t1_no
                self.t1_actions[:self.t1_size] = state.pop("t1_actions")
                self.t1_rewards[:self.t1_size] = state.pop("t1_rewards")
                self.t1_dones[:self.t1_size] = state.pop("t1_dones")
                self.t1_int_rewards[:self.t1_size] = state.pop("t1_int_rewards")
                self.t1_epistemic_errors[:self.t1_size] = state.pop("t1_epistemic_errors")
                t1_ms = state.pop("t1_is_milestone", None)
                if t1_ms is not None:
                    self.t1_is_milestone[:self.t1_size] = t1_ms
                else:
                    self.t1_is_milestone[:self.t1_size] = False
                t1_ep = state.pop("t1_episode_id", None)
                if t1_ep is not None:
                    self.t1_episode_id[:self.t1_size] = t1_ep
                else:
                    self.t1_episode_id[:self.t1_size] = -1
                t1_st = state.pop("t1_step_in_ep", None)
                if t1_st is not None:
                    self.t1_step_in_ep[:self.t1_size] = t1_st
                else:
                    self.t1_step_in_ep[:self.t1_size] = -1

            if self.t2_size > 0 and "t2_obs" in state:
                t2_o = state.pop("t2_obs")
                if t2_o is not None:
                    self.t2_obs[:self.t2_size] = t2_o
                t2_no = state.pop("t2_next_obs")
                if t2_no is not None:
                    self.t2_next_obs[:self.t2_size] = t2_no
                self.t2_actions[:self.t2_size] = state.pop("t2_actions")
                self.t2_rewards[:self.t2_size] = state.pop("t2_rewards")
                self.t2_dones[:self.t2_size] = state.pop("t2_dones")
                self.t2_int_rewards[:self.t2_size] = state.pop("t2_int_rewards")
                self.t2_epistemic_errors[:self.t2_size] = state.pop("t2_epistemic_errors")
                self.t2_is_milestone[:self.t2_size] = state.pop("t2_is_milestone")
                t2_ep = state.pop("t2_episode_id", None)
                if t2_ep is not None:
                    self.t2_episode_id[:self.t2_size] = t2_ep
                else:
                    self.t2_episode_id[:self.t2_size] = -1
                t2_st = state.pop("t2_step_in_ep", None)
                if t2_st is not None:
                    self.t2_step_in_ep[:self.t2_size] = t2_st
                else:
                    self.t2_step_in_ep[:self.t2_size] = -1

        if "recent_novelties" in state:
            self._recent_novelties = deque(state["recent_novelties"], maxlen=2000)

    def clear(self) -> None:
        """Reset and empty buffer contents."""
        self.t1_ptr = 0
        self.t1_size = 0
        self.t2_size = 0
        self._recent_novelties.clear()
