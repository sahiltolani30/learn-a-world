"""Sequence sampler for contiguous multi-horizon JEPA training."""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from src.data.replay_buffer import ReplayBuffer


class SequenceSampler:
    """Samples valid contiguous trajectory slices from ReplayBuffer.
    
    Guarantees:
    - Never crosses episode boundaries (terminated or truncated).
    - Preserves exact temporal alignment: context [t-3..t], future actions [t..t+7],
      targets at offsets 1, 2, 4, 8.
    - Context left-padding with validity mask for early-episode anchors.
    - 70% uniform / 30% recent sampling mixture.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        context_length: int = 4,
        prediction_horizon: int = 8,
        recent_fraction: float = 0.30,
    ):
        self.replay = replay_buffer
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.recent_fraction = recent_fraction

    def is_valid_anchor(self, t_idx: int) -> bool:
        """Verify whether an anchor index t_idx can support a full future rollout."""
        # Must have horizon transitions in buffer
        if self.replay.size < self.prediction_horizon:
            return False

        ep_id = self.replay.episode_ids[t_idx]
        t_step = self.replay.timesteps[t_idx]

        for k in range(self.prediction_horizon):
            idx = (t_idx + k) % self.replay.capacity
            # Episode continuity check
            if self.replay.episode_ids[idx] != ep_id:
                return False
            if self.replay.timesteps[idx] != t_step + k:
                return False
            # Intermediate steps cannot be done
            if k < self.prediction_horizon - 1:
                if self.replay.terminated[idx] or self.replay.truncated[idx]:
                    return False
        return True

    def get_valid_anchor_indices(self) -> List[int]:
        """Scan buffer for all valid anchor indices."""
        valid = []
        for i in range(self.replay.size):
            # Map index in active buffer
            if self.replay.total_added <= self.replay.capacity:
                idx = i
            else:
                idx = (self.replay.write_idx - self.replay.size + i) % self.replay.capacity
            if self.is_valid_anchor(idx):
                valid.append(idx)
        return valid

    def sample_one_sequence(self, valid_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """Sample a single contiguous sequence."""
        if valid_indices is None or len(valid_indices) == 0:
            valid_indices = self.get_valid_anchor_indices()
            if len(valid_indices) == 0:
                raise ValueError("No valid sequences found in replay buffer yet.")

        # 70% uniform, 30% recent
        if np.random.rand() < self.recent_fraction:
            # Pick from most recent 30% of valid indices
            recent_count = max(1, int(len(valid_indices) * 0.30))
            anchor_idx = valid_indices[np.random.randint(len(valid_indices) - recent_count, len(valid_indices))]
        else:
            anchor_idx = valid_indices[np.random.randint(0, len(valid_indices))]

        # Build context [t-3..t]
        ep_id = self.replay.episode_ids[anchor_idx]
        current_t = self.replay.timesteps[anchor_idx]
        
        obs_context = np.zeros((self.context_length, *self.replay.obs_shape), dtype=np.uint8)
        context_mask = np.zeros(self.context_length, dtype=bool)
        previous_actions = np.zeros(self.context_length, dtype=np.int64)

        # Slot context_length - 1 is O[t]
        obs_context[-1] = self.replay.obs_buffer[self.replay.transition_obs_idx[anchor_idx]]
        context_mask[-1] = True

        for c in range(1, self.context_length):
            slot = self.context_length - 1 - c
            prev_t = current_t - c
            if prev_t >= 0:
                prev_idx = (anchor_idx - c) % self.replay.capacity
                if (self.replay.episode_ids[prev_idx] == ep_id and 
                    self.replay.timesteps[prev_idx] == prev_t):
                    obs_context[slot] = self.replay.obs_buffer[self.replay.transition_obs_idx[prev_idx]]
                    context_mask[slot] = True
                    previous_actions[slot + 1] = int(self.replay.actions[prev_idx])

        # Build future actions [t..t+7]
        actions_future = np.zeros(self.prediction_horizon, dtype=np.int64)
        for k in range(self.prediction_horizon):
            idx = (anchor_idx + k) % self.replay.capacity
            actions_future[k] = int(self.replay.actions[idx])

        # Targets at offsets 1, 2, 4, 8
        target_offsets = [1, 2, 4, 8]
        targets = {}
        for off in target_offsets:
            step_idx = (anchor_idx + off - 1) % self.replay.capacity
            # Target frame is next_obs of step (anchor_idx + off - 1)
            t_frame_idx = self.replay.transition_next_obs_idx[step_idx]
            targets[off] = self.replay.obs_buffer[t_frame_idx]

        # Auxiliary targets for first step
        first_step_idx = anchor_idx
        reward_first = float(self.replay.rewards[first_step_idx])
        done_first = float(self.replay.terminated[first_step_idx] or self.replay.truncated[first_step_idx])

        return {
            "obs_context": obs_context,
            "context_mask": context_mask,
            "previous_actions": previous_actions,
            "actions_future": actions_future,
            "target_1": targets[1],
            "target_2": targets[2],
            "target_4": targets[4],
            "target_8": targets[8],
            "reward_first": np.array([reward_first], dtype=np.float32),
            "done_first": np.array([done_first], dtype=np.float32),
        }

    def sample_batch(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Sample a batch of sequences and convert to torch tensors with channel-first images."""
        valid_indices = self.get_valid_anchor_indices()
        if len(valid_indices) < batch_size:
            raise ValueError(f"Insufficient valid sequences ({len(valid_indices)}) for batch size {batch_size}")

        samples = [self.sample_one_sequence(valid_indices) for _ in range(batch_size)]
        
        # Collate
        # obs_context: [B, 4, 3, 64, 64] float32 in [0, 1]
        raw_obs_context = np.stack([s["obs_context"] for s in samples], axis=0)  # [B, 4, 64, 64, 3]
        obs_context_ch = np.transpose(raw_obs_context, (0, 1, 4, 2, 3))         # [B, 4, 3, 64, 64]
        
        def to_ch_first(arr_list):
            arr = np.stack(arr_list, axis=0) # [B, 64, 64, 3]
            return np.transpose(arr, (0, 3, 1, 2)) # [B, 3, 64, 64]

        batch = {
            "obs_context": torch.from_numpy(obs_context_ch).float().div_(255.0),
            "context_mask": torch.from_numpy(np.stack([s["context_mask"] for s in samples], axis=0)),
            "previous_actions": torch.from_numpy(np.stack([s["previous_actions"] for s in samples], axis=0)).long(),
            "actions_future": torch.from_numpy(np.stack([s["actions_future"] for s in samples], axis=0)).long(),
            "target_1": torch.from_numpy(to_ch_first([s["target_1"] for s in samples])).float().div_(255.0),
            "target_2": torch.from_numpy(to_ch_first([s["target_2"] for s in samples])).float().div_(255.0),
            "target_4": torch.from_numpy(to_ch_first([s["target_4"] for s in samples])).float().div_(255.0),
            "target_8": torch.from_numpy(to_ch_first([s["target_8"] for s in samples])).float().div_(255.0),
            "reward_first": torch.from_numpy(np.stack([s["reward_first"] for s in samples], axis=0)).float(),
            "done_first": torch.from_numpy(np.stack([s["done_first"] for s in samples], axis=0)).float(),
        }

        if device is not None:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        return batch
