"""
Replay Buffer for Learn-a-World.
Stores self-play experience (obs, action, next_obs, reward, done) as compact uint8 numpy arrays.
Features:
- Fixed capacity with FIFO ring-buffer overwrite.
- Sharded chunked persistence (.npz) for Colab Google Drive / Local storage.
- Fast PyTorch tensor batch sampling with automatic device transfer.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing image transitions.
    Observations stored as compact uint8 arrays (H, W, C) to minimize RAM footprint.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_shape: Tuple[int, int, int] = (64, 64, 3),
        action_dim: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Preallocate memory blocks
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.integer],
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Appends a single transition into the buffer."""
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of transitions.
        Observations are converted from uint8 [0, 255] to float32 [0.0, 1.0].
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions; buffer only has {self.size}.")

        target_device = torch.device(device) if device is not None else self.device
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert to float32 tensors with channels-first layout [B, C, H, W] for PyTorch CNNs
        b_obs = torch.from_numpy(self.obs[indices]).permute(0, 3, 1, 2).float() / 255.0
        b_next_obs = torch.from_numpy(self.next_obs[indices]).permute(0, 3, 1, 2).float() / 255.0
        b_actions = torch.from_numpy(self.actions[indices]).long()
        b_rewards = torch.from_numpy(self.rewards[indices]).float()
        b_dones = torch.from_numpy(self.dones[indices]).bool()

        return (
            b_obs.to(target_device),
            b_actions.to(target_device),
            b_next_obs.to(target_device),
            b_rewards.to(target_device),
            b_dones.to(target_device),
        )

    def save_chunk(self, file_path: Union[str, Path]) -> None:
        """Saves current buffer content as a compressed .npz shard."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            obs=self.obs[: self.size],
            next_obs=self.next_obs[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            dones=self.dones[: self.size],
            ptr=self.ptr,
            size=self.size,
        )

    def load_chunk(self, file_path: Union[str, Path]) -> None:
        """Loads a compressed .npz shard into the buffer."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Buffer chunk not found: {file_path}")

        data = np.load(file_path)
        loaded_size = int(data["size"])
        assert loaded_size <= self.capacity, f"Loaded size {loaded_size} exceeds capacity {self.capacity}"

        self.obs[:loaded_size] = data["obs"]
        self.next_obs[:loaded_size] = data["next_obs"]
        self.actions[:loaded_size] = data["actions"]
        self.rewards[:loaded_size] = data["rewards"]
        self.dones[:loaded_size] = data["dones"]
        self.size = loaded_size
        self.ptr = int(data["ptr"]) % self.capacity

    def __len__(self) -> int:
        return self.size
