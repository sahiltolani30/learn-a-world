"""Sequence-aware circular replay buffer storing raw uint8 observations."""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer storing raw RGB observations once.
    
    Stores up to `capacity` transitions and `capacity + 1` unique frames.
    Tracks episode boundaries to guarantee sampled sequences never cross episodes.
    """

    def __init__(self, capacity: int = 50000, obs_shape: Tuple[int, int, int] = (64, 64, 3)):
        self.capacity = capacity
        self.obs_shape = obs_shape
        
        # Buffer arrays
        # We preallocate observation storage
        self.obs_buffer = np.zeros((capacity + 1, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.timesteps = np.zeros(capacity, dtype=np.int32)
        self.transition_ids = np.zeros(capacity, dtype=np.int64)
        
        # Pointers and size
        self.write_idx = 0
        self.obs_write_idx = 0
        self.size = 0
        self.total_added = 0
        
        # Mapping from transition index to (obs_idx, next_obs_idx)
        self.transition_obs_idx = np.zeros(capacity, dtype=np.int32)
        self.transition_next_obs_idx = np.zeros(capacity, dtype=np.int32)
        
        # Episode tracking
        self.current_episode_id = 0
        self.current_timestep = 0
        self._last_obs_idx: Optional[int] = None

    def start_episode(self, initial_obs: np.ndarray, episode_id: Optional[int] = None) -> None:
        """Register the initial observation of a new episode."""
        if episode_id is not None:
            self.current_episode_id = episode_id
        else:
            self.current_episode_id += 1
            
        self.current_timestep = 0
        idx = self.obs_write_idx
        self.obs_buffer[idx] = initial_obs
        self._last_obs_idx = idx
        self.obs_write_idx = (self.obs_write_idx + 1) % (self.capacity + 1)

    def add(
        self,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Add a transition resulting from taking action from the previous observation."""
        if self._last_obs_idx is None:
            raise RuntimeError("Must call start_episode() before adding transitions!")
            
        t_idx = self.write_idx
        next_obs_idx = self.obs_write_idx
        self.obs_buffer[next_obs_idx] = next_obs
        self.obs_write_idx = (self.obs_write_idx + 1) % (self.capacity + 1)
        
        self.transition_obs_idx[t_idx] = self._last_obs_idx
        self.transition_next_obs_idx[t_idx] = next_obs_idx
        self.actions[t_idx] = action
        self.rewards[t_idx] = reward
        self.terminated[t_idx] = terminated
        self.truncated[t_idx] = truncated
        self.episode_ids[t_idx] = self.current_episode_id
        self.timesteps[t_idx] = self.current_timestep
        self.transition_ids[t_idx] = self.total_added
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        
        self.current_timestep += 1
        if terminated or truncated:
            self._last_obs_idx = None
        else:
            self._last_obs_idx = next_obs_idx

    def add_episode_trajectory(
        self,
        obs_sequence: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        episode_id: Optional[int] = None,
    ) -> None:
        """Add a complete contiguous episode trajectory into the circular buffer.
        
        Guarantees all transitions within the episode are stored consecutively.
        """
        T = len(actions)
        if T == 0:
            return
        self.start_episode(obs_sequence[0], episode_id=episode_id)
        for t in range(T):
            self.add(
                action=int(actions[t]),
                reward=float(rewards[t]),
                next_obs=obs_sequence[t + 1],
                terminated=bool(terminated[t]),
                truncated=bool(truncated[t]),
            )

    def get_state(self) -> Dict[str, Any]:
        """Capture state dict for checkpointing."""
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "obs_buffer": self.obs_buffer[:self.size + 1].copy() if self.total_added < self.capacity else self.obs_buffer.copy(),
            "actions": self.actions[:self.size].copy() if self.total_added < self.capacity else self.actions.copy(),
            "rewards": self.rewards[:self.size].copy() if self.total_added < self.capacity else self.rewards.copy(),
            "terminated": self.terminated[:self.size].copy() if self.total_added < self.capacity else self.terminated.copy(),
            "truncated": self.truncated[:self.size].copy() if self.total_added < self.capacity else self.truncated.copy(),
            "episode_ids": self.episode_ids[:self.size].copy() if self.total_added < self.capacity else self.episode_ids.copy(),
            "timesteps": self.timesteps[:self.size].copy() if self.total_added < self.capacity else self.timesteps.copy(),
            "transition_ids": self.transition_ids[:self.size].copy() if self.total_added < self.capacity else self.transition_ids.copy(),
            "transition_obs_idx": self.transition_obs_idx[:self.size].copy() if self.total_added < self.capacity else self.transition_obs_idx.copy(),
            "transition_next_obs_idx": self.transition_next_obs_idx[:self.size].copy() if self.total_added < self.capacity else self.transition_next_obs_idx.copy(),
            "write_idx": self.write_idx,
            "obs_write_idx": self.obs_write_idx,
            "size": self.size,
            "total_added": self.total_added,
            "current_episode_id": self.current_episode_id,
            "current_timestep": self.current_timestep,
            "last_obs_idx": self._last_obs_idx,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state dict from checkpoint."""
        self.capacity = state["capacity"]
        self.obs_shape = tuple(state["obs_shape"])
        self.size = state["size"]
        self.write_idx = state["write_idx"]
        self.obs_write_idx = state["obs_write_idx"]
        self.total_added = state["total_added"]
        self.current_episode_id = state["current_episode_id"]
        self.current_timestep = state["current_timestep"]
        self._last_obs_idx = state["last_obs_idx"]
        
        s_obs = state["obs_buffer"]
        self.obs_buffer[:len(s_obs)] = s_obs
        s_act = state["actions"]
        self.actions[:len(s_act)] = s_act
        s_rew = state["rewards"]
        self.rewards[:len(s_rew)] = s_rew
        s_term = state["terminated"]
        self.terminated[:len(s_term)] = s_term
        s_trunc = state["truncated"]
        self.truncated[:len(s_trunc)] = s_trunc
        s_ep = state["episode_ids"]
        self.episode_ids[:len(s_ep)] = s_ep
        s_time = state["timesteps"]
        self.timesteps[:len(s_time)] = s_time
        if "transition_ids" in state:
            s_tid = state["transition_ids"]
            self.transition_ids[:len(s_tid)] = s_tid
        s_to = state["transition_obs_idx"]
        self.transition_obs_idx[:len(s_to)] = s_to
        s_tno = state["transition_next_obs_idx"]
        self.transition_next_obs_idx[:len(s_tno)] = s_tno
