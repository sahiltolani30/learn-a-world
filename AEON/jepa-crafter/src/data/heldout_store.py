"""Independent held-out episode store for strict out-of-distribution JEPA validation."""

from typing import List, Dict, Any, Optional, Set
import numpy as np
import torch

from src.env.crafter_adapter import CrafterAdapter
from src.env.opaque_actions import OpaqueActionMapping
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler


class HeldoutEpisodeStore:
    """Manages validation episodes generated from a disjoint seed range.
    
    Guarantees:
    - Heldout episodes are collected from seeds in [heldout_seed_start, heldout_seed_start + episode_count).
    - Heldout episodes never enter the training replay buffer.
    - Automated assertion verifies disjointness against training episode/seed manifests.
    """

    def __init__(
        self,
        heldout_seed_start: int = 900000,
        heldout_episode_count: int = 10,
        max_steps_per_episode: int = 100,
        opaque_mapping: Optional[OpaqueActionMapping] = None,
        context_length: int = 4,
        prediction_horizon: int = 8,
    ):
        self.heldout_seed_start = heldout_seed_start
        self.heldout_episode_count = heldout_episode_count
        self.max_steps_per_episode = max_steps_per_episode
        self.opaque_mapping = opaque_mapping
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon

        self.heldout_seeds: Set[int] = {
            heldout_seed_start + i for i in range(heldout_episode_count)
        }

        # Dedicated private replay buffer for validation only
        self.buffer = ReplayBuffer(
            capacity=heldout_episode_count * (max_steps_per_episode + 5),
            obs_shape=(64, 64, 3),
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.buffer,
            context_length=context_length,
            prediction_horizon=prediction_horizon,
            recent_fraction=0.0,  # uniform across all heldout episodes
        )
        self.is_collected = False

    def collect_heldout_episodes(self) -> int:
        """Collect heldout episodes using fixed random policy and dedicated seed range."""
        if self.is_collected:
            return self.buffer.size

        total_transitions = 0
        for ep_idx, seed in enumerate(sorted(self.heldout_seeds)):
            adapter = CrafterAdapter(
                seed=seed,
                opaque_mapping=self.opaque_mapping,
            )
            obs = adapter.reset(seed=seed)
            self.buffer.start_episode(obs, episode_id=seed)

            for step in range(self.max_steps_per_episode):
                action = int(np.random.randint(0, 17))
                next_obs, reward, terminated, truncated = adapter.step(action)
                self.buffer.add(
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                )
                total_transitions += 1
                if terminated or truncated:
                    break
            adapter.close()

        self.is_collected = True
        return total_transitions

    def sample_batch(self, batch_size: int = 16, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """Sample a contiguous multi-horizon validation batch exclusively from heldout store."""
        if not self.is_collected:
            self.collect_heldout_episodes()
        return self.sampler.sample_batch(batch_size=batch_size, device=device)

    def verify_disjoint_from(self, training_seeds: Set[int]) -> bool:
        """Verify that training seeds and heldout seeds share zero overlap."""
        overlap = self.heldout_seeds.intersection(training_seeds)
        if overlap:
            raise ValueError(f"CRITICAL LEAK: Training seeds overlap with heldout seeds: {overlap}")
        return True

    def get_manifest(self) -> Dict[str, Any]:
        return {
            "heldout_seed_start": self.heldout_seed_start,
            "heldout_episode_count": self.heldout_episode_count,
            "heldout_seeds": sorted(list(self.heldout_seeds)),
            "total_transitions": self.buffer.size,
        }
