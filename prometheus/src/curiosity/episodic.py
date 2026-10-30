"""
src/curiosity/episodic.py

Per-episode k-NN novelty in latent space (NGU-style, simplified).
Calculates distance to nearest neighbor keys within the current episode,
incentivizing intra-episodic exploration and novel interaction breakthrough.
Uses float16 ring storage to maintain sub-millisecond query latency.
"""

from typing import Optional, Union
import numpy as np


class EpisodicNovelty:
    """
    Per-episode k-NN novelty in latent space.

    novelty(z) = mean distance to the k nearest stored keys (float, unnormalized),
    then the key is inserted into an episodic ring buffer. Reset at every episode start.
    Uses float16 storage for cache efficiency and fast Euclidean distance computation.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        max_keys: int = 4096,
        k: int = 10,
        seed: int = 0,
        sentinel_value: float = 1000.0,
    ):
        """
        Initialize episodic memory.

        Args:
            latent_dim: Dimension of latent embedding vector z.
            max_keys: Maximum keys stored in ring buffer per episode.
            k: Number of nearest neighbors to average for novelty.
            seed: Optional seed for reproducibility.
            sentinel_value: Value returned during the first k steps of an episode.
        """
        self.latent_dim = latent_dim
        self.max_keys = max_keys
        self.k = k
        self.sentinel_value = float(sentinel_value)
        self.rng = np.random.RandomState(seed)

        self.keys = np.empty((self.max_keys, self.latent_dim), dtype=np.float16)
        self.ptr: int = 0
        self.count: int = 0

    def reset(self) -> None:
        """Reset episodic keys for a new episode."""
        self.ptr = 0
        self.count = 0

    def novelty_and_insert(self, z: Union[np.ndarray, list]) -> float:
        """
        Compute mean distance to k nearest stored keys and insert z into ring buffer.

        Args:
            z: Latent vector of shape (latent_dim,) or (1, latent_dim).

        Returns:
            novelty_score: Mean Euclidean distance to k nearest neighbors,
                           or sentinel_value if fewer than k keys exist.
        """
        z_arr = np.asarray(z, dtype=np.float32).reshape(-1)
        if z_arr.shape[0] != self.latent_dim:
            raise ValueError(f"Expected z of dimension {self.latent_dim}, got {z_arr.shape[0]}")

        # If count < k: early steps in the episode are maximally novel
        if self.count < self.k:
            self._insert(z_arr)
            return self.sentinel_value

        # Compute Euclidean distance to stored keys
        keys_f32 = self.keys[:self.count].astype(np.float32)
        diff = keys_f32 - z_arr
        dists = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 0.0))

        # Mean of k smallest distances
        if self.count > self.k:
            k_smallest = np.partition(dists, self.k - 1)[:self.k]
        else:
            k_smallest = dists

        score = float(np.mean(k_smallest))

        # Insert z into episodic ring buffer
        self._insert(z_arr)

        return score

    def _insert(self, z_arr: np.ndarray) -> None:
        """Insert vector into float16 circular buffer."""
        self.keys[self.ptr] = z_arr.astype(np.float16)
        self.ptr = (self.ptr + 1) % self.max_keys
        self.count = min(self.count + 1, self.max_keys)
