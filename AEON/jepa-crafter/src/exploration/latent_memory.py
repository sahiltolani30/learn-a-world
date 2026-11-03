"""Episodic and reservoir latent memory for novelty estimation."""
from typing import Optional
import torch
import torch.nn.functional as F


class LatentMemory:
    """Stores pooled latent vectors and computes k-NN novelty in latent space."""

    def __init__(self, capacity: int = 5000, latent_dim: int = 128, k_neighbors: int = 5):
        self.capacity = capacity
        self.latent_dim = latent_dim
        self.k_neighbors = k_neighbors
        self.memory = torch.zeros(capacity, latent_dim)
        self.size = 0
        self.write_ptr = 0

    def add(self, latent: torch.Tensor) -> None:
        """Add a pooled latent vector [128] or [1, 128] to memory."""
        v = latent.detach().cpu().flatten()
        v_norm = F.normalize(v, dim=-1, eps=1e-6)
        
        self.memory[self.write_ptr] = v_norm
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_novelty(self, latent: torch.Tensor) -> float:
        """Compute k-NN cosine distance novelty against stored latents.
        
        Args:
            latent: [128] or [1, 128]
        Returns:
            novelty: float (mean distance to k nearest neighbors, in [0, 2])
        """
        if self.size < self.k_neighbors:
            return 1.0  # Default initial novelty

        v = latent.detach().cpu().flatten()
        v_norm = F.normalize(v, dim=-1, eps=1e-6)

        stored = self.memory[:self.size]  # [N, 128]
        # Cosine similarity: [N]
        cos_sim = torch.mv(stored, v_norm)
        # Cosine distance: 1.0 - cos_sim
        cos_dist = 1.0 - cos_sim

        topk_dist, _ = torch.topk(cos_dist, k=self.k_neighbors, largest=False)
        return float(topk_dist.mean().item())

    def reset_episode(self) -> None:
        """Optional: clear episode memory if configured as episodic-only."""
        pass
