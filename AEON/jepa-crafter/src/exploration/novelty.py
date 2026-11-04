"""Running normalizer and novelty estimator."""
import numpy as np
import torch
from src.exploration.latent_memory import LatentMemory


class RunningNormalizer:
    """Tracks running mean and variance to normalize signals and clip outliers."""

    def __init__(self, clip_val: float = 5.0, eps: float = 1e-4):
        self.clip_val = clip_val
        self.eps = eps
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, val: float) -> None:
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return max(float(np.sqrt(self.var)), self.eps)

    def normalize(self, val: float, update: bool = False) -> float:
        """Normalize val to zero-mean unit-variance clipped to [-clip_val, clip_val].
        
        Args:
            val: float to normalize
            update: if True, updates running statistics. Default is False (frozen evaluation).
        """
        if update:
            self.update(val)
        normed = (val - self.mean) / self.std
        return float(np.clip(normed, -self.clip_val, self.clip_val))

    def get_state(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "clip_val": self.clip_val,
            "eps": self.eps,
        }

    def set_state(self, state: dict) -> None:
        self.count = state["count"]
        self.mean = state["mean"]
        self.m2 = state["m2"]
        self.clip_val = state.get("clip_val", 5.0)
        self.eps = state.get("eps", 1e-4)


class NoveltyEstimator:
    """Calculates running-normalized latent novelty."""

    def __init__(self, capacity: int = 5000, latent_dim: int = 128, clip_val: float = 5.0):
        self.memory = LatentMemory(capacity=capacity, latent_dim=latent_dim)
        self.normalizer = RunningNormalizer(clip_val=clip_val)

    def evaluate(self, latent: torch.Tensor, update_memory: bool = False, update_normalizer: bool = False) -> float:
        raw_novelty = self.memory.compute_novelty(latent)
        if update_memory:
            self.memory.add(latent)
        return self.normalizer.normalize(raw_novelty, update=update_normalizer)

    def insert_observation(self, latent: torch.Tensor) -> float:
        """Insert a real observed latent into memory and update running normalizer statistics."""
        raw_novelty = self.memory.compute_novelty(latent)
        self.memory.add(latent)
        return self.normalizer.normalize(raw_novelty, update=True)

    def add(self, latent: torch.Tensor) -> None:
        self.memory.add(latent)

    def get_state(self) -> dict:
        return {
            "memory": self.memory.get_state(),
            "normalizer": self.normalizer.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self.memory.set_state(state["memory"])
        self.normalizer.set_state(state["normalizer"])
