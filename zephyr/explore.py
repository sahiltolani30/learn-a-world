"""Zephyr exploration: episodic kNN novelty in belief space.

Intrinsic motivation for the sparse frontier (stone/coal discovery). The other
exploration signal — dynamics-ensemble disagreement — lives in
``zephyr.heads.DynamicsEnsemble`` + ``zephyr.ac.ensemble_disagreement``.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from zephyr.config import ExplorationConfig


class EpisodicNovelty:
    """kNN-distance novelty over within-episode belief vectors (E3B-style).

    Score ~ [0, 1): mean distance to the k nearest stored beliefs, squashed by
    x/(1+x). Returns 0.0 until the memory holds at least ``k + 1`` entries.
    Reset per episode — it measures within-episode revisititation pressure only.
    """

    def __init__(self, cfg: ExplorationConfig) -> None:
        self.k = int(cfg.knn_k)
        self.capacity = int(cfg.knn_memory)
        self._memory: deque = deque(maxlen=self.capacity)

    def reset(self) -> None:
        self._memory.clear()

    def add(self, h: np.ndarray) -> None:
        self._memory.append(np.asarray(h, dtype=np.float32).ravel().copy())

    def score(self, h: np.ndarray) -> float:
        """Mean distance to the k nearest stored beliefs, squashed to [0, 1)."""
        if len(self._memory) < self.k + 1:
            return 0.0
        x = np.asarray(h, dtype=np.float32).ravel()
        stack = np.stack(list(self._memory), axis=0)
        d2 = np.sum((stack - x[None, :]) ** 2, axis=1)
        kth = np.partition(d2, self.k - 1)[: self.k]
        mean_dist = float(np.sqrt(kth.mean()))
        return mean_dist / (1.0 + mean_dist)

    def __len__(self) -> int:
        return len(self._memory)
