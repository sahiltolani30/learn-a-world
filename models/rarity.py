"""
models/rarity.py — Rarity Weight Scheduler for Learn-a-World v3 (Tranche T2 Part 4)

Calculates dynamic per-achievement rarity bonus weights from cumulative unlock counts.

Standing Rules:
- All achievements referenced by INDEX (int) only (0 to NUM_ACHIEVEMENTS-1).
- ZERO achievement names in code logic.
- Weight formula: w_k = base_weight / ((count_k + 10) ** decay)
"""

from typing import Optional, Union
import numpy as np
import torch


class RarityWeightScheduler:
    """
    Computes per-achievement rarity bonus weights based on running cumulative unlock counts.
    All indexing is purely integer-based (0 to num_achievements - 1).

    Formula:
        w_k = base_weight / ((count_k + epsilon) ** decay)

    Default:
        base_weight = 5.0
        decay = 0.5 (square-root scaling)
        epsilon = 10.0
    """

    def __init__(
        self,
        num_achievements: int = 22,
        base_weight: float = 5.0,
        decay: float = 0.5,
        epsilon: float = 10.0,
        initial_counts: Optional[np.ndarray] = None,
    ):
        self.num_achievements = num_achievements
        self.base_weight = float(base_weight)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        if initial_counts is not None:
            self.counts = np.array(initial_counts, dtype=np.float64).flatten()
            assert len(self.counts) == num_achievements, (
                f"Expected initial_counts of length {num_achievements}, got {len(self.counts)}"
            )
        else:
            self.counts = np.zeros(num_achievements, dtype=np.float64)

        self._weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Computes weight array from current counts."""
        return (self.base_weight / ((self.counts + self.epsilon) ** self.decay)).astype(np.float32)

    def update(self, new_event_counts: np.ndarray) -> None:
        """
        Updates cumulative counts by adding newly observed event counts.
        new_event_counts: [num_achievements] array of newly unlocked events.
        """
        counts_arr = np.array(new_event_counts, dtype=np.float64).flatten()
        assert len(counts_arr) == self.num_achievements, (
            f"Counts length mismatch: expected {self.num_achievements}, got {len(counts_arr)}"
        )
        self.counts += counts_arr
        self._weights = self._compute_weights()

    def set_counts(self, counts: np.ndarray) -> None:
        """Sets cumulative counts directly (e.g. loaded from manifest)."""
        counts_arr = np.array(counts, dtype=np.float64).flatten()
        assert len(counts_arr) == self.num_achievements, (
            f"Counts length mismatch: expected {self.num_achievements}, got {len(counts_arr)}"
        )
        self.counts = counts_arr
        self._weights = self._compute_weights()

    def get_weights(self) -> np.ndarray:
        """Returns current weights as numpy array [num_achievements]."""
        return self._weights.copy()

    def get_weights_tensor(self, device: torch.device) -> torch.Tensor:
        """Returns weights as torch.Tensor on specified device."""
        return torch.from_numpy(self._weights).to(device=device, dtype=torch.float32)

    def compute_bonus(self, events: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Computes rarity bonus for observed binary events or predicted probabilities.
        events: [num_achievements] or [..., num_achievements]
        """
        if isinstance(events, torch.Tensor):
            w = self.get_weights_tensor(events.device)
            return float(torch.sum(events * w).item())
        else:
            e = np.array(events, dtype=np.float32)
            return float(np.sum(e * self._weights))

    def compute_bonus_tensor(self, event_probs: torch.Tensor) -> torch.Tensor:
        """
        Batched tensor bonus computation for MPC imagined trajectories.
        event_probs: [..., num_achievements]
        returns: [...] tensor of rarity bonuses
        """
        w = self.get_weights_tensor(event_probs.device)
        return torch.sum(event_probs * w, dim=-1)
