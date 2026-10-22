"""
Agent Belief & Recurrent State Tracker for Learn-a-World v3 (R2).
Maintains both:
1. Unlocked achievement mask m in {0, 1}^22
2. Recurrent memory state h in R^256 (temporal context for nights, occlusions, and long chains)
WITHOUT accessing ground-truth environment info.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22


class AgentBeliefV3:
    """
    Epistemic & Recurrent state tracker for v3 planning.
    Tracks unlocked achievements and temporal hidden state strictly from transitions.
    """

    def __init__(self, recurrent_dim: int = 256):
        self.recurrent_dim = recurrent_dim
        self.mask = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.h = np.zeros(recurrent_dim, dtype=np.float32)
        self.history = []

    def reset(self) -> None:
        """Resets belief mask and recurrent hidden state at episode start."""
        self.mask.fill(0.0)
        self.h.fill(0.0)
        self.history.clear()

    def update(
        self,
        event_probs: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_h: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Updates belief mask and advances recurrent hidden state:
        event_probs: [22] probabilities from EventHead
        reward: scalar reward received from environment
        next_h: [recurrent_dim] updated hidden state from RecurrentCore
        """
        if isinstance(event_probs, torch.Tensor):
            p = event_probs.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            p = np.array(event_probs, dtype=np.float32).reshape(-1)

        assert len(p) == NUM_ACHIEVEMENTS, f"Expected {NUM_ACHIEVEMENTS} event probs, got {len(p)}"

        if next_h is not None:
            if isinstance(next_h, torch.Tensor):
                self.h = next_h.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                self.h = np.array(next_h, dtype=np.float32).reshape(-1)

        locked_indices = np.where(self.mask < 0.5)[0]
        newly_unlocked = []

        if len(locked_indices) > 0:
            # 1. Primary rule: mark unlocked if p_i > 0.5
            for idx in locked_indices:
                if p[idx] > 0.5:
                    self.mask[idx] = 1.0
                    newly_unlocked.append(idx)

            # 2. Fallback rule: reward fired (>= 0.9) but no event probability exceeded threshold
            if float(reward) >= 0.9 and len(newly_unlocked) == 0:
                locked_probs = p[locked_indices]
                best_locked_idx = locked_indices[int(np.argmax(locked_probs))]
                self.mask[best_locked_idx] = 1.0
                newly_unlocked.append(best_locked_idx)

        self.history.append({
            "event_probs": p.copy(),
            "reward": float(reward),
            "newly_unlocked": newly_unlocked,
            "mask": self.mask.copy(),
        })

    def get_mask(self) -> np.ndarray:
        """Returns copy of unlocked mask [22]."""
        return self.mask.copy()

    def get_mask_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns mask as PyTorch tensor."""
        t = torch.from_numpy(self.mask).float()
        if device is not None:
            t = t.to(device)
        return t

    def get_h(self) -> np.ndarray:
        """Returns copy of recurrent hidden state [recurrent_dim]."""
        return self.h.copy()

    def get_h_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns recurrent hidden state as PyTorch tensor."""
        t = torch.from_numpy(self.h).float()
        if device is not None:
            t = t.to(device)
        return t
