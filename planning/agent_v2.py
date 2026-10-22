"""
Agent Belief Tracker for Learn-a-World v2.
Maintains the agent's estimated unlocked-achievement mask m in {0, 1}^22
WITHOUT accessing ground-truth environment info.

Update Rule:
1. Primary: EventHead predicted probability p_i > 0.5 -> mark achievement i unlocked.
2. Fallback: If received reward >= 0.9 (an achievement definitely occurred) but no
   locked p_i > 0.5, mark the highest-probability locked achievement as unlocked.
"""

from typing import Optional, Union
import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22

class AgentBelief:
    """
    Epistemic state tracker for task-conditioned planning.
    Tracks unlocked achievements strictly from transition observations and rewards.
    """

    def __init__(self):
        self.mask = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.history = []

    def reset(self) -> None:
        """Resets belief mask to initial state at episode start."""
        self.mask.fill(0.0)
        self.history.clear()

    def update(
        self,
        event_probs: Union[np.ndarray, torch.Tensor],
        reward: float,
    ) -> None:
        """
        Updates belief state following environment transition:
        event_probs: [22] probabilities from EventHead
        reward: scalar reward received from environment
        """
        if isinstance(event_probs, torch.Tensor):
            p = event_probs.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            p = np.array(event_probs, dtype=np.float32).reshape(-1)

        assert len(p) == NUM_ACHIEVEMENTS, f"Expected {NUM_ACHIEVEMENTS} event probs, got {len(p)}"

        locked_indices = np.where(self.mask < 0.5)[0]
        if len(locked_indices) == 0:
            return  # All 22 already unlocked

        newly_unlocked = []

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
        """Returns a copy of the current 22-dim float32 unlocked mask."""
        return self.mask.copy()

    def get_mask_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns the mask as a PyTorch tensor on the requested device."""
        t = torch.from_numpy(self.mask).float()
        if device is not None:
            t = t.to(device)
        return t
