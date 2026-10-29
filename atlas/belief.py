"""AchievementBelief: strict m_bin and m_soft tracking.

Fresh implementation; does not import planning/.
Consumes ONLY the model's own event head predictions and scalar reward.
Refuses privileged environment information by construction.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import torch

from atlas.config import NUM_ACHIEVEMENTS, BeliefConfig


class AchievementBelief:
    """Tracks binary and soft achievement belief over the 22 Crafter achievements."""

    def __init__(
        self,
        unlock_threshold: float = 0.25,
        reward_fallback: float = 0.9,
    ):
        self.unlock_threshold = float(unlock_threshold)
        self.reward_fallback = float(reward_fallback)
        self.m_bin: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.m_soft: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.last_fallback: List[int] = []

    def has(self, dim: int) -> bool:
        """Returns True if achievement dim has been unlocked this episode (V2.2 F11)."""
        if 0 <= dim < NUM_ACHIEVEMENTS:
            return bool(self.m_bin[dim] > 0.5)
        return False

    @classmethod
    def from_config(cls, cfg: Optional[BeliefConfig] = None) -> AchievementBelief:
        cfg = cfg or BeliefConfig()
        return cls(
            unlock_threshold=cfg.unlock_threshold,
            reward_fallback=cfg.reward_fallback,
        )

    def reset(self) -> None:
        """Clears belief at episode boundaries."""
        self.m_bin.fill(0.0)
        self.m_soft.fill(0.0)
        self.last_fallback = []

    def update(self, event_probs: np.ndarray | torch.Tensor, reward: float) -> List[int]:
        """Update beliefs from model predictions and scalar reward.

        The ONLY caller-supplied signals are the model's own predicted event probs
        and the scalar reward. No environment info/inventory/achievements allowed.

        Args:
            event_probs: predicted probabilities for the 22 achievements, shape (22,).
            reward: scalar reward obtained from the transition.

        Returns:
            List of achievement indices newly unlocked in this step.
        """
        if isinstance(event_probs, torch.Tensor):
            p = event_probs.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            p = np.asarray(event_probs, dtype=np.float32).squeeze()

        if p.shape != (NUM_ACHIEVEMENTS,):
            raise ValueError(f"Expected event_probs shape ({NUM_ACHIEVEMENTS},), got {p.shape}")

        reward = float(reward)

        # 1) Soft belief tracks running max of predicted p
        self.m_soft = np.maximum(self.m_soft, p)

        # 2) Binary thresholding on locked dims
        newly_unlocked: List[int] = []
        for i in range(NUM_ACHIEVEMENTS):
            if self.m_bin[i] < 0.5 and p[i] > self.unlock_threshold:
                self.m_bin[i] = 1.0
                newly_unlocked.append(i)

        # 3) Scalar-reward fallback: if reward >= 0.9 and no unlock triggered by threshold,
        # pin argmax p among locked dims
        self.last_fallback = []
        if reward >= self.reward_fallback and len(newly_unlocked) == 0:
            locked_dims = [i for i in range(NUM_ACHIEVEMENTS) if self.m_bin[i] < 0.5]
            if locked_dims:
                best_sub_idx = int(np.argmax(p[locked_dims]))
                best_dim = locked_dims[best_sub_idx]
                self.m_bin[best_dim] = 1.0
                self.m_soft[best_dim] = max(float(self.m_soft[best_dim]), 1.0)
                self.last_fallback = [best_dim]
                newly_unlocked.append(best_dim)

        return newly_unlocked

    def get_mask_tensor(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        """Returns binary mask tensor [22] on requested device."""
        t = torch.from_numpy(self.m_bin).float()
        return t.to(device) if device is not None else t

    def get_soft_mask_tensor(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        """Returns soft mask tensor [22] on requested device."""
        t = torch.from_numpy(self.m_soft).float()
        return t.to(device) if device is not None else t
