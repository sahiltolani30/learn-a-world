"""VitalsReadout: 16-dimensional self-state estimator.

Predicts internal item inventory (12 slots, log1p) and vitals (health, food, drink, energy /16).
Diagnostics / goal-scoring only. NEVER an input to OptionNet, ContinueNet, or epistemic heads.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import STATE_DIM, STATE_VECTOR_DIM, ReadoutConfig


class VitalsReadout(nn.Module):
    """Trunk MLP(384 -> 256 -> 256) -> mean[16] and unc[16]."""

    def __init__(self, cfg: Optional[ReadoutConfig] = None):
        super().__init__()
        self.cfg = cfg or ReadoutConfig()
        h_dim = self.cfg.hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.head_mean = nn.Linear(h_dim, STATE_VECTOR_DIM)
        self.head_logvar = nn.Linear(h_dim, STATE_VECTOR_DIM)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat[B, 16], unc[B, 16])."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        h = self.trunk(s)
        mean = self.head_mean(h)
        unc = F.softplus(self.head_logvar(h)) + 1e-4
        if s.ndim == 1:
            return mean.squeeze(0), unc.squeeze(0)
        return mean, unc

    def compute_loss(
        self,
        s: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked MSE loss between predicted mean and targets."""
        x_hat, _ = self.forward(s)
        if valid_mask is None:
            return F.mse_loss(x_hat, targets)
        diff_sq = (x_hat - targets) ** 2
        masked_diff = diff_sq * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
        return masked_diff.sum() / denom
