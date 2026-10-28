"""
Survival Continuation Discount Head for Godfather JEPA.

Predicts state-dependent continuation probability c(s_t) in [0, 1].
In imagination rollouts, cumulative discounting is computed as:
    Gamma_t = prod_{k=0}^t gamma * c(s_k)

When mortality occurs (nightfall mob attack, health depletion), c(s_t) -> 0,
immediately zeroing out all downstream imagined returns and penalizing death.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SurvivalContinuationHead(nn.Module):
    """
    MLP predicting continuation probability c(s_t) in [0, 1] from composite state.
    """

    def __init__(self, state_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            s: Composite state tensor [z_t; h_t] of shape (..., 1024).
        Returns:
            continuation_prob: Probability of continuation in [0, 1] of shape (...).
        """
        return self.net(s).squeeze(-1)

    def compute_discounts(
        self,
        states: torch.Tensor,
        base_gamma: float = 0.997,
    ) -> torch.Tensor:
        """
        Computes state-dependent discount factors across an imagined rollout sequence.
        Args:
            states: (H, B, 1024)
            base_gamma: Environmental discount factor (default 0.997).
        Returns:
            discounts: (H, B) state-dependent discount factors.
        """
        contin_probs = self.forward(states)  # (H, B)
        return base_gamma * contin_probs
