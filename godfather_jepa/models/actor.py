"""
Amortized Policy Actor for Godfather JEPA.

Outputs log-probabilities over 17 discrete Crafter actions in O(1) time,
eliminating runtime MPC planning horizon bottlenecks.
Trained directly inside latent imagination rollouts using multi-step lambda-returns.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AmortizedPolicyActor(nn.Module):
    """
    Amortized Actor MLP producing action log-probabilities with uniform exploration mixing.
    """

    def __init__(
        self,
        state_dim: int = 1024,
        num_actions: int = 17,
        hidden_dim: int = 256,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.unimix = unimix

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            s: Composite state tensor [z_t; h_t] of shape (B, 1024) or (H, B, 1024).
        Returns:
            log_probs: Log-probabilities over 17 actions of shape (..., 17).
        """
        logits = self.net(s)
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0.0:
            probs = (1.0 - self.unimix) * probs + (self.unimix / self.num_actions)
        return torch.log(probs + 1e-8)

    def sample_action(self, s: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Samples actions for environmental interaction.
        Args:
            s: Composite state of shape (B, 1024).
            deterministic: If True, takes argmax.
        Returns:
            actions: Discrete action indices of shape (B,).
        """
        log_probs = self.forward(s)
        if deterministic:
            return log_probs.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=log_probs)
        return dist.sample()
