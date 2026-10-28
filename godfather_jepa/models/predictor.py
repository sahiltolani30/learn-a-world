"""
Bounded Residual Dynamics Predictor for Godfather JEPA.

Predicts next-step latent state z_{t+1} given current latent z_t, recurrent belief h_t,
and action a_t via a bounded residual delta network:
    z_{t+1} = LayerNorm(z_t + tanh(MLP([z_t, h_t, a_t])))
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoundedResidualPredictor(nn.Module):
    """
    Residual predictor network with bounded delta to guarantee numerical rollout stability.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        belief_dim: int = 512,
        num_actions: int = 17,
        act_emb_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        self.num_actions = num_actions

        self.act_emb = nn.Embedding(num_actions, act_emb_dim)
        in_dim = latent_dim + belief_dim + act_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward step predicting z_{t+1}.
        Args:
            z: Current latent vector of shape (B, 512).
            h: Current belief state vector of shape (B, 512).
            action: Discrete action index tensor of shape (B,).
        Returns:
            z_next: Predicted next latent vector of shape (B, 512).
        """
        a_vec = self.act_emb(action)
        inp = torch.cat([z, h, a_vec], dim=-1)
        delta = torch.tanh(self.mlp(inp))
        z_next = self.ln(z + delta)
        return z_next
