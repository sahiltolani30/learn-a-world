"""
Dense Extrinsic Reward Predictor Head for Godfather JEPA.

Predicts scalar environment reward r_hat in R given composite state s_t = [z_t, h_t]
and discrete action a_t:
    r_hat = MLP([s_t, Emb(a_t)])

Trained via regression against observed scalar rewards (+1.0 achievements, vital shifts),
providing true extrinsic reward gradients in latent imagination to eliminate pure-curiosity dithering.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseRewardHead(nn.Module):
    """
    MLP reward predictor mapping (state, action) to scalar reward.
    """

    def __init__(
        self,
        state_dim: int = 1024,
        num_actions: int = 17,
        act_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.act_emb = nn.Embedding(num_actions, act_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting scalar reward.
        Args:
            state: Composite state tensor of shape (..., 1024).
            action: Discrete action index tensor of shape (...).
        Returns:
            predicted_reward: Scalar reward tensor of shape (...).
        """
        a_vec = self.act_emb(action)
        inp = torch.cat([state, a_vec], dim=-1)
        return self.net(inp).squeeze(-1)
