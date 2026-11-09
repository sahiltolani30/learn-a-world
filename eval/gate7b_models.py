"""Gate 7B Model Heads: Specialized Critics and Policy Architectures.

Provides modular, parameter-matched architectures for the Gate 7B Actor-Critic study:
- DiscreteQCritic: Twin Q-networks for discrete action spaces (maps state -> [B, num_actions]).
- ExpectileValueCritic: State-value network for IQL expectile regression.
- DistributionalValueCritic: Quantile-regression critic predicting return distribution.
- ResidualPolicyHead: Parameter-matched residual MLP policy head.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiscreteQCritic(nn.Module):
    """Twin Q-networks for discrete action spaces.
    
    Predicts Q(s, a) for all 17 actions simultaneously:
    Q_1(s): [B, 448] -> [B, 17]
    Q_2(s): [B, 448] -> [B, 17]
    """

    def __init__(self, state_dim: int = 448, num_actions: int = 17, hidden_dim: int = 384) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (q1_all_actions, q2_all_actions) each of shape [B, num_actions]."""
        return self.q1(state), self.q2(state)

    def evaluate_actions(self, state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Gathers Q(s, a) for specific taken actions.
        
        Args:
            state: [B, state_dim]
            actions: [B] (long)
        Returns:
            q1_a, q2_a: each [B]
        """
        q1_all, q2_all = self.forward(state)
        act = actions.unsqueeze(-1)
        q1_a = q1_all.gather(-1, act).squeeze(-1)
        q2_a = q2_all.gather(-1, act).squeeze(-1)
        return q1_a, q2_a


class ExpectileValueCritic(nn.Module):
    """State-value network V(s) for IQL expectile regression.
    
    Predicts expected upper expectile of return:
    V(s): [B, 448] -> [B, 1]
    """

    def __init__(self, state_dim: int = 448, hidden_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        """Returns V(s) of shape [B]."""
        return self.net(state).squeeze(-1)


class DistributionalValueCritic(nn.Module):
    """Quantile regression distributional value critic (QR-DQN style).
    
    Predicts N return quantiles:
    Z(s): [B, 448] -> [B, num_quantiles]
    """

    def __init__(self, state_dim: int = 448, num_quantiles: int = 8, hidden_dim: int = 384) -> None:
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_quantiles),
        )
        # Register midpoints of quantile intervals: tau_i = (2i - 1) / (2N)
        taus = torch.arange(0.5, num_quantiles, 1.0, dtype=torch.float32) / num_quantiles
        self.register_buffer("tau_hats", taus)

    def forward(self, state: Tensor) -> Tensor:
        """Returns quantiles of shape [B, num_quantiles]."""
        return self.net(state)

    def expected_value(self, state: Tensor) -> Tensor:
        """Returns expected return E[Z(s)] = mean(quantiles) of shape [B]."""
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)


class ResidualPolicyHead(nn.Module):
    """Parameter-matched Residual MLP policy head.
    
    Adds a residual skip connection in the intermediate representation:
    Linear(448, 384) -> SiLU -> [Linear(384, 384) + Identity] -> SiLU -> Linear(384, 17)
    """

    def __init__(self, state_dim: int = 448, num_actions: int = 17, hidden_dim: int = 384) -> None:
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)
        self.res_block = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state: Tensor) -> Tensor:
        h = F.silu(self.fc_in(state))
        h = F.silu(h + self.res_block(h))
        return self.fc_out(h)
