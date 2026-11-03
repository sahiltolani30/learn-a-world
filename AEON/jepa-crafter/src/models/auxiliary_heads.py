"""Auxiliary prediction heads for scalar reward and episode termination."""
from typing import NamedTuple
import torch
import torch.nn as nn


class AuxiliaryOutputs(NamedTuple):
    reward: torch.Tensor           # [B, 1] predicted scalar reward
    done_probability: torch.Tensor # [B, 1] predicted termination probability in [0, 1]


class AuxiliaryHeads(nn.Module):
    """Small MLP heads predicting next-step reward and termination from predicted latent and context."""

    def __init__(self, visual_dim: int = 128, context_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        in_dim = visual_dim + context_dim  # 384

        self.reward_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.termination_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled_latent: torch.Tensor, context_state: torch.Tensor) -> AuxiliaryOutputs:
        """
        Args:
            pooled_latent: [B, 128]
            context_state: [B, 256]
        Returns:
            AuxiliaryOutputs(reward=[B, 1], done_probability=[B, 1])
        """
        feat = torch.cat([pooled_latent, context_state], dim=-1)  # [B, 384]
        rew = self.reward_head(feat)
        term = self.termination_head(feat)
        return AuxiliaryOutputs(reward=rew, done_probability=term)
