"""
Epistemic Predictor Ensemble for Autonomous Curiosity in Godfather JEPA.

Maintains K independent predictor heads initialized with diverse weights.
Measures epistemic dynamics disagreement as the ensemble variance across heads:
    r_curiosity = (1 / K) * sum_k ||z_pred^(k) - z_bar||^2

Provides targeted exploration drive toward unmodeled workstation interactions
(e.g. craft pickaxe, mine stone) while remaining immune to grass-running novelty traps.
"""
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
from godfather_jepa.models.predictor import BoundedResidualPredictor


class EpistemicPredictorEnsemble(nn.Module):
    """
    Ensemble of K bounded residual predictors generating epistemic disagreement curiosity.
    """

    def __init__(
        self,
        k_heads: int = 5,
        latent_dim: int = 512,
        belief_dim: int = 512,
        num_actions: int = 17,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.k_heads = k_heads
        self.heads = nn.ModuleList([
            BoundedResidualPredictor(
                latent_dim=latent_dim,
                belief_dim=belief_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
            )
            for _ in range(k_heads)
        ])

    def forward(self, z: torch.Tensor, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Computes forward predictions across all K heads.
        Args:
            z: Latent vector of shape (B, 512).
            h: Belief vector of shape (B, 512).
            action: Discrete action tensor of shape (B,).
        Returns:
            preds: Tensor of shape (K, B, 512).
        """
        preds = [head(z, h, action) for head in self.heads]
        return torch.stack(preds, dim=0)

    def disagreement(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Computes epistemic disagreement reward across ensemble heads.
        Args:
            preds: Predicted next latents of shape (K, B, 512).
        Returns:
            disagreement: Scalar curiosity reward per batch item of shape (B,).
        """
        # std across K heads, mean across latent dimensions
        return preds.std(dim=0).mean(dim=-1)

    def prediction_loss(self, preds: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        """
        Computes joint mean squared error loss between all heads and EMA target encoder latents.
        Args:
            preds: (K, B, 512)
            target_z: (B, 512)
        Returns:
            loss: scalar MSE loss averaged across K heads and batch.
        """
        k = preds.shape[0]
        targets_expanded = target_z.unsqueeze(0).expand(k, -1, -1)
        loss = nn.functional.mse_loss(preds, targets_expanded)
        return loss
