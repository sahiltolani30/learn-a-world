"""
Decoupled VICReg (Variance-Invariance-Covariance Regularization) Loss for Godfather JEPA.

Enforces information-theoretic non-collapse in latent representations:
1. Variance hinge loss: Prevents dimensional collapse by forcing the standard
   deviation of each latent feature along the batch dimension to exceed gamma (default 1.0).
2. Covariance decorrelation loss: Prevents informational collapse and redundancy by
   penalizing off-diagonal elements of the covariance matrix.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def decoupled_vicreg_loss(
    p: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes decoupled variance hinge and covariance decorrelation losses.

    Args:
        p: Latent projections tensor of shape (batch_size, latent_dim).
        gamma: Target standard deviation threshold (default 1.0).
        eps: Small numerical stability epsilon.

    Returns:
        Tuple of (variance_loss, covariance_loss) as scalar torch tensors.
    """
    if p.dim() != 2:
        raise ValueError(f"Expected 2D tensor (batch_size, latent_dim), got shape {p.shape}")

    n, d = p.shape
    if n <= 1:
        # Autograd-safe zero return for single-element or empty batches
        zero = (p * 0.0).sum()
        return zero, zero

    # 1. Variance hinge loss
    std = torch.sqrt(p.var(dim=0, unbiased=False) + eps)
    l_var = torch.mean(F.relu(gamma - std))

    # 2. Covariance decorrelation loss
    centered = p - p.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n - 1)
    diag = torch.diagonal(cov)
    off_diag = cov - torch.diag(diag)
    l_cov = (off_diag ** 2).sum() / max(d * (d - 1), 1)

    return l_var, l_cov


class DecoupledVICRegLoss(nn.Module):
    """Module wrapper for decoupled VICReg regularization."""

    def __init__(self, gamma: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04):
        super().__init__()
        self.gamma = gamma
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_var, l_cov = decoupled_vicreg_loss(p, gamma=self.gamma)
        total = self.var_weight * l_var + self.cov_weight * l_cov
        return total, l_var, l_cov


# Alias conforming to Interface Contract 3
vicreg_loss = decoupled_vicreg_loss

__all__ = ["decoupled_vicreg_loss", "vicreg_loss", "DecoupledVICRegLoss"]

