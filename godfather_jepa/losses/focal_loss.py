"""
Class-Weighted Focal Loss for Survival Continuation Head in Godfather JEPA.

Addresses the severe class imbalance of episode termination in Crafter (terminal states
occur in <0.5% of transitions). Focal weighting with alpha=25.0 and gamma=2.0
prevents trivial survival prediction and forces the continuation head to accurately
detect nightfall mob mortality walls.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def class_weighted_focal_loss(
    pred_probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 25.0,
    gamma: float = 2.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes class-weighted binary focal cross-entropy loss.

    Args:
        pred_probs: Predicted continuation probabilities in [0, 1] of shape (B,).
        targets: Ground-truth binary targets (1 = alive, 0 = terminal/death) of shape (B,).
        alpha: Positive class / rare class weight multiplier (default 25.0).
        gamma: Focal modulation exponent (default 2.0).
        eps: Small numerical clamp epsilon.

    Returns:
        Scalar focal loss tensor.
    """
    p = torch.clamp(pred_probs, eps, 1.0 - eps)
    pt = targets * p + (1.0 - targets) * (1.0 - p)
    w = (1.0 - targets) * alpha + targets * 1.0
    focal_weight = w * ((1.0 - pt) ** gamma)
    bce = -(targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))
    return torch.mean(focal_weight * bce)


class ClassWeightedFocalLoss(nn.Module):
    """Module wrapper for class-weighted binary focal loss."""

    def __init__(self, alpha: float = 25.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(pred_probs, targets, alpha=self.alpha, gamma=self.gamma)
