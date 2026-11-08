"""
genesis_jepa.modules.losses — Normalized L1 Loss & Soft-Capped Anti-Persistence Loss
===================================================================================
Mathematical Formulations:
1. Normalized L1 Loss:
   Targets conditional medians rather than means, snapping to discrete physical branches.
       L_1^{norm}(hat{z}, z^*) = (1 / sqrt(D)) * ||hat{z} - z^*||_1

2. Soft-Capped Anti-Persistence Loss:
   Prevents trivial identity copying by weighting active interaction tokens up to 5.0x:
       w_t^{(k)} = 1.0 + 4.0 * tanh( ||Delta z_t^{*(k)}||_2 / (sigma_{Delta, t} + eps) ) in [1.0, 5.0]
   Where:
       Delta z_t^{*(k)} = z_{t+1}^{*(k)} - z_t^{*(k)}
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def normalized_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes scale-invariant normalized L1 loss over latent tokens.
    Args:
        pred: (B, K, D) or (..., D)
        target: Same shape as pred
    Returns:
        Per-token normalized L1 loss: (B, K) or (...)
    """
    D = pred.shape[-1]
    norm_factor = 1.0 / math.sqrt(D)
    # L1 norm across the feature dimension
    return norm_factor * torch.sum(torch.abs(pred - target), dim=-1)


def compute_anti_persistence_weights(
    z_prev_target: torch.Tensor,
    z_curr_target: torch.Tensor,
    floor: float = 1.0,
    gain: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes soft-capped anti-persistence weights based on ground-truth target motion.
    Args:
        z_prev_target: Target tokens at time t, shape (B, K, D)
        z_curr_target: Target tokens at time t+1, shape (B, K, D)
    Returns:
        w: Tensor of shape (B, K) with values strictly in [floor, floor + gain] == [1.0, 5.0]
    """
    # Delta vector of true target
    delta_target = z_curr_target - z_prev_target  # (B, K, D)
    token_delta_norms = torch.norm(delta_target, p=2, dim=-1)  # (B, K)

    # Mean delta across all tokens in the batch
    mean_delta = torch.mean(token_delta_norms, dim=-1, keepdim=True)  # (B, 1)

    # Soft-capped hyperbolic tangent weighting
    scaled_ratio = token_delta_norms / (mean_delta + eps)
    weights = floor + gain * torch.tanh(scaled_ratio)  # Strictly in [1.0, 5.0]

    return weights


class WorldModelGate1Loss(nn.Module):
    """
    Combined Gate 1 World Model Loss Module.
    """

    def __init__(
        self,
        floor: float = 1.0,
        gain: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.floor = floor
        self.gain = gain
        self.eps = eps

    def forward(
        self,
        pred_z: torch.Tensor,
        curr_target_z: torch.Tensor,
        prev_target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_z: Predicted next tokens (B, K, D)
            curr_target_z: True next tokens from EMA target encoder (B, K, D)
            prev_target_z: True current tokens from EMA target encoder (B, K, D)
        Returns:
            loss: Scalar loss
            metrics: Diagnostic metrics dictionary
        """
        # Per-token normalized L1 loss: (B, K)
        token_losses = normalized_l1_loss(pred_z, curr_target_z)

        # Anti-persistence weights: (B, K) in [1.0, 5.0]
        with torch.no_grad():
            weights = compute_anti_persistence_weights(
                prev_target_z,
                curr_target_z,
                floor=self.floor,
                gain=self.gain,
                eps=self.eps,
            )

        # Weighted loss
        weighted_loss = torch.mean(weights * token_losses)

        # Persistence baseline check (what if model just copied prev_z?)
        with torch.no_grad():
            static_copy_error = torch.mean(normalized_l1_loss(prev_target_z, curr_target_z))
            unweighted_loss = torch.mean(token_losses)
            anti_persistence_ratio = float(unweighted_loss / (static_copy_error + 1e-6))

        metrics = {
            "wm_loss": float(weighted_loss.item()),
            "unweighted_l1": float(unweighted_loss.item()),
            "static_baseline": float(static_copy_error.item()),
            "anti_persistence_ratio": anti_persistence_ratio,
            "mean_weight": float(weights.mean().item()),
            "max_weight": float(weights.max().item()),
        }

        return weighted_loss, metrics
