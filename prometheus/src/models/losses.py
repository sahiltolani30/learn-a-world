"""
src/models/losses.py

Anti-Collapse Regularization & Loss Functions for JEPA World Model.
Implements:
- VICRegLoss: Variance-Invariance-Covariance regularization with multi-head ensemble support.
- Non-collapse verification and assertion utilities (strict variance floor check > 0.1).
- Numerical stability guarantees (safe division, NaN prevention, B < 2 guard).
"""

from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization Loss (VICReg) for JEPA.
    
    Paper: Bardes, Ponce, LeCun (ICLR 2022) - 'VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning'.
    
    Adapted for Action-Conditioned JEPA World Models with:
    - Multi-head ensemble predictions: shape (K, B, D) vs target (B, D).
    - Configurable multi-head reduction ('mean_heads' or 'average_first').
    - Numerical stability safeguards: epsilon=1e-4 inside sqrt to eliminate NaN gradients.
    - Small-batch edge case handling: graceful fallback for B < 2.
    - Full metric telemetry dictionary returning diagnostic scalars.
    
    Args:
        sim_coeff (float): Weight on invariance loss (lambda). Default: 1.0.
        var_coeff (float): Weight on variance hinge loss (mu). Default: 1.0.
        cov_coeff (float): Weight on covariance decorrelation loss (nu). Default: 0.04.
        gamma (float): Target standard deviation threshold. Default: 1.0.
        epsilon (float): Small constant inside variance sqrt for numerical stability. Default: 1e-4.
        multi_head_reduction (str): Strategy for multi-head inputs:
            - 'mean_heads': Supervise each ensemble head independently and average losses.
            - 'average_first': Average predictions across heads before computing VICReg.
            Default: 'mean_heads'.
        regularize_target (bool): Whether to include target latents in variance/covariance loss.
            Default: True.
        normalize_invariance (bool): If True, divides invariance loss by D (MSE).
            If False, computes squared L2 norm sum_j (z_hat - z)^2 matching standard VICReg.
            Default: True.
    """

    def __init__(
        self,
        sim_coeff: float = 1.0,
        var_coeff: float = 1.0,
        cov_coeff: float = 0.04,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
        multi_head_reduction: str = "mean_heads",
        regularize_target: bool = True,
        normalize_invariance: bool = True,
    ):
        super().__init__()
        self.sim_coeff = float(sim_coeff)
        self.var_coeff = float(var_coeff)
        self.cov_coeff = float(cov_coeff)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.multi_head_reduction = multi_head_reduction
        self.regularize_target = regularize_target
        self.normalize_invariance = normalize_invariance

        if self.multi_head_reduction not in ("mean_heads", "average_first"):
            raise ValueError(
                f"Unsupported multi_head_reduction '{multi_head_reduction}'. "
                "Must be 'mean_heads' or 'average_first'."
            )

    def _compute_var_cov(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes batch variance and covariance losses in a vectorized manner.
        
        Args:
            z (torch.Tensor): Latent tensor of shape (K, B, D).
            
        Returns:
            var_loss (torch.Tensor): Scalar variance hinge loss.
            cov_loss (torch.Tensor): Scalar covariance decorrelation loss.
            min_std (torch.Tensor): Diagnostic min standard deviation across features.
            mean_std (torch.Tensor): Diagnostic mean standard deviation across features.
        """
        K, B, D = z.shape

        # Guard against B < 2: sample variance across batch is undefined
        if B < 2:
            zero_scalar = (z * 0.0).sum()
            return zero_scalar, zero_scalar, zero_scalar, zero_scalar

        # 1. Centered latents along batch dimension (dim=1)
        z_c = z - z.mean(dim=1, keepdim=True)  # Shape: (K, B, D)

        # 2. Sample variance along batch dimension: sum((z - mu)^2) / (B - 1)
        var = (z_c ** 2).sum(dim=1) / (B - 1)  # Shape: (K, D)

        # 3. Standard deviation with epsilon INSIDE square root
        # Gradient d/dx sqrt(x + eps) = 1 / (2 * sqrt(x + eps)). At x=0, grad is 1/(2*sqrt(eps)) < inf
        std = torch.sqrt(var + self.epsilon)  # Shape: (K, D)

        # 4. Variance hinge loss: max(0, gamma - std)
        var_loss_per_head = torch.mean(F.relu(self.gamma - std), dim=-1)  # Shape: (K,)
        var_loss = var_loss_per_head.mean()

        # 5. Covariance matrix along batch dimension: C = (z_c^T @ z_c) / (B - 1)
        # Vectorized batch matrix multiplication: (K, D, B) @ (K, B, D) -> (K, D, D)
        if D > 1:
            cov = torch.bmm(z_c.transpose(1, 2), z_c) / (B - 1)  # Shape: (K, D, D)
            cov_sq = cov.pow(2)  # Shape: (K, D, D)
            diag_sq = torch.diagonal(cov, dim1=1, dim2=2).pow(2)  # Shape: (K, D)
            # Sum of squared off-diagonal entries divided by D
            cov_loss_per_head = (cov_sq.sum(dim=(1, 2)) - diag_sq.sum(dim=-1)) / D  # Shape: (K,)
            cov_loss = cov_loss_per_head.mean()
        else:
            cov_loss = (z * 0.0).sum()

        min_std = std.min().detach()
        mean_std = std.mean().detach()

        return var_loss, cov_loss, min_std, mean_std

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward computation of VICReg loss.
        
        Args:
            z_pred (torch.Tensor): Predicted latents, either:
                - 2D tensor of shape (B, D) (single head or pre-averaged)
                - 3D tensor of shape (K, B, D) (multi-head ensemble, e.g. K=4)
            z_target (torch.Tensor): Target latents from EMA TargetEncoder, either:
                - 2D tensor of shape (B, D)
                - 3D tensor of shape (K, B, D)
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'loss': Total combined scalar loss for optimizer backward.
                - 'invariance': Invariance (prediction error) loss.
                - 'variance': Variance hinge loss.
                - 'covariance': Covariance decorrelation loss.
                - 'invariance_loss': Alias for invariance.
                - 'variance_loss': Alias for variance.
                - 'covariance_loss': Alias for covariance.
                - 'pred_var': Variance loss on predicted representations.
                - 'pred_cov': Covariance loss on predicted representations.
                - 'target_var': Variance loss on target representations.
                - 'target_cov': Covariance loss on target representations.
                - 'min_std_pred': Minimum standard deviation across predictor dimensions.
                - 'mean_std_pred': Mean standard deviation across predictor dimensions.
                - 'min_std_target': Minimum standard deviation across target dimensions.
                - 'mean_std_target': Mean standard deviation across target dimensions.
        """
        # Canonicalize z_pred to (K, B, D)
        if z_pred.ndim == 2:
            z_pred_3d = z_pred.unsqueeze(0)  # (1, B, D)
        elif z_pred.ndim == 3:
            if self.multi_head_reduction == "average_first":
                z_pred_3d = z_pred.mean(dim=0, keepdim=True)  # (1, B, D)
            else:
                z_pred_3d = z_pred  # (K, B, D)
        else:
            raise ValueError(
                f"z_pred must be 2D (B, D) or 3D (K, B, D), got shape {tuple(z_pred.shape)}"
            )

        # Canonicalize z_target to (K, B, D)
        if z_target.ndim == 2:
            z_target_3d = z_target.unsqueeze(0)  # (1, B, D)
        elif z_target.ndim == 3:
            z_target_3d = z_target  # (K, B, D)
        else:
            raise ValueError(
                f"z_target must be 2D (B, D) or 3D (K, B, D), got shape {tuple(z_target.shape)}"
            )

        K_p, B_p, D_p = z_pred_3d.shape
        K_t, B_t, D_t = z_target_3d.shape

        if B_p != B_t:
            raise ValueError(f"Batch size mismatch: z_pred B={B_p}, z_target B={B_t}")
        if D_p != D_t:
            raise ValueError(f"Latent dimension mismatch: z_pred D={D_p}, z_target D={D_t}")

        # 1. Invariance Loss: (1/B) * sum_i || z_pred_i - z_target_i ||_2^2
        diff = z_pred_3d - z_target_3d  # Shape: (K_p, B, D)
        if self.normalize_invariance:
            inv_loss_per_head = torch.mean(diff ** 2, dim=(-2, -1))  # Mean over B and D
        else:
            inv_loss_per_head = torch.sum(diff ** 2, dim=-1).mean(dim=-1)  # L2^2 mean over B
        inv_loss = inv_loss_per_head.mean()

        # 2. Variance and Covariance on Predicted Latents
        v_pred, c_pred, min_std_p, mean_std_p = self._compute_var_cov(z_pred_3d)

        # 3. Variance and Covariance on Target Latents
        if self.regularize_target:
            v_targ, c_targ, min_std_t, mean_std_t = self._compute_var_cov(z_target_3d)
            var_loss = 0.5 * (v_pred + v_targ)
            cov_loss = 0.5 * (c_pred + c_targ)
        else:
            v_targ = (z_target_3d * 0.0).sum()
            c_targ = (z_target_3d * 0.0).sum()
            min_std_t = torch.tensor(0.0, device=z_pred.device)
            mean_std_t = torch.tensor(0.0, device=z_pred.device)
            var_loss = v_pred
            cov_loss = c_pred

        # 4. Total Weighted Loss
        total_loss = (
            self.sim_coeff * inv_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return {
            "loss": total_loss,
            "invariance": inv_loss,
            "variance": var_loss,
            "covariance": cov_loss,
            "invariance_loss": inv_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss,
            "pred_var": v_pred,
            "pred_cov": c_pred,
            "target_var": v_targ,
            "target_cov": c_targ,
            "min_std_pred": min_std_p,
            "mean_std_pred": mean_std_p,
            "min_std_target": min_std_t,
            "mean_std_target": mean_std_t,
        }


# ==============================================================================
# Non-Collapse Verification & Representation Integrity Utilities
# ==============================================================================

def verify_non_collapse(
    z: torch.Tensor,
    threshold: Optional[float] = None,
    min_var_threshold: float = 0.1,
    strict: bool = True,
    eps: float = 1e-4,
) -> Dict[str, Any]:
    """
    Audits a batch of latent vectors to mathematically verify non-collapse.
    
    Verifies that:
    1. Every feature dimension j=1..D maintains empirical variance > threshold.
    2. Effective rank (participation ratio) confirms multi-dimensional feature spread.
    
    Args:
        z (torch.Tensor): Latent tensor of shape (B, D) or (K, B, D).
        threshold (float, optional): Alias for min_var_threshold. Default: 0.1.
        min_var_threshold (float): Minimum acceptable sample variance. Default: 0.1.
        strict (bool): If True, raises AssertionError on failure. If False, returns report.
        eps (float): Epsilon for effective rank stability. Default: 1e-4.
        
    Returns:
        Dict[str, Any]: Diagnostic dictionary with metrics and collapse status.
        
    Raises:
        ValueError: If batch size B < 2.
        AssertionError: If strict=True and any dimension has variance <= threshold.
    """
    effective_threshold = float(threshold if threshold is not None else min_var_threshold)

    if z.ndim == 2:
        B, D = z.shape
    elif z.ndim == 3:
        _, B, D = z.shape
    else:
        raise ValueError(f"Expected 2D (B, D) or 3D (K, B, D) tensor, got {tuple(z.shape)}")

    if B < 2:
        raise ValueError(
            f"Batch size B must be at least 2 to evaluate representation variance; got B={B}"
        )

    # Compute unbiased sample variance across the batch dimension (dim=-2)
    variances = torch.var(z, dim=-2, unbiased=True)  # Shape: (D,) or (K, D)
    min_var = variances.min().item()
    mean_var = variances.mean().item()
    max_var = variances.max().item()

    # Check for collapsed dimensions
    collapsed_mask = variances <= effective_threshold
    num_collapsed = collapsed_mask.sum().item()
    is_valid = num_collapsed == 0

    # Effective rank (Participation Ratio): PR = (Tr(C))^2 / Tr(C^2)
    if z.ndim == 2:
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / (B - 1)
        tr_c = torch.diagonal(cov).sum()
        tr_c2 = cov.pow(2).sum()
        eff_rank = ((tr_c ** 2) / (tr_c2 + eps)).item()
    else:
        eff_rank = 0.0  # Multi-head summary

    report = {
        "is_valid": is_valid,
        "min_variance": min_var,
        "mean_variance": mean_var,
        "max_variance": max_var,
        "num_collapsed_dims": num_collapsed,
        "effective_rank": eff_rank,
        "threshold": effective_threshold,
        "variances": variances.detach().cpu(),
    }

    if strict and not is_valid:
        raise AssertionError(
            f"Latent representation collapse detected! {num_collapsed} feature dimension(s) "
            f"have variance <= {effective_threshold}. "
            f"Min variance: {min_var:.6f}, Mean variance: {mean_var:.6f}, "
            f"Threshold: {effective_threshold:.4f}."
        )

    return report


def assert_non_collapse(
    z: torch.Tensor,
    threshold: Optional[float] = None,
    min_var_threshold: float = 0.1,
) -> None:
    """Convenience wrapper strictly asserting non-collapse."""
    verify_non_collapse(
        z,
        threshold=threshold,
        min_var_threshold=min_var_threshold,
        strict=True,
    )
