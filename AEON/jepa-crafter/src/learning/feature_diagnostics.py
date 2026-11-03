"""Diagnostics for representation collapse, feature variance, and effective rank."""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F


def compute_feature_diagnostics(
    online_tokens: torch.Tensor,
    target_tokens: Optional[torch.Tensor] = None,
    predicted_tokens: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute anti-collapse and geometric diagnostics on latent features.
    
    Args:
        online_tokens: [B, 64, 128]
        target_tokens: optional [B, 64, 128]
        predicted_tokens: optional [B, 64, 128]
    Returns:
        dict of scalar metrics: feature_mean, feature_std, min_channel_std,
        pairwise_cosine_sim, effective_rank, predictor_target_sim.
    """
    B, N, D = online_tokens.shape
    flat_tokens = online_tokens.reshape(B * N, D)  # [M, 128]

    # Feature mean & standard deviation
    f_mean = flat_tokens.mean().item()
    f_std = flat_tokens.std().item()

    # Per-channel standard deviation
    per_channel_std = flat_tokens.std(dim=0)  # [128]
    min_channel_std = per_channel_std.min().item()
    mean_channel_std = per_channel_std.mean().item()
    collapsed_channels = (per_channel_std < 0.01).sum().item() / D

    # Average pairwise cosine similarity between different frames (using pooled vectors)
    pooled_vectors = online_tokens.mean(dim=1)  # [B, 128]
    pooled_norm = F.normalize(pooled_vectors, dim=-1, eps=1e-6)
    sim_matrix = torch.mm(pooled_norm, pooled_norm.t())  # [B, B]
    # Mask out diagonal
    mask = ~torch.eye(B, dtype=bool, device=online_tokens.device)
    if mask.sum() > 0:
        avg_pairwise_cos = sim_matrix[mask].mean().item()
    else:
        avg_pairwise_cos = 1.0

    # Effective rank / participation ratio:
    # Covariance matrix C of centered features: [D, D]
    centered = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
    cov = torch.mm(centered.t(), centered) / max(1, flat_tokens.size(0) - 1)
    # Eigenvalues of cov
    try:
        eigenvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        # Participation ratio: (sum lambda_i)^2 / sum(lambda_i^2)
        trace_cov = eigenvals.sum()
        sum_sq = (eigenvals ** 2).sum()
        if sum_sq > 1e-8:
            effective_rank = (trace_cov ** 2 / sum_sq).item()
        else:
            effective_rank = 0.0
    except Exception:
        effective_rank = 0.0

    diag = {
        "latent_mean": f_mean,
        "latent_std": f_std,
        "min_channel_std": min_channel_std,
        "mean_channel_std": mean_channel_std,
        "collapsed_channel_fraction": collapsed_channels,
        "avg_pairwise_cosine_sim": avg_pairwise_cos,
        "effective_rank": effective_rank,
    }

    # Predictor to target similarity if both provided
    if target_tokens is not None and predicted_tokens is not None:
        p_norm = F.normalize(predicted_tokens, dim=-1, eps=1e-6)
        t_norm = F.normalize(target_tokens, dim=-1, eps=1e-6)
        pred_targ_sim = (p_norm * t_norm).sum(dim=-1).mean().item()
        diag["pred_target_cosine_sim"] = pred_targ_sim

    # Check for hard collapse warning
    collapse_warning = False
    warning_reasons = []
    if f_std < 0.05:
        collapse_warning = True
        warning_reasons.append("Feature std < 0.05")
    if avg_pairwise_cos > 0.98:
        collapse_warning = True
        warning_reasons.append("Pairwise cosine sim > 0.98")
    if effective_rank < 2.0:
        collapse_warning = True
        warning_reasons.append("Effective rank < 2.0")

    diag["collapse_warning"] = 1.0 if collapse_warning else 0.0
    if collapse_warning:
        diag["collapse_reasons"] = "; ".join(warning_reasons)

    return diag
