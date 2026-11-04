"""Diagnostic and validation tools for MOSAIC Phase P2 representations and dynamics.

Measures:
1. Action-conditioning sensitivity (paired action divergence test).
2. Held-out latent prediction versus persistence baseline.
3. Representation collapse metrics (feature std, covariance, effective rank).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F


def evaluate_action_conditioning(
    world_model: torch.nn.Module,
    belief: torch.Tensor,
    action_token_1: int = 0,
    action_token_2: int = 1,
) -> Dict[str, float]:
    """Test whether dynamics predictions diverge under distinct actions at the same belief state.

    Args:
        world_model: MosaicWorldModel instance
        belief: [B, belief_dim] current predictive belief
        action_token_1: first candidate action integer
        action_token_2: second candidate action integer
    Returns:
        dict with action divergence metrics (must be strictly > 0 to pass action conditioning).
    """
    B = belief.size(0)
    device = belief.device

    act_1 = torch.full((B,), int(action_token_1), dtype=torch.long, device=device)
    act_2 = torch.full((B,), int(action_token_2), dtype=torch.long, device=device)

    with torch.no_grad():
        out_1 = world_model.imagine_transition(belief, act_1)
        out_2 = world_model.imagine_transition(belief, act_2)

    # Belief divergence
    b_diff = (out_1["next_belief_mean"] - out_2["next_belief_mean"]).pow(2).sum(dim=-1).sqrt().mean().item()

    # Latent prediction divergence
    l_diff = (out_1["latent_pred_mean"] - out_2["latent_pred_mean"]).pow(2).sum(dim=-1).sqrt().mean().item()

    # Reward divergence
    r_diff = (out_1["reward_mean"] - out_2["reward_mean"]).abs().mean().item()

    return {
        "belief_divergence": float(b_diff),
        "latent_divergence": float(l_diff),
        "reward_divergence": float(r_diff),
        "is_action_sensitive": bool(b_diff > 1e-4 or l_diff > 1e-4),
    }


def evaluate_heldout_prediction(
    world_model: torch.nn.Module,
    obs_t: torch.Tensor,
    action_t: torch.Tensor,
    obs_next: torch.Tensor,
) -> Dict[str, float]:
    """Compare world model prediction error against a persistence baseline.

    Args:
        world_model: MosaicWorldModel instance
        obs_t: [B, 3, 64, 64] frame at step t
        action_t: [B] action token at step t
        obs_next: [B, 3, 64, 64] ground-truth frame at step t+1
    Returns:
        model_error: cosine distance between predicted latent and true target latent
        persistence_error: cosine distance between current frame latent and next frame latent
        beats_persistence: bool indicating whether model_error < persistence_error
    """
    B = obs_t.size(0)
    device = obs_t.device

    with torch.no_grad():
        _, curr_pooled = world_model.encoder(obs_t)
        _, next_target_pooled = world_model.target_encoder(obs_next)

        # Persistence baseline: predict next latent = current latent
        p_curr = F.normalize(curr_pooled, p=2, dim=-1)
        p_next = F.normalize(next_target_pooled, p=2, dim=-1)
        persistence_error = (1.0 - (p_curr * p_next).sum(dim=-1)).mean().item()

        # Step belief and imagine next latent
        h_0, z_0, b_0 = world_model.initial_belief(B, device=device)
        dummy_action = torch.zeros(B, dtype=torch.long, device=device)
        dummy_reward = torch.zeros(B, 1, device=device)

        b_t, _, _, _, _ = world_model.step_belief(
            obs_uint8_or_float=obs_t,
            action_token_prev=dummy_action,
            reward_prev=dummy_reward,
            h_prev=h_0,
            z_prev=z_0,
            use_posterior=True,
        )

        dyn_out = world_model.imagine_transition(b_t, action_t)
        p_pred = F.normalize(dyn_out["latent_pred_mean"], p=2, dim=-1)
        model_error = (1.0 - (p_pred * p_next).sum(dim=-1)).mean().item()

    return {
        "model_error": float(model_error),
        "persistence_error": float(persistence_error),
        "error_delta": float(persistence_error - model_error),
        "beats_persistence": bool(model_error < persistence_error),
    }


def evaluate_representation_collapse(features: torch.Tensor) -> Dict[str, Any]:
    """Detect dimensional or variance collapse in latent representations.

    Args:
        features: [N, D] tensor of representations (e.g., pooled features or spatial tokens)
    Returns:
        std_mean: mean standard deviation across dimensions (should be > 0.05)
        std_min: minimum standard deviation across dimensions
        cov_offdiag_mean: off-diagonal squared covariance (decorrelation metric, lower is better)
        effective_rank: effective dimension rank derived from singular value entropy
        is_collapsed: True if features have collapsed to a constant or subspace
    """
    N, D = features.shape
    if N <= 1:
        return {"is_collapsed": True, "reason": "insufficient_samples"}

    with torch.no_grad():
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        std = torch.sqrt(centered.pow(2).sum(dim=0) / (N - 1) + 1e-8)
        std_mean = std.mean().item()
        std_min = std.min().item()

        # Covariance matrix: [D, D]
        cov = (centered.t() @ centered) / (N - 1)
        diag = torch.diag(cov)
        off_diag = cov - torch.diag(diag)
        cov_offdiag_mean = off_diag.pow(2).mean().item()

        # Singular values for effective rank
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            norm_s = S / (S.sum() + 1e-8)
            entropy = -(norm_s * torch.log(norm_s + 1e-8)).sum().item()
            effective_rank = float(torch.exp(torch.tensor(entropy)).item())
        except Exception:
            effective_rank = 1.0

    is_collapsed = bool(std_mean < 0.02 or std_min < 1e-4 or effective_rank < 2.0)

    return {
        "std_mean": float(std_mean),
        "std_min": float(std_min),
        "cov_offdiag_mean": float(cov_offdiag_mean),
        "effective_rank": float(effective_rank),
        "is_collapsed": is_collapsed,
    }
