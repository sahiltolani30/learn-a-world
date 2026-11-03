"""Loss functions for JEPA future latent prediction and auxiliary heads."""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_feature_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask_fraction: float = 0.50,
) -> torch.Tensor:
    """Compute cosine distance loss over a random subset of spatial tokens.
    
    Args:
        predicted_tokens: [B, 64, 128]
        target_tokens: [B, 64, 128] (stop-gradient EMA targets)
        mask_fraction: fraction of spatial tokens to include in loss (default 0.50)
    Returns:
        loss: scalar tensor
    """
    B, N, D = predicted_tokens.shape
    device = predicted_tokens.device

    # L2 normalize along feature dimension
    pred_norm = F.normalize(predicted_tokens, dim=-1, eps=1e-6)
    target_norm = F.normalize(target_tokens.detach(), dim=-1, eps=1e-6)

    # Cosine distance per token: 1.0 - cos_sim
    # Shape: [B, 64]
    cos_dist = 1.0 - (pred_norm * target_norm).sum(dim=-1)

    # Sample random subset of spatial tokens of size int(N * mask_fraction)
    num_keep = max(1, int(N * mask_fraction))
    # Random permutation of token indices per batch item
    rand_scores = torch.rand(B, N, device=device)
    _, keep_indices = torch.topk(rand_scores, k=num_keep, dim=-1)  # [B, num_keep]

    selected_dist = torch.gather(cos_dist, dim=-1, index=keep_indices)  # [B, num_keep]
    return selected_dist.mean()


class JepaLoss(nn.Module):
    """Multi-horizon weighted JEPA loss with auxiliary reward and termination losses."""

    def __init__(
        self,
        horizons: List[int] = [1, 2, 4, 8],
        raw_weights: List[float] = [1.00, 0.50, 0.25, 0.125],
        target_mask_fraction: float = 0.50,
        reward_weight: float = 0.10,
        termination_weight: float = 0.05,
    ):
        super().__init__()
        self.horizons = horizons
        self.target_mask_fraction = target_mask_fraction
        self.reward_weight = reward_weight
        self.termination_weight = termination_weight

        # Normalize horizon weights
        w_sum = sum(raw_weights)
        self.horizon_weights = {h: w / w_sum for h, w in zip(horizons, raw_weights)}

    def forward(
        self,
        predictions_by_member: List[Dict[int, torch.Tensor]],
        targets_by_horizon: Dict[int, torch.Tensor],
        bootstrap_masks: Optional[List[torch.Tensor]] = None,
        auxiliary_outputs: Optional[List] = None,
        rewards_first: Optional[torch.Tensor] = None,
        done_first: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute full weighted loss across horizons and ensemble members.
        
        Args:
            predictions_by_member: list of dicts mapping horizon -> predicted z_hat [B, 64, 128]
            targets_by_horizon: dict mapping horizon -> target z_bar [B, 64, 128]
            bootstrap_masks: optional list of binary masks [B] for each member
            auxiliary_outputs: optional list of AuxiliaryOutputs from each member for step 1
            rewards_first: [B, 1] ground truth scalar reward
            done_first: [B, 1] ground truth done flag
        Returns:
            dict containing total_loss, jepa_loss, reward_loss, termination_loss, and per-horizon losses.
        """
        device = targets_by_horizon[self.horizons[0]].device
        num_members = len(predictions_by_member)

        total_jepa_loss = torch.tensor(0.0, device=device)
        per_horizon_losses = {h: torch.tensor(0.0, device=device) for h in self.horizons}

        for m_idx, member_preds in enumerate(predictions_by_member):
            member_jepa_loss = torch.tensor(0.0, device=device)
            mask = bootstrap_masks[m_idx] if bootstrap_masks is not None else None

            for h in self.horizons:
                if h in member_preds and h in targets_by_horizon:
                    pred_h = member_preds[h]
                    targ_h = targets_by_horizon[h]
                    h_loss = cosine_feature_loss(pred_h, targ_h, mask_fraction=self.target_mask_fraction)

                    if mask is not None:
                        h_loss = (h_loss * mask.float()).sum() / (mask.float().sum() + 1e-6)

                    member_jepa_loss = member_jepa_loss + self.horizon_weights[h] * h_loss
                    per_horizon_losses[h] = per_horizon_losses[h] + h_loss / num_members

            total_jepa_loss = total_jepa_loss + member_jepa_loss / num_members

        # Auxiliary losses
        total_rew_loss = torch.tensor(0.0, device=device)
        total_term_loss = torch.tensor(0.0, device=device)

        if auxiliary_outputs is not None and rewards_first is not None and done_first is not None:
            for aux in auxiliary_outputs:
                rew_loss = F.smooth_l1_loss(aux.reward, rewards_first)
                term_loss = F.binary_cross_entropy(aux.done_probability.clamp(1e-6, 1.0 - 1e-6), done_first)
                total_rew_loss = total_rew_loss + rew_loss / len(auxiliary_outputs)
                total_term_loss = total_term_loss + term_loss / len(auxiliary_outputs)

        total_loss = (
            total_jepa_loss
            + self.reward_weight * total_rew_loss
            + self.termination_weight * total_term_loss
        )

        metrics = {
            "loss": total_loss,
            "jepa_loss": total_jepa_loss,
            "reward_loss": total_rew_loss,
            "termination_loss": total_term_loss,
        }
        for h, l in per_horizon_losses.items():
            metrics[f"loss_h{h}"] = l

        return metrics
