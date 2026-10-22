"""
Task Prediction Heads for Learn-a-World v2.
Implements:
1. EventHead: Predicts per-achievement unlock probability p_i in [0, 1]^22 given [z_t, emb(a_t), m_t].
   Loss: Focal Binary Cross-Entropy with gamma=2.0.
2. HealthHead: Predicts non-achievement health/survival reward delta h_hat in [-1, 1].
   Loss: Huber (SmoothL1) Loss.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACHIEVEMENTS = 22

def focal_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Numerically stable Focal BCE loss for sparse multi-label achievement prediction.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    modulating_factor = (1.0 - p_t) ** gamma

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * modulating_factor * bce
    else:
        loss = modulating_factor * bce

    return loss.mean()

class EventHead(nn.Module):
    """
    Achievement Event Prediction Head.
    Estimates P(achievement i unlocks on transition t -> t+1 | z_t, a_t, mask_before_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        num_achievements: int = NUM_ACHIEVEMENTS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim
        self.num_achievements = num_achievements

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim + num_achievements
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_achievements),
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [B, latent_dim] or [B, T, latent_dim]
        action: [B] or [B, T] (int64) OR [..., action_emb_dim]
        mask_before: [B, 22] or [B, T, 22] (float32)
        Returns: logits of shape [..., 22]
        """
        if action.dtype in (torch.int64, torch.int32):
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        features = torch.cat([z, a_emb, mask_before], dim=-1)
        return self.net(features)

    def predict_proba(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
    ) -> torch.Tensor:
        """Returns calibrated probabilities in [0, 1]^22."""
        logits = self.forward(z, action, mask_before)
        return torch.sigmoid(logits)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Computes focal BCE loss against binary event indicators."""
        return focal_binary_cross_entropy(logits, targets, gamma=gamma)

class HealthHead(nn.Module):
    """
    Survival / Vital Reward Head.
    Predicts scalar non-achievement reward delta (health, thirst, hunger effects).
    Target: reward - (number of events unlocked this step), clipped to [-1, 1].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [..., latent_dim]
        action: [...] int64 OR [..., action_emb_dim]
        Returns: scalar prediction of shape [...]
        """
        if action.dtype in (torch.int64, torch.int32):
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        features = torch.cat([z, a_emb], dim=-1)
        out = self.net(features)
        return out.squeeze(-1)

    def compute_loss(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Huber (Smooth L1) loss."""
        return F.smooth_l1_loss(pred, targets, beta=1.0)
