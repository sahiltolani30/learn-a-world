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

def two_hot_encode(
    x: torch.Tensor,
    num_bins: int = 51,
    v_min: float = 0.0,
    v_max: float = 12.0,
) -> torch.Tensor:
    """
    Two-Hot encoding of continuous return targets into discrete probability bins.
    Used in DreamerV2 / TD-MPC2 for non-divergent value regression with high variance returns.
    """
    x_clamped = x.clamp(v_min, v_max)
    delta = (v_max - v_min) / (num_bins - 1)
    k = ((x_clamped - v_min) / delta).floor().long()
    k = k.clamp(0, num_bins - 2)

    weight_upper = ((x_clamped - (v_min + k.float() * delta)) / delta).clamp(0.0, 1.0)
    weight_lower = (1.0 - weight_upper).clamp(0.0, 1.0)

    shape = list(x.shape) + [num_bins]
    target_dist = torch.zeros(shape, device=x.device, dtype=torch.float32)

    # Scatter lower and upper bin probabilities
    target_dist.scatter_(-1, k.unsqueeze(-1), weight_lower.unsqueeze(-1))
    target_dist.scatter_(-1, (k + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))

    return target_dist

def two_hot_decode(
    logits: torch.Tensor,
    num_bins: int = 51,
    v_min: float = 0.0,
    v_max: float = 12.0,
) -> torch.Tensor:
    """Decodes discrete bin logits into expected scalar values."""
    probs = F.softmax(logits, dim=-1)
    bins = torch.linspace(v_min, v_max, num_bins, device=logits.device)
    return (probs * bins).sum(dim=-1)

class QEnsemble(nn.Module):
    """
    K-head Q-Value Ensemble with Two-Hot discrete regression over [v_min, v_max].
    Provides both expected return (q_mean) and epistemic uncertainty (q_std)
    to penalize out-of-distribution trajectories in the MPC planner.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        num_heads: int = 3,
        num_bins: int = 51,
        v_min: float = 0.0,
        v_max: float = 12.0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim
        self.num_heads = num_heads
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_bins),
            )
            for _ in range(num_heads)
        ])

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z: [..., latent_dim]
        action: [...] int64 OR [..., action_emb_dim]
        Returns: logits of shape [num_heads, ..., num_bins]
        """
        if action.dtype in (torch.int64, torch.int32):
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        features = torch.cat([z, a_emb], dim=-1)
        head_logits = [head(features) for head in self.heads]
        return torch.stack(head_logits, dim=0)

    def q_values(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Decodes TwoHot logits for each ensemble head to scalar Q-values. Shape: [K, ...]"""
        logits = self.forward(z, action)
        probs = F.softmax(logits, dim=-1)
        bins = torch.linspace(self.v_min, self.v_max, self.num_bins, device=z.device)
        return (probs * bins).sum(dim=-1)

    def q_mean(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Mean Q-value across ensemble heads. Shape: [...]"""
        q_vals = self.q_values(z, action)
        return q_vals.mean(dim=0)

    def q_std(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Epistemic uncertainty (standard deviation) across ensemble heads. Shape: [...]"""
        q_vals = self.q_values(z, action)
        return q_vals.std(dim=0)

    def compute_loss(self, logits: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross-Entropy loss between each head's logits and TwoHot encoded targets.
        logits: [K, ..., num_bins]
        target_values: [...]
        """
        target_dist = two_hot_encode(target_values, num_bins=self.num_bins, v_min=self.v_min, v_max=self.v_max)
        K = logits.shape[0]
        loss = 0.0
        for k in range(K):
            log_probs = F.log_softmax(logits[k], dim=-1)
            loss = loss - (target_dist * log_probs).sum(dim=-1).mean()
        return loss / K

class PolicyPrior(nn.Module):
    """
    Behaviour-Cloned Policy Prior Head.
    Predicts logits over 17 Crafter actions given latent state z and unlocked mask m:
    pi_prior(a | z, m). Guides CEM candidate generation in the MPC planner.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 17,
        num_achievements: int = NUM_ACHIEVEMENTS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.num_achievements = num_achievements

        in_dim = latent_dim + num_achievements
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        z: [..., latent_dim]
        mask: [..., 22] float32
        Returns: action logits of shape [..., 17]
        """
        features = torch.cat([z, mask], dim=-1)
        return self.net(features)

    def sample(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples categorical action indices given z and mask."""
        logits = self.forward(z, mask)
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        # Reshape to 2D for torch.multinomial then restore shape
        orig_shape = probs.shape[:-1]
        probs_2d = probs.reshape(-1, self.num_actions)
        sampled = torch.multinomial(probs_2d, num_samples=1).squeeze(-1)
        return sampled.reshape(orig_shape)

    def compute_loss(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Cross-Entropy loss against teacher actions."""
        return F.cross_entropy(logits.reshape(-1, self.num_actions), target_actions.reshape(-1))

