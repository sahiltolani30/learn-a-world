"""
Task Prediction Heads for Learn-a-World v2.
Implements:
1. EventHead: Predicts per-achievement unlock probability p_i in [0, 1]^22 given [z_t, emb(a_t), m_t].
   Loss: Focal Binary Cross-Entropy with gamma=2.0.
2. HealthHead: Predicts non-achievement health/survival reward delta h_hat in [-1, 1].
   Loss: Huber (SmoothL1) Loss.
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
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

    def set_calibrator(self, calibrator: Optional[Any] = None) -> None:
        """Attaches an EventHeadCalibrator instance."""
        self.calibrator = calibrator

    def predict_proba(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
        use_calibration: bool = True,
    ) -> torch.Tensor:
        """Returns probabilities in [0, 1]^22, calibrated if calibrator is attached and enabled."""
        logits = self.forward(z, action, mask_before)
        if use_calibration and hasattr(self, "calibrator") and self.calibrator is not None:
            return self.calibrator.apply(logits)
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


class StateDeltaHead(nn.Module):
    """Predicts a generic action-conditioned state/resource delta.

    The head has no Crafter names or recipe logic.  ``state_dim`` is an adapter
    dimension supplied by the dataset (16 for the optional Crafter diagnostic
    inventory target).  At inference it consumes only the learned latent state
    and action, never the ground-truth state.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        state_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Small normal initialization: preserves action conditioning while staying close to zero delta
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dtype in (torch.int64, torch.int32):
            action = self.action_emb(action.squeeze(-1) if action.ndim > 1 else action)
        features = torch.cat([state, action], dim=-1)
        return self.net(features)

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.mse_loss(prediction, target, reduction="none")
        if valid_mask is not None:
            mask = valid_mask.float().reshape(-1, 1)
            loss = loss.reshape(-1, self.state_dim) * mask
            denom = mask.sum().clamp_min(1.0) * self.state_dim
            return loss.sum() / denom
        return loss.mean()

class StateReadoutHead(nn.Module):
    """Predicts a generic current-state/resource estimate and uncertainty from composite state.

    Input: composite state s_t = [z_t, h_t]
    Output: estimated state x_hat_t of dimension state_dim (e.g. 16), uncertainty of shape [..., state_dim]
    The head has zero Crafter names, recipe tables, or object-specific logic.
    At inference it consumes only learned representations [z_t, h_t], never environment inventory dictionaries.
    """

    def __init__(
        self,
        latent_dim: int,
        state_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.state_mean = nn.Linear(hidden_dim, state_dim)
        self.state_logvar = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        x_hat = self.state_mean(features)
        uncertainty = F.softplus(self.state_logvar(features)) + 1e-4
        return x_hat, uncertainty

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.mse_loss(prediction, target, reduction="none")
        if valid_mask is not None:
            mask = valid_mask.float().reshape(-1, 1)
            loss = loss.reshape(-1, self.state_dim) * mask
            denom = mask.sum().clamp_min(1.0) * self.state_dim
            return loss.sum() / denom
        return loss.mean()


class TransitionHead(nn.Module):
    """Generic reward, termination, and uncertainty prediction head.

    ``reward`` is an environment-observed scalar target and ``done`` is the
    transition boundary target.  The uncertainty output is a learned positive
    scale (used for diagnostics and future risk-aware planning), not an input
    from the environment.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if action.dtype in (torch.int64, torch.int32):
            action = self.action_emb(action.squeeze(-1) if action.ndim > 1 else action)
        out = self.net(torch.cat([state, action], dim=-1))
        reward = out[..., 0]
        done_logits = out[..., 1]
        uncertainty = F.softplus(out[..., 2]) + 1e-4
        return reward, done_logits, uncertainty

    def compute_loss(
        self,
        reward_pred: torch.Tensor,
        done_logits: torch.Tensor,
        uncertainty: torch.Tensor,
        reward_target: torch.Tensor,
        done_target: torch.Tensor,
    ) -> torch.Tensor:
        reward_loss = F.smooth_l1_loss(reward_pred, reward_target, beta=1.0)
        done_loss = F.binary_cross_entropy_with_logits(done_logits, done_target.float())
        # Gaussian NLL gives the uncertainty head a meaningful training signal.
        residual = reward_pred - reward_target
        scale = uncertainty.clamp_min(1e-4)
        uncertainty_loss = (residual.square() / (2.0 * scale.square()) + scale.log()).mean()
        return reward_loss + done_loss + 0.1 * uncertainty_loss

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

class ReturnNormalizer:
    """
    DreamerV3-style percentile return normalizer.
    Tracks running 5th and 95th percentiles of returns using EMA.
    Normalizes targets into [0, v_max] so two-hot discrete bins have maximum resolution.
    """

    def __init__(
        self,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        decay: float = 0.99,
        target_max: float = 12.0,
    ):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.decay = decay
        self.target_max = target_max
        self.low = 0.0
        self.high = 1.0
        self.initialized = False

    def update(self, returns: torch.Tensor) -> None:
        with torch.no_grad():
            r_flat = returns.detach().float().cpu().numpy().flatten()
            if len(r_flat) == 0:
                return
            p_low = float(np.percentile(r_flat, self.low_percentile))
            p_high = float(np.percentile(r_flat, self.high_percentile))
            if not self.initialized:
                self.low = p_low
                self.high = max(p_high, p_low + 1.0)
                self.initialized = True
            else:
                self.low = self.decay * self.low + (1.0 - self.decay) * p_low
                self.high = self.decay * self.high + (1.0 - self.decay) * p_high

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        scale = max(self.high - self.low, 1.0)
        norm = (returns - self.low) / scale * (self.target_max * 0.8)
        return norm.clamp(0.0, self.target_max)

    def denormalize(self, norm_values: torch.Tensor) -> torch.Tensor:
        scale = max(self.high - self.low, 1.0)
        return norm_values / (self.target_max * 0.8) * scale + self.low

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

    def q_min(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Clipped Double-Q minimum return across ensemble heads. Shape: [...]"""
        q_vals = self.q_values(z, action)
        return q_vals.min(dim=0).values

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

class GRUBeliefModule(nn.Module):
    """
    Recurrent Memory Belief Module for Temporal Representation (Phase 3).
    h_t = GRUCell([z_t, emb(a_{t-1})], h_{t-1})
    Maintains a 128-dimensional hidden state across sequence rollouts.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        self.gru = nn.GRUCell(latent_dim + action_emb_dim, hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        action_prev: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z: [B, latent_dim]
        action_prev: [B] int64
        h_prev: [B, hidden_dim] or None
        Returns: h_t of shape [B, hidden_dim]
        """
        if h_prev is None:
            h_prev = torch.zeros(z.shape[0], self.hidden_dim, device=z.device)
        if action_prev.dtype in (torch.int64, torch.int32):
            a_emb = self.action_emb(action_prev)
        else:
            a_emb = action_prev
        inputs = torch.cat([z, a_emb], dim=-1)
        return self.gru(inputs, h_prev)


class TimeToEventHead(nn.Module):
    """
    Generic Time-To-Event Prediction Head.
    Predicts future proximity / discounted time-to-event for all 22 achievements,
    indexed strictly 0..21 (never by name, zero object-specific logic).
    Target tau_i in [0, 1] is gamma^{Delta t_i} where Delta t_i is the forward steps
    until achievement index i unlocks in the trajectory (0 if never occurs).
    Operates on composite recurrent state s_t = [z_t, h_t].
    """

    def __init__(
        self,
        state_dim: int = 384,
        num_achievements: int = NUM_ACHIEVEMENTS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_achievements = num_achievements
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_achievements),
            nn.Sigmoid(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [..., state_dim]
        Returns: [..., num_achievements] continuous proximity in [0, 1]
        """
        return self.net(s)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Huber loss across all 22 heads by index."""
        return F.smooth_l1_loss(pred, target)

