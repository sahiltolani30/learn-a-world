"""CriticNet: learned long-horizon value via two-hot distributional regression (V3 F1).

Predicts expected return over horizon H using a discrete two-hot distribution
over symlog returns. Provides baseline expectations for AWR policy weighting
and payoff-driven goal scoring without backbone backprop.
"""

from __future__ import annotations

import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import STATE_DIM, SkillsConfig


def symlog(x: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
    """Symmetric logarithm: sign(x) * log(1 + |x|)."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    else:
        sign = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        return sign * np.log1p(abs(x))


def symexp(x: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
    """Inverse symmetric logarithm: sign(x) * (exp(|x|) - 1)."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.expm1(torch.abs(x))
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.expm1(np.abs(x))
    else:
        sign = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        return sign * np.expm1(abs(x))


def two_hot_encode(
    x: torch.Tensor,
    num_bins: int = 65,
    v_min: float = -5.0,
    v_max: float = 10.0,
) -> torch.Tensor:
    """Two-hot discrete encoding of continuous values into prob distribution over bins."""
    x_clamped = x.clamp(v_min, v_max)
    delta = (v_max - v_min) / float(num_bins - 1)
    k = ((x_clamped - v_min) / delta).floor().long()
    k = k.clamp(0, num_bins - 2)

    weight_upper = ((x_clamped - (v_min + k.float() * delta)) / delta).clamp(0.0, 1.0)
    weight_lower = (1.0 - weight_upper).clamp(0.0, 1.0)

    shape = list(x.shape) + [num_bins]
    target_dist = torch.zeros(shape, device=x.device, dtype=torch.float32)

    target_dist.scatter_(-1, k.unsqueeze(-1), weight_lower.unsqueeze(-1))
    target_dist.scatter_(-1, (k + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))
    return target_dist


def two_hot_decode(
    logits: torch.Tensor,
    num_bins: int = 65,
    v_min: float = -5.0,
    v_max: float = 10.0,
) -> torch.Tensor:
    """Decodes discrete logits into expected scalar continuous value."""
    probs = F.softmax(logits, dim=-1)
    bins = torch.linspace(v_min, v_max, num_bins, device=logits.device)
    return (probs * bins).sum(dim=-1)


def critic_loss(online_logits: torch.Tensor, target_twohot: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss against two-hot targets."""
    log_probs = F.log_softmax(online_logits, dim=-1)
    return -(target_twohot * log_probs).sum(dim=-1).mean()


class CriticNet(nn.Module):
    """Trunk MLP(384 -> 256 -> 256) LayerNorm+SiLU; head: 65-bin two-hot over symlog returns.

    Bins span [-5, 10] on symlog(1 + discounted return); maintains EMA target copy (decay 0.995).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
        num_bins: int = 65,
        v_min: float = -5.0,
        v_max: float = 10.0,
        ema_decay: float = 0.995,
        cfg: Optional[SkillsConfig] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_bins = getattr(cfg, "critic_bins", num_bins) if cfg else num_bins
        self.v_min = getattr(cfg, "critic_v_min", v_min) if cfg else v_min
        self.v_max = getattr(cfg, "critic_v_max", v_max) if cfg else v_max
        self.ema_decay = getattr(cfg, "critic_ema_decay", ema_decay) if cfg else ema_decay

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(self.hidden_dim, self.num_bins)

        self.register_buffer("bins", torch.linspace(self.v_min, self.v_max, self.num_bins))

        # Target network copy for stable bootstrapping / evaluation
        self.target_trunk = copy.deepcopy(self.trunk)
        self.target_head = copy.deepcopy(self.head)
        for p in self.target_trunk.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns logits over return bins of shape [..., num_bins]."""
        was_1d = (s.ndim == 1)
        if was_1d:
            s = s.unsqueeze(0)
        h = self.trunk(s)
        logits = self.head(h)
        return logits.squeeze(0) if was_1d else logits

    def forward_target(self, s: torch.Tensor) -> torch.Tensor:
        """Returns logits from EMA target network."""
        was_1d = (s.ndim == 1)
        if was_1d:
            s = s.unsqueeze(0)
        h = self.target_trunk(s)
        logits = self.target_head(h)
        return logits.squeeze(0) if was_1d else logits

    def value(self, s: torch.Tensor, use_target: bool = False) -> Union[torch.Tensor, float]:
        """Returns scalar expected value in raw discounted return scale."""
        was_1d = (s.ndim == 1)
        logits = self.forward_target(s) if use_target else self.forward(s)
        if was_1d:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        expected_symlog = (probs * self.bins).sum(dim=-1)
        val = symexp(expected_symlog)
        if was_1d:
            return float(val.squeeze(0).item())
        return val

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None) -> None:
        """Applies EMA update: target = decay * target + (1 - decay) * online."""
        d = self.ema_decay if decay is None else decay
        for p_online, p_target in zip(self.trunk.parameters(), self.target_trunk.parameters()):
            p_target.data.mul_(d).add_(p_online.data, alpha=1.0 - d)
        for p_online, p_target in zip(self.head.parameters(), self.target_head.parameters()):
            p_target.data.mul_(d).add_(p_online.data, alpha=1.0 - d)
