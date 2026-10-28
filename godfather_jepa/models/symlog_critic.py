"""
65-Bin Symlog Two-Hot Categorical Critic for Godfather JEPA.

Replaces flawed scalar MSE critics (which collapsed to a flat ~0.73 constant with Std < 0.02)
with a 65-bin two-hot symlog categorical distribution in [-20, 20]:
1. Compresses large returns via symlog(x) = sign(x) * ln(|x| + 1).
2. Spreads value probability mass over adjacent support bins (two-hot representation).
3. Trained via cross-entropy loss against multi-step lambda-returns, preserving distribution
   spread (Std > 0.20) and providing sharp gradients for tool milestone discovery.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Computes symmetrical log transform: symlog(x) = sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(y: torch.Tensor) -> torch.Tensor:
    """Inverts symmetrical log transform: symexp(y) = sign(y) * (exp(|y|) - 1)."""
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)


class SymlogTwoHotCoder:
    """
    Two-hot categorical encoder and decoder for symlog value distributions.
    """

    def __init__(self, num_bins: int = 65, v_min: float = -20.0, v_max: float = 20.0):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.bins = torch.linspace(v_min, v_max, num_bins)
        self.delta = (v_max - v_min) / (num_bins - 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encodes scalar return values into two-hot probability distributions.
        Args:
            values: (B,) scalar values.
        Returns:
            twohot: (B, num_bins) two-hot target distribution.
        """
        sym_vals = symlog(values).clamp(self.v_min, self.v_max)
        device = values.device
        rel_pos = (sym_vals - self.v_min) / self.delta
        idx_low = rel_pos.floor().long().clamp(0, self.num_bins - 2)
        idx_high = idx_low + 1
        w_high = rel_pos - idx_low.float()
        w_low = 1.0 - w_high

        twohot = torch.zeros(values.shape[0], self.num_bins, device=device)
        twohot.scatter_add_(1, idx_low.unsqueeze(1), w_low.unsqueeze(1))
        twohot.scatter_add_(1, idx_high.unsqueeze(1), w_high.unsqueeze(1))
        return twohot

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decodes categorical logits into scalar expected values.
        Args:
            logits: (B, num_bins) unnormalized log-probabilities.
        Returns:
            expected_values: (B,) scalar expected returns in real value space.
        """
        probs = F.softmax(logits, dim=-1)
        bins = self.bins.to(logits.device)
        sym_mean = (probs * bins).sum(dim=-1)
        return symexp(sym_mean)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes cross-entropy loss between critic logits and two-hot encoded targets.
        Args:
            logits: (B, num_bins)
            targets: (B,) scalar return targets.
        Returns:
            scalar cross-entropy loss.
        """
        twohot = self.encode(targets)
        return -(twohot * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class SymlogCategoricalCritic(nn.Module):
    """
    Categorical Critic MLP estimating value distributions over 65 symlog bins.
    """

    def __init__(self, state_dim: int = 1024, hidden_dim: int = 512, num_bins: int = 65):
        super().__init__()
        self.state_dim = state_dim
        self.num_bins = num_bins
        self.coder = SymlogTwoHotCoder(num_bins=num_bins)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            state: Composite state vector [z_t; h_t] of shape (B, 1024).
        Returns:
            logits: Unnormalized logits over 65 bins of shape (B, 65).
        """
        return self.net(state)

    def expected_value(self, state: torch.Tensor) -> torch.Tensor:
        """Returns scalar estimated value for state."""
        logits = self.forward(state)
        return self.coder.decode(logits)
