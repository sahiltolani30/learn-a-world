"""
genesis_jepa.modules.simnorm — Simplicial Normalization (SimNorm)
================================================================
Mathematical Invariant:
For an activation vector partitioned into M heads of dimension L:
    p_{m, l} = softmax(v_{m, l} / tau) in Delta^{L-1}
The Euclidean norm is strictly upper-bounded:
    ||SimNorm(v)||_2 = sqrt( sum_{m=1}^M ||p_m||_2^2 ) <= sqrt(M * 1.0) = sqrt(M)
This eliminates exploding gradients and bounds state divergence unconditionally.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """
    Simplicial Normalization layer.
    Maps an unconstrained R^{dim} vector to M probability simplexes Delta^{L-1},
    where dim = num_heads * head_dim.
    """

    def __init__(self, num_heads: int, head_dim: int, temp: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temp = temp
        self.total_dim = num_heads * head_dim
        self.max_norm = math.sqrt(num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., total_dim)
        Returns:
            Tensor of shape (..., total_dim) where each head is a probability distribution.
        """
        orig_shape = x.shape
        # Flatten leading dimensions for head splitting
        x_reshaped = x.view(-1, self.num_heads, self.head_dim)
        # Apply temperature-scaled softmax per head across head_dim
        simplicial = F.softmax(x_reshaped / self.temp, dim=-1)
        # Restore original tensor shape
        return simplicial.view(orig_shape)

    @property
    def maximum_l2_norm(self) -> float:
        """The mathematical supremum of ||SimNorm(x)||_2."""
        return self.max_norm
