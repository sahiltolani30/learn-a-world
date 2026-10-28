"""
Expander Projector for Godfather JEPA.

Projects 512-dim visual latents into high-dimensional representation space (1024-dim)
for non-contrastive VICReg variance and covariance decorrelation penalties.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ExpanderProjector(nn.Module):
    """
    Multi-layer projection head mapping visual latent z into projector space.
    """

    def __init__(self, in_dim: int = 512, proj_dim: int = 1024):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization guaranteeing healthy feature spread."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            z: Latent tensor of shape (B, in_dim=512)
        Returns:
            p: Projection tensor of shape (B, proj_dim=1024)
        """
        return self.net(z)
