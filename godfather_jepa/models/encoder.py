"""
Visual Conv4 Backbone Encoder for Godfather JEPA.

Processes 64x64x3 RGB Crafter frames into:
1. Low-dimensional invariant latent vector z_t in R^512 via LayerNorm.
2. High-resolution spatial feature map feat in R^(256 x 8 x 8) preserving
   2D translation equivariance for spatial belief tracking.
"""
from __future__ import annotations

from typing import Tuple, Union
import math
import torch
import torch.nn as nn


class VisualConv4Encoder(nn.Module):
    """
    4-layer strided convolutional backbone for 64x64 RGB observations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        features_dim: int = 256,
        use_group_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.features_dim = features_dim
        self.use_group_norm = use_group_norm

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, features_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.fc = nn.Linear(features_dim * 8 * 8, latent_dim)
        self.ln = nn.LayerNorm(latent_dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization guaranteeing healthy variance and norm invariant."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    bound = 1.0 / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    bound = 1.0 / math.sqrt(m.weight.shape[1])
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            obs: Either (B, 64, 64, 3) uint8 or (B, 3, 64, 64) float in [0, 1].
        Returns:
            Tuple of:
                z: Latent vector of shape (B, latent_dim=512)
                feat: Spatial feature map of shape (B, 256, 8, 8)
        """
        if obs.dim() != 4:
            raise ValueError(f"Expected 4D observation tensor, got shape {obs.shape}")

        if obs.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            x = obs.permute(0, 3, 1, 2)
        else:
            x = obs

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0

        feat = self.conv(x)  # (B, 256, 8, 8)
        flat = feat.reshape(feat.shape[0], -1)  # (B, 16384)
        z = self.ln(self.fc(flat))  # (B, 512)
        return z, feat


class SpatialFeatureProjector(nn.Module):
    """
    Point-wise 1x1 convolution projection bridging high-dimensional visual backbone
    features (256x8x8) into the compact SpatialConvGRU memory channel space (64x8x8).

    Preserves 2D spatial translation equivariance via 1x1 point-wise kernel.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Backbone feature map of shape (B, in_channels=256, 8, 8).
        Returns:
            feat_proj: Projected spatial map of shape (B, out_channels=64, 8, 8).
        """
        return self.act(self.norm(self.proj(feat)))

