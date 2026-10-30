"""
src/models/encoder.py

Visual Observation Encoder (ResNetTinyEncoder / ObservationEncoder) for the JEPA World Model.
Maps raw pixel observations (Crafter 64x64x3 or Atari 84x84x4) into a continuous
latent representation z in R^{latent_dim} with LayerNorm.

Preserves spatial feature maps across 3 downsampling stages to retain fine-grained status
bars (health, hunger, thirst, energy) located in the lower HUD rows.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and an identity skip connection.
    Activations use inplace=False to prevent autograd in-place modification errors
    during backward passes through residual branches.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        return residual + out


class ObservationEncoder(nn.Module):
    """
    ResNetTiny visual encoder mapping pixel frames to latent representations.
    
    Architecture:
    - 3 Downsampling residual stages:
        Stage 1: Conv2d(in_channels, c1, 4, stride=2, padding=1) -> ReLU -> ResBlock(c1)
        Stage 2: Conv2d(c1, c2, 4, stride=2, padding=1) -> ReLU -> ResBlock(c2)
        Stage 3: Conv2d(c2, c3, 4, stride=2, padding=1) -> ReLU -> ResBlock(c3)
    - Spatial preservation pooling & 1x1 conv preserving spatial grid:
        AdaptiveAvgPool2d((spatial_dim, spatial_dim)) -> Conv2d(c3, head_channels, 1) -> ReLU
    - Dense projection to latent dimension:
        Linear(head_channels * spatial_dim * spatial_dim, latent_dim) -> LayerNorm(latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        channels: Tuple[int, int, int] = (32, 64, 64),
        spatial_dim: int = 8,
        head_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.head_channels = head_channels
        c1, c2, c3 = channels

        # Stage 1: (B, in_channels, H, W) -> (B, c1, H/2, W/2)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            ResBlock(c1),
        )

        # Stage 2: (B, c1, H/2, W/2) -> (B, c2, H/4, W/4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            ResBlock(c2),
        )

        # Stage 3: (B, c2, H/4, W/4) -> (B, c3, H/8, W/8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            ResBlock(c3),
        )

        # Spatial preservation head: maintains (spatial_dim, spatial_dim) resolution
        # Preserves HUD rows in distinct spatial feature cells
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c3, head_channels, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        # Dense projection to latent dimension
        flat_dim = head_channels * spatial_dim * spatial_dim
        self.proj = nn.Linear(flat_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping pixel observation to latent representation.
        
        Args:
            x: Tensor of shape (B, C, H, W) or (C, H, W).
               Supports float32 in [0, 1] or uint8 in [0, 255].
        Returns:
            z: Normalized latent state tensor of shape (B, latent_dim) or (latent_dim,).
        """
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected 3D (C, H, W) or 4D (B, C, H, W) observation, got shape {tuple(x.shape)}"
            )

        # Defensive dtype conversion
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif not x.is_floating_point():
            x = x.float()

        # 3 downsampling residual stages
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)

        # Spatial preservation and projection
        h = self.spatial_pool(h)
        h = self.spatial_conv(h)
        h = torch.flatten(h, start_dim=1)
        z = self.proj(h)
        z = self.ln(z)

        if unbatched:
            z = z.squeeze(0)
        return z


# Interface alias matching PROJECT.md and task specification
ResNetTinyEncoder = ObservationEncoder
