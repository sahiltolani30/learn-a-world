"""Visual encoder and perceptual anchor decoder for MOSAIC (Phase P2)."""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block with GroupNorm and GELU."""

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + residual)


class AttentionPool(nn.Module):
    """Attention pooling layer over spatial tokens producing a global vector [B, latent_dim]."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.scale = 1.0 / (latent_dim ** 0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, latent_dim] where N is number of spatial tokens (e.g., 64).
        Returns:
            pooled: [B, latent_dim]
        """
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, latent_dim]
        attn = torch.bmm(q, tokens.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, tokens).squeeze(1)
        return pooled


class VisualEncoder(nn.Module):
    """Convolutional encoder producing an 8x8 spatial grid of 128-dim tokens and pooled feature.

    Input:
        x: RGB tensor [B, 3, 64, 64] with values in [0, 1] or uint8.
    Outputs:
        spatial_tokens: [B, 64, 128] representing the 8x8 spatial patch grid with positional embeddings.
        pooled_feature: [B, 128] global vector obtained via attention pooling.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Downsample 1: 64x64 -> 32x32 (channels: 3 -> 64)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            ConvBlock(64, num_groups=8),
            ConvBlock(64, num_groups=8),
        )

        # Downsample 2: 32x32 -> 16x16 (channels: 64 -> 96)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            ConvBlock(96, num_groups=8),
            ConvBlock(96, num_groups=8),
        )

        # Downsample 3: 16x16 -> 8x8 (channels: 96 -> 128)
        self.down2 = nn.Sequential(
            nn.Conv2d(96, latent_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            ConvBlock(latent_dim, num_groups=8),
            ConvBlock(latent_dim, num_groups=8),
        )

        # 2D Positional Embeddings for 8x8 grid (64 tokens)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, latent_dim) * 0.02)
        self.attention_pool = AttentionPool(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        h = self.stem(x)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)  # [B, 128, 8, 8]

        B, C, H, W = h.shape
        # Flatten spatial grid to tokens: [B, H*W, C] -> [B, 64, 128]
        tokens = h.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_emb

        pooled = self.attention_pool(tokens)  # [B, 128]
        return tokens, pooled


class AnchorDecoder(nn.Module):
    """Lightweight perceptual anchor decoder reconstructing 64x64x3 RGB observations.

    Used to anchor representation learning and prevent feature collapse.
    Can operate on spatial tokens [B, 64, 128] or projected belief states.
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Input: [B, 128, 8, 8]
        # Upsample 1: 8x8 -> 16x16
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            ConvBlock(96, num_groups=8),
        )

        # Upsample 2: 16x16 -> 32x32
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ConvBlock(64, num_groups=8),
        )

        # Upsample 3: 32x32 -> 64x64
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range matching normalized pixel space
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, 64, 128] spatial tokens or reshaped feature map.
        Returns:
            recon: [B, 3, 64, 64] reconstructed normalized RGB image.
        """
        B = tokens.size(0)
        # Reshape tokens back to [B, 128, 8, 8]
        feat_map = tokens.transpose(1, 2).reshape(B, self.latent_dim, 8, 8)
        h = self.up1(feat_map)
        h = self.up2(h)
        recon = self.up3(h)
        return recon
