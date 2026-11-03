"""Visual convolutional encoder and attention pooling for JEPA-Crafter."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block with GroupNorm and GELU."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + res)


class AttentionPool(nn.Module):
    """Attention pooling layer over spatial tokens producing a global vector [B, latent_dim]."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.scale = 1.0 / (latent_dim ** 0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, 64, 128]
        Returns:
            pooled: [B, 128]
        """
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, 128]
        # Attention scores: [B, 1, 64]
        attn = torch.bmm(q, tokens.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        # Pooled vector: [B, 1, 128] -> [B, 128]
        pooled = torch.bmm(attn, tokens).squeeze(1)
        return pooled


class VisualEncoder(nn.Module):
    """Convolutional encoder producing an 8x8 spatial grid of 128-dim tokens (64 tokens).
    
    Architecture (Section 5.2):
    - Stem: 3x3 conv stride 2, 64 channels; GroupNorm(8); GELU -> [64, 32, 32]
    - Block 1: two 3x3 residual convs, 64 channels -> [64, 32, 32]
    - Downsample 1: 3x3 conv stride 2, 96 channels; GroupNorm(8); GELU -> [96, 16, 16]
    - Block 2: two 3x3 residual convs, 96 channels -> [96, 16, 16]
    - Downsample 2: 3x3 conv stride 2, 128 channels; GroupNorm(8); GELU -> [128, 8, 8]
    - Block 3: two 3x3 residual convs, 128 channels -> [128, 8, 8]
    - Tokenize: flatten spatial grid and add learned 8x8 2D positional embedding -> [64, 128]
    - AttentionPool: global pooled vector u_t [128]
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Stem: 3 -> 64 (downsample 64x64 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            ConvBlock(64, num_groups=8),
            ConvBlock(64, num_groups=8),
        )

        # Downsample 1: 64 -> 96 (downsample 32x32 -> 16x16)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            ConvBlock(96, num_groups=8),
            ConvBlock(96, num_groups=8),
        )

        # Downsample 2: 96 -> 128 (downsample 16x16 -> 8x8)
        self.down2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, num_groups=8),
            ConvBlock(128, num_groups=8),
        )

        # Learned 2D positional embedding: 8x8 = 64 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 64, latent_dim) * 0.02)
        self.pool = AttentionPool(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, 64, 64] float32 in [0, 1]
        Returns:
            tokens: [B, 64, 128] spatial feature tokens
            pooled: [B, 128] attention-pooled global vector
        """
        h = self.stem(x)        # [B, 64, 32, 32]
        h = self.block1(h)      # [B, 64, 32, 32]
        h = self.down1(h)       # [B, 96, 16, 16]
        h = self.block2(h)      # [B, 96, 16, 16]
        h = self.down2(h)       # [B, 128, 8, 8]
        h = self.block3(h)      # [B, 128, 8, 8]

        # Reshape to [B, 64, 128]
        B, C, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = tokens + self.pos_embed

        pooled = self.pool(tokens)
        return tokens, pooled
