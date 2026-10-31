"""
src/models/encoder.py

Visual Observation Encoder (ResNetTinyEncoder / ObservationEncoder) for the JEPA World Model.
Upgrade 1 (MAP - Spatial Tokens + Masked Prediction):
- Preserves 8x8 spatial token grid through stage3 trunk.
- token_proj: Conv2d(64, token_dim, 1) -> 64 spatial tokens of dimension 64. No flatten.
- pool: nn.MultiheadAttention(token_dim, num_heads=4, batch_first=True) with learned query pooling.
- pooled_proj: Linear(token_dim, latent_dim) + LayerNorm(latent_dim) -> 256-d pooled latent representation.
- Interface contract:
  * encoder(x) -> (B, 256) pooled vector (downstream compatibility).
  * encoder.forward_tokens(x) -> ((B, 64, token_dim), (B, 256)).
  * encoder.pool_tokens(tokens) -> (B, 256).
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
    ResNetTiny visual encoder mapping pixel frames to spatial tokens and pooled latent representations.
    
    Architecture (U1 MAP):
    - 3 Downsampling residual stages:
        Stage 1: Conv2d(in_channels, c1, 4, stride=2, padding=1) -> ReLU -> ResBlock(c1)
        Stage 2: Conv2d(c1, c2, 4, stride=2, padding=1) -> ReLU -> ResBlock(c2)
        Stage 3: Conv2d(c2, c3, 4, stride=2, padding=1) -> ReLU -> ResBlock(c3)
    - Spatial tokens:
        token_proj: Conv2d(c3, token_dim, 1) -> (B, 64, token_dim) (8x8 grid = 64 tokens)
    - Learned query attention pooling:
        pool: MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        pooled_proj: Linear(token_dim, latent_dim) -> LayerNorm(latent_dim) -> (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        channels: Tuple[int, int, int] = (32, 64, 64),
        spatial_dim: int = 8,
        head_channels: int = 32,
        token_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.head_channels = head_channels
        self.token_dim = token_dim
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

        # U1: 8x8 spatial preservation pool to guarantee (spatial_dim, spatial_dim) resolution
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))

        # U1: token_proj Conv2d(64, token_dim, 1) mapping feature map to token channels
        self.token_proj = nn.Conv2d(c3, token_dim, kernel_size=1)

        # W1a-3: Mean pooling projection to 256-d latent dimension + LayerNorm
        self.pooled_proj = nn.Linear(token_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

        # W1a-0: Reconstruction auxiliary head (each token decodes its own 8x8 patch)
        self.recon_head = nn.Linear(token_dim, 3 * 8 * 8)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        W1a-0: tokens (B, 64, token_dim) -> reconstructed frame (B, 3, 64, 64) in [0,1]-scale.
        """
        B = tokens.size(0)
        patches = self.recon_head(tokens).view(B, 8, 8, 3, 8, 8)
        frame = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 64, 64)
        return frame

    def _preprocess_obs(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Convert input observation to standard (B, C, H, W) float32 tensor."""
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected 3D (C, H, W) or 4D (B, C, H, W) observation, got shape {tuple(x.shape)}"
            )

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif not x.is_floating_point():
            x = x.float()
        return x, unbatched

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Pass observation through 3 downsampling stages and return (B, 64, token_dim)."""
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.spatial_pool(h)  # (B, c3, 8, 8)
        token_map = self.token_proj(h)  # (B, token_dim, 8, 8)
        # Flatten spatial dimensions to form token sequence: (B, token_dim, 64) -> (B, 64, token_dim)
        tokens = token_map.flatten(2).transpose(1, 2)
        return tokens

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Pool a sequence of spatial tokens into a single 256-d latent vector via mean pooling (W1a-3).
        
        Args:
            tokens: Tensor of shape (B, num_tokens, token_dim) or (num_tokens, token_dim).
        Returns:
            pooled: Tensor of shape (B, latent_dim) or (latent_dim,).
        """
        unbatched = False
        if tokens.dim() == 2:
            unbatched = True
            tokens = tokens.unsqueeze(0)

        # W1a-3: Mean pooling preserves every spatial region equally
        pooled_token = tokens.mean(dim=1)  # (B, token_dim)
        z = self.ln(self.pooled_proj(pooled_token))  # (B, latent_dim)

        if unbatched:
            z = z.squeeze(0)
        return z

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both spatial tokens and pooled 256-d latent.
        
        Args:
            x: Raw pixel observation tensor (B, C, H, W) or (C, H, W).
        Returns:
            tokens: (B, 64, token_dim) spatial tokens.
            pooled: (B, latent_dim) normalized pooled latent state.
        """
        x_proc, unbatched = self._preprocess_obs(x)
        tokens = self._extract_tokens(x_proc)  # (B, 64, token_dim)
        pooled = self.pool_tokens(tokens)  # (B, latent_dim)

        if unbatched:
            tokens = tokens.squeeze(0)
            pooled = pooled.squeeze(0)
        return tokens, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the standard 256-d pooled latent representation.
        Preserves complete backward compatibility with all downstream modules.
        
        Args:
            x: Raw pixel observation tensor (B, C, H, W) or (C, H, W).
        Returns:
            z: Normalized latent state tensor of shape (B, latent_dim) or (latent_dim,).
        """
        _, pooled = self.forward_tokens(x)
        return pooled


# Interface alias matching PROJECT.md and task specification
ResNetTinyEncoder = ObservationEncoder
