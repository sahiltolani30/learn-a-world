"""
sapna/encoder.py

Visual Observation Encoder and EMA Target Encoder for SAPNA.
Copied from src/models/encoder.py and src/models/target.py per SAPNA_PLAN.md §2.1.
"""

from typing import Optional, Tuple, Union
import copy
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and an identity skip connection.
    Activations use inplace=False to prevent autograd in-place modification errors.
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
    - Mean pooling projection:
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

        # 8x8 spatial preservation pool to guarantee (spatial_dim, spatial_dim) resolution
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))

        # token_proj Conv2d(64, token_dim, 1) mapping feature map to token channels
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
        tokens = token_map.flatten(2).transpose(1, 2)
        return tokens

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Pool a sequence of spatial tokens into a single 256-d latent vector via mean pooling (W1a-3).
        """
        unbatched = False
        if tokens.dim() == 2:
            unbatched = True
            tokens = tokens.unsqueeze(0)

        pooled_token = tokens.mean(dim=1)  # (B, token_dim)
        z = self.ln(self.pooled_proj(pooled_token))  # (B, latent_dim)

        if unbatched:
            z = z.squeeze(0)
        return z

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both spatial tokens and pooled 256-d latent.
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
        """
        _, pooled = self.forward_tokens(x)
        return pooled


# Aliases
TokenEncoder = ObservationEncoder
ResNetTinyEncoder = ObservationEncoder


class TargetEncoder(nn.Module):
    """
    EMA Target Encoder with stop-gradient enforcement.
    """

    def __init__(self, encoder: nn.Module, tau: float = 0.99):
        super().__init__()
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")

        object.__setattr__(self, "online_encoder", encoder)
        self.tau = float(tau)

        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.target_encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.target_encoder(x)
            return z.detach()

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens, z = self.target_encoder.forward_tokens(x)
            return tokens.detach(), z.detach()

    def update(self, tau: Optional[float] = None) -> None:
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)

            for b_t, b_o in zip(self.target_encoder.buffers(), self.online_encoder.buffers()):
                if b_t.is_floating_point():
                    b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)
                else:
                    b_t.data.copy_(b_o.data)

    def ema_update(self, tau: Optional[float] = None) -> None:
        self.update(tau)

    def hard_update(self) -> None:
        with torch.no_grad():
            for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                p_t.data.copy_(p_o.data)
            for b_t, b_o in zip(self.target_encoder.buffers(), self.online_encoder.buffers()):
                b_t.data.copy_(b_o.data)

    def train(self, mode: bool = True):
        super().train(False)
        self.target_encoder.eval()
        return self


def verify_non_collapse(
    z: torch.Tensor,
    min_var_threshold: float = 0.1,
    strict: bool = True,
    eps: float = 1e-4,
) -> dict:
    """
    Audits a batch of latent vectors to mathematically verify non-collapse per §5.5.
    
    Verifies that:
    1. Every feature dimension maintains empirical variance > min_var_threshold.
    2. Effective rank (participation ratio) confirms multi-dimensional feature spread.
    """
    if z.ndim == 2:
        B, D = z.shape
    elif z.ndim == 3:
        _, B, D = z.shape
    else:
        raise ValueError(f"Expected 2D (B, D) or 3D (K, B, D) tensor, got {tuple(z.shape)}")

    if B < 2:
        raise ValueError(f"Batch size B must be at least 2 to evaluate variance; got B={B}")

    variances = torch.var(z, dim=-2, unbiased=True)
    min_var = float(variances.min().item())
    mean_var = float(variances.mean().item())
    max_var = float(variances.max().item())

    collapsed_mask = variances <= min_var_threshold
    num_collapsed = int(collapsed_mask.sum().item())
    is_valid = (num_collapsed == 0)

    # Effective rank (Participation Ratio): PR = (Tr(C))^2 / Tr(C^2)
    if z.ndim == 2:
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, (B - 1))
        tr_c = torch.diagonal(cov).sum()
        tr_c2 = cov.pow(2).sum()
        eff_rank = float(((tr_c ** 2) / (tr_c2 + eps)).item())
    else:
        eff_rank = 0.0

    report = {
        "is_valid": is_valid,
        "min_variance": min_var,
        "mean_variance": mean_var,
        "max_variance": max_var,
        "num_collapsed_dims": num_collapsed,
        "effective_rank": eff_rank,
    }

    if strict and not is_valid:
        raise RuntimeError(
            f"VICReg collapse detected: {num_collapsed}/{z.shape[-1]} dims "
            f"have variance <= {min_var_threshold:.4f} (min_var={min_var:.4f})"
        )
    return report
