"""
CNN Observation Encoder for Learn-a-World.
Maps top-down observation images (64x64x3) into abstract latent vectors z (128d).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    Compact Convolutional Encoder for 64x64 visual observations.
    Includes LayerNorm on the final latent representation to standardize scale.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        features: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 4 strided convolutional layers: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        layers = []
        curr_c = in_channels
        for f in features:
            layers.append(nn.Conv2d(curr_c, f, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            curr_c = f

        self.conv_net = nn.Sequential(*layers)
        self.pre_fc_ln = nn.LayerNorm(features[-1] * 4 * 4)
        self.fc = nn.Linear(features[-1] * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, C, H, W] in range [0, 1]
        Returns:
            z: Latent representation of shape [B, latent_dim]
        """
        feats = self.conv_net(x)
        flat = feats.flatten(start_dim=1)
        flat = self.pre_fc_ln(flat)
        z = self.fc(flat)
        return z
