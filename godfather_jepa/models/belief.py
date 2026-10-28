"""
Spatial ConvGRU and Temporal Vector Belief Core for Godfather JEPA.

Maintains 2D translation equivariance and object permanence:
1. SpatialConvGRU: Tracks persistent spatial representations across 9x7 camera viewports
   modulated by ego-motion actions (preserving crafting tables and stone nodes when moving).
2. VectorBeliefCore: Recurrent temporal core integrating visual latents, actions, and
   spatial belief pool into a unified recurrent state vector h_t in R^512.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class SpatialConvGRU(nn.Module):
    """
    Convolutional Gated Recurrent Unit operating on spatial feature grids (8x8)
    modulated by ego-motion discrete actions.
    """

    def __init__(
        self,
        channels: int = 64,
        in_channels: Optional[int] = None,
        num_actions: int = 17,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels if in_channels is not None else channels
        self.num_actions = num_actions
        self.act_emb = nn.Embedding(num_actions, channels)

        if self.in_channels != channels:
            from godfather_jepa.models.encoder import SpatialFeatureProjector
            self.input_proj = SpatialFeatureProjector(self.in_channels, channels)
        else:
            self.input_proj = nn.Identity()

        padding = kernel_size // 2
        combined_channels = channels * 2
        self.convz = nn.Conv2d(combined_channels, channels, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(combined_channels, channels, kernel_size=kernel_size, padding=padding)
        self.convh = nn.Conv2d(combined_channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(
        self,
        feat_proj: torch.Tensor,
        action: torch.Tensor,
        H_spatial: Optional[torch.Tensor] = None,
        H: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward step.
        Args:
            feat_proj: Spatial feature map of shape (B, channels=64, 8, 8)
                       or (B, in_channels, 8, 8) if in_channels was configured.
            action: Discrete action tensor of shape (B,).
            H_spatial: Previous spatial belief state of shape (B, channels=64, 8, 8), or None.
            H: Alias for H_spatial for backward compatibility.
        Returns:
            Updated spatial belief state H_next of shape (B, channels=64, 8, 8).
        """
        h_prev = H_spatial if H_spatial is not None else H

        if feat_proj.shape[1] == self.in_channels and self.in_channels != self.channels:
            feat_proj = self.input_proj(feat_proj)

        b, c, h, w = feat_proj.shape
        if h_prev is None:
            h_prev = torch.zeros(b, c, h, w, device=feat_proj.device, dtype=feat_proj.dtype)

        act_map = self.act_emb(action).unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        inp = feat_proj + act_map
        combined = torch.cat([inp, h_prev], dim=1)

        z_gate = torch.sigmoid(self.convz(combined))
        r_gate = torch.sigmoid(self.convr(combined))

        combined_h = torch.cat([inp, r_gate * h_prev], dim=1)
        h_cand = torch.tanh(self.convh(combined_h))

        H_next = (1.0 - z_gate) * h_prev + z_gate * h_cand
        return H_next


class VectorBeliefCore(nn.Module):
    """
    Recurrent Vector Belief Core fusing visual latent, action embedding, and spatial pool.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_actions: int = 17,
        spatial_channels: int = 64,
        act_emb_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_emb = nn.Embedding(num_actions, act_emb_dim)
        self.cell = nn.GRUCell(latent_dim + act_emb_dim + spatial_channels, latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        H_spatial: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward step.
        Args:
            z: Latent vector of shape (B, 512).
            action: Discrete action indices of shape (B,).
            H_spatial: Spatial belief grid of shape (B, 64, 8, 8).
            h: Previous vector belief state of shape (B, 512), or None.
        Returns:
            Updated vector belief state h_next of shape (B, 512).
        """
        b = z.shape[0]
        if h is None:
            h = torch.zeros(b, self.latent_dim, device=z.device, dtype=z.dtype)

        spatial_pool = H_spatial.mean(dim=[-2, -1])  # (B, 64)
        act_vec = self.act_emb(action)              # (B, 64)
        fused = torch.cat([z, act_vec, spatial_pool], dim=-1)

        h_next = self.cell(fused, h)
        return h_next

    @staticmethod
    def get_unified_state(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concatenates latent visual state z and recurrent belief state h into unified state s in R^(B x 1024).
        """
        return torch.cat([z, h], dim=-1)

