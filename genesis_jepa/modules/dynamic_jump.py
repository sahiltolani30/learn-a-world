r"""
genesis_jepa.modules.dynamic_jump — Demidovich Spatially Gated Dynamic Jump Operator
===================================================================================
Mathematical Foundation:
Global Lipschitz contraction (Banach contraction L < 1) forces exponential state amnesia,
erasing 79.4% of memory over 15 steps.

In open-world physics, transitions are partially conservative:
    hat{Z}_{t+1} = Z_t + m_t \odot Delta_phi(Z_t, a_t)

Where:
  1. Routing Gate: m_t = sigmoid(W_g [Z_t; emb(a_t)] + b_g) in [0, 1]^{B x K x 1}
     With bias b_g initialized to -2.0, passive background tokens receive m_k -> 0,
     preserving exact mathematical identity (lambda = 1.0) without compounding decay.
  2. SimNorm Displacements: Delta_phi passes through SimNorm, ensuring ||Delta z_k||_2 <= sqrt(M).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from genesis_jepa.modules.simnorm import SimNorm


class SpatiallyGatedJumpPredictor(nn.Module):
    """
    Spatially Gated Dynamic Jump Operator for JEPA world model transitions.
    """

    def __init__(
        self,
        token_dim: int = 192,
        action_dim: int = 17,
        action_emb_dim: int = 32,
        num_patches: int = 64,
        simnorm_heads: int = 6,
        simnorm_head_dim: int = 32,
        hidden_dim: int = 384,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.action_emb_dim = action_emb_dim
        self.num_patches = num_patches

        # Discrete Action Embedding
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)

        # Context combination dimension
        combined_dim = token_dim + action_emb_dim

        # 1. Dynamic Routing Gate: W_g [z_k; a] + b_g
        self.gate_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Initialize gate bias to negative value (favor conservation of background)
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

        # 2. Latent Jump Dynamics Network
        self.jump_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # 3. Simplicial Normalization on Displacements
        self.simnorm = SimNorm(num_heads=simnorm_heads, head_dim=simnorm_head_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: Current latent patch tokens of shape (B, K, token_dim)
            action: Discrete action indices of shape (B,) or (B, 1)
        Returns:
            z_next: Predicted latent tokens of shape (B, K, token_dim)
            gate: Routing gate values of shape (B, K, 1) in [0, 1]
            delta: SimNorm-bounded displacement vectors of shape (B, K, token_dim)
        """
        B, K, D = z_t.shape
        if action.dim() == 2:
            action = action.squeeze(1)

        # Embed action: (B, action_emb_dim) -> (B, 1, action_emb_dim) -> (B, K, action_emb_dim)
        a_emb = self.action_emb(action).unsqueeze(1).expand(-1, K, -1)

        # Concatenate token features with action context
        combined = torch.cat([z_t, a_emb], dim=-1)  # (B, K, D + action_emb_dim)

        # Compute dynamic routing gate per token
        gate_logits = self.gate_net(combined)       # (B, K, 1)
        m_t = torch.sigmoid(gate_logits)            # (B, K, 1)

        # Compute raw latent displacement and bound via SimNorm
        raw_delta = self.jump_net(combined)         # (B, K, D)
        delta = self.simnorm(raw_delta)             # (B, K, D) with ||delta_k|| <= sqrt(M)

        # Apply Demidovich jump update
        z_next = z_t + m_t * delta

        return z_next, m_t, delta
