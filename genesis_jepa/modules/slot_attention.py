"""
genesis_jepa.modules.slot_attention — Inverted Competitive Slot Attention
========================================================================
Mathematical Formulation:
Standard spatial self-attention normalizes across spatial tokens (sum_{p} A_{s, p} = 1.0),
which dilutes 3x3 pixel dropped items (SNR -26.58 dB) across 64 background patches.

Inverted Slot Attention normalizes softmax across SLOTS for each PATCH:
    A_{s, p} = exp(q_s^T k_p / sqrt(d)) / ( sum_{m=1}^S exp(q_m^T k_p / sqrt(d)) )
Invariant:
    sum_{s=1}^S A_{s, p} == 1.0  (for all patches p in {1, ..., K})

This allows a single foreground slot to claim 100% of a small dropped item patch,
preserving O(1) feature magnitude, while background slots absorb grass and water.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedCompetitiveSlotAttention(nn.Module):
    """
    Inverted Slot Attention where slots compete for image patches.
    """

    def __init__(
        self,
        num_slots: int = 12,
        dim: int = 192,
        iters: int = 3,
        hidden_dim: int = 384,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = 1.0 / math.sqrt(dim)

        # Slot initialization: learnable Gaussian parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        # Linear projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Recurrent state transition
        self.gru = nn.GRUCell(dim, dim)

        # Feedforward refinement
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (B, K, dim) representing K patch tokens.
        Returns:
            slots: Tensor of shape (B, S, dim) representing S discrete object slots.
        """
        B, K, D = inputs.shape
        inputs = self.norm_input(inputs)

        # Keys and values from patch tokens
        k = self.to_k(inputs)  # (B, K, D)
        v = self.to_v(inputs)  # (B, K, D)

        # Sample initial slot states
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # (B, S, D)

        for _ in range(self.iters):
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)  # (B, S, D)

            # Attention scores: Q @ K^T -> (B, S, K)
            dots = torch.einsum("bsd,bkd->bsk", q, k) * self.scale

            # INVERTED SOFTMAX: Normalization across SLOTS (dim=1) for each patch (dim=2)
            # sum_{s=1}^S attn[b, s, p] == 1.0
            attn = F.softmax(dots, dim=1) + self.eps

            # Weighted sum over patches
            # Normalization factor for each slot across patches
            attn_norm = attn / torch.sum(attn, dim=-1, keepdim=True)
            updates = torch.einsum("bsk,bkd->bsd", attn_norm, v)  # (B, S, D)

            # GRU update
            slots_flat = slots.view(-1, D)
            updates_flat = updates.view(-1, D)
            slots_next = self.gru(updates_flat, slots_flat)
            slots = slots_next.view(B, self.num_slots, D)

            # Residual MLP refinement
            slots = slots + self.mlp(self.norm_slots(slots))

        return slots
