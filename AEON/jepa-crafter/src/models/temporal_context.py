"""Temporal context GRU for summarizing recent observations and actions."""
from typing import Tuple, Optional
import torch
import torch.nn as nn


class TemporalContext(nn.Module):
    """Temporal context state summarizing recent visual latents and action embeddings."""

    def __init__(self, visual_dim: int = 128, action_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input_dim = visual_dim + action_dim  # 192
        self.gru_cell = nn.GRUCell(self.input_dim, hidden_dim)

        # Learned start / unobserved context embedding
        self.start_visual = nn.Parameter(torch.zeros(1, visual_dim))
        self.start_action = nn.Parameter(torch.zeros(1, action_dim))

    def step(self, visual_pooled: torch.Tensor, action_emb: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Single-step GRU update: h_{t+1} = GRUCell([u_{t+1}; a_t], h_t).
        
        Args:
            visual_pooled: [B, 128]
            action_emb: [B, 64]
            h_prev: [B, 256]
        Returns:
            h_next: [B, 256]
        """
        inp = torch.cat([visual_pooled, action_emb], dim=-1)  # [B, 192]
        return self.gru_cell(inp, h_prev)

    def forward_history(
        self,
        visual_history: torch.Tensor,
        action_history: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process observed history to produce context hidden state h_t.
        
        Args:
            visual_history: [B, T_ctx, 128]
            action_history: [B, T_ctx, 64]
            validity_mask: [B, T_ctx] bool tensor where True indicates observed frame
        Returns:
            h_t: [B, 256] context state at final context step
        """
        B, T_ctx, _ = visual_history.shape
        device = visual_history.device
        h = torch.zeros(B, self.hidden_dim, device=device)

        start_v = self.start_visual.expand(B, -1)
        start_a = self.start_action.expand(B, -1)

        for t in range(T_ctx):
            v_t = visual_history[:, t]
            a_t = action_history[:, t]

            if validity_mask is not None:
                mask_t = validity_mask[:, t].unsqueeze(-1).float()  # [B, 1]
                v_t = mask_t * v_t + (1.0 - mask_t) * start_v
                a_t = mask_t * a_t + (1.0 - mask_t) * start_a

            h_next = self.step(v_t, a_t, h)
            if validity_mask is not None:
                # If frame was unobserved, keep h unchanged (or updated from start token)
                h = mask_t * h_next + (1.0 - mask_t) * h
            else:
                h = h_next

        return h
