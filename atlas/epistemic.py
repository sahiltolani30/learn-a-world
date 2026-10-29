"""EpistemicNet: K bootstrap residual predictors + kNN novelty.

Measures world model epistemic uncertainty via ensemble disagreement over next-latent residuals.
Intrinsic curiosity signal: r_int = disagreement + knn_beta * knn_novelty.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import (
    LATENT_DIM,
    NUM_ACTIONS,
    STATE_DIM,
    EpistemicConfig,
)


class EpistemicNet(nn.Module):
    """Ensemble of K MLP heads predicting latent transition residuals dz_k(s, a).

    Disagreement = std_k(next_latent_k).mean(dim=-1).
    Since next_latent_k = frozen_pred(z, h, a) + dz_k, std_k(next_latent_k) == std_k(dz_k).
    """

    def __init__(self, cfg: Optional[EpistemicConfig] = None):
        super().__init__()
        self.cfg = cfg or EpistemicConfig()
        k = self.cfg.num_heads
        h_dim = self.cfg.hidden_dim
        action_emb_dim = 32

        self.action_emb = nn.Embedding(NUM_ACTIONS, action_emb_dim)
        in_dim = STATE_DIM + action_emb_dim

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(h_dim, LATENT_DIM),
            )
            for _ in range(k)
        ])

    def predict_residuals(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Returns predicted residuals for all K heads: shape [K, B, LATENT_DIM]."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if a.ndim == 0:
            a = a.unsqueeze(0)

        a_emb = self.action_emb(a)
        feat = torch.cat([s, a_emb], dim=-1)

        residuals = [head(feat) for head in self.heads]
        return torch.stack(residuals, dim=0)

    def disagreement(
        self,
        z_or_s: torch.Tensor,
        h_or_a: Optional[torch.Tensor] = None,
        a_opt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes epistemic disagreement: std over K heads, mean over latent dims.

        Accepts either:
            disagreement(s, a)
        or:
            disagreement(z, h, a)
        Returns: [B] scalar disagreement per batch item.
        """
        if a_opt is not None:
            # Called as disagreement(z, h, a)
            z = z_or_s
            h = h_or_a
            a = a_opt
            s = torch.cat([z, h], dim=-1)
        else:
            # Called as disagreement(s, a)
            s = z_or_s
            a = h_or_a

        if s.ndim == 1:
            s = s.unsqueeze(0)
        if a.ndim == 0:
            a = a.unsqueeze(0)

        res = self.predict_residuals(s, a)  # [K, B, 128]
        # Standard deviation across heads K
        std_k = torch.std(res, dim=0, unbiased=True)  # [B, 128]
        return std_k.mean(dim=-1)  # [B]

    def disagreement_all_actions(self, s: torch.Tensor) -> torch.Tensor:
        """Computes disagreement for all 17 discrete actions: returns [B, 17]."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]
        a_all = torch.arange(NUM_ACTIONS, device=s.device).unsqueeze(0).expand(B, NUM_ACTIONS)  # [B, 17]
        s_exp = s.unsqueeze(1).expand(B, NUM_ACTIONS, STATE_DIM)  # [B, 17, 384]

        flat_s = s_exp.reshape(-1, STATE_DIM)
        flat_a = a_all.reshape(-1)

        d_flat = self.disagreement(flat_s, flat_a)
        return d_flat.reshape(B, NUM_ACTIONS)

    def compute_loss(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        target_residual: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss across all K heads against true residual z_{t+1} - frozen_pred."""
        res = self.predict_residuals(s, a)  # [K, B, 128]
        target = target_residual.unsqueeze(0).expand_as(res)
        return F.mse_loss(res, target)


def knn_novelty(
    h: Union[np.ndarray, torch.Tensor],
    memory: Sequence[Union[np.ndarray, torch.Tensor]],
    k: int = 4,
) -> float:
    """Mean Euclidean distance to the k nearest neighbors in episode memory."""
    if len(memory) == 0:
        return 0.0

    if isinstance(h, torch.Tensor):
        h_vec = h.detach().cpu().numpy().reshape(-1)
    else:
        h_vec = np.asarray(h, dtype=np.float32).reshape(-1)

    mem_arr = []
    for item in memory:
        if isinstance(item, torch.Tensor):
            mem_arr.append(item.detach().cpu().numpy().reshape(-1))
        else:
            mem_arr.append(np.asarray(item, dtype=np.float32).reshape(-1))

    M = np.stack(mem_arr, axis=0)  # [N, dim]
    dists = np.linalg.norm(M - h_vec, axis=-1)  # [N]
    actual_k = min(k, len(dists))
    top_k = np.partition(dists, actual_k - 1)[:actual_k]
    return float(np.mean(top_k))
