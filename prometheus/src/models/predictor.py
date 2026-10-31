"""
src/models/predictor.py

Action-Conditioned Latent Dynamics Predictor for Crafter & Atari JEPA World Model.
Contains:
1. SpatialTransformerPredictor (Upgrade 1 & 2): 2-layer Pre-LN Transformer Encoder over 64 spatial
   tokens, with discrete action embedding (dim 32) and GRU memory injection (dim 256).
   Supports 35% token masking, masked MSE prediction loss, and 4-head epistemic disagreement.
2. ResidualMLPPredictor: Legacy 4-head Residual MLP Predictor Ensemble for baseline compatibility.
3. ActionEmbedding: Differentiable embedding for discrete actions.
4. sample_token_mask: Uniformly samples 35% masked positions per update.
"""

from typing import Tuple, Optional, Union, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_token_mask(
    batch_size: int,
    num_tokens: int = 64,
    mask_ratio: float = 0.35,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Uniformly sample a binary mask hiding exactly floor(mask_ratio * num_tokens) tokens per sample.
    
    Args:
        batch_size: Number of items in batch.
        num_tokens: Total number of spatial tokens (default: 64).
        mask_ratio: Fraction of tokens to mask (default: 0.35 -> 22 tokens).
        device: PyTorch tensor device.
        
    Returns:
        mask: Boolean tensor of shape (batch_size, num_tokens), where True denotes masked tokens.
    """
    num_masked = int(math.floor(mask_ratio * num_tokens))
    if num_masked <= 0:
        return torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
    if num_masked >= num_tokens:
        return torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device)

    rand = torch.rand(batch_size, num_tokens, device=device)
    _, topk_indices = torch.topk(rand, k=num_masked, dim=-1)

    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=topk_indices, value=True)
    return mask


class ActionEmbedding(nn.Module):
    """
    Learnable discrete action embedding module.
    Supports 1D integer indices, 2D tensors, and one-hot float probability vectors.
    """

    def __init__(self, num_actions: int = 17, action_dim: int = 32):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.embedding = nn.Embedding(num_actions, action_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Embed discrete action into continuous space R^{action_dim}."""
        if a.dim() == 2 and a.size(-1) == 1:
            a = a.squeeze(-1)

        if a.dim() == 1:
            if a.dtype not in (torch.int64, torch.int32):
                a = a.long()
            if (a < 0).any() or (a >= self.num_actions).any():
                raise IndexError(
                    f"Action index out of range [0, {self.num_actions - 1}]. "
                    f"Min action: {a.min().item()}, Max action: {a.max().item()}"
                )
            return self.embedding(a)
        elif a.dim() == 2 and a.size(-1) == self.num_actions:
            return torch.matmul(a.float(), self.embedding.weight)
        else:
            raise ValueError(
                f"Unsupported action tensor shape {a.shape} for num_actions={self.num_actions}."
            )


class SpatialTransformerPredictor(nn.Module):
    """
    Action-Conditioned Spatial Transformer Predictor Ensemble for JEPA World Model.
    
    Architecture (U1 MAP + U2 MEMORY):
    - Input: current tokens (B, 64, 64) + action token (B, 1, 64) + memory token (B, 1, 64).
    - 2-layer Pre-LN Transformer Encoder (d_model=64, 4 heads, ff=256, dropout=0.0, GELU).
    - Positional embeddings for up to 128 positions.
    - 4 independent linear output heads predicting delta updates for spatial tokens.
    - 35% token masking with masked-only prediction loss.
    - Epistemic disagreement curiosity computed on pooled masked positions.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        action_dim: int = 32,
        num_heads: int = 4,
        token_dim: int = 64,
        num_tokens: int = 64,
        mem_hidden: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dim_feedforward: int = 256,
        mask_ratio: float = 0.35,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.mem_hidden = mem_hidden
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim_feedforward

        # Action and Memory projection to token_dim
        self.action_embedding = ActionEmbedding(num_actions=num_actions, action_dim=action_dim)
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.default_mem = nn.Parameter(torch.randn(1, mem_hidden) * 0.05)
        self.mem_proj = nn.Linear(mem_hidden, token_dim)

        # Learned positional embeddings (covering 64 spatial + action + memory tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, token_dim) * 0.02)

        # 2-layer Transformer Encoder (Pre-LN, dropout=0.0 for deterministic dynamics)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 4 Ensemble Output Heads (predicting deltas for spatial tokens)
        self.heads = nn.ModuleList([
            nn.Linear(token_dim, token_dim)
            for _ in range(num_heads)
        ])

        self._init_weights()

    def _init_weights(self):
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next-frame spatial tokens conditioned on action and memory.
        
        Args:
            tokens: Current spatial tokens of shape (B, 64, token_dim).
            a: Discrete action tensor (B,) or (B, 1) or (B, num_actions).
            h: Optional memory state tensor of shape (B, mem_hidden).
            
        Returns:
            mean_pred: Consensus ensemble predicted tokens of shape (B, 64, token_dim).
            all_preds: Stacked head predictions of shape (num_heads, B, 64, token_dim).
        """
        B = tokens.size(0)

        # 1. Embed action and project to token_dim
        a_emb = self.action_embedding(a)  # (B, action_dim)
        a_tok = self.action_proj(a_emb).unsqueeze(1)  # (B, 1, token_dim)

        # 2. Project memory state to token_dim (uses learnable default_mem if h is None)
        if h is None:
            h = self.default_mem.expand(B, -1)
        m_tok = self.mem_proj(h).unsqueeze(1)  # (B, 1, token_dim)

        # 3. Concatenate tokens: [spatial_tokens (64), action_token (1), memory_token (1)] -> (B, 66, token_dim)
        seq = torch.cat([tokens, a_tok, m_tok], dim=1)
        seq_len = seq.size(1)
        seq = seq + self.pos_embed[:, :seq_len, :]

        # 4. Pass through Transformer Encoder
        out_seq = self.transformer(seq)

        # 5. Extract spatial token positions
        spatial_out = out_seq[:, :self.num_tokens, :]  # (B, 64, token_dim)

        # 6. Predict residual deltas across 4 ensemble heads: pred_k = tokens + head_k(spatial_out)
        all_preds = torch.stack([
            tokens + head(spatial_out)
            for head in self.heads
        ], dim=0)  # (num_heads, B, 64, token_dim)

        mean_pred = all_preds.mean(dim=0)  # (B, 64, token_dim)
        return mean_pred, all_preds

    def forward(
        self,
        z_or_tokens: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-mode forward pass supporting both (B, 64, token_dim) spatial tokens and legacy (B, 256) vectors.
        """
        if z_or_tokens.dim() == 3:
            if z_or_tokens.size(1) != self.num_tokens or z_or_tokens.size(2) != self.token_dim:
                raise ValueError(
                    f"Expected spatial tokens of shape (B, {self.num_tokens}, {self.token_dim}), got {z_or_tokens.shape}"
                )
            # Spatial token input (B, 64, token_dim)
            return self.forward_tokens(z_or_tokens, a, h=h)
        elif z_or_tokens.dim() == 2:
            if z_or_tokens.size(1) != self.latent_dim:
                raise ValueError(
                    f"Expected latent state z of shape (B, {self.latent_dim}), got {z_or_tokens.shape}"
                )
            # 256-d vector input (B, 256) -> project to tokens, predict, pool back to 256-d
            B = z_or_tokens.size(0)
            tokens = z_or_tokens.view(B, 4, self.token_dim).repeat(1, 16, 1)
            mean_tokens, all_tokens = self.forward_tokens(tokens, a, h=h)
            
            mean_z = mean_tokens[:, :4, :].reshape(B, self.latent_dim)
            all_z = all_tokens[:, :, :4, :].reshape(self.num_heads, B, self.latent_dim)
            return mean_z, all_z
        else:
            raise ValueError(f"Expected 2D or 3D latent input, got shape {tuple(z_or_tokens.shape)}")

    def compute_masked_loss(
        self,
        all_pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked token prediction MSE loss.
        Loss is computed strictly on masked positions only:
        normalized per-sample, then averaged across the batch and ensemble heads.
        
        Args:
            all_pred_tokens: (num_heads, B, 64, token_dim) or (B, 64, token_dim).
            target_tokens: (B, 64, token_dim) true target tokens from TargetEncoder.
            mask: (B, 64) boolean tensor where True = masked position.
            
        Returns:
            L_tokens: Scalar tensor loss.
        """
        if all_pred_tokens.dim() == 3:
            all_pred_tokens = all_pred_tokens.unsqueeze(0)

        num_heads = all_pred_tokens.size(0)
        mask_float = mask.float()
        num_masked_per_sample = mask_float.sum(dim=-1).clamp(min=1.0)  # (B,)

        head_losses = []
        for k in range(num_heads):
            # Squared error per token: (B, 64)
            sq_diff = ((all_pred_tokens[k] - target_tokens) ** 2).sum(dim=-1)
            # Masked sum per sample
            masked_sq_diff = (sq_diff * mask_float).sum(dim=-1)  # (B,)
            sample_loss = masked_sq_diff / num_masked_per_sample  # (B,)
            head_losses.append(sample_loss.mean())

        return torch.stack(head_losses).mean()

    def compute_epistemic_disagreement(
        self,
        all_preds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute epistemic disagreement (variance across heads) on pooled masked positions.
        Consumes either (num_heads, B, 64, token_dim) token grids or (num_heads, B, latent_dim) vectors.
        
        Returns:
            epistemic_var: Non-negative tensor of shape (B,).
        """
        if all_preds.dim() == 4:
            # (num_heads, B, 64, token_dim)
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()  # (B, 64, 1)
                mask_cnt = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
                pooled = (all_preds * mask_f.unsqueeze(0)).sum(dim=2) / mask_cnt.unsqueeze(0)
            else:
                pooled = all_preds.mean(dim=2)  # (num_heads, B, token_dim)

            mean_pred = pooled.mean(dim=0, keepdim=True)
            var = ((pooled - mean_pred) ** 2).sum(dim=-1).mean(dim=0)  # (B,)
            return var
        elif all_preds.dim() == 3:
            # (num_heads, B, latent_dim)
            mean_pred = all_preds.mean(dim=0, keepdim=True)
            return ((all_preds - mean_pred) ** 2).sum(dim=-1).mean(dim=0)
        else:
            raise ValueError(f"Unsupported shape for all_preds: {all_preds.shape}")

    def predict_next(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method returning consensus mean prediction."""
        mean_pred, _ = self.forward(z, a, h=h)
        return mean_pred

    def rollout(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        batch_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive multi-step latent rollout for CEM / MPC planning.
        """
        if actions.dim() == 1:
            action_seq = actions.unsqueeze(0)
            H = 1
        elif actions.dim() == 2:
            if actions.size(0) == actions.size(1) == z_0.size(0):
                action_seq = actions.transpose(0, 1) if batch_first else actions
                H = action_seq.size(0)
            elif actions.size(1) == z_0.size(0):
                action_seq = actions
                H = actions.size(0)
            elif actions.size(0) == z_0.size(0):
                action_seq = actions.transpose(0, 1)
                H = action_seq.size(0)
            else:
                raise ValueError(f"Unsupported action sequence shape {actions.shape}")
        else:
            raise ValueError(f"Unsupported action sequence shape {actions.shape}")

        traj_mean = []
        traj_all = []
        curr_z = z_0
        curr_h = h_0

        for t in range(H):
            m_pred, a_pred = self.forward(curr_z, action_seq[t], h=curr_h)
            traj_mean.append(m_pred)
            traj_all.append(a_pred)
            curr_z = m_pred

        stacked_mean = torch.stack(traj_mean, dim=0)  # (H, B, ...)
        stacked_all = torch.stack(traj_all, dim=1)   # (num_heads, H, B, ...)
        return stacked_mean, stacked_all


# -----------------------------------------------------------------------------
# Legacy Baseline 4-Head Residual MLP Predictor (Preserved for backward tests)
# -----------------------------------------------------------------------------

class ResidualMLPPredictorHead(nn.Module):
    """
    Individual Residual MLP Head predicting latent transition deltas.
    Architecture: Linear(in_dim, hidden_dim) -> LN -> GELU -> Linear(hidden_dim, hidden_dim) -> LN -> GELU -> Linear.
    """

    def __init__(
        self,
        in_dim: int = 288,
        hidden_dim: int = 192,
        out_dim: int = 256,
        head_id: int = 0,
    ):
        super().__init__()
        self.head_id = head_id
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        return self.fc3(h)


class ResidualMLPPredictor(nn.Module):
    """
    4-Head Residual MLP Predictor Ensemble for Action-Conditioned JEPA World Model.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        action_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 192,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.action_embedding = ActionEmbedding(num_actions=num_actions, action_dim=action_dim)
        in_dim = latent_dim + action_dim
        self.num_tokens = 64
        self.token_dim = 64

        self.heads = nn.ModuleList([
            ResidualMLPPredictorHead(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                head_id=k,
            )
            for k in range(num_heads)
        ])

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = tokens.size(0)
        z = tokens[:, :4, :].reshape(B, self.latent_dim)
        mean_z, all_z = self.forward(z, a, h=h)
        mean_tokens = mean_z.view(B, 4, -1).repeat(1, 16, 1)
        all_tokens = all_z.view(self.num_heads, B, 4, -1).repeat(1, 1, 16, 1)
        return mean_tokens, all_tokens

    def compute_masked_loss(
        self,
        all_pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if all_pred_tokens.dim() == 4:
            all_pred_tokens = all_pred_tokens.mean(dim=0)
        mask_f = mask.unsqueeze(-1).float()
        diff = ((all_pred_tokens - target_tokens) ** 2) * mask_f
        return diff.sum() / mask_f.sum().clamp(min=1.0)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 3:
            if z.size(1) == 64 and z.size(2) == 64:
                return self.forward_tokens(z, a, h=h)
            raise ValueError(f"Expected 2D latent state of shape (B, {self.latent_dim}), got {z.shape}")
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent state z of shape (B, {self.latent_dim}), got {z.shape}"
            )

        a_emb = self.action_embedding(a)
        x = torch.cat([z, a_emb], dim=-1)
        deltas = torch.stack([head(x) for head in self.heads], dim=0)
        all_preds = z.unsqueeze(0) + deltas
        mean_pred = all_preds.mean(dim=0)
        return mean_pred, all_preds

    def compute_epistemic_disagreement(self, all_preds: torch.Tensor) -> torch.Tensor:
        if all_preds.dim() == 4:
            pooled = all_preds.mean(dim=2)  # (num_heads, B, token_dim)
            mean = pooled.mean(dim=0, keepdim=True)
            return ((pooled - mean) ** 2).sum(dim=-1).mean(dim=0)
        mean = all_preds.mean(dim=0, keepdim=True)
        return ((all_preds - mean) ** 2).sum(dim=-1).mean(dim=0)

    def predict_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mean_pred, _ = self.forward(z, a)
        return mean_pred

    def rollout(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        batch_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if actions.dim() == 1:
            action_seq = actions.unsqueeze(0)
            H = 1
        elif actions.dim() == 2:
            if actions.size(0) == actions.size(1) == z_0.size(0):
                action_seq = actions.transpose(0, 1) if batch_first else actions
                H = action_seq.size(0)
            elif actions.size(1) == z_0.size(0):
                action_seq = actions
                H = actions.size(0)
            elif actions.size(0) == z_0.size(0):
                action_seq = actions.transpose(0, 1)
                H = action_seq.size(0)
            else:
                raise ValueError(f"Unsupported action sequence shape {actions.shape}")
        else:
            raise ValueError(f"Unsupported action sequence shape {actions.shape}")

        traj_mean = []
        traj_all = []
        curr_z = z_0

        for t in range(H):
            m_pred, a_pred = self.forward(curr_z, action_seq[t])
            traj_mean.append(m_pred)
            traj_all.append(a_pred)
            curr_z = m_pred

        stacked_mean = torch.stack(traj_mean, dim=0)
        stacked_all = torch.stack(traj_all, dim=1)
        return stacked_mean, stacked_all


# Interface alias matching PROJECT.md
LatentPredictor = ResidualMLPPredictor
