"""
sapna/dynamics.py

Dynamics model components for SAPNA:
- ActionEmbedding
- TemporalCore (GRUCell 288 -> 256)
- SpatialPredictor (SpatialTransformerPredictor with forward_tokens_clean per §2.5 item 5)
- RewardHead (symlog two-hot 255 bins)
- DoneHead (Bernoulli logits)
"""

from typing import Optional, Tuple, Union
import math
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
    Learnable discrete action embedding module (dim 32).
    """

    def __init__(self, num_actions: int = 17, action_dim: int = 32):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.embedding = nn.Embedding(num_actions, action_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 2 and a.size(-1) == 1:
            a = a.squeeze(-1)

        if a.dim() == 1 or (a.dim() == 0):
            if a.dim() == 0:
                a = a.unsqueeze(0)
            if a.dtype not in (torch.int64, torch.int32):
                a = a.long()
            if (a < 0).any() or (a >= self.num_actions).any():
                raise IndexError(
                    f"Action index out of range [0, {self.num_actions - 1}]. "
                    f"Min: {a.min().item()}, Max: {a.max().item()}"
                )
            return self.embedding(a)
        elif a.dim() == 2 and a.size(-1) == self.num_actions:
            return torch.matmul(a.float(), self.embedding.weight)
        else:
            raise ValueError(
                f"Unsupported action tensor shape {a.shape} for num_actions={self.num_actions}."
            )


class TemporalCore(nn.Module):
    """
    Temporal memory core: nn.GRUCell(input_size=288, hidden_size=256).
    h_{t+1} = GRUCell(concat[z_t (256), a_emb_t (32)], h_t)
    """

    def __init__(self, input_size: int = 288, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(
        self,
        z_or_concat: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        a_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if a_emb is not None:
            inp = torch.cat([z_or_concat, a_emb], dim=-1)
        else:
            inp = z_or_concat

        if h is None:
            h = torch.zeros(inp.size(0), self.hidden_size, device=inp.device, dtype=inp.dtype)

        return self.cell(inp, h)


class SpatialPredictor(nn.Module):
    """
    Action-Conditioned Spatial Transformer Predictor Ensemble for SAPNA.
    Copied from SpatialTransformerPredictor with forward_tokens_clean per §2.5 item 5.
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

        # Action and Memory projections
        self.action_embedding = ActionEmbedding(num_actions=num_actions, action_dim=action_dim)
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.a_proj = nn.Linear(action_dim, token_dim)
        self.film = nn.Linear(action_dim, 2 * token_dim)
        self.default_mem = nn.Parameter(torch.randn(1, mem_hidden) * 0.05)
        self.mem_proj = nn.Linear(mem_hidden, token_dim)

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 128, token_dim) * 0.02)

        # 2-layer Pre-LN Transformer Encoder
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

        # 4 Ensemble Output Heads
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
        Training forward pass: builds predictions on the post-broadcast grid.
        Returns:
            mean_pred: (B, 64, token_dim)
            all_preds: (4, B, 64, token_dim)
        """
        B = tokens.size(0)

        a_emb = self.action_embedding(a)  # (B, action_dim)
        a_broadcast = self.a_proj(a_emb)  # (B, token_dim)
        post_tokens = tokens + a_broadcast.unsqueeze(1)
        film = self.film(a_emb)
        gamma_f, beta_f = film.chunk(2, dim=-1)
        post_tokens = post_tokens * (1.0 + gamma_f).unsqueeze(1) + beta_f.unsqueeze(1)
        a_tok = self.action_proj(a_emb).unsqueeze(1)

        if h is None:
            h = self.default_mem.expand(B, -1)
        m_tok = self.mem_proj(h).unsqueeze(1)

        seq = torch.cat([post_tokens, a_tok, m_tok], dim=1)
        seq_len = seq.size(1)
        seq = seq + self.pos_embed[:, :seq_len, :]

        out_seq = self.transformer(seq)
        spatial_out = out_seq[:, :self.num_tokens, :]

        all_preds = torch.stack([
            post_tokens + head(spatial_out)
            for head in self.heads
        ], dim=0)

        mean_pred = all_preds.mean(dim=0)
        return mean_pred, all_preds

    def forward_tokens_clean(
        self,
        tokens: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Clean imagination forward pass per §2.5 item 5.
        Residual predictions are built on the PRE-broadcast grid (tokens_input)
        to prevent linear accumulation of action broadcast residue in closed-loop rollouts.
        """
        B = tokens.size(0)
        tokens_input = tokens  # Pre-broadcast grid

        a_emb = self.action_embedding(a)
        a_broadcast = self.a_proj(a_emb)
        post_tokens = tokens + a_broadcast.unsqueeze(1)
        film = self.film(a_emb)
        gamma_f, beta_f = film.chunk(2, dim=-1)
        post_tokens = post_tokens * (1.0 + gamma_f).unsqueeze(1) + beta_f.unsqueeze(1)
        a_tok = self.action_proj(a_emb).unsqueeze(1)

        if h is None:
            h = self.default_mem.expand(B, -1)
        m_tok = self.mem_proj(h).unsqueeze(1)

        seq = torch.cat([post_tokens, a_tok, m_tok], dim=1)
        seq_len = seq.size(1)
        seq = seq + self.pos_embed[:, :seq_len, :]

        out_seq = self.transformer(seq)
        spatial_out = out_seq[:, :self.num_tokens, :]

        # Built on PRE-broadcast grid: pred = tokens_input + head(spatial_out)
        all_preds = torch.stack([
            tokens_input + head(spatial_out)
            for head in self.heads
        ], dim=0)

        mean_pred = all_preds.mean(dim=0)
        return mean_pred, all_preds

    def forward(
        self,
        z_or_tokens: torch.Tensor,
        a: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_or_tokens.dim() == 3:
            return self.forward_tokens(z_or_tokens, a, h=h)
        elif z_or_tokens.dim() == 2:
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
        position_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked token prediction MSE loss.
        """
        if all_pred_tokens.dim() == 3:
            all_pred_tokens = all_pred_tokens.unsqueeze(0)

        num_heads = all_pred_tokens.size(0)
        mask_float = mask.float()

        if position_weights is not None:
            eff_mask = mask_float * position_weights
            norm_per_sample = eff_mask.sum(dim=-1).clamp(min=1e-6)
        else:
            eff_mask = mask_float
            norm_per_sample = mask_float.sum(dim=-1).clamp(min=1.0)

        head_losses = []
        for k in range(num_heads):
            sq_diff = ((all_pred_tokens[k] - target_tokens) ** 2).sum(dim=-1)
            masked_sq_diff = (sq_diff * eff_mask).sum(dim=-1)
            sample_loss = masked_sq_diff / norm_per_sample
            head_losses.append(sample_loss.mean())

        return torch.stack(head_losses).mean()

    def compute_epistemic_disagreement(
        self,
        all_preds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dyn_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute epistemic disagreement across heads.
        """
        if all_preds.dim() == 4:
            if dyn_w is not None:
                w = dyn_w.unsqueeze(0).unsqueeze(-1)
                pooled = (all_preds * w).sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)
            elif mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                mask_cnt = mask_f.sum(dim=1).clamp(min=1.0)
                pooled = (all_preds * mask_f.unsqueeze(0)).sum(dim=2) / mask_cnt.unsqueeze(0)
            else:
                pooled = all_preds.mean(dim=2)

            mean_pred = pooled.mean(dim=0, keepdim=True)
            var = ((pooled - mean_pred) ** 2).sum(dim=-1).mean(dim=0)
            return var
        elif all_preds.dim() == 3:
            mean_pred = all_preds.mean(dim=0, keepdim=True)
            return ((all_preds - mean_pred) ** 2).sum(dim=-1).mean(dim=0)
        else:
            raise ValueError(f"Unsupported shape for all_preds: {all_preds.shape}")


# Alias
SpatialTransformerPredictor = SpatialPredictor


class RewardHead(nn.Module):
    """
    Reward head: Linear(512, 256) -> ELU -> Linear(256, 255).
    Input: ARRIVAL state concat[pool_tokens(mean_pred_t), h_{t+1}] (post-action).
    Predicts symlog two-hot distribution over 255 bins for r_{t+1}.
    """

    def __init__(self, in_features: int = 512, hidden_features: int = 256, num_bins: int = 255):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ELU(),
            nn.Linear(hidden_features, num_bins),
        )

    def forward(self, arr_state: torch.Tensor) -> torch.Tensor:
        return self.net(arr_state)


class DoneHead(nn.Module):
    """
    Done head: Linear(512, 256) -> ELU -> Linear(256, 1).
    Input: ARRIVAL state concat[pool_tokens(mean_pred_t), h_{t+1}].
    Predicts Bernoulli termination logits of arriving at that state.
    """

    def __init__(self, in_features: int = 512, hidden_features: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ELU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, arr_state: torch.Tensor) -> torch.Tensor:
        return self.net(arr_state)
