"""Action-conditioned latent predictor member using FiLM and Transformer blocks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictorMember(nn.Module):
    """Single predictor member forecasting next spatial latent tokens from current latents, context, and action."""

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 256,
        action_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Conditioning MLP: [context, action] -> scale (gamma) and shift (beta)
        cond_in = context_dim + action_dim  # 320
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )

        # 2 Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Retained for strict state_dict compatibility with historical checkpoints;
        # not applied in forward() to ensure predictions match unnormalized target representation space.
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        current_tokens: torch.Tensor,
        context_state: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_tokens: [B, 64, 128]
            context_state: [B, 256]
            action_emb: [B, 64]
        Returns:
            predicted_tokens: [B, 64, 128]
        """
        # Condition projection
        cond_input = torch.cat([context_state, action_emb], dim=-1)  # [B, 320]
        film_params = self.cond_mlp(cond_input)                       # [B, 256]
        gamma, beta = film_params.chunk(2, dim=-1)                   # [B, 128], [B, 128]
        gamma = gamma.unsqueeze(1)                                    # [B, 1, 128]
        beta = beta.unsqueeze(1)                                      # [B, 1, 128]

        # Feature-wise modulation (FiLM)
        modulated = (1.0 + gamma) * current_tokens + beta            # [B, 64, 128]

        # Residual transformer prediction: z_hat = z_t + delta
        delta = self.transformer(modulated)                          # [B, 64, 128]
        return current_tokens + delta
