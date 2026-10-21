r"""
Action-Conditioned Latent Predictor for Learn-a-World.
Predicts next latent representation \hat{z}_{t+1} from current latent z_t and action a_t.
Features:
- Action embedding table for discrete actions.
- Residual skip connection (predicts \Delta z rather than absolute position).
- LayerNorm and GELU activations for training stability.
"""

from typing import Tuple
import torch
import torch.nn as nn

class LatentPredictor(nn.Module):
    r"""
    Predicts next latent state \hat{z}_{t+1} conditioned on action a_t.
    Uses residual formulation: \hat{z}_{t+1} = z_t + \Delta z(z_t, a_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 4,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            z: Current latent vector [B, latent_dim]
            actions: Action indices [B] or [B, 1]
        Returns:
            next_z: Predicted next latent vector [B, latent_dim]
        """
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        a_emb = self.action_embedding(actions)
        za = torch.cat([z, a_emb], dim=-1)
        delta_z = self.net(za)
        next_z = z + delta_z
        return next_z
