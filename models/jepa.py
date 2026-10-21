"""
JEPA (Joint-Embedding Predictive Architecture) World Model.
Combines:
1. Online Encoder (trained via backprop)
2. EMA Target Encoder (momentum updated, stop-gradient target)
3. Action-Conditioned Latent Predictor
4. VICReg-Style Variance Regularization (mathematically prevents representation collapse)
"""

import copy
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ConvEncoder
from models.predictor import LatentPredictor

class JEPAWorldModel(nn.Module):
    """
    JEPA World Model:
    Learns world dynamics purely in latent space without pixel reconstruction.
    Uses VICReg variance regularization to guarantee healthy, non-collapsed representations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_actions: int = 4,
        action_emb_dim: int = 32,
        predictor_hidden_dim: int = 256,
        predictor_layers: int = 2,
        ema_decay: float = 0.99,
        var_weight: float = 1.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.var_weight = var_weight
        self.target_std = target_std

        # Online branch (trained via gradients)
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
        )

        # Target branch (stop-gradient, updated via EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        r"""
        Momentum update: \xi \leftarrow \tau \xi + (1 - \tau) \theta.
        """
        tau = decay if decay is not None else self.ema_decay
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_encoder.parameters(), self.encoder.parameters()
            ):
                target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

    def encode(self, x: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Encodes observation frame into latent representation z."""
        if use_target:
            with torch.no_grad():
                return self.target_encoder(x)
        return self.encoder(x)

    def compute_latent_variance(self, z: torch.Tensor) -> float:
        r"""
        Computes average per-dimension standard deviation across the batch:
        \sigma(Z) = \frac{1}{D} \sum_{j=1}^D \sqrt{\text{Var}(z_{:, j}) + \epsilon}
        A value approaching 0 indicates representation collapse.
        """
        if z.size(0) <= 1:
            return 0.0
        std = torch.sqrt(z.var(dim=0, unbiased=True) + 1e-4)
        return float(std.mean().item())

    def compute_variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        VICReg-style variance loss: penalizes per-dimension batch std below target_std.
        L_var = (1/D) * sum_j max(0, target_std - std_j(Z))
        Forces different states to have distinct representations, preventing collapse.
        """
        if z.size(0) <= 1:
            return torch.tensor(0.0, device=z.device)
        std = torch.sqrt(z.var(dim=0, unbiased=True) + 1e-4)
        return torch.mean(F.relu(self.target_std - std))

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes JEPA latent prediction loss, variance regularization, and diagnostics.
        """
        # 1. Encode current observation through online encoder
        z_t = self.encoder(obs)

        # 2. Predict next latent conditioned on action
        pred_z_next = self.predictor(z_t, actions)

        # 3. Compute target latent with target encoder (stop-gradient)
        with torch.no_grad():
            target_z_next = self.target_encoder(next_obs)

        # 4. Latent prediction loss (MSE)
        pred_loss = F.mse_loss(pred_z_next, target_z_next)

        # 5. Anti-collapse variance regularization on online latents
        var_loss = self.compute_variance_loss(z_t) + self.compute_variance_loss(pred_z_next)
        total_loss = pred_loss + self.var_weight * var_loss

        # 6. Measure representation health
        online_var = self.compute_latent_variance(z_t)
        target_var = self.compute_latent_variance(target_z_next)

        metrics = {
            "loss": float(total_loss.item()),
            "pred_loss": float(pred_loss.item()),
            "var_loss": float(var_loss.item()),
            "online_latent_variance": online_var,
            "target_latent_variance": target_var,
            "collapsed": bool(online_var < 0.1),
        }

        return total_loss, metrics
