"""
src/curiosity/rnd.py

Latent Random Network Distillation (Latent RND) for JEPA World Model.

Fulfills requirements:
- R2: Tabula rasa lifetime novelty estimation in compact JEPA latent space.
- Provides auxiliary novelty scores via prediction error of a trainable predictor
  distilling a fixed, randomly initialized target network.
- Operating directly in latent space z avoids pixel-level noise artifacts.
"""

from typing import Optional, Iterator, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.curiosity.ensemble import RunningMeanStd


class LatentRND(nn.Module):
    """
    Random Network Distillation in JEPA Latent Space.
    
    Consists of:
    1. Fixed Target Network T_rand: Randomly initialized and permanently frozen.
    2. Trainable Predictor Network P_rnd: Trained via MSE to match target output on visited latents.
    
    Novelty Score:
        r_rnd(z) = (1 / d_rnd) * || P_rnd(z) - T_rand(z) ||_2^2
        
    As states are visited repeatedly across episodes, P_rnd learns to match T_rand,
    reducing prediction error (lifetime novelty decay). Unvisited or rare states
    produce high distillation error.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        rnd_dim: int = 128,
        hidden_dim: int = 256,
        normalize_by_dim: bool = True,
        use_running_stats: bool = True,
        clip_novelty: Optional[float] = 10.0,
    ):
        """
        Initialize Latent RND module.
        
        Args:
            latent_dim: Dimension of input latent state z (default 256).
            rnd_dim: Output projection dimension for target and predictor (default 128).
            hidden_dim: Hidden layer width for target and predictor networks.
            normalize_by_dim: If True, divides squared L2 norm by rnd_dim.
            use_running_stats: If True, tracks RunningMeanStd for stationary scaling.
            clip_novelty: Maximum bound for returned novelty score.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.rnd_dim = rnd_dim
        self.hidden_dim = hidden_dim
        self.normalize_by_dim = normalize_by_dim
        self.use_running_stats = use_running_stats
        self.clip_novelty = clip_novelty

        # 1. Fixed Random Target Network (Permanently Frozen)
        self.target_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, rnd_dim),
        )
        self._init_target_weights()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()

        # 2. Trainable Predictor Network
        self.predictor_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rnd_dim),
        )
        self._init_predictor_weights()

        # 3. Optional Running Normalizer
        if use_running_stats:
            self.running_stats = RunningMeanStd(shape=(), clip_max=clip_novelty)
        else:
            self.running_stats = None

    def _init_target_weights(self) -> None:
        """Initialize target network with orthogonal weights."""
        for m in self.target_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_predictor_weights(self) -> None:
        """Initialize predictor network with standard Xavier uniform weights."""
        for m in self.predictor_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """Return iterator over trainable predictor parameters only."""
        return self.predictor_net.parameters()

    def train(self, mode: bool = True):
        """Override train mode to keep target network permanently in eval mode."""
        super().train(mode)
        self.target_net.eval()
        return self

    def compute_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared distillation error for training the predictor network.
        
        Args:
            z: Latent state tensor of shape (B, latent_dim).
            
        Returns:
            Scalar MSE loss tensor with gradient flow to predictor_net.
        """
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent state z of shape (B, {self.latent_dim}), got {z.shape}"
            )

        with torch.no_grad():
            target_out = self.target_net(z.detach())

        pred_out = self.predictor_net(z.detach())
        loss = F.mse_loss(pred_out, target_out)
        return loss

    def compute_novelty(
        self,
        z: torch.Tensor,
        update_stats: bool = False,
    ) -> torch.Tensor:
        """
        Compute latent RND novelty score for a batch of latent states.
        
        Args:
            z: Latent state tensor of shape (B, latent_dim) or (latent_dim,).
            update_stats: If True and training, updates running normalization stats.
            
        Returns:
            novelty: Non-negative tensor of shape (B,) or scalar.
        """
        was_single = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            was_single = True
        elif z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent state z of shape (B, {self.latent_dim}), got {z.shape}"
            )

        with torch.no_grad():
            target_out = self.target_net(z)
            pred_out = self.predictor_net(z)

            diff_sq = (pred_out - target_out) ** 2  # (B, rnd_dim)

            if self.normalize_by_dim:
                novelty = diff_sq.mean(dim=-1)  # (B,)
            else:
                novelty = diff_sq.sum(dim=-1)  # (B,)

            novelty = torch.clamp(novelty, min=0.0)

            if self.running_stats is not None:
                if update_stats and self.training:
                    self.running_stats.update(novelty)
                novelty = self.running_stats.normalize(novelty, center=False, clip=self.clip_novelty)
                novelty = torch.clamp(novelty, min=0.0)

            if self.clip_novelty is not None:
                novelty = torch.clamp(novelty, max=self.clip_novelty)

        if was_single:
            return novelty.squeeze(0)
        return novelty

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward alias returning novelty score."""
        return self.compute_novelty(z, update_stats=self.training)

    def update_predictor(
        self,
        optimizer: torch.optim.Optimizer,
        z: torch.Tensor,
    ) -> float:
        """
        Perform one optimization step on the predictor network.
        
        Args:
            optimizer: Optimizer configured with self.predictor_parameters().
            z: Batch of latent representations.
            
        Returns:
            Scalar distillation loss value.
        """
        optimizer.zero_grad()
        loss = self.compute_loss(z)
        loss.backward()
        optimizer.step()
        return loss.item()
