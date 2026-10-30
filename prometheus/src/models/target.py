"""
src/models/target.py

Target Encoder wrapper for the JEPA World Model.
Maintains an Exponential Moving Average (EMA) clone of the online ObservationEncoder
with stop-gradient (torch.no_grad()) semantics to provide non-oscillating target latents.
"""

from typing import Optional
import copy
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """
    EMA Target Encoder with stop-gradient enforcement.
    
    Attributes:
        online_encoder: Reference to the trainable online ObservationEncoder.
        tau: Polyak EMA momentum coefficient in [0.0, 1.0] (default 0.99).
        target_encoder: Detached deep copy of the online encoder.
    """

    def __init__(self, encoder: nn.Module, tau: float = 0.99):
        super().__init__()
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")

        # Store reference to online encoder via object.__setattr__ to prevent PyTorch nn.Module
        # from registering online_encoder as a child submodule in self._modules.
        # This guarantees self.parameters() yields only detached target weights and prevents
        # self.train() or self.eval() from mutating online_encoder.training mode.
        object.__setattr__(self, "online_encoder", encoder)
        self.tau = float(tau)

        # Create detached copy of the encoder
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Target network operates permanently in eval mode
        self.target_encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass strictly under torch.no_grad().
        
        Args:
            x: Raw pixel observation tensor (B, C, H, W) or (C, H, W).
        Returns:
            z_target: Detached target latent tensor (B, latent_dim) or (latent_dim,).
        """
        with torch.no_grad():
            z = self.target_encoder(x)
            return z.detach()

    def update(self, tau: Optional[float] = None) -> None:
        """
        Polyak EMA update: theta_target = tau * theta_target + (1 - tau) * theta_online.
        
        Args:
            tau: Optional override for momentum coefficient. If None, uses self.tau.
        """
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            # Update parameters
            for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)

            # Update non-trainable buffers (e.g. running stats)
            for b_t, b_o in zip(self.target_encoder.buffers(), self.online_encoder.buffers()):
                if b_t.is_floating_point():
                    b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)
                else:
                    b_t.data.copy_(b_o.data)

    def hard_update(self) -> None:
        """
        Instant hard synchronization: theta_target = theta_online.
        Used at model initialization and checkpoint restoration.
        """
        with torch.no_grad():
            for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                p_t.data.copy_(p_o.data)
            for b_t, b_o in zip(self.target_encoder.buffers(), self.online_encoder.buffers()):
                b_t.data.copy_(b_o.data)

    def train(self, mode: bool = True):
        """
        Overrides train mode to ensure target network permanently stays in eval mode.
        """
        super().train(False)
        self.target_encoder.eval()
        return self
