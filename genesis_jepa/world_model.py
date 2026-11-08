"""
genesis_jepa.world_model — Genesis JEPA World Model Architecture
================================================================
Implements Gate 1 World Model with:
  1. Contrast-Normalized Vision Transformer (online & EMA target encoders)
  2. Inverted Competitive Slot Attention (S=12 slots)
  3. Spatially Gated Dynamic Jump Predictor (SimNorm bounded updates)
  4. Normalized L1 & Soft-Capped Anti-Persistence Losses
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn

from genesis_jepa.config import GenesisConfig
from genesis_jepa.modules.dynamic_jump import SpatiallyGatedJumpPredictor
from genesis_jepa.modules.losses import WorldModelGate1Loss, normalized_l1_loss
from genesis_jepa.modules.visual_encoder import GenesisVisualEncoder


class GenesisJEPA(nn.Module):
    """
    True Joint Embedding Predictive Architecture (JEPA) for open-world environments.
    """

    def __init__(self, config: GenesisConfig | None = None) -> None:
        super().__init__()
        self.config = config or GenesisConfig()

        # 1. Online Visual Encoder (trained via backprop)
        self.encoder = GenesisVisualEncoder(
            in_channels=self.config.obs_shape[-1],
            patch_size=self.config.patch_size,
            token_dim=self.config.token_dim,
            num_patches=self.config.num_patches,
            num_slots=self.config.num_slots,
            contrast_norm=self.config.contrast_norm,
        )

        # 2. Target Visual Encoder (EMA updated, stop-gradient)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. Spatially Gated Dynamic Jump Operator
        self.predictor = SpatiallyGatedJumpPredictor(
            token_dim=self.config.token_dim,
            action_dim=self.config.num_actions,
            action_emb_dim=self.config.action_emb_dim,
            num_patches=self.config.num_patches,
            simnorm_heads=self.config.simnorm_heads,
            simnorm_head_dim=self.config.simnorm_head_dim,
            gate_bias_init=self.config.jump_gate_bias_init,
        )

        # 4. Gate 1 Loss Function
        self.loss_fn = WorldModelGate1Loss(
            floor=self.config.anti_persistence_floor,
            gain=self.config.anti_persistence_gain,
        )

    def encode(self, obs: torch.Tensor, use_target: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes observation into (patch_tokens, slot_tokens).
        """
        enc = self.target_encoder if use_target else self.encoder
        return enc(obs)

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts next state latent tokens from current tokens and action.
        """
        return self.predictor(z_t, action)

    def rollout(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Executes multi-step autoregressive rollout in latent space.
        Args:
            z_init: Initial tokens of shape (B, K, D)
            actions: Sequence of actions of shape (B, H)
        Returns:
            trajectory: List of H predicted token tensors [z_1, z_2, ..., z_H]
        """
        B, H = actions.shape
        z_curr = z_init
        trajectory = []
        for h in range(H):
            act = actions[:, h]
            z_curr, _, _ = self.predictor(z_curr, act)
            trajectory.append(z_curr)
        return trajectory

    @torch.no_grad()
    def update_target_ema(self, decay: float | None = None) -> None:
        """
        Updates the target encoder parameters using exponential moving average.
        """
        if decay is None:
            decay = self.config.target_ema_decay

        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    def compute_step_loss(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes 1-step JEPA loss with target encoder stop-gradient.
        """
        # Online encoder embeds obs_t
        online_z_t, _ = self.encoder(obs_t)

        # Predict next tokens with dynamic jump operator
        pred_z_next, gate, delta = self.predictor(online_z_t, action_t)

        # Target encoder embeds obs_t and obs_next with stop-gradient
        with torch.no_grad():
            target_z_t, _ = self.target_encoder(obs_t)
            target_z_next, _ = self.target_encoder(obs_next)

        # Compute Gate 1 loss with soft-capped anti-persistence weighting
        loss, metrics = self.loss_fn(pred_z_next, target_z_next, target_z_t)

        metrics["mean_gate_activation"] = float(gate.mean().item())
        metrics["mean_displacement_norm"] = float(torch.norm(delta, dim=-1).mean().item())

        return loss, metrics

    def step(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Alias for compute_step_loss."""
        return self.compute_step_loss(obs_t, action_t, obs_next)
