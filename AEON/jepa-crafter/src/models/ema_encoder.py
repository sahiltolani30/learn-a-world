"""EMA target visual encoder."""
import copy
from typing import Tuple
import torch
import torch.nn as nn
from src.models.visual_encoder import VisualEncoder


class EmaTargetEncoder(nn.Module):
    """Target visual encoder running under stop-gradient, updated via exponential moving average."""

    def __init__(self, online_encoder: VisualEncoder):
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute stop-gradient target spatial latents and pooled vector."""
        self.encoder.eval()
        tokens, pooled = self.encoder(x)
        return tokens.detach(), pooled.detach()

    @torch.no_grad()
    def update(self, online_encoder: VisualEncoder, momentum: float) -> None:
        """Update target weights: theta_target = momentum * theta_target + (1 - momentum) * theta_online."""
        for target_p, online_p in zip(self.encoder.parameters(), online_encoder.parameters()):
            target_p.data.mul_(momentum).add_(online_p.data, alpha=(1.0 - momentum))
