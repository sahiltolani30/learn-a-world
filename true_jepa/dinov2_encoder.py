"""Frozen Meta AI DINOv2 ViT-S/14 encoder with trainable spatial adapter.

This module implements Candidate B for ORBIT Gate 3 representation selection.
The backbone is frozen and kept in eval mode. Observations are resized to 224x224
via nearest-neighbor interpolation to preserve Crafter's full pixel-art FOV without
cropping. Patch tokens (16x16=256) are mapped via spatial pooling and projection
to the common 8x8x192 representation.
"""
from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from true_jepa.config import JEPAConfig

if TYPE_CHECKING:
    from true_jepa.model import LatentFrame


class DINOv2SpatialAdapter(nn.Module):
    """Maps 16x16 DINOv2 patch tokens (384-dim) to 8x8 ORBIT tokens (192-dim)."""

    def __init__(self, in_dim: int = 384, token_dim: int = 192) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.token_dim = token_dim
        self.spatial_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Linear(in_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.pool_norm = nn.LayerNorm(token_dim)

    def forward(self, patch_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """Maps [B, 256, 384] to tokens [B, 64, 192] and pooled [B, 192]."""
        batch_size = patch_tokens.shape[0]
        # Reshape [B, 256, 384] -> [B, 384, 16, 16]
        grid = patch_tokens.permute(0, 2, 1).reshape(batch_size, self.in_dim, 16, 16)
        # Spatial average pooling 2x2: [B, 384, 16, 16] -> [B, 384, 8, 8]
        pooled_grid = self.spatial_pool(grid)
        # Reshape to token sequence: [B, 64, 384]
        tokens = pooled_grid.flatten(2).transpose(1, 2)
        # Linear projection and normalization: [B, 64, 192]
        tokens = self.norm(self.proj(tokens))
        pooled = self.pool_norm(tokens.mean(dim=1))
        return tokens, pooled


class DINOv2VisualEncoder(nn.Module):
    """Frozen ViT-S/14 backbone + trainable spatial adapter."""

    def __init__(
        self,
        cfg: JEPAConfig,
        backbone: Optional[nn.Module] = None,
        hub_repo: str = "facebookresearch/dinov2",
        model_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_dim = cfg.token_dim

        if backbone is not None:
            self.backbone = backbone
        else:
            cache_dirs = [
                os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"),
                "/root/.cache/torch/hub/facebookresearch_dinov2_main",
                "/content/.cache/torch/hub/facebookresearch_dinov2_main",
            ]
            local_cache = next((d for d in cache_dirs if os.path.exists(d)), None)

            if local_cache is None:
                target_cache = cache_dirs[0]
                import subprocess
                print(f"[DINOv2] Pre-cloning dinov2 to {target_cache} to bypass GitHub 504 Gateway Timeouts...")
                os.makedirs(os.path.dirname(target_cache), exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov2.git", target_cache],
                    check=False,
                )
                if os.path.exists(target_cache):
                    local_cache = target_cache

            if local_cache is not None and os.path.exists(local_cache):
                self.backbone = torch.hub.load(local_cache, model_name, source="local", pretrained=True)
            else:
                self.backbone = torch.hub.load(hub_repo, model_name, pretrained=True)

        # Freeze backbone parameters permanently
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # ImageNet normalization constants as buffers
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.adapter = DINOv2SpatialAdapter(in_dim=384, token_dim=cfg.token_dim)

    def train(self, mode: bool = True) -> "DINOv2VisualEncoder":
        """Keep the backbone in eval mode permanently even when adapter trains."""
        super().train(mode)
        self.backbone.eval()
        return self

    def preprocess(self, obs_uint8: Tensor) -> Tensor:
        """Full-frame nearest-neighbor resize to 224x224 and ImageNet normalization."""
        if obs_uint8.ndim != 4:
            raise ValueError(f"expected [B,H,W,C], got {tuple(obs_uint8.shape)}")
        # Convert [B, H, W, C] -> [B, C, H, W]
        x = obs_uint8.permute(0, 3, 1, 2).contiguous()
        # Nearest-neighbor resize from 64x64 to 224x224
        x = F.interpolate(x, size=(224, 224), mode="nearest")
        # Normalize to [0.0, 1.0] float32
        x = x.to(torch.float32) / 255.0
        # ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    def forward(self, obs_or_tokens: Tensor) -> LatentFrame:
        """Processes observations through frozen backbone and trainable adapter.
        
        Accepts either raw uint8 observations [B, 64, 64, 3] or pre-extracted
        frozen patch tokens [B, 256, 384] for bitwise-equivalent accelerated training.
        """
        if obs_or_tokens.ndim == 4:
            x_norm = self.preprocess(obs_or_tokens)
            with torch.no_grad():
                features = self.backbone.forward_features(x_norm)
                patch_tokens = features["x_norm_patchtokens"]
                if features.get("x_norm_regtokens") is not None:
                    if features["x_norm_regtokens"].shape[1] > 0:
                        raise RuntimeError("Unexpected register tokens found in DINOv2 model")
        elif obs_or_tokens.ndim == 3 and obs_or_tokens.shape[-1] == 384:
            patch_tokens = obs_or_tokens
        else:
            raise ValueError(f"Unexpected input shape to DINOv2VisualEncoder: {tuple(obs_or_tokens.shape)}")

        # Adapter forward pass (trainable)
        tokens, pooled = self.adapter(patch_tokens)
        from true_jepa.model import LatentFrame
        return LatentFrame(tokens=tokens, pooled=pooled)

    def backbone_sha256(self) -> str:
        """Computes SHA-256 hash across all frozen backbone parameters."""
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.backbone.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()
