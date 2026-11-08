"""Core visual JEPA and action-conditioned latent world model.

The model predicts the EMA target encoder's representation of the next frame,
never its pixels.  The predictor is closed-loop: its predicted token map is fed
back into the next belief update during multi-step training and imagination.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from true_jepa.config import JEPAConfig
from true_jepa.dinov2_encoder import DINOv2VisualEncoder


@dataclass
class LatentFrame:
    """Dense and pooled representation of one observation."""

    tokens: Tensor
    pooled: Tensor


class VisualEncoder(nn.Module):
    """Small patch encoder suitable for a free T4 while retaining spatial tokens."""

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        height, width, channels = cfg.obs_shape
        if height % cfg.patch_size or width % cfg.patch_size:
            raise ValueError("obs_shape must be divisible by patch_size")
        self.patch_size = cfg.patch_size
        self.token_dim = cfg.token_dim
        token_count = (height // cfg.patch_size) * (width // cfg.patch_size)
        self.proj = nn.Conv2d(
            channels,
            cfg.token_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, token_count, cfg.token_dim))
        self.norm = nn.LayerNorm(cfg.token_dim)
        self.pool_norm = nn.LayerNorm(cfg.token_dim)

    def forward(self, obs_uint8: Tensor) -> LatentFrame:
        if obs_uint8.ndim != 4:
            raise ValueError(f"expected [B,H,W,C], got {tuple(obs_uint8.shape)}")
        x = obs_uint8.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        tokens = self.proj(x).flatten(2).transpose(1, 2) + self.position
        tokens = self.norm(tokens)
        pooled = self.pool_norm(tokens.mean(dim=1))
        return LatentFrame(tokens=tokens, pooled=pooled)


class BeliefCore(nn.Module):
    """Recurrent memory updated only from visual latents and public scalars."""

    def __init__(self, cfg: JEPAConfig, num_actions: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, cfg.token_dim // 2)
        self.reward_projection = nn.Sequential(
            nn.Linear(1, cfg.token_dim // 4), nn.SiLU(),
        )
        input_dim = cfg.token_dim + cfg.token_dim // 2 + cfg.token_dim // 4
        self.cell = nn.GRUCell(input_dim, cfg.belief_dim)

    def initial(self, batch: int, device: Union[str, torch.device]) -> Tensor:
        return torch.zeros(batch, self.cell.hidden_size, device=device)

    def forward(
        self,
        frame: LatentFrame,
        previous_action: Tensor,
        reward_prev: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        action = self.action_embedding(previous_action.long())
        reward = self.reward_projection(reward_prev.float().unsqueeze(-1))
        return self.cell(torch.cat([frame.pooled, action, reward], dim=-1), hidden)


class ActionConditionedPredictor(nn.Module):
    """Predict the next dense latent map from the current map, belief, and action."""

    def __init__(self, cfg: JEPAConfig, num_actions: int, residual: bool = False) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.action_embedding = nn.Embedding(num_actions, cfg.token_dim)
        self.belief_projection = nn.Linear(cfg.belief_dim, cfg.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.predictor_heads,
            dim_feedforward=cfg.predictor_mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.predictor_layers)
        self.output_norm = nn.LayerNorm(cfg.token_dim)
        if self.residual:
            self.delta_proj = nn.Linear(cfg.token_dim, cfg.token_dim)
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)

    def forward(self, frame: LatentFrame, action: Tensor, hidden: Tensor) -> LatentFrame:
        context = self.action_embedding(action.long()).unsqueeze(1)
        context = context + self.belief_projection(hidden).unsqueeze(1)
        transformed = self.transformer(frame.tokens + context)
        transformed = self.output_norm(transformed)
        if self.residual:
            delta = self.delta_proj(transformed)
            predicted_tokens = frame.tokens + delta
        else:
            predicted_tokens = transformed
        return LatentFrame(tokens=predicted_tokens, pooled=predicted_tokens.mean(dim=1))


class TransitionEnsemble(nn.Module):
    """Independent latent transition probes used only for epistemic exploration."""

    def __init__(self, cfg: JEPAConfig, num_actions: int) -> None:
        super().__init__()
        members = []
        input_dim = cfg.token_dim + cfg.belief_dim + cfg.token_dim // 2
        for _ in range(cfg.ensemble_size):
            members.append(
                nn.Sequential(
                    nn.Linear(input_dim, cfg.predictor_mlp_dim),
                    nn.SiLU(),
                    nn.Linear(cfg.predictor_mlp_dim, cfg.token_dim),
                )
            )
        self.members = nn.ModuleList(members)
        self.action_embedding = nn.Embedding(num_actions, cfg.token_dim // 2)

    def forward(self, frame: LatentFrame, action: Tensor, hidden: Tensor) -> Tensor:
        action_emb = self.action_embedding(action.long())
        source = torch.cat([frame.pooled, hidden, action_emb], dim=-1)
        return torch.stack([member(source) for member in self.members], dim=0)


class ScalarHeads(nn.Module):
    """Reward, continuation, value, and policy heads over learned state."""

    def __init__(self, cfg: JEPAConfig, num_actions: int) -> None:
        super().__init__()
        state_dim = cfg.token_dim + cfg.belief_dim
        self.reward = nn.Sequential(nn.Linear(state_dim + cfg.token_dim // 2, cfg.predictor_mlp_dim), nn.SiLU(), nn.Linear(cfg.predictor_mlp_dim, 1))
        self.continuation = nn.Sequential(nn.Linear(state_dim + cfg.token_dim // 2, cfg.predictor_mlp_dim), nn.SiLU(), nn.Linear(cfg.predictor_mlp_dim, 1))
        self.policy = nn.Sequential(nn.Linear(state_dim, cfg.predictor_mlp_dim), nn.SiLU(), nn.Linear(cfg.predictor_mlp_dim, num_actions))
        self.value = nn.Sequential(nn.Linear(state_dim, cfg.predictor_mlp_dim), nn.SiLU(), nn.Linear(cfg.predictor_mlp_dim, 1))
        self.action_embedding = nn.Embedding(num_actions, cfg.token_dim // 2)

    def state(self, frame: LatentFrame, hidden: Tensor) -> Tensor:
        return torch.cat([frame.pooled, hidden], dim=-1)

    def transition(self, frame: LatentFrame, hidden: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        state = self.state(frame, hidden)
        conditioned = torch.cat([state, self.action_embedding(action.long())], dim=-1)
        return self.reward(conditioned).squeeze(-1), self.continuation(conditioned).squeeze(-1)


class TrueJEPA(nn.Module):
    """Online encoder, EMA target encoder, recurrent predictor, and control heads."""

    checkpoint_format = "true_jepa_v1"

    def __init__(
        self,
        cfg: JEPAConfig,
        num_actions: int,
        encoder_type: str = "scratch",
    ) -> None:
        super().__init__()
        if num_actions < 2:
            raise ValueError("num_actions must be discovered from an environment")
        self.cfg = cfg
        self.num_actions = int(num_actions)
        self.encoder_type = str(encoder_type).lower()

        if self.encoder_type == "scratch":
            self.encoder = VisualEncoder(cfg)
            self.target_encoder = copy.deepcopy(self.encoder)
        elif self.encoder_type == "dinov2":
            self.encoder = DINOv2VisualEncoder(cfg)
            # Share frozen backbone to conserve memory, deepcopy the trainable adapter
            self.target_encoder = DINOv2VisualEncoder(cfg, backbone=self.encoder.backbone)
            self.target_encoder.adapter = copy.deepcopy(self.encoder.adapter)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'scratch' or 'dinov2'.")

        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.belief = BeliefCore(cfg, num_actions)
        self.predictor = ActionConditionedPredictor(
            cfg, num_actions, residual=getattr(cfg, "predictor_residual", False)
        )
        self.ensemble = TransitionEnsemble(cfg, num_actions)
        self.heads = ScalarHeads(cfg, num_actions)

    def train(self, mode: bool = True) -> "TrueJEPA":
        """Keep the EMA target and frozen backbone in evaluation mode even when the online model trains."""

        super().train(mode)
        self.target_encoder.eval()
        if hasattr(self.encoder, "backbone"):
            self.encoder.backbone.eval()
        return self

    def encode(self, obs_uint8: Tensor) -> LatentFrame:
        return self.encoder(obs_uint8)

    @torch.no_grad()
    def target_encode(self, obs_uint8: Tensor) -> LatentFrame:
        return self.target_encoder(obs_uint8)

    def initial_belief(self, batch: int, device: Union[str, torch.device]) -> Tensor:
        return self.belief.initial(batch, device)

    def belief_step(
        self,
        frame: LatentFrame,
        previous_action: Tensor,
        reward_prev: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        return self.belief(frame, previous_action, reward_prev, hidden)

    def predict(self, frame: LatentFrame, action: Tensor, hidden: Tensor) -> LatentFrame:
        return self.predictor(frame, action, hidden)

    def state(self, frame: LatentFrame, hidden: Tensor) -> Tensor:
        return self.heads.state(frame, hidden)

    def policy_logits(self, frame: LatentFrame, hidden: Tensor) -> Tensor:
        return self.heads.policy(self.state(frame, hidden))

    def transition_heads(
        self, frame: LatentFrame, hidden: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self.heads.transition(frame, hidden, action)

    def update_target(self, update: int) -> None:
        if self.cfg.ema_ramp_updates > 0:
            fraction = min(1.0, update / float(self.cfg.ema_ramp_updates))
        else:
            fraction = 1.0
        decay = self.cfg.ema_decay_start + fraction * (
            self.cfg.ema_decay_end - self.cfg.ema_decay_start
        )
        with torch.no_grad():
            online_params = dict(self.encoder.named_parameters())
            target_params = dict(self.target_encoder.named_parameters())
            if online_params.keys() != target_params.keys():
                raise RuntimeError("online and target encoder parameter schemas differ")
            for key, value in online_params.items():
                if value.requires_grad:
                    target_params[key].mul_(decay).add_(value, alpha=1.0 - decay)
            online_buffers = dict(self.encoder.named_buffers())
            target_buffers = dict(self.target_encoder.named_buffers())
            for key, value in online_buffers.items():
                target_buffers[key].copy_(value)

    def save_checkpoint(
        self, path: str, update: int, optimizer: Optional[Dict[str, object]] = None
    ) -> None:
        payload: Dict[str, object] = {
            "format": self.checkpoint_format,
            "update": int(update),
            "config": {key: value for key, value in self.cfg.__dict__.items()},
            "num_actions": self.num_actions,
            "encoder_type": self.encoder_type,
            "online": self.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Union[str, torch.device] = "cpu"
    ) -> Tuple["TrueJEPA", Dict[str, object]]:
        import gzip
        from pathlib import Path
        p = Path(path)
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                payload = torch.load(f, map_location="cpu", weights_only=False)
        elif not p.exists() and Path(str(path) + ".gz").exists():
            with gzip.open(str(path) + ".gz", "rb") as f:
                payload = torch.load(f, map_location="cpu", weights_only=False)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != cls.checkpoint_format:
            raise ValueError("checkpoint is not a true_jepa_v1 checkpoint")
        cfg = JEPAConfig(**payload["config"])
        encoder_type = payload.get("encoder_type", "scratch")
        model = cls(cfg, int(payload["num_actions"]), encoder_type=encoder_type)
        model.load_state_dict(payload["online"])
        model.target_encoder.load_state_dict(payload["target_encoder"])
        model.target_encoder.requires_grad_(False)
        model.target_encoder.eval()
        if hasattr(model.encoder, "backbone"):
            model.encoder.backbone.eval()
            model.target_encoder.backbone.eval()
        model.to(device)
        meta = {
            "update": int(payload["update"]),
            "optimizer": payload.get("optimizer"),
        }
        return model, meta
