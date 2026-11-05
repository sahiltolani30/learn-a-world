"""Zephyr world model: conv encoder, belief GRU, JEPA predictor, EMA target.

The online encoder, its EMA deepcopy, the belief GRU, and the latent predictor
live here. Multi-step free-running rollouts are composed by callers via
``predict`` + ``belief_step`` (the predicted z-hat feeds the next belief step).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from zephyr.config import NUM_ACTIONS, EncoderConfig, WorldModelConfig, ZephyrConfig

# Checkpoint layout tag written by, and required by, this code (matches
# ZephyrConfig.format_version). It identifies the serialization format itself
# and must only change when the layout changes.
_CHECKPOINT_FORMAT = "zephyr_v1"


class ZephyrEncoder(nn.Module):
    """Conv encoder: uint8 [N, 64, 64, 3] HWC -> float32 [N, latent] LayerNorm'd.

    Four stride-2 convs (channels from cfg), SiLU + GroupNorm(8) after each,
    then Linear -> latent -> LayerNorm. Input conversion (float/255, HWC->CHW)
    happens internally.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = cfg.in_channels
        for out_ch in cfg.channels:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            blocks.append(nn.SiLU())
            blocks.append(nn.GroupNorm(8, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        side = cfg.img_size // (2 ** len(cfg.channels))
        self.flat_dim = cfg.channels[-1] * side * side
        self.fc = nn.Linear(self.flat_dim, cfg.latent_dim)
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, obs_uint8: Tensor) -> Tensor:
        x = obs_uint8.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.out_norm(self.fc(x))


class ZephyrGRU(nn.Module):
    """Belief GRU over concat(z, action_embedding); batch or sequence inputs.

    forward accepts z [B, latent] with action [B], or z [B, T, latent] with
    action [B, T] (or [B], broadcast over T) and h [B, belief]; it loops over T
    internally and returns h_next matching the shape of ``z``.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.belief_dim = cfg.belief_dim
        self.action_embed_dim = cfg.action_embed_dim
        self.action_embed = nn.Embedding(NUM_ACTIONS, cfg.action_embed_dim)
        self.cell = nn.GRUCell(cfg.latent_dim + cfg.action_embed_dim, cfg.belief_dim)

    @property
    def initial_input_dim(self) -> int:
        """GRU input size: latent + action embedding."""
        return self.latent_dim + self.action_embed_dim

    def init_state(self, batch: int, device: Union[str, torch.device]) -> Tensor:
        return torch.zeros(batch, self.belief_dim, device=device)

    def forward(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor:
        if z.dim() == 2:
            emb = self.action_embed(action.long())
            return self.cell(torch.cat([z, emb], dim=-1), h)
        batch, seq_len = z.shape[0], z.shape[1]
        actions = action.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(1).expand(batch, seq_len)
        h_cur = h
        outs: List[Tensor] = []
        for t in range(seq_len):
            emb = self.action_embed(actions[:, t])
            h_cur = self.cell(torch.cat([z[:, t], emb], dim=-1), h_cur)
            outs.append(h_cur)
        return torch.stack(outs, dim=1)


class ZephyrPredictor(nn.Module):
    """JEPA predictor: (z, action, h) -> z_hat, LayerNorm'd to match encoder z.

    MLP [latent+embed+belief -> predictor_hidden -> predictor_hidden -> latent]
    with LayerNorm + SiLU on the input/hidden blocks and a LayerNorm'd output.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        in_dim = cfg.latent_dim + cfg.action_embed_dim + cfg.belief_dim
        self.action_embed = nn.Embedding(NUM_ACTIONS, cfg.action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.predictor_hidden),
            nn.LayerNorm(cfg.predictor_hidden),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden, cfg.predictor_hidden),
            nn.LayerNorm(cfg.predictor_hidden),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor:
        emb = self.action_embed(action.long())
        return self.net(torch.cat([z, emb, h], dim=-1))


class ZephyrWM(nn.Module):
    """Online encoder + EMA target encoder + belief GRU + JEPA predictor."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ZephyrEncoder(cfg.encoder)
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_encoder.requires_grad_(False)
        self.ema_encoder.eval()
        self.gru = ZephyrGRU(cfg.world_model)
        self.predictor = ZephyrPredictor(cfg.world_model)

    def encode(self, obs_u8: Tensor) -> Tensor:
        """Online encoder forward (gradients flow)."""
        return self.encoder(obs_u8)

    def ema_encode(self, obs_u8: Tensor) -> Tensor:
        """Target encoder forward under no_grad."""
        with torch.no_grad():
            return self.ema_encoder(obs_u8)

    def belief_step(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor:
        """One GRU belief update; see ZephyrGRU.forward for shapes."""
        return self.gru(z, action, h)

    def predict(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor:
        """Predict the next latent z_hat from (z, action, h)."""
        return self.predictor(z, action, h)

    def init_state(self, batch: int, device: Union[str, torch.device]) -> Tensor:
        return self.gru.init_state(batch, device)

    def update_ema(self, step: int) -> None:
        """Polyak-update every EMA parameter/buffer from the online encoder.

        Decay ramps linearly from ema_decay_start to ema_decay_max over
        ema_ramp_steps (config); at step 0 the decay equals ema_decay_start.
        """
        wm_cfg = self.cfg.world_model
        if wm_cfg.ema_ramp_steps > 0:
            ramp = min(1.0, float(step) / float(wm_cfg.ema_ramp_steps))
        else:
            ramp = 1.0
        decay = wm_cfg.ema_decay_start + (wm_cfg.ema_decay_max - wm_cfg.ema_decay_start) * ramp
        with torch.no_grad():
            src = self.encoder.state_dict()
            tgt = self.ema_encoder.state_dict()
            for key, value in src.items():
                target = tgt[key]
                if torch.is_floating_point(value):
                    target.mul_(decay).add_(value, alpha=1.0 - decay)
                else:
                    target.copy_(value)

    def save_checkpoint(self, path: str, step: int, extra: Optional[dict] = None) -> None:
        """Save config + online module states; the EMA encoder is rebuilt on load."""
        payload: Dict[str, object] = {
            "format": _CHECKPOINT_FORMAT,
            "step": int(step),
            "config": self.cfg.to_dict(),
            "state": {
                "encoder": self.encoder.state_dict(),
                "gru": self.gru.state_dict(),
                "predictor": self.predictor.state_dict(),
            },
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Union[str, torch.device] = "cpu", strict: bool = True
    ) -> Tuple["ZephyrWM", Dict[str, object]]:
        """Load a checkpoint saved by :meth:`save_checkpoint`.

        Returns (model, meta) where meta has ``step`` and ``extra``. The EMA
        encoder is reconstructed as a fresh deepcopy of the loaded encoder.
        Raises ValueError on a wrong ``format`` key.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # trusted local file
        if ckpt.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"unexpected checkpoint format {ckpt.get('format')!r}; "
                f"expected {_CHECKPOINT_FORMAT!r}"
            )
        cfg = ZephyrConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.encoder.load_state_dict(ckpt["state"]["encoder"], strict=strict)
        model.gru.load_state_dict(ckpt["state"]["gru"], strict=strict)
        model.predictor.load_state_dict(ckpt["state"]["predictor"], strict=strict)
        model.ema_encoder = copy.deepcopy(model.encoder)
        model.ema_encoder.requires_grad_(False)
        model.ema_encoder.eval()
        model.to(device)
        meta: Dict[str, object] = {"step": int(ckpt["step"]), "extra": ckpt.get("extra")}
        return model, meta
