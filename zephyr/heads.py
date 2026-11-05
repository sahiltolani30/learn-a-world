"""Zephyr heads: reward, continue, readout, effect, critic, actor, ensemble.

All state heads read ``s = concat(z, h)`` (use ``ZephyrHeads.state_input``);
the dynamics ensemble instead takes (z, action, h) per its contract. Heads are
shape-agnostic over leading dims ([B, ...] and [B, T, ...] both work).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zephyr.config import ZephyrConfig
from zephyr.losses import TwoHotCoder

# Checkpoint layout tag written by, and required by, this code (matches
# ZephyrConfig.format_version). Must only change when the layout changes.
_CHECKPOINT_FORMAT = "zephyr_v1"

# State keys of the container checkpoint. 'contin' (not 'continue') because
# 'continue' is a Python keyword.
_HEAD_KEYS: Tuple[str, ...] = (
    "reward", "contin", "readout", "effect", "critic", "actor", "ensemble",
)


def _state_in_dim(cfg: ZephyrConfig) -> int:
    return cfg.world_model.latent_dim + cfg.world_model.belief_dim


def _head_hidden(cfg: ZephyrConfig) -> int:
    """Hidden width for 2-layer state heads (pinned spec: 1024 -> 512 -> out).

    HeadsConfig has no hidden field, so the width is sourced from
    world_model.latent_dim (512 by default) rather than introducing a constant.
    """
    return cfg.world_model.latent_dim


class _TwoHotStateHead(nn.Module):
    """Shared 2-layer MLP emitting symlog two-hot logits over ``bins``."""

    def __init__(self, cfg: ZephyrConfig, bins: int, symlog_range: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_state_in_dim(cfg), _head_hidden(cfg)),
            nn.SiLU(),
            nn.Linear(_head_hidden(cfg), bins),
        )
        self.coder = TwoHotCoder(symlog_range=symlog_range, num_bins=bins)

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)


class RewardHead(_TwoHotStateHead):
    """s [.., 1024] -> reward two-hot logits [.., reward_bins]."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__(cfg, cfg.heads.reward_bins, cfg.heads.reward_symlog_range)


class CriticHead(_TwoHotStateHead):
    """s [.., 1024] -> value two-hot logits [.., critic_bins]."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__(cfg, cfg.heads.critic_bins, cfg.heads.critic_symlog_range)


class ContinueHead(nn.Module):
    """s [.., 1024] -> termination logit [..] (sigmoid for P(done))."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_state_in_dim(cfg), _head_hidden(cfg)),
            nn.SiLU(),
            nn.Linear(_head_hidden(cfg), 1),
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s).squeeze(-1)


class ReadoutHead(nn.Module):
    """s -> predicted 16-d inventory vector (linear output; targets transformed)."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_state_in_dim(cfg), _head_hidden(cfg)),
            nn.SiLU(),
            nn.Linear(_head_hidden(cfg), cfg.heads.inventory_dim),
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)


class EffectHead(nn.Module):
    """(s, action) -> predicted 16-d inventory delta; action enters one-hot."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.num_actions = cfg.heads.num_actions
        in_dim = _state_in_dim(cfg) + cfg.heads.num_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, _head_hidden(cfg)),
            nn.SiLU(),
            nn.Linear(_head_hidden(cfg), cfg.heads.inventory_dim),
        )

    def forward(self, s: Tensor, action: Tensor) -> Tensor:
        one_hot = F.one_hot(action.long(), self.num_actions).to(s.dtype)
        return self.net(torch.cat([s, one_hot], dim=-1))


class ActorHead(nn.Module):
    """s -> categorical logits over the Crafter action set."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_state_in_dim(cfg), _head_hidden(cfg)),
            nn.SiLU(),
            nn.Linear(_head_hidden(cfg), cfg.heads.num_actions),
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)


class DynamicsEnsemble(nn.Module):
    """Bootstrap ensemble of latent dynamics MLPs (Plan2Explore disagreement).

    Input is (z, action, h) — NOT s — with a per-member action embedding. Each
    member maps [latent+embed+belief -> ensemble_hidden -> ensemble_hidden ->
    latent]; forward returns per-member predictions stacked [K, ..., latent].
    """

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        wm = cfg.world_model
        in_dim = wm.latent_dim + wm.action_embed_dim + wm.belief_dim
        self.members = nn.ModuleList()
        self.action_embeds = nn.ModuleList()
        for _ in range(cfg.explore.ensemble_k):
            self.action_embeds.append(
                nn.Embedding(cfg.heads.num_actions, wm.action_embed_dim)
            )
            self.members.append(
                nn.Sequential(
                    nn.Linear(in_dim, cfg.explore.ensemble_hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.explore.ensemble_hidden, cfg.explore.ensemble_hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.explore.ensemble_hidden, wm.latent_dim),
                )
            )

    def forward(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor:
        action = action.long()
        preds = []
        for embed, member in zip(self.action_embeds, self.members):
            preds.append(member(torch.cat([z, embed(action), h], dim=-1)))
        return torch.stack(preds, dim=0)


class ZephyrHeads(nn.Module):
    """Container for all Zephyr heads plus the dynamics ensemble."""

    def __init__(self, cfg: ZephyrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.reward = RewardHead(cfg)
        self.contin = ContinueHead(cfg)  # 'contin': 'continue' is a Python keyword
        self.readout = ReadoutHead(cfg)
        self.effect = EffectHead(cfg)
        self.critic = CriticHead(cfg)
        self.actor = ActorHead(cfg)
        self.ensemble = DynamicsEnsemble(cfg)

    @staticmethod
    def state_input(z: Tensor, h: Tensor) -> Tensor:
        """s = concat(z, h) on the last dim -> [.., latent + belief]."""
        return torch.cat([z, h], dim=-1)

    def save_checkpoint(self, path: str, step: int, extra: Optional[dict] = None) -> None:
        """Save config + per-head states under the pinned state keys."""
        payload: Dict[str, object] = {
            "format": _CHECKPOINT_FORMAT,
            "step": int(step),
            "config": self.cfg.to_dict(),
            "state": {name: getattr(self, name).state_dict() for name in _HEAD_KEYS},
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Union[str, torch.device] = "cpu", strict: bool = True
    ) -> Tuple["ZephyrHeads", Dict[str, object]]:
        """Load a checkpoint saved by :meth:`save_checkpoint`.

        Returns (heads, meta) where meta has ``step`` and ``extra``. Raises
        ValueError on a wrong ``format`` key.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # trusted local file
        if ckpt.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"unexpected checkpoint format {ckpt.get('format')!r}; "
                f"expected {_CHECKPOINT_FORMAT!r}"
            )
        cfg = ZephyrConfig.from_dict(ckpt["config"])
        heads = cls(cfg)
        for name in _HEAD_KEYS:
            getattr(heads, name).load_state_dict(ckpt["state"][name], strict=strict)
        heads.to(device)
        meta: Dict[str, object] = {"step": int(ckpt["step"]), "extra": ckpt.get("extra")}
        return heads, meta
