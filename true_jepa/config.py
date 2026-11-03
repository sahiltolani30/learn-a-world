"""Configuration for the clean JEPA implementation.

There are no environment-semantic constants here.  The action count is
discovered from the environment and passed into the model constructor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class JEPAConfig:
    """Model dimensions and EMA/prediction settings."""

    obs_shape: Tuple[int, int, int] = (64, 64, 3)
    patch_size: int = 8
    token_dim: int = 192
    belief_dim: int = 256
    predictor_heads: int = 4
    predictor_layers: int = 2
    predictor_mlp_dim: int = 384
    ensemble_size: int = 3
    ema_decay_start: float = 0.996
    ema_decay_end: float = 0.999
    ema_ramp_updates: int = 20_000
    rollout_horizon: int = 8


@dataclass(frozen=True)
class TrainConfig:
    """T4-friendly defaults for online self-play and latent training."""

    batch_size: int = 8
    sequence_length: int = 16
    replay_capacity_frames: int = 120_000
    learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 10.0
    jepa_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 0.25
    ensemble_weight: float = 0.25
    actor_entropy_weight: float = 0.01
    gamma: float = 0.997
    amp: bool = True


def as_dict(cfg: object) -> Dict[str, object]:
    """Serialize a dataclass configuration without importing legacy code."""

    return asdict(cfg)
