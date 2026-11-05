"""Zephyr configuration — the single source of truth for every module and script.

Nothing in zephyr/ may define its own hyperparameters. Read them from here so that
checkpoints, training runs, and benchmark reports are all reproducible from one object.

Py3.9+ compatible (local laptop runs 3.9; Colab runs 3.11+): no match statements,
no runtime `X | Y` annotations outside `from __future__ import annotations` scope.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical constants (identical everywhere: env, buffer, heads, screen)
# ---------------------------------------------------------------------------

NUM_ACTIONS = 17
NUM_ACHIEVEMENTS = 22
IMG_SIZE = 64

# Privileged-diagnostic inventory vector layout (training targets ONLY — never an
# inference input). Order is fixed here and used by collector, buffer, readout and
# effect heads alike.
VITALS: Tuple[str, ...] = ("health", "food", "drink", "energy")
ITEMS: Tuple[str, ...] = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)
INVENTORY_KEYS: Tuple[str, ...] = VITALS + ITEMS  # 16-d
assert len(INVENTORY_KEYS) == 16

# Canonical 22 Crafter achievements (Hafner 2021) — MUST equal eval.metrics
# CRAFTER_ACHIEVEMENTS; screen.py asserts equality so scoring order is identical.
ACHIEVEMENTS: Tuple[str, ...] = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)
assert len(ACHIEVEMENTS) == NUM_ACHIEVEMENTS

VITALS_MAX: float = 9.0  # crafter vitals live on a 0..9 scale


def inventory_dict_to_vector(inv: Optional[Dict[str, float]]) -> np.ndarray:
    """Flatten the env's diagnostic inventory dict into the fixed 16-d vector.

    PRIVILEGED: this function exists for training-target extraction (diagnostic mode).
    It must never be called on the strict-inference action path.
    Missing keys count as 0. Vitals are scaled to [0, 1] (divided by VITALS_MAX).
    """
    vec = np.zeros(len(INVENTORY_KEYS), dtype=np.float32)
    if not inv:
        return vec
    for i, key in enumerate(INVENTORY_KEYS):
        val = float(inv.get(key, 0.0))
        if key in VITALS:
            val = val / VITALS_MAX
        vec[i] = val
    return vec


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderConfig:
    img_size: int = IMG_SIZE
    in_channels: int = 3
    channels: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 512


@dataclass(frozen=True)
class WorldModelConfig:
    latent_dim: int = 512
    belief_dim: int = 512
    action_embed_dim: int = 64
    predictor_hidden: int = 1024
    # EMA target-encoder schedule: linear ramp start -> max over ema_ramp_steps.
    ema_decay_start: float = 0.996
    ema_decay_max: float = 0.999
    ema_ramp_steps: int = 20000
    # Anti-collapse (VICReg-lite) applied in the JEPA loss.
    var_weight_pred: float = 1.0    # variance regularizer on predicted z-hat
    cov_weight_pred: float = 0.02   # covariance regularizer on predicted z-hat
    var_weight_online: float = 0.25  # variance regularizer on online encoder z
    var_gamma: float = 1.0
    # Free-running multi-step loss: per-offset weight = horizon_decay ** (k - 1).
    horizon_decay: float = 0.5


@dataclass(frozen=True)
class HeadsConfig:
    num_actions: int = NUM_ACTIONS
    num_achievements: int = NUM_ACHIEVEMENTS
    inventory_dim: int = 16
    # symlog twohot reward / critic coders
    reward_bins: int = 65
    reward_symlog_range: float = 20.0
    critic_bins: int = 65
    critic_symlog_range: float = 20.0
    # termination head: rare-positive handling (fixes T2's zero-recall failure)
    continue_pos_weight: float = 25.0
    continue_focal_gamma: float = 2.0
    # loss weights
    reward_loss_weight: float = 1.0
    continue_loss_weight: float = 1.0
    readout_loss_weight: float = 1.0
    effect_loss_weight: float = 0.25
    ensemble_loss_weight: float = 1.0


@dataclass(frozen=True)
class ActorCriticConfig:
    imagination_horizon: int = 15
    gamma: float = 0.997
    td_lambda: float = 0.95
    actor_entropy_weight: float = 0.01
    actor_unimix: float = 0.01
    actor_loss_weight: float = 1.0
    critic_loss_weight: float = 0.5
    slow_target_decay: float = 0.995
    # Imagination start states: take every N-th teacher-forced timestep of the batch.
    imag_start_stride: int = 4


@dataclass(frozen=True)
class ExplorationConfig:
    ensemble_k: int = 5
    ensemble_hidden: int = 256
    disagreement_weight: float = 1.0
    knn_k: int = 8
    knn_memory: int = 512
    knn_weight: float = 1.0
    # Intrinsic-motivation schedule (frames are env steps since run start).
    beta_start: float = 1.0
    beta_decay_frames: int = 2_000_000
    beta_min: float = 0.05

    def beta(self, frames: int) -> float:
        if self.beta_decay_frames <= 0:
            return self.beta_min
        t = min(1.0, frames / self.beta_decay_frames)
        return self.beta_start + (self.beta_min - self.beta_start) * t


@dataclass(frozen=True)
class DataConfig:
    num_envs: int = 4
    episode_max_steps: int = 1000
    batch_size: int = 16
    seq_len: int = 32
    # In-memory replay bound in FRAMES (uint8 64x64x3 = 12 KB/frame; 300k ≈ 3.7 GB).
    # Episodes beyond this FIFO-evict from RAM; shards on disk remain the archive.
    replay_capacity_frames: int = 300_000
    prioritize_achievement_frac: float = 0.1
    val_ratio: float = 0.1
    split_seed: int = 4242
    train_seed_base: int = 7000
    dev_screen_seed_start: int = 6150
    dev_screen_episodes: int = 10
    benchmark_seed_start: int = 2050
    benchmark_episodes: int = 50


@dataclass(frozen=True)
class OptimConfig:
    lr_world: float = 1e-4       # encoder + GRU + predictor + ensemble
    lr_heads: float = 3e-4       # reward/continue/readout/effect
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip: float = 10.0
    adam_eps: float = 1e-8
    fp16: bool = True            # autocast; honored only on CUDA


@dataclass(frozen=True)
class ZephyrConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    explore: ExplorationConfig = field(default_factory=ExplorationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    protocol_id: str = "strict_visual_zephyr_clean_v1"
    format_version: str = "zephyr_v1"

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict) -> "ZephyrConfig":
        kwargs = {}
        for fld in ("encoder", "world_model", "heads", "ac", "explore", "data", "optim"):
            sub = raw.get(fld)
            if sub is None:
                continue
            section = {"encoder": EncoderConfig, "world_model": WorldModelConfig,
                       "heads": HeadsConfig, "ac": ActorCriticConfig,
                       "explore": ExplorationConfig, "data": DataConfig,
                       "optim": OptimConfig}[fld]
            kwargs[fld] = section(**{k: v for k, v in sub.items()
                                     if k in section.__dataclass_fields__})
        for fld in ("protocol_id", "format_version"):
            if fld in raw:
                kwargs[fld] = raw[fld]
        return cls(**kwargs)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "ZephyrConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
