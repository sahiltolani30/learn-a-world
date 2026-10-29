"""ATLAS configuration (single source of truth, dataclasses)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

NUM_ACTIONS = 17
CRAFTER_ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword",
]
LATENT_DIM = 128
RECURRENT_DIM = 256
STATE_DIM = LATENT_DIM + RECURRENT_DIM  # composite s = [z; h]
NUM_ACHIEVEMENTS = 22
STATE_VECTOR_DIM = 16  # 12 items + 4 vitals (training-target readout only)


def resolve_drive_root(custom_path: Optional[str | Path] = None) -> Optional[Path]:
    """Resolves the main Google Drive root folder learn-a-world."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    env = os.environ.get("DRIVE_ROOT")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        Path("/content/drive/MyDrive/learn-a-world"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_atlas_drive_dir(custom_path: Optional[str | Path] = None) -> Optional[Path]:
    """Returns own subfolder atlas inside main folder learn-a-world on Drive."""
    root = resolve_drive_root(custom_path)
    if root is not None:
        p = root / "atlas"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return None


def _default_checkpoint() -> str:
    """Checkpoint selection: env override -> Drive champion -> local repo checkpoints."""
    env = os.environ.get("ATLAS_CHECKPOINT")
    if env:
        return env
    drive_root = resolve_drive_root()
    candidates = []
    if drive_root is not None:
        candidates.extend([
            drive_root / "checkpoints" / "jepa_crafter_v3_state_readout_v2.pt",
            drive_root / "checkpoints" / "jepa_crafter_v3_phase2.pt",
            drive_root / "checkpoints" / "jepa_crafter_v3_t2_final.pt",
        ])
    candidates.extend([
        REPO_ROOT / "checkpoints" / "jepa_crafter_v3_state_readout_v2.pt",
        REPO_ROOT / "checkpoints" / "jepa_crafter_v3_phase2.pt",
    ])
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[-1])


@dataclass
class BackboneConfig:
    checkpoint: str = field(default_factory=_default_checkpoint)
    device: str = "auto"  # auto -> mps if available else cpu
    frozen: bool = True


@dataclass
class BeliefConfig:
    unlock_threshold: float = 0.25      # uniform calibrated threshold (strict mode)
    reward_fallback: float = 0.9        # scalar-reward fallback pin (same rule as T2 strict)


@dataclass
class TopoMapConfig:
    scene_change_threshold: float = 0.45   # cosine distance trigger for a new keyframe
    min_keyframe_interval: int = 3          # steps between forced keyframes
    max_keyframes: int = 512                # per-episode ring capacity
    persistent_capacity: int = 4096         # cross-episode store capacity
    match_threshold: float = 0.93           # cos(z, key) considered "the same place"


@dataclass
class SkillsConfig:
    goal_emb_dim: int = 128
    hidden_dim: int = 256
    her_horizons: Tuple[int, ...] = (5, 10, 20)
    arrival_cosine: float = 0.93
    max_option_steps: int = 25
    stagnation_patience: int = 8
    lr: float = 3e-4


@dataclass
class EpistemicConfig:
    num_heads: int = 4
    hidden_dim: int = 128
    lr: float = 3e-4
    knn_capacity: int = 256
    knn_beta: float = 2.0


@dataclass
class SafetyConfig:
    hidden_dim: int = 256
    lr: float = 3e-4
    focal_weight: float = 25.0
    risk_threshold: float = 0.5     # intended action death-risk that triggers the override
    risk_margin: float = 0.15       # alternative must be safer by this much


@dataclass
class ReadoutConfig:
    hidden_dim: int = 256
    lr: float = 3e-4


@dataclass
class GoalsConfig:
    max_goals: int = 256
    vital_spike: float = 0.15       # health-head positive spike that marks a vital-restoring place
    frontier_quantile: float = 0.85
    w_unlock: float = 1.0
    w_reach: float = 0.6
    w_explor: float = 0.5
    w_danger: float = 0.4
    predecessor_window: int = 30    # steps for "d precedes e" mining
    probe_steps: int = 6            # PROBE-mode duration after arrival
    probe_entropy: float = 0.25     # exploration mix inside PROBE mode


@dataclass
class CollectConfig:
    out_dir: str = str(REPO_ROOT / "data" / "atlas")
    max_steps: int = 1000
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_iters: int = 40   # self-learn iterations
    beta_start: float = 1.0         # intrinsic reward scale
    beta_end: float = 0.15
    episodes_per_iter: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs_per_iter: int = 6
    device: str = "auto"
    cache_latents: bool = True


@dataclass
class ScreenConfig:
    seeds: Tuple[int, int] = (2050, 2099)   # locked benchmark seeds
    dev_seeds: Tuple[int, int] = (6150, 6159)
    max_steps: int = 1000
    num_episodes: int = 50


@dataclass
class AtlasConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    placemap: TopoMapConfig = field(default_factory=TopoMapConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    epistemic: EpistemicConfig = field(default_factory=EpistemicConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    goals: GoalsConfig = field(default_factory=GoalsConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
