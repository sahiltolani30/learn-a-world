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
    prior_weight: float = 0.0  # NEW: 0.0 = old behavior-brain has ZERO influence


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
    her_horizons: Tuple[int, ...] = (5, 10)  # dropped k=20 (measured arrivals are short-range)
    arrival_cosine: float = 0.85  # CHANGED from 0.93 (0.93 was unreachable; cos plateaus ~0.92)
    max_option_steps: int = 25
    stagnation_patience: int = 24  # V3 F6: raised from 8 to 24 to avoid aborting valid detours
    lr: float = 3e-4
    # NEW fields:
    wander_steps: int = 6  # V3 F6: reduced from 10 to 6
    label_smoothing: float = 0.05  # policy CE label smoothing
    max_action_share: float = 0.25  # not used for subsampling; kept for reference/tests
    weight_clip: float = 50.0  # max per-sample imitation weight
    her_reward_weight_max: float = 4.0
    her_unlock_weight: float = 2.0
    progress_weight_gain: float = 3.0  # V2.2 F4
    option_budget_min: int = 25        # V2.2 F9
    option_budget_max: int = 60        # V2.2 F9
    # V3 F1/F2 Critic & AWR:
    critic_lr: float = 3e-4
    critic_gamma: float = 0.97
    critic_horizon: int = 25
    critic_bins: int = 65
    critic_v_min: float = -5.0
    critic_v_max: float = 10.0
    critic_epochs_per_iter: int = 8
    critic_ema_decay: float = 0.995


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
    # PLUS:
    max_goals_per_dim: int = 15      # cap per achievement-dimension among "event" goals
    same_kind_dedupe: float = 0.90   # cos threshold to reject near-duplicate same-kind goals
    frontier_min_share: float = 0.20 # bank must keep >= 20% frontier goals
    frontier_target_share: float = 0.35   # album share frontier may occupy at steady state
    vital_target_share: float = 0.15
    lift_cap: float = 10.0
    lift_min_base_count: int = 20    # min firings of d to mine edge (d,e)
    lift_min_pair_support: int = 3   # min co-occurrence count k
    lift_ci_threshold: float = 1.2   # ci_low must exceed this to influence scoring
    # V2.2 F3 Selection rotation:
    selection_rotate_share: float = 0.25
    selection_frontier_share: float = 0.10
    # V2.2 F10 & F11:
    episode_scope_frequency: float = 0.30     # below this -> episode-scoped goal (F10)
    readout_in_goal_gating: bool = True       # learned material gating (F11)
    prerequisite_routing: bool = True         # route to prerequisite before target (F11)
    material_gating_learned: bool = True   # F11-fix: learned costs gate probes, never recipes
    material_gate_margin: float = 0.2      # log1p-scale tolerance on have-vs-cost
    # V3 additions:
    vital_select_below: float = 0.35       # V3 F7: homeostatic pressure threshold
    vital_window: int = 30                 # V3 F7: steps window for realized-vital mining
    vital_min_delta: float = 1.0           # V3 F7: min vital unit increase to mine goal
    blind_probe_boost: float = 1.0         # V3 F4: curiosity boost in blind contexts
    w_value: float = 0.5                   # V3 F3: goal value weighting
    max_cycles: int = 4                    # V3 F3: max material repeat cycles
    probe_affordance_min: float = 0.05     # V3 F8: craft execution affordance threshold


@dataclass
class CollectConfig:
    out_dir: str = str(REPO_ROOT / "data" / "atlas")
    max_steps: int = 1000
    epsilon_start: float = 0.30
    epsilon_end: float = 0.10        # CHANGED from 0.05 (floor stays high)
    epsilon_decay_iters: int = 100   # CHANGED from 40
    beta_start: float = 1.0         # intrinsic reward scale
    beta_end: float = 0.30           # CHANGED from 0.15
    episodes_per_iter: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs_per_iter: int = 6
    skill_epochs_per_iter: int = 16  # V2.2 F5 (OptionNet + ReachNet)
    device: str = "auto"
    cache_latents: bool = True
    her_max_pairs: int = 32_768


@dataclass
class ScreenConfig:
    seeds: Tuple[int, int] = (2050, 2099)   # locked benchmark seeds
    dev_seeds: Tuple[int, int] = (6150, 6159)
    max_steps: int = 1000
    num_episodes: int = 50


@dataclass
class HealthConfig:
    window_steps: int = 200
    action_share_alarm: float = 0.60
    confidence_alarm: float = 0.85
    flatline_iters: int = 15
    flatline_improvement: float = 0.10
    epsilon_boost: float = 2.0
    boost_steps: int = 100


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
    health: HealthConfig = field(default_factory=HealthConfig)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
