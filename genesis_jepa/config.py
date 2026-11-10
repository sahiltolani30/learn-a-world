"""
genesis_jepa.config — Hyperparameter and Architecture Configuration for Genesis JEPA
=====================================================================================
Governing Protocol: AGENTS.md & GEMINI.md (Zero Hardcoding, Zero Cheat Tables)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GenesisConfig:
    # 1. Visual Retina & Patch Tokenizer
    obs_shape: Tuple[int, int, int] = (64, 64, 3)
    patch_size: int = 8
    num_patches: int = 64           # (64 / 8) * (64 / 8) = 64 tokens
    token_dim: int = 192            # Patch token embedding dimension
    contrast_norm: bool = True      # Intra-frame spatial contrast normalization

    # 2. Inverted Competitive Slot Attention
    num_slots: int = 12             # S = 12 slots competing for 64 patches
    slot_dim: int = 192             # Slot latent representation dimension
    slot_iters: int = 3             # Recurrent slot refinement iterations

    # 3. SimNorm Dynamics & Spatially Gated Jump Operator
    simnorm_heads: int = 6          # M = 6 simplicial heads
    simnorm_head_dim: int = 32      # L = 32 per head (6 * 32 = 192)
    simnorm_temp: float = 1.0       # Temperature for simplicial softmax
    jump_gate_bias_init: float = -2.0  # Initial bias towards identity for background
    rollout_horizon: int = 4        # H = 4 maximum unroll horizon (anti-compounding)
    anti_persistence_floor: float = 1.0
    anti_persistence_ceil: float = 5.0
    anti_persistence_gain: float = 4.0

    # 4. Target Encoder EMA
    target_ema_decay: float = 0.99  # Exponential moving average decay for JEPA target

    # 5. Discrete Action Conditioning
    num_actions: int = 17           # 17 Crafter discrete actions
    action_emb_dim: int = 32        # Action embedding vector length

    # 6. Action-Free 2D Fourier Phase Correlation SLAM
    slam_grid_size: int = 8         # 8 x 8 spatial patch grid
    keyframe_distance_threshold: float = 3.0  # Tiles before keyframe node creation
    sr_discount: float = 0.95       # Successor Representation discount factor

    # 7. Distributional Symlog Critic
    symlog_bins: int = 65           # 65-bin categorical distribution
    symlog_v_min: float = -20.0     # Minimum return in symlog space
    symlog_v_max: float = 20.0     # Maximum return in symlog space
    symlog_center_bin: int = 32     # Zero return lands on atom 32 (0-indexed)
    stratified_milestone_quota: float = 0.50  # 50% milestone transitions per batch

    # 8. Offline Policy & Regularization
    awr_temperature: float = 3.0
    awr_clip_min: float = -2.0
    awr_clip_max: float = 2.0
    progress_velocity_gain: float = 3.0  # Weight multiplier for approach progress Δcos
    unimix_floor: float = 0.10     # Categorical exploration floor for milestone discovery
    entropy_canary_floor: float = 1.50   # Nats
    max_action_frequency_cap: float = 0.60  # Canary tripwire: prevents collapse to any single action

    # 9. Training Loop & Optimization
    lr_world_model: float = 3e-4
    lr_critic: float = 3e-4
    lr_actor: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    weight_decay: float = 1e-4
    checkpoint_dir: str = "genesis_checkpoints"

    # 10. Latent Planning & Mental Simulation (Model-Predictive Control)
    planning_mode: str = "hybrid_mppi"  # "reflex", "lookahead_1step", "hybrid_mppi", "multistep_rollout"
    plan_horizon: int = 3               # Multi-step mental simulation horizon H
    plan_temperature: float = 1.0       # Temperature for rollout Q-value softmax
    plan_num_candidates: int = 16       # Number of trajectory branches in multi-step rollout
