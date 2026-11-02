"""
sapna/config.py

Single source of truth for all SAPNA hyperparameters per SAPNA_PLAN.md §3.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class SAPNAConfig:
    """Exact hyperparameters for SAPNA (Spatial Anticipatory Predictive Neural Agent)."""

    # Environment & Collection
    num_envs: int = 8
    action_repeat: int = 2
    frame_stack: int = 1
    base_seed: int = 42
    seed: int = 42

    # Sequence dimensions
    seq_len: int = 16  # T
    batch_seqs: int = 8  # B
    token_dim: int = 64
    pooled_dim: int = 256
    num_tokens: int = 64
    num_actions: int = 17
    action_dim: int = 32
    mem_hidden: int = 256

    # World Model
    mask_ratio: float = 0.35  # 22 of 64 tokens
    ema_tau: float = 0.99  # TargetEncoder EMA tau per optimizer step
    vicreg_sim: float = 1.0
    vicreg_var: float = 1.0
    vicreg_cov: float = 0.04
    token_var_floor: float = 0.05
    recon_coeff: float = 0.3

    # Distributional Reward & Values
    reward_bins: int = 255
    reward_bins_range: float = 20.0  # [-20, 20] in symlog space

    # Imagination & Policy
    imag_horizon: int = 15  # H
    lam: float = 0.95  # lambda-returns
    gamma: float = 0.997
    entropy_coeff: float = 0.01  # actent calibrated from 3e-4 to prevent mode collapse on sparse rewards
    unimix: float = 0.01
    eps_base: float = 0.0
    curiosity_weight: float = 0.0  # off in v1

    # Learning Rates & Optimization
    lr_wm: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    slow_critic_tau: float = 0.98  # SlowCritic EMA tau
    grad_clip: float = 1.0
    update_every: int = 1  # 1 vec-step = 16 raw env steps

    # Replay Buffer
    buffer_capacity: int = 100_000
    milestone_frac: float = 0.30  # 70% uniform / 30% milestone
    warmup_steps: int = 200  # vec-steps uniform-random

    # Schedules & Intervals
    total_steps: int = 25_000  # Default G1 validation
    eval_interval: int = 5_000
    save_interval: int = 25_000

    # System & Execution
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir: str = "learn-a-world/sapna/checkpoints_v1"
    no_resume: bool = False
