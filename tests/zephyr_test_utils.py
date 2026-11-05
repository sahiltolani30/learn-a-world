"""Shared tiny-config helpers for Zephyr tests (fast CPU smoke scale)."""
from __future__ import annotations

from dataclasses import replace

from zephyr.config import ZephyrConfig


def tiny_cfg() -> ZephyrConfig:
    """A minutes-scale ZephyrConfig: tiny dims, CPU-friendly, no fp16."""
    cfg = ZephyrConfig()
    return replace(
        cfg,
        world_model=replace(
            cfg.world_model,
            latent_dim=64, belief_dim=64, action_embed_dim=16,
            predictor_hidden=128, ema_ramp_steps=100,
        ),
        encoder=replace(cfg.encoder, channels=(8, 16, 16, 32), latent_dim=64),
        heads=replace(
            cfg.heads, reward_bins=33, critic_bins=33,
            reward_symlog_range=10.0, critic_symlog_range=10.0,
        ),
        ac=replace(cfg.ac, imagination_horizon=5, imag_start_stride=2),
        explore=replace(cfg.explore, ensemble_k=2, ensemble_hidden=64),
        data=replace(
            cfg.data, batch_size=4, seq_len=8, num_envs=1,
            replay_capacity_frames=20_000, episode_max_steps=120,
        ),
        optim=replace(cfg.optim, fp16=False),
    )
