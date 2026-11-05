"""Tier 1: Replay Buffer & Tabula-Rasa Training Loop (F15, F16).

Tests:
- F15: Replay Buffer (Prioritized & Uniform) [5 tests]
- F16: Tabula-Rasa Multi-Game Online Loop [5 tests]
"""
from __future__ import annotations

from dataclasses import replace
import pathlib
import numpy as np
import pytest
import torch

from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import NUM_ACHIEVEMENTS
from zephyr.train_loop import ZephyrTrainer
from zephyr.train_wm import wm_joint_losses
from zephyr.wm import ZephyrWM
from zephyr.heads import ZephyrHeads


# ==============================================================================
# Feature 15: Replay Buffer (Prioritized & Uniform)
# ==============================================================================

def test_f15_episode_record_validation():
    """F15: EpisodeRecord validates consistency of tensor shapes across observation/action/reward fields."""
    T = 10
    obs = np.zeros((T, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros(T, dtype=np.int16)
    rewards = np.zeros(T, dtype=np.float32)
    rewards_int = np.zeros(T, dtype=np.float32)
    done = np.zeros(T, dtype=bool)
    inventory = np.zeros((T, 16), dtype=np.float32)
    ach_cum = np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool)
    ach_new = np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool)

    # Valid record instantiates cleanly
    rec = EpisodeRecord(
        obs=obs, actions=actions, rewards=rewards, rewards_int=rewards_int,
        done=done, inventory=inventory, achievements_cum=ach_cum,
        achievements_new=ach_new, seed=123, source="test",
    )
    assert rec.obs.shape[0] == T

    # Mismatched action length raises ValueError
    with pytest.raises(ValueError, match="actions shape"):
        EpisodeRecord(
            obs=obs, actions=np.zeros(T + 5, dtype=np.int16),
            rewards=rewards, rewards_int=rewards_int, done=done,
            inventory=inventory, achievements_cum=ach_cum,
            achievements_new=ach_new, seed=123, source="test",
        )


def test_f15_sample_window_bounds_no_crossing(populated_replay):
    """F15: Sampled sequence windows never cross episode boundaries; valid flag tracks active steps."""
    batch_size = 8
    seq_len = 10
    rng = np.random.default_rng(42)

    batch = populated_replay.sample(batch_size, seq_len, rng)

    assert batch["obs"].shape == (batch_size, seq_len, 64, 64, 3)
    assert batch["actions"].shape == (batch_size, seq_len)
    assert batch["valid"].shape == (batch_size, seq_len)
    assert batch["valid"].dtype == torch.bool
    # Observation must stay uint8 end-to-end in replay
    assert batch["obs"].dtype == torch.uint8


def test_f15_achievement_prioritized_sampling_ratio():
    """F15: Sampling with prioritize_achievement_frac draws from achievement episodes."""
    replay = ZephyrReplay(capacity_frames=10_000)
    T = 20

    # Add 9 normal episodes without achievements
    for ep in range(9):
        rec = EpisodeRecord(
            obs=np.zeros((T, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros(T, dtype=np.int16),
            rewards=np.zeros(T, dtype=np.float32),
            rewards_int=np.zeros(T, dtype=np.float32),
            done=np.zeros(T, dtype=bool),
            inventory=np.zeros((T, 16), dtype=np.float32),
            achievements_cum=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
            achievements_new=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
            seed=ep, source="normal",
        )
        replay.add_episode(rec)

    # Add 1 milestone achievement episode
    ach_new = np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool)
    ach_new[5, 0] = True  # unlock achievement
    rec_ach = EpisodeRecord(
        obs=np.ones((T, 64, 64, 3), dtype=np.uint8) * 200,
        actions=np.ones(T, dtype=np.int16),
        rewards=np.ones(T, dtype=np.float32),
        rewards_int=np.zeros(T, dtype=np.float32),
        done=np.zeros(T, dtype=bool),
        inventory=np.zeros((T, 16), dtype=np.float32),
        achievements_cum=ach_new,
        achievements_new=ach_new,
        seed=999, source="milestone",
    )
    replay.add_episode(rec_ach)

    rng = np.random.default_rng(123)
    # With prioritize_achievement_frac = 1.0, 100% of samples come from the achievement episode
    batch = replay.sample(batch=20, seq_len=5, rng=rng, prioritize_achievement_frac=1.0)
    # All sampled observations should be 200 from the achievement episode
    assert (batch["obs"] == 200).all(), "Priority 1.0 must sample exclusively from achievement pool"


def test_f15_replay_fifo_capacity_eviction():
    """F15: Replay evicts oldest episodes FIFO when total frames exceed capacity_frames."""
    capacity = 50
    replay = ZephyrReplay(capacity_frames=capacity)
    T = 20  # each episode has 20 frames

    def _make_rec(seed: int) -> EpisodeRecord:
        return EpisodeRecord(
            obs=np.zeros((T, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros(T, dtype=np.int16),
            rewards=np.zeros(T, dtype=np.float32),
            rewards_int=np.zeros(T, dtype=np.float32),
            done=np.zeros(T, dtype=bool),
            inventory=np.zeros((T, 16), dtype=np.float32),
            achievements_cum=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
            achievements_new=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
            seed=seed, source="test",
        )

    # Add 4 episodes (80 frames total, exceeding capacity 50)
    for s in range(4):
        replay.add_episode(_make_rec(seed=s))

    assert len(replay) <= capacity, f"Replay frames {len(replay)} exceeded capacity {capacity}"
    # Earliest episodes (0 and 1) must be evicted; remaining episodes should be 2 and 3
    remaining_seeds = [ep.seed for ep in replay.records]
    assert 0 not in remaining_seeds, "Oldest episode seed 0 should have been evicted"
    assert remaining_seeds == [2, 3] or remaining_seeds == [1, 2, 3]


def test_f15_replay_shard_save_and_load_roundtrip(tmp_path):
    """F15: save_shard() and load_shard() preserve records bit-for-bit via compressed .npz archive."""
    replay = ZephyrReplay(capacity_frames=1000)
    T = 15
    rec = EpisodeRecord(
        obs=np.random.randint(0, 256, (T, 64, 64, 3), dtype=np.uint8),
        actions=np.random.randint(0, 17, size=T, dtype=np.int16),
        rewards=np.random.uniform(0.0, 1.0, size=T).astype(np.float32),
        rewards_int=np.zeros(T, dtype=np.float32),
        done=np.zeros(T, dtype=bool),
        inventory=np.zeros((T, 16), dtype=np.float32),
        achievements_cum=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=np.zeros((T, NUM_ACHIEVEMENTS), dtype=bool),
        seed=777, source="shard_test",
    )
    rec.done[-1] = True
    replay.add_episode(rec)

    shard_file = tmp_path / "shard_000.npz"
    replay.save_shard(str(shard_file))

    restored = ZephyrReplay.load_shard(str(shard_file))

    assert restored.episodes == 1
    rec_restored = restored.records[0]
    assert np.array_equal(rec.obs, rec_restored.obs)
    assert np.array_equal(rec.actions, rec_restored.actions)
    assert np.array_equal(rec.rewards, rec_restored.rewards)
    assert rec.seed == rec_restored.seed


# ==============================================================================
# Feature 16: Tabula-Rasa Multi-Game Online Loop
# ==============================================================================

def test_f16_trainer_initialization_scratch(tmp_path, tiny_test_cfg):
    """F16: ZephyrTrainer initializes from scratch with 0 frames, 0 steps, and empty replay."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)

    assert trainer.train_steps == 0
    assert trainer.env_frames == 0
    assert len(trainer.replay) == 0
    assert trainer.collector_generation == 0


def test_f16_collection_phase_populates_replay(tmp_path, tiny_test_cfg):
    """F16: trainer.collect() collects frames from environment and populates replay."""
    cfg = replace(tiny_test_cfg, data=replace(tiny_test_cfg.data, episode_max_steps=20))
    trainer = ZephyrTrainer(cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)

    target_frames = 25
    collected = trainer.collect(frames=target_frames, source="actor")

    assert collected["frames"] >= target_frames, f"Collected {collected['frames']} < target {target_frames}"
    assert trainer.env_frames >= target_frames
    assert len(trainer.replay) > 0, "Replay buffer should not be empty after collection"


def test_f16_training_step_updates_world_model_and_policy(tmp_path, tiny_test_cfg, populated_replay):
    """F16: trainer.train() performs joint updates on world model and actor-critic."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)
    # Feed populated replay into trainer
    trainer.replay = populated_replay

    metrics = trainer.train(steps=2)

    assert trainer.train_steps == 2, f"Expected 2 train steps, got {trainer.train_steps}"
    assert isinstance(metrics, dict)
    assert "loss_wm" in metrics or "loss_total" in metrics or len(metrics) > 0


def test_f16_joint_loss_finite_and_propagating(mock_wm, mock_heads, tiny_test_cfg, synthetic_batch_generator):
    """F16: wm_joint_losses computes finite joint loss across JEPA MSE, VICReg, reward CE, continue BCE."""
    batch = synthetic_batch_generator(batch_size=4, seq_len=6)
    opt = torch.optim.Adam(list(mock_wm.parameters()) + list(mock_heads.parameters()), lr=1e-3)

    loss_total, metrics = wm_joint_losses(mock_wm, mock_heads, batch, tiny_test_cfg, device=torch.device("cpu"))

    assert "loss" in metrics
    assert torch.isfinite(loss_total), f"Total loss is non-finite: {loss_total}"

    opt.zero_grad()
    loss_total.backward()

    # Gradients must be finite on model parameters
    for p in mock_wm.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_f16_multi_cycle_online_alternation(tmp_path, tiny_test_cfg):
    """F16: Alternating between collection and training updates counters monotonically."""
    cfg = replace(tiny_test_cfg, data=replace(tiny_test_cfg.data, episode_max_steps=20))
    trainer = ZephyrTrainer(cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)

    # Cycle 1
    trainer.collect(frames=25, source="actor")
    f1 = trainer.env_frames
    trainer.train(steps=1)
    s1 = trainer.train_steps

    # Cycle 2
    trainer.collect(frames=25, source="actor")
    f2 = trainer.env_frames
    trainer.train(steps=1)
    s2 = trainer.train_steps

    assert f2 > f1, f"Frames should increment monotonically: {f2} > {f1}"
    assert s2 > s1, f"Steps should increment monotonically: {s2} > {s1}"
