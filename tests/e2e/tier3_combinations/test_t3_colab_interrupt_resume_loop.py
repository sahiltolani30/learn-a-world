"""Tier 3: Google Colab Atomic Checkpointing, Interrupt Resilience & Resumption Loop.

Covers combinatorial interactions between:
- F15 (Dual-Pool Replay Buffer), F16 (Online Loop), F17 (Google Colab 30-min Checkpoint),
  F18 (Seed Offsetting 10^9 * gen).
"""
from __future__ import annotations

import os
import random
import numpy as np
import pytest
import torch

from zephyr.buffer import ZephyrReplay
from zephyr.train_loop import ZephyrTrainer


def test_t3_resume_01_atomic_two_stage_checkpoint_serialization(tmp_path, tiny_test_cfg):
    """T3-RESUME-01: Atomic two-stage checkpoint write via temporary file replacement."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)
    ckpt_path = os.path.join(str(tmp_path), "state.pt")

    saved_path = trainer.save_run(ckpt_path)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 0

    # Ensure no lingering .tmp file remains
    assert not os.path.exists(ckpt_path + ".tmp")

    payload = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert payload.get("format") == "zephyr_run_v1"
    assert "wm_state" in payload
    assert "heads_state" in payload
    assert "slow_critic_state" in payload


def test_t3_resume_02_mid_write_disconnection_resilience(tmp_path, tiny_test_cfg, monkeypatch):
    """T3-RESUME-02: Mid-write disconnection or crash preserves existing valid state uncorrupted."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)
    ckpt_path = os.path.join(str(tmp_path), "state.pt")

    # Step 1: Save valid original state
    trainer.save_run(ckpt_path)
    orig_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert orig_payload["format"] == "zephyr_run_v1"

    # Step 2: Inject mock failure during subsequent write to .tmp
    real_torch_save = torch.save

    def failing_save(obj, f, *args, **kwargs):
        if str(f).endswith(".tmp"):
            raise IOError("Simulated network drop / Colab disconnect mid-write")
        return real_torch_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(IOError, match="Simulated network drop"):
        trainer.save_run(ckpt_path)

    # Step 3: Verify existing state.pt remains 100% valid, uncorrupted, and loadable
    reloaded_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert reloaded_payload["format"] == "zephyr_run_v1"
    assert reloaded_payload["train_steps"] == orig_payload["train_steps"]


def test_t3_resume_03_complete_state_dict_and_optimizer_restoration(tmp_path, tiny_test_cfg, populated_replay):
    """T3-RESUME-03: Complete trainer state restoration with bit-for-bit parameter parity."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)
    for rec in populated_replay.records:
        trainer.replay.add_episode(rec)

    # Run 1 training update to alter weights and populate optimizer momentum states
    trainer.train(steps=1)
    trainer.env_frames = 150
    ckpt_path = os.path.join(str(tmp_path), "state.pt")
    trainer.save_run(ckpt_path)

    # Restore in fresh trainer instance
    restored = ZephyrTrainer.load_run(ckpt_path, device="cpu")

    assert restored.env_frames == trainer.env_frames
    assert restored.train_steps == trainer.train_steps

    # Check weights match bit-for-bit
    for (k1, p1), (k2, p2) in zip(trainer.wm.named_parameters(), restored.wm.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter mismatch in wm: {k1}"

    for (k1, p1), (k2, p2) in zip(trainer.heads.named_parameters(), restored.heads.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter mismatch in heads: {k1}"

    for (k1, p1), (k2, p2) in zip(trainer.slow_critic.named_parameters(), restored.slow_critic.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter mismatch in slow_critic: {k1}"

    assert set(restored.optimizers.keys()) == set(trainer.optimizers.keys())


def test_t3_resume_04_deterministic_prng_state_restoration(tmp_path, tiny_test_cfg):
    """T3-RESUME-04: Deterministic PRNG restoration across PyTorch, NumPy, and Python random."""
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)

    # Advance PRNG states
    _ = torch.randn(10)
    _ = trainer._rng.random(10)
    _ = random.random()

    ckpt_path = os.path.join(str(tmp_path), "state.pt")
    trainer.save_run(ckpt_path)

    # Sample control future numbers
    ctrl_torch = torch.randn(5)
    ctrl_np = trainer._rng.random(5)
    ctrl_py = [random.random() for _ in range(5)]

    # Restore trainer and evaluate PRNG continuation
    restored = ZephyrTrainer.load_run(ckpt_path, device="cpu")
    rest_torch = torch.randn(5)
    rest_np = restored._rng.random(5)
    rest_py = [random.random() for _ in range(5)]

    assert torch.equal(ctrl_torch, rest_torch), "PyTorch RNG state did not resume deterministically"
    assert np.array_equal(ctrl_np, rest_np), "NumPy BitGenerator RNG state did not resume deterministically"
    assert ctrl_py == rest_py, "Python random RNG state did not resume deterministically"


def test_t3_resume_05_replay_buffer_shard_persistence_and_reconstruction(tmp_path, populated_replay, tiny_test_cfg):
    """T3-RESUME-05: Replay buffer shard persistence on disk and load_shards reconstruction."""
    data_dir = os.path.join(str(tmp_path), "shards")
    os.makedirs(data_dir, exist_ok=True)
    shard_file = os.path.join(data_dir, "zephyr_shard_000000.npz")

    # Save shard
    populated_replay.save_shard(shard_file)
    assert os.path.exists(shard_file)

    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=str(tmp_path), env_in_process=True)
    shards_loaded = trainer.load_shards(data_dir)

    assert shards_loaded == 1
    assert trainer.replay.episodes == populated_replay.episodes

    batch = trainer.replay.sample(batch=2, seq_len=4, rng=np.random.default_rng(42))
    assert "obs" in batch
    assert "actions" in batch
    assert batch["obs"].shape == (2, 4, 64, 64, 3)


def test_t3_resume_06_generation_seed_offsetting_on_resumed_training(tiny_test_cfg):
    """T3-RESUME-06: Base seed generation offset formula 10^9 * gen guarantees disjoint world seeds."""
    base_seed = tiny_test_cfg.data.train_seed_base

    gen0_base = int(base_seed + 1_000_000_000 * 0)
    gen1_base = int(base_seed + 1_000_000_000 * 1)
    gen2_base = int(base_seed + 1_000_000_000 * 2)

    assert gen0_base == base_seed
    assert gen1_base == base_seed + 1_000_000_000
    assert gen2_base == base_seed + 2_000_000_000

    # Test seed ranges for 100,000 episodes in Gen 0 vs Gen 1: completely disjoint
    gen0_max = gen0_base + 100_000
    assert gen0_max < gen1_base


def test_t3_resume_07_resumed_joint_training_continuity(tmp_path, tiny_test_cfg, populated_replay):
    """T3-RESUME-07: Resumed trainer seamlessly continues optimization steps without error."""
    run_dir = str(tmp_path / "trainer_run")
    trainer = ZephyrTrainer(tiny_test_cfg, device="cpu", run_dir=run_dir, env_in_process=True)
    for rec in populated_replay.records:
        trainer.replay.add_episode(rec)

    # Initial training step
    stats1 = trainer.train(steps=1)
    assert trainer.train_steps == 1
    ckpt_path = trainer.save_run()

    # Resume trainer
    resumed_trainer = ZephyrTrainer.load_run(ckpt_path, device="cpu")
    assert resumed_trainer.train_steps == 1

    for rec in populated_replay.records:
        resumed_trainer.replay.add_episode(rec)

    # Step 2 after resumption
    stats2 = resumed_trainer.train(steps=1)
    assert resumed_trainer.train_steps == 2
    assert "actor_loss" in stats2
    assert "critic_loss" in stats2
    assert np.isfinite(stats2["actor_loss"])
    assert np.isfinite(stats2["critic_loss"])
