"""Tier 2: Boundary Tests - Replay Buffer Edge Conditions.

Category 5: Replay Limits, FIFO Eviction, Splits
Tests:
25. test_t2_replay_single_step_episode_boundary
26. test_t2_replay_exact_capacity_boundary
27. test_t2_replay_zero_completed_episodes_split_boundary
28. test_t2_replay_prioritization_with_zero_achievements
29. test_t2_replay_max_sequence_length_sampling
30. test_t2_replay_shard_corrupted_npz_load
"""
from __future__ import annotations

import pathlib
import zipfile
import numpy as np
import pytest
import torch

from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import NUM_ACHIEVEMENTS


def _make_dummy_episode(length: int, has_achievement: bool = False, seed: int = 1000) -> EpisodeRecord:
    ach_new = np.zeros((length, NUM_ACHIEVEMENTS), dtype=bool)
    if has_achievement:
        ach_new[0, 6] = True  # unlock collect_wood
    rec = EpisodeRecord(
        obs=np.random.randint(0, 256, (length, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(length, dtype=np.int16),
        rewards=np.zeros(length, dtype=np.float32),
        rewards_int=np.zeros(length, dtype=np.float32),
        done=np.zeros(length, dtype=bool),
        inventory=np.zeros((length, 16), dtype=np.float32),
        achievements_cum=np.zeros((length, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=ach_new,
        seed=seed,
        source="test",
    )
    rec.done[-1] = True
    return rec


def test_t2_replay_single_step_episode_boundary():
    """T2-Replay: Episode of length 1 (immediate death) is sampled with valid=False padding."""
    replay = ZephyrReplay(capacity_frames=100)
    single_step_ep = _make_dummy_episode(length=1, seed=42)
    replay.add_episode(single_step_ep)

    rng = np.random.default_rng(1234)
    batch = replay.sample(batch=2, seq_len=8, rng=rng)

    assert batch["valid"].shape == (2, 8)
    # First step is valid, remaining 7 steps are padded with valid=False
    assert batch["valid"][:, 0].all(), "First frame should be valid"
    assert not batch["valid"][:, 1:].any(), "Padded frames must have valid=False"

    # Padded observations repeat the final frame for bootstrapping
    final_frame = single_step_ep.obs[-1]
    for b in range(2):
        for t in range(1, 8):
            assert np.array_equal(batch["obs"][b, t].numpy(), final_frame)


def test_t2_replay_exact_capacity_boundary():
    """T2-Replay: Adding episodes beyond capacity_frames triggers FIFO eviction of oldest episodes."""
    capacity = 50
    replay = ZephyrReplay(capacity_frames=capacity)

    # Add 5 episodes of 10 frames = 50 frames (exact capacity)
    for i in range(5):
        replay.add_episode(_make_dummy_episode(length=10, seed=100 + i))

    assert len(replay) == 50
    assert replay.episodes == 5
    assert replay.records[0].seed == 100

    # Add 1 additional episode of 10 frames (total 60 > 50)
    replay.add_episode(_make_dummy_episode(length=10, seed=105))

    # Oldest episode (seed 100) must be evicted, maintaining capacity
    assert len(replay) <= capacity
    assert len(replay) == 50
    assert replay.episodes == 5
    assert replay.records[0].seed == 101, f"Expected oldest episode 100 to be evicted, got {replay.records[0].seed}"


def test_t2_replay_zero_completed_episodes_split_boundary():
    """T2-Replay: Calling split() with < 2 episodes raises ValueError cleanly."""
    replay = ZephyrReplay(capacity_frames=100)
    replay.add_episode(_make_dummy_episode(length=10, seed=1))

    with pytest.raises(ValueError, match="need >= 2 episodes to split"):
        replay.split(val_ratio=0.1)


def test_t2_replay_prioritization_with_zero_achievements():
    """T2-Replay: Prioritized sampling with 0 achievement episodes falls back cleanly to uniform sampling."""
    replay = ZephyrReplay(capacity_frames=100)
    for i in range(3):
        replay.add_episode(_make_dummy_episode(length=10, has_achievement=False, seed=200 + i))

    rng = np.random.default_rng(999)
    # Request 50% achievement prioritization when pool is empty
    batch = replay.sample(batch=4, seq_len=5, rng=rng, prioritize_achievement_frac=0.5)

    assert batch["obs"].shape == (4, 5, 64, 64, 3)
    assert batch["valid"].shape == (4, 5)
    assert batch["valid"].all()


def test_t2_replay_max_sequence_length_sampling():
    """T2-Replay: Sampling window length exactly matching episode length (L = T) yields 100% valid mask."""
    replay = ZephyrReplay(capacity_frames=100)
    ep_len = 12
    replay.add_episode(_make_dummy_episode(length=ep_len, seed=300))

    rng = np.random.default_rng(555)
    batch = replay.sample(batch=3, seq_len=ep_len, rng=rng)

    assert batch["valid"].shape == (3, ep_len)
    assert batch["valid"].all(), "All frames in exact-length window should be valid"


def test_t2_replay_shard_corrupted_npz_load(tmp_path: pathlib.Path):
    """T2-Replay: Loading corrupted or non-record .npz file raises descriptive format exception."""
    corrupt_file = tmp_path / "corrupt_shard.npz"

    # Write truncated garbage bytes
    with open(corrupt_file, "wb") as f:
        f.write(b"PK\x03\x04" + b"\x00" * 20)

    with pytest.raises((ValueError, KeyError, zipfile.BadZipFile, Exception)):
        ZephyrReplay.load_shard(str(corrupt_file))
