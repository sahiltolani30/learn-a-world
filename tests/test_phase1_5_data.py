"""Focused tests for Phase 1.5 shard discovery and deduplication."""

import shutil

import numpy as np

from training.phase1_5_data import discover_active_shards
from training.sequence_buffer import EpisodeBuffer, NUM_ACHIEVEMENTS


def _buffer(length=20, labeled=False, action_value=0):
    buf = EpisodeBuffer(capacity_transitions=100)
    events = np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8)
    events[3, 0] = 1
    kwargs = {}
    if labeled:
        kwargs["state_delta"] = np.ones((length, 16), dtype=np.float32)
        kwargs["state_delta_valid"] = np.ones(length, dtype=np.bool_)
    buf.add_episode(
        obs=np.zeros((length, 64, 64, 3), dtype=np.uint8),
        actions=np.full(length, action_value, dtype=np.int64),
        rewards=np.zeros(length, dtype=np.float32),
        dones=np.array([False] * (length - 1) + [True]),
        events=events,
        mask_before=np.zeros_like(events),
        source="test",
        **kwargs,
    )
    return buf


def test_phase1_5_prefers_final_and_labeled_duplicate(tmp_path):
    numbered = tmp_path / "shard_structured_random_0000.npz"
    final = tmp_path / "shard_structured_random_final.npz"
    unlabeled_copy = tmp_path / "combined_shards" / "shard_curiosity_v3_t01_0000.npz"
    labeled_copy = tmp_path / "phase1_transitions" / "shard_curiosity_v3_t01_0000.npz"
    unlabeled_copy.parent.mkdir()
    labeled_copy.parent.mkdir()

    _buffer(length=10).save_shard(numbered)
    _buffer(length=20, action_value=1).save_shard(final)
    _buffer(length=20).save_shard(unlabeled_copy)
    _buffer(length=20, labeled=True).save_shard(labeled_copy)

    manifest = discover_active_shards([tmp_path])
    selected = {item["filename"]: item for item in manifest["selected_shards"]}

    assert "shard_structured_random_final.npz" in selected
    assert "shard_structured_random_0000.npz" not in selected
    assert selected["shard_curiosity_v3_t01_0000.npz"]["has_state_delta"]
    assert manifest["totals"]["valid_state_delta_transitions"] == 20
    assert manifest["totals"]["transitions"] == 40


def test_phase1_5_collapses_byte_identical_copies(tmp_path):
    source = tmp_path / "a" / "shard_x_0000.npz"
    duplicate = tmp_path / "b" / "shard_x_0000.npz"
    source.parent.mkdir()
    duplicate.parent.mkdir()
    _buffer().save_shard(source)
    shutil.copy2(source, duplicate)

    manifest = discover_active_shards([tmp_path])
    assert len(manifest["selected_shards"]) == 1
    assert manifest["totals"]["transitions"] == 20
    assert any(item["reason"] == "duplicate_transition_content" for item in manifest["excluded_shards"])
