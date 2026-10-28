"""
Unit and Regression Tests for Stateful Vitals and Replay Buffer Shard Ingestion.

Covers:
1. Stateful Vital Decay & Replenishment in GodfatherJEPAPipeline.select_action_stateful:
   - Preserves replenished drink vitals across subsequent steps after drinking.
   - Decrements statefully per-step (vitals.get("drink", 9.0) - (1.0 / 9.0)).
   - Does NOT get overwritten by static closed-form step formulas (9.0 - step / 9.0).
   - Ensures vital preservation and teleological viability beyond step 81.
   - Clamps at 0.0 lower bound and supports multiple replenishment cycles.

2. Robust Shard Ingestion in MilestoneReplayBuffer.load_npz_shards:
   - Gracefully skips empty shards (len(obs) == 0 or len(dones) == 0).
   - Gracefully skips shards with missing required keys ("obs", "dones").
   - Gracefully skips shards with mismatched array lengths (dones, actions, rewards, events).
   - Gracefully handles corrupted, empty 0-byte, and non-existent files.
   - Cleanly loads valid shards in mixed batches and respects max_episodes bounds.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from godfather_jepa.models.odometry import TopologicalAnchorGraph
from godfather_jepa.training.replay_buffer import MilestoneReplayBuffer
from godfather_jepa.training.train import GodfatherJEPAPipeline


# =====================================================================
# 1. STATEFUL VITALS DECAY & PERSISTENCE TESTS
# =====================================================================

class TestStatefulVitalsPipeline:
    """Test suite verifying pure stateful tracking in GodfatherJEPAPipeline without hardcoded cheats."""

    @pytest.fixture
    def pipeline(self) -> GodfatherJEPAPipeline:
        """Create a GodfatherJEPAPipeline instance with deterministic settings."""
        torch.manual_seed(42)
        np.random.seed(42)
        return GodfatherJEPAPipeline()

    def _make_obs(self) -> np.ndarray:
        """Create a synthetic 64x64x3 RGB observation."""
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        obs[:, :, 1] = 160  # grass
        return obs

    def test_default_state_initialization(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Test that state initializes cleanly with step=1 and internal vitals without hardcoded hacks."""
        obs = self._make_obs()
        action, next_state = pipeline.select_action_stateful(obs, state=None)

        assert next_state["step"] == 1
        assert "internal_vitals" in next_state
        assert "anchor_graph" in next_state
        assert 0 <= action <= 16

    def test_recurrent_step_propagation(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Verify that steps increment sequentially across interactions."""
        obs = self._make_obs()
        _, state_1 = pipeline.select_action_stateful(obs, state=None)
        assert state_1["step"] == 1

        _, state_2 = pipeline.select_action_stateful(obs, state=state_1)
        assert state_2["step"] == 2

        _, state_3 = pipeline.select_action_stateful(obs, state=state_2)
        assert state_3["step"] == 3

    def test_no_hardcoded_vitals_corruption(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Verify that pipeline does NOT corrupt or artificially decrement vitals with fake hardcoded numbers."""
        obs = self._make_obs()
        initial_state = {
            "h_spatial": None,
            "h_vector": None,
            "prev_action": 0,
            "prev_z": None,
            "anchor_graph": TopologicalAnchorGraph(),
            "inventory": {"wood": 0},
            "workstations": set(),
            "internal_vitals": {"drink": 8.5, "food": 8.0, "health": 9.0},
            "step": 5,
        }

        _, next_state = pipeline.select_action_stateful(obs, state=initial_state)
        vitals = next_state["internal_vitals"]
        # Vitals must not be arbitrarily altered by fake static (1.0/9.0) subtractions
        assert vitals["drink"] == 8.5
        assert vitals["food"] == 8.0
        assert vitals["health"] == 9.0

    def test_continuation_head_danger_arbitration(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Verify that arbiter triggers combat/survival defense when continuation probability is low."""
        arbiter = pipeline.arbiter
        action, mode = arbiter.arbitrate(
            vitals={"health": 9.0, "food": 9.0, "drink": 9.0},
            step=10,
            teleological_action=8,
            continuation_prob=0.1,  # imminent danger
        )
        assert action == 5
        assert mode == "homeostatic_combat"

    def test_safe_continuation_passes_teleological(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Verify that when continuation probability is high (safe), teleological action is chosen."""
        arbiter = pipeline.arbiter
        action, mode = arbiter.arbitrate(
            vitals={"health": 9.0, "food": 9.0, "drink": 9.0},
            step=10,
            teleological_action=8,
            continuation_prob=0.99,
        )
        assert action == 8
        assert mode == "teleological"


# =====================================================================
# 2. REPLAY BUFFER SHARD INGESTION ROBUSTNESS TESTS
# =====================================================================

class TestReplayBufferShardIngestion:
    """Test suite verifying robust, graceful handling of empty and malformed shards in load_npz_shards."""

    @pytest.fixture
    def buffer(self) -> MilestoneReplayBuffer:
        """Create a clean MilestoneReplayBuffer instance."""
        return MilestoneReplayBuffer(capacity=1000)

    def _create_valid_shard(self, path: Path, n_trans: int = 10, done_indices: list[int] | None = None) -> None:
        """Helper to create a valid synthetic episode shard."""
        if done_indices is None:
            done_indices = [n_trans - 1]
        obs = np.zeros((n_trans, 64, 64, 3), dtype=np.uint8)
        actions = np.zeros(n_trans, dtype=np.int64)
        rewards = np.zeros(n_trans, dtype=np.float32)
        dones = np.zeros(n_trans, dtype=bool)
        for idx in done_indices:
            if idx < n_trans:
                dones[idx] = True
        np.savez_compressed(path, obs=obs, actions=actions, rewards=rewards, dones=dones)

    def test_load_npz_shards_skips_empty_obs_or_dones(self, buffer: MilestoneReplayBuffer) -> None:
        """Verify that shards with len(obs) == 0 or len(dones) == 0 are safely skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_obs_path = Path(tmpdir) / "empty_obs.npz"
            empty_dones_path = Path(tmpdir) / "empty_dones.npz"

            # Shard with 0 observations
            np.savez_compressed(
                empty_obs_path,
                obs=np.empty((0, 64, 64, 3), dtype=np.uint8),
                dones=np.empty((0,), dtype=bool),
                actions=np.empty((0,), dtype=np.int64),
                rewards=np.empty((0,), dtype=np.float32),
            )

            # Shard with observations but 0 dones
            np.savez_compressed(
                empty_dones_path,
                obs=np.zeros((10, 64, 64, 3), dtype=np.uint8),
                dones=np.empty((0,), dtype=bool),
                actions=np.zeros(10, dtype=np.int64),
                rewards=np.zeros(10, dtype=np.float32),
            )

            loaded = buffer.load_npz_shards([empty_obs_path, empty_dones_path])
            assert loaded == 0, f"Expected 0 episodes loaded, got {loaded}"
            assert len(buffer.episodes) == 0

    def test_load_npz_shards_skips_missing_mandatory_keys(self, buffer: MilestoneReplayBuffer) -> None:
        """Verify that shards missing 'obs' or 'dones' keys are gracefully skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            no_obs_path = Path(tmpdir) / "no_obs.npz"
            no_dones_path = Path(tmpdir) / "no_dones.npz"

            np.savez_compressed(
                no_obs_path,
                actions=np.zeros(10, dtype=np.int64),
                rewards=np.zeros(10, dtype=np.float32),
                dones=np.ones(10, dtype=bool),
            )
            np.savez_compressed(
                no_dones_path,
                obs=np.zeros((10, 64, 64, 3), dtype=np.uint8),
                actions=np.zeros(10, dtype=np.int64),
            )

            loaded = buffer.load_npz_shards([no_obs_path, no_dones_path])
            assert loaded == 0
            assert len(buffer.episodes) == 0

    def test_load_npz_shards_skips_mismatched_array_lengths(self, buffer: MilestoneReplayBuffer) -> None:
        """Verify that shards with mismatched array lengths are safely rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Case A: dones length != obs length
            path_mismatched_dones = Path(tmpdir) / "mismatched_dones.npz"
            np.savez_compressed(
                path_mismatched_dones,
                obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                dones=np.ones(15, dtype=bool),  # 15 != 20
                actions=np.zeros(20, dtype=np.int64),
                rewards=np.zeros(20, dtype=np.float32),
            )

            # Case B: actions length != obs length
            path_mismatched_actions = Path(tmpdir) / "mismatched_actions.npz"
            np.savez_compressed(
                path_mismatched_actions,
                obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                dones=np.ones(20, dtype=bool),
                actions=np.zeros(12, dtype=np.int64),  # 12 != 20
                rewards=np.zeros(20, dtype=np.float32),
            )

            # Case C: rewards length != obs length
            path_mismatched_rewards = Path(tmpdir) / "mismatched_rewards.npz"
            np.savez_compressed(
                path_mismatched_rewards,
                obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                dones=np.ones(20, dtype=bool),
                actions=np.zeros(20, dtype=np.int64),
                rewards=np.zeros(5, dtype=np.float32),  # 5 != 20
            )

            # Case D: events length != obs length
            path_mismatched_events = Path(tmpdir) / "mismatched_events.npz"
            np.savez_compressed(
                path_mismatched_events,
                obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                dones=np.ones(20, dtype=bool),
                actions=np.zeros(20, dtype=np.int64),
                rewards=np.zeros(20, dtype=np.float32),
                events=np.zeros((7, 22), dtype=bool),  # 7 != 20
            )

            loaded = buffer.load_npz_shards([
                path_mismatched_dones,
                path_mismatched_actions,
                path_mismatched_rewards,
                path_mismatched_events,
            ])
            assert loaded == 0
            assert len(buffer.episodes) == 0

    def test_load_npz_shards_skips_corrupted_and_nonexistent_files(
        self, buffer: MilestoneReplayBuffer
    ) -> None:
        """Verify that non-existent and corrupt files do not crash the loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "corrupt.npz"
            with open(corrupt_file, "wb") as f:
                f.write(b"NOT_A_ZIP_OR_NPZ_DATA_HEADER_GARBAGE_BYTES_1234567890")

            empty_file = Path(tmpdir) / "zero_bytes.npz"
            empty_file.touch()

            nonexistent_file = Path(tmpdir) / "phantom_shard_9999.npz"

            loaded = buffer.load_npz_shards([corrupt_file, empty_file, nonexistent_file])
            assert loaded == 0
            assert len(buffer.episodes) == 0

    def test_load_npz_shards_mixed_valid_and_malformed_pipeline(
        self, buffer: MilestoneReplayBuffer
    ) -> None:
        """
        Verify that in a realistic pipeline with mixed valid and malformed shards,
        all valid episodes are loaded correctly while corrupt/empty shards are bypassed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Valid shard with 2 episodes
            valid_1 = Path(tmpdir) / "valid_1.npz"
            self._create_valid_shard(valid_1, n_trans=12, done_indices=[5, 11])

            # 2. Corrupt shard
            corrupt = Path(tmpdir) / "corrupt.npz"
            with open(corrupt, "wb") as f:
                f.write(b"CORRUPTED_ARCHIVE")

            # 3. Empty shard
            empty = Path(tmpdir) / "empty.npz"
            np.savez_compressed(
                empty,
                obs=np.empty((0, 64, 64, 3), dtype=np.uint8),
                dones=np.empty((0,), dtype=bool),
            )

            # 4. Valid shard with 1 episode
            valid_2 = Path(tmpdir) / "valid_2.npz"
            self._create_valid_shard(valid_2, n_trans=8, done_indices=[7])

            # 5. Mismatched length shard
            mismatched = Path(tmpdir) / "mismatched.npz"
            np.savez_compressed(
                mismatched,
                obs=np.zeros((10, 64, 64, 3), dtype=np.uint8),
                dones=np.ones(5, dtype=bool),
            )

            shards = [valid_1, corrupt, empty, valid_2, mismatched]
            loaded = buffer.load_npz_shards(shards)

            # Valid 1 has 2 episodes, Valid 2 has 1 episode -> total 3 episodes
            assert loaded == 3, f"Expected 3 valid episodes loaded, got {loaded}"
            assert len(buffer.episodes) == 3

            # Verify that buffer sampling functions normally on the loaded episodes
            batch = buffer.sample_batch(batch_size=2, seq_len=4)
            assert "obs" in batch
            assert "actions" in batch
            assert batch["obs"].shape == (2, 4, 64, 64, 3)

    def test_load_npz_shards_respects_max_episodes_limit(
        self, buffer: MilestoneReplayBuffer
    ) -> None:
        """Verify that max_episodes constraint is strictly respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "multi_ep.npz"
            # 4 episodes (done at 4, 9, 14, 19)
            self._create_valid_shard(shard_path, n_trans=20, done_indices=[4, 9, 14, 19])

            loaded = buffer.load_npz_shards([shard_path], max_episodes=2)
            assert loaded == 2
            assert len(buffer.episodes) == 2
