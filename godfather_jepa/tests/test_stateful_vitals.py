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
    """Test suite verifying stateful vital tracking in GodfatherJEPAPipeline."""

    @pytest.fixture
    def pipeline(self) -> GodfatherJEPAPipeline:
        """Create a GodfatherJEPAPipeline instance with deterministic settings."""
        torch.manual_seed(42)
        np.random.seed(42)
        return GodfatherJEPAPipeline()

    def _make_water_obs(self) -> np.ndarray:
        """Create a synthetic 64x64x3 RGB observation with water at center-bottom."""
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        # Arbiter water detection looks at obs[30:34, 30:34]: blue > 140, red < 80
        obs[30:34, 30:34, 0] = 20   # low red
        obs[30:34, 30:34, 1] = 60   # green
        obs[30:34, 30:34, 2] = 220  # high blue (water)
        return obs

    def _make_land_obs(self) -> np.ndarray:
        """Create a synthetic 64x64x3 RGB observation representing dry land."""
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        # Grass / land: green dominance, low blue
        obs[:, :, 0] = 50
        obs[:, :, 1] = 160
        obs[:, :, 2] = 40
        return obs

    def test_default_state_initial_decrement(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Test that vitals start at 9.0 and decrement statefully on land."""
        land_obs = self._make_land_obs()
        action, next_state = pipeline.select_action_stateful(land_obs, state=None)

        vitals = next_state["internal_vitals"]
        expected_drink = max(0.0, 9.0 - (1.0 / 9.0))
        expected_food = max(0.0, 9.0 - (1.0 / 25.0))

        assert pytest.approx(vitals["drink"], rel=1e-5) == expected_drink
        assert pytest.approx(vitals["food"], rel=1e-5) == expected_food
        assert next_state["step"] == 1

        # Second step on land: should decrement again from stateful values
        action2, next_state2 = pipeline.select_action_stateful(land_obs, state=next_state)
        vitals2 = next_state2["internal_vitals"]
        assert pytest.approx(vitals2["drink"], rel=1e-5) == max(0.0, expected_drink - (1.0 / 9.0))
        assert pytest.approx(vitals2["food"], rel=1e-5) == max(0.0, expected_food - (1.0 / 25.0))
        assert next_state2["step"] == 2

    def test_drinking_replenishes_and_persists_across_subsequent_steps(
        self, pipeline: GodfatherJEPAPipeline
    ) -> None:
        """
        Verify that drinking replenishes drink to 9.0 and that subsequent steps on land
        decrement from 9.0 statefully rather than being overwritten by static closed-form step formulas.
        """
        water_obs = self._make_water_obs()
        land_obs = self._make_land_obs()

        # Step 60: Agent is thirsty (drink=1.0 <= 3.0) and stands next to water
        initial_state = {
            "h_spatial": None,
            "h_vector": None,
            "prev_action": 0,
            "prev_z": None,
            "anchor_graph": TopologicalAnchorGraph(),
            "inventory": {"wood": 0, "stone": 0, "wood_pickaxe": 0},
            "workstations": set(),
            "internal_vitals": {"drink": 1.0, "food": 9.0, "health": 9.0},
            "step": 60,
        }

        # Step 60: Agent drinks water
        act_60, state_61 = pipeline.select_action_stateful(water_obs, state=initial_state)
        assert act_60 == 5, f"Expected action 5 (drink), got {act_60}"
        assert state_61["internal_vitals"]["drink"] == 9.0, "Drink should be replenished to 9.0"

        # Step 61: Agent moves to land
        # In the old buggy implementation: 9.0 - (61 / 9.0) = 2.222 (immediately depleted!)
        # In the stateful fix: 9.0 - (1.0 / 9.0) = 8.8888...
        act_61, state_62 = pipeline.select_action_stateful(land_obs, state=state_61)
        drink_61 = state_62["internal_vitals"]["drink"]
        assert drink_61 > 8.0, (
            f"Drink vital at step 61 was overwritten! Expected ~8.889, got {drink_61}"
        )
        assert pytest.approx(drink_61, rel=1e-4) == (9.0 - (1.0 / 9.0))

        # Steps 62-75: Agent continues traversing land for 14 steps
        curr_state = state_62
        for s in range(62, 76):
            act_s, curr_state = pipeline.select_action_stateful(land_obs, state=curr_state)
            drink_val = curr_state["internal_vitals"]["drink"]
            expected_drink_s = 9.0 - ((s - 60) * (1.0 / 9.0))
            assert pytest.approx(drink_val, rel=1e-4) == expected_drink_s
            assert drink_val > 5.0, (
                f"Vital decayed prematurely at step {s}: drink={drink_val}"
            )

    def test_vital_preservation_beyond_step_81(self, pipeline: GodfatherJEPAPipeline) -> None:
        """
        In the old buggy implementation, step >= 81 resulted in max(0, 9 - 81/9) = 0.0 permanently.
        Verify that with stateful decrement, replenishing at step >= 81 maintains healthy vitals.
        """
        water_obs = self._make_water_obs()
        land_obs = self._make_land_obs()

        # Step 90: Beyond the step 81 wall
        late_state = {
            "h_spatial": None,
            "h_vector": None,
            "prev_action": 0,
            "prev_z": None,
            "anchor_graph": TopologicalAnchorGraph(),
            "inventory": {"wood": 0, "stone": 0, "wood_pickaxe": 0},
            "workstations": set(),
            "internal_vitals": {"drink": 0.5, "food": 5.0, "health": 9.0},
            "step": 90,
        }

        # Step 90: Drink at water
        act_90, state_91 = pipeline.select_action_stateful(water_obs, state=late_state)
        assert act_90 == 5
        assert state_91["internal_vitals"]["drink"] == 9.0

        # Step 91: Next step on land
        act_91, state_92 = pipeline.select_action_stateful(land_obs, state=state_91)
        drink_91 = state_92["internal_vitals"]["drink"]
        assert drink_91 > 8.0, (
            f"Beyond step 81, drink collapsed to {drink_91} instead of ~8.889!"
        )
        assert pytest.approx(drink_91, rel=1e-4) == (9.0 - (1.0 / 9.0))

        # Step 92: Another step on land
        act_92, state_93 = pipeline.select_action_stateful(land_obs, state=state_92)
        drink_92 = state_93["internal_vitals"]["drink"]
        assert pytest.approx(drink_92, rel=1e-4) == (9.0 - 2.0 * (1.0 / 9.0))

    def test_vital_lower_bound_clamping(self, pipeline: GodfatherJEPAPipeline) -> None:
        """Verify that vitals clamp cleanly at 0.0 and never become negative."""
        land_obs = self._make_land_obs()
        near_zero_state = {
            "h_spatial": None,
            "h_vector": None,
            "prev_action": 0,
            "prev_z": None,
            "anchor_graph": TopologicalAnchorGraph(),
            "inventory": {"wood": 0, "stone": 0, "wood_pickaxe": 0},
            "workstations": set(),
            "internal_vitals": {"drink": 0.05, "food": 0.01, "health": 9.0},
            "step": 20,
        }

        _, next_state = pipeline.select_action_stateful(land_obs, state=near_zero_state)
        vitals = next_state["internal_vitals"]
        assert vitals["drink"] == 0.0, f"Expected drink clamped at 0.0, got {vitals['drink']}"
        assert vitals["food"] == 0.0, f"Expected food clamped at 0.0, got {vitals['food']}"

    def test_cyclic_drinking_and_land_traversal(self, pipeline: GodfatherJEPAPipeline) -> None:
        """
        Verify repeated cycles of drinking and land exploration over multiple episode steps:
        when thirst decays below threshold (<= 3.0), encountering water triggers drinking
        and fully restores drink vital to 9.0.
        """
        water_obs = self._make_water_obs()
        land_obs = self._make_land_obs()

        state = {
            "h_spatial": None,
            "h_vector": None,
            "prev_action": 0,
            "prev_z": None,
            "anchor_graph": TopologicalAnchorGraph(),
            "inventory": {"wood": 0, "stone": 0, "wood_pickaxe": 0},
            "workstations": set(),
            "internal_vitals": {"drink": 2.0, "food": 9.0, "health": 9.0},
            "step": 10,
        }

        # Cycle 1: Drink (drink <= 3.0 near water)
        act_d1, state = pipeline.select_action_stateful(water_obs, state=state)
        assert act_d1 == 5
        assert state["internal_vitals"]["drink"] == 9.0

        # Cycle 1: 5 steps on land
        for _ in range(5):
            _, state = pipeline.select_action_stateful(land_obs, state=state)
        expected_drink = 9.0 - 5 * (1.0 / 9.0)
        assert pytest.approx(state["internal_vitals"]["drink"], rel=1e-4) == expected_drink

        # Simulate further land journey until thirst drops below threshold
        state["internal_vitals"]["drink"] = 2.5

        # Cycle 2: Drink again when thirsty and near water
        act_d2, state = pipeline.select_action_stateful(water_obs, state=state)
        assert act_d2 == 5
        assert state["internal_vitals"]["drink"] == 9.0

        # Cycle 2: 3 steps on land
        for _ in range(3):
            _, state = pipeline.select_action_stateful(land_obs, state=state)
        assert pytest.approx(state["internal_vitals"]["drink"], rel=1e-4) == (9.0 - 3 * (1.0 / 9.0))


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
